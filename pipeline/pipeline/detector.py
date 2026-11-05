import time
import torch
import torch.nn.functional as F
from yolox.exp import get_exp
from yolox.utils import fuse_model, postprocess


class Predictor:
    def __init__(self, model, exp, device, fp16):
        self.model, self.exp, self.device, self.fp16 = model, exp, device, fp16
        self.dtype = torch.float16 if fp16 else torch.float32
        mean = torch.tensor((0.485, 0.456, 0.406), device=device, dtype=self.dtype).view(1, 3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225), device=device, dtype=self.dtype).view(1, 3, 1, 1)
        self.mean, self.std = mean, std
        self.t = {"pre": 0.0, "fwd": 0.0, "nms": 0.0}      # cộng dồn (giây)
        self.profile = True

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter()

    def _preproc_gpu(self, frame):
        """Tương đương yolox preproc: resize giữ tỉ lệ, pad 114 góc trên-trái, BGR->RGB, /255, chuẩn hoá."""
        h, w = frame.shape[:2]
        th, tw = self.exp.test_size
        r = min(th / h, tw / w)
        nh, nw = int(h * r), int(w * r)
        x = torch.from_numpy(frame).to(self.device, non_blocking=True)      # uint8 HxWx3 (nhỏ hơn float32 ~4 lần)
        x = x.permute(2, 0, 1).unsqueeze(0).to(self.dtype)                  # 1x3xHxW
        x = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)
        canvas = torch.full((1, 3, th, tw), 114.0, device=self.device, dtype=self.dtype)
        canvas[:, :, :nh, :nw] = x
        canvas = canvas.flip(1) / 255.0                                     # BGR -> RGB
        return (canvas - self.mean) / self.std

    @torch.no_grad()
    def __call__(self, frame):
        h, w = frame.shape[:2]
        t0 = self._sync() if self.profile else 0
        img = self._preproc_gpu(frame)
        t1 = self._sync() if self.profile else 0
        out = self.model(img)
        t2 = self._sync() if self.profile else 0
        out = postprocess(out, self.exp.num_classes, self.exp.test_conf, self.exp.nmsthre)
        t3 = self._sync() if self.profile else 0
        if self.profile:
            self.t["pre"] += t1 - t0; self.t["fwd"] += t2 - t1; self.t["nms"] += t3 - t2
        return out[0], (h, w)


def build_detector(cfg):
    use_cuda = cfg.device == "gpu" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True
    exp = get_exp(cfg.exp_file, None)
    if cfg.conf_thresh is not None:
        exp.test_conf = cfg.conf_thresh
    if cfg.test_size is not None:
        exp.test_size = tuple(cfg.test_size)
    model = exp.get_model().to(device).eval()
    try:
        ckpt = torch.load(cfg.ckpt, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(cfg.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if cfg.fuse:
        model = fuse_model(model)
    fp16 = cfg.fp16 and use_cuda
    if fp16:
        model = model.half()
    print(f"[detector] device={device} fp16={fp16} test_size={exp.test_size} "
          f"test_conf={exp.test_conf} nms={exp.nmsthre}")
    pred = Predictor(model, exp, device, fp16)
    dummy = torch.zeros((720, 1280, 3), dtype=torch.uint8).numpy()
    for _ in range(3):
        pred(dummy)
    pred.t = {k: 0.0 for k in pred.t}
    return exp, pred
