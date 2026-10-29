import torch
from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, postprocess


class Predictor:
    """Bản gọn của Predictor trong ByteTrack/tools/demo_track.py (không phụ thuộc thư mục tools/)."""

    def __init__(self, model, exp, device, fp16):
        self.model, self.exp, self.device, self.fp16 = model, exp, device, fp16
        self.rgb_means, self.std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    @torch.no_grad()
    def __call__(self, frame):
        h, w = frame.shape[:2]
        img, _ = preproc(frame, self.exp.test_size, self.rgb_means, self.std)
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()
        out = self.model(img)
        out = postprocess(out, self.exp.num_classes, self.exp.test_conf, self.exp.nmsthre)
        return out[0], (h, w)          # out[0]: tensor (N, 7) hoặc None nếu không có detection


def build_detector(cfg):
    use_cuda = cfg.device == "gpu" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    exp = get_exp(cfg.exp_file, None)
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
    return exp, Predictor(model, exp, device, fp16)
