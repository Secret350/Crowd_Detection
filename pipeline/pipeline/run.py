import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from yolox.tracker.byte_tracker import BYTETracker
from yolox.utils.visualize import plot_tracking

from .analyzer import CrowdAnalyzer
from .config import Config
from .detector import build_detector
from .extras import CrowdExtras
from .status import write_status
from .video_io import close_writer, open_reader, open_writer


def process_video(input_path, out_dir, cfg=None, skip=1, on_progress=None):
    """skip = n: chỉ xử lý 1 trong mỗi n frame. Video output giữ nguyên độ dài thực (fps xuất = fps gốc / n)."""
    cfg = cfg or Config()
    skip = max(int(skip), 1)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_status(out, state="running", progress=0.0)

    exp, detect = build_detector(cfg)
    cap, m = open_reader(input_path)
    eff_fps = m["fps"] / skip
    tracker = BYTETracker(cfg, frame_rate=eff_fps)
    analyzer = CrowdAnalyzer(cfg, (m["w"], m["h"]), eff_fps)

    w_bbox = open_writer(out / "output_bbox.mp4", m["w"], m["h"], f"{eff_fps:.3f}")
    w_heat = (open_writer(out / "output_heatmap.mp4", m["w"], m["h"], f"{eff_fps:.3f}")
              if analyzer.enabled else None)
    extras = CrowdExtras(analyzer, (m["w"], m["h"]), eff_fps, out, cfg)   # heatmap nền trắng + flow + đếm người

    cuda = torch is not None and torch.cuda.is_available()
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    tm = {"detect": 0.0, "track": 0.0, "analyze": 0.0, "write": 0.0}
    idx, done, t0, fps_ema = -1, 0, time.time(), 0.0
    try:
        while True:
            if not cap.grab():
                break
            idx += 1
            if idx % skip:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                break
            t_sec = idx / m["fps"]

            t1 = time.perf_counter()
            det, (h, w) = detect(frame)
            sync()
            t2 = time.perf_counter()
            tm["detect"] += t2 - t1

            tlwhs, ids = [], []
            if det is not None:
                for t in tracker.update(det, [h, w], exp.test_size):
                    x, y, bw, bh = t.tlwh
                    if bw * bh > cfg.min_box_area and bw / bh <= cfg.aspect_ratio_thresh:
                        tlwhs.append(t.tlwh)
                        ids.append(t.track_id)
            t3 = time.perf_counter()
            tm["track"] += t3 - t2

            heat = analyzer.update(frame, tlwhs, ids, t_sec)
            t4 = time.perf_counter()
            tm["analyze"] += t4 - t3

            dt = t4 - t1
            fps_ema = 1.0 / dt if fps_ema == 0 else 0.9 * fps_ema + 0.1 / dt
            vis = plot_tracking(frame, tlwhs, ids, frame_id=done + 1, fps=fps_ema)
            w_bbox.stdin.write(np.ascontiguousarray(vis).tobytes())
            if w_heat is not None:
                w_heat.stdin.write(np.ascontiguousarray(heat).tobytes())
            extras.update(tlwhs, ids, t_sec)        # phải đứng sau analyzer.update() của CÙNG frame
            tm["write"] += time.perf_counter() - t4

            done += 1
            if done % 10 == 0:
                p = min(idx / m["total"], 0.99)
                write_status(out, state="running", progress=round(p, 3))
                if on_progress:
                    on_progress(p)
    finally:
        cap.release()
        close_writer(w_bbox)
        if w_heat is not None:
            close_writer(w_heat)
        extras.finish()

    events = analyzer.finish()
    if analyzer.enabled:
        (out / "gathering_log.json").write_text(json.dumps(events, ensure_ascii=False, indent=2))

    secs = time.time() - t0
    video_s = (idx + 1) / m["fps"]
    meta = {
        "frames_read": idx + 1, "frames_processed": done, "skip": skip,
        "seconds": round(secs, 1), "video_seconds": round(video_s, 1),
        "slowdown_x": round(secs / max(video_s, 1e-6), 2),
        "proc_fps": round(done / max(secs, 1e-6), 2),
        "src_fps": m["fps"], "size": [m["w"], m["h"]],
        "cuda": cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "vram_peak_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20) if cuda else None,
        "stage_ms_per_frame": {k: round(v / max(done, 1) * 1000, 1) for k, v in tm.items()},
        "gathering_events": len(events),
        "detect_breakdown_ms": {k: round(v / max(done, 1) * 1000, 1)
                        for k, v in getattr(detect, "t", {}).items()},
    }
    (out / "meta.json").write_text(json.dumps(meta))
    print(json.dumps(meta))
    write_status(out, state="done", progress=1.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--ckpt")
    p.add_argument("--device", choices=["gpu", "cpu"])
    p.add_argument("--skip", type=int, default=1, help="xử lý 1 trong mỗi N frame (vd 3 cho video 30 fps)")
    p.add_argument("--conf", type=float, help="ngưỡng confidence của detector (ByteTrack chỉ cần >= 0.1)")
    p.add_argument("--test-size", type=int, nargs=2, metavar=("H", "W"), help="vd 736 1312")
    a = p.parse_args()
    over = {k: v for k, v in {"ckpt": a.ckpt, "device": a.device, "conf_thresh": a.conf,
                              "test_size": tuple(a.test_size) if a.test_size else None}.items() if v is not None}
    process_video(a.input, a.out, replace(Config(), **over), skip=a.skip)


if __name__ == "__main__":
    main()