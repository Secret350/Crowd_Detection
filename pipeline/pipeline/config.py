from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]      # thư mục chứa pipeline/ và ByteTrack/
BT_ROOT = ROOT / "ByteTrack"


@dataclass
class Config:
    # --- detector ---
    exp_file: str = str(BT_ROOT / "exps/example/mot/yolox_x_mix_mot20_ch.py")
    ckpt: str = str(ROOT / "models/bytetrack_x_mot20.pth")
    device: str = "gpu"
    fp16: bool = True
    fuse: bool = True
    conf_thresh: Optional[float] = None
    test_size: Optional[Tuple[int, int]] = None

    # --- ByteTrack (BYTETracker đọc các thuộc tính này trực tiếp từ object cfg) ---
    mot20: bool = True
    track_thresh: float = 0.6
    track_buffer: int = 30
    match_thresh: float = 0.8

    # --- lọc box ---
    min_box_area: float = 100
    aspect_ratio_thresh: float = 1.6
