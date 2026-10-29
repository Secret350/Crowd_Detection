"""Pipeline Crowd Detection: ByteTrack (MOT20) -> analyzer -> output video/log."""
import sys
from pathlib import Path

_BT = Path(__file__).resolve().parents[1] / "ByteTrack"
if _BT.exists() and str(_BT) not in sys.path:
    sys.path.insert(0, str(_BT))
import sys
from pathlib import Path

import numpy as np

#Do xung đột phiên bản numpy nên cần thêm vòng lặp dưới đây để phù hợp với phiên bản numpy của repo ByteTrack
for _name, _type in (("float", float), ("int", int), ("bool", bool), ("object", object)):
    if not hasattr(np, _name):
        setattr(np, _name, _type)

_BT = Path(__file__).resolve().parents[1] / "ByteTrack"
if _BT.exists() and str(_BT) not in sys.path:
    sys.path.insert(0, str(_BT))