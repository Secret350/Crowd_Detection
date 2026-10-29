import json
import os
from pathlib import Path


def write_status(out_dir, **kw):
    """Ghi status.json (worker/API đọc không bao giờ gặp file dở)."""
    out_dir = Path(out_dir)
    tmp = out_dir / "status.tmp"
    tmp.write_text(json.dumps(kw, ensure_ascii=False))
    os.replace(tmp, out_dir / "status.json")
