"""Chạy trong thư mục webapp:  python patch_webapp.py   (hoặc: python patch_webapp.py đường/dẫn/main.py)
Sửa main.py tại chỗ, KHÔNG đụng build_cmd của bạn. Có thể chạy lại nhiều lần (bước nào xong rồi thì bỏ qua):
  1) results : nhận thêm heatmap nền trắng / flow / crowd_stats; không mã hoá lại video (pipeline đã ghi H.264)
  2) progress: đọc status.json của pipeline để thanh tiến độ chạy thật"""
import re
import sys
from pathlib import Path

p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("main.py")
s = p.read_text(encoding="utf-8")


def step_results(s):
    new_patterns = '''RESULT_PATTERNS = {
    "heatmap_white_video": ["*heatmap_white*.mp4", "*heat*white*.mp4"],
    "flow_video":          ["*flow*.mp4"],
    "bbox_video":          ["*bbox*.mp4", "*track*.mp4"],
    "heatmap_video":       ["*heat*.mp4", "*density*.mp4"],
    "stats":               ["crowd_stats.json"],
    "log":                 ["*gather*.json", "*gather*.csv", "*gather*.txt", "*.log"],
}'''
    new_collect = '''def collect_results(out_dir: Path) -> dict:
    found, claimed = {}, set()
    for key, patterns in RESULT_PATTERNS.items():
        for pat in patterns:
            hits = [h for h in sorted(out_dir.glob(pat))
                    if not h.name.startswith("web_") and h not in claimed]
            if hits:
                found[key] = hits[0]
                claimed.add(hits[0])
                break
    return found


'''
    s, n1 = re.subn(r"RESULT_PATTERNS = \{.*?\n\}", lambda m: new_patterns, s, count=1, flags=re.S)
    s, n2 = re.subn(r"def collect_results\(out_dir: Path\) -> dict:\n.*?\n\n\n(?=def run_job)",
                    lambda m: new_collect, s, count=1, flags=re.S)
    s, n3 = re.subn(r"(KEEP_HOURS = 24[^\n]*\n)",
                    lambda m: m.group(1) + 'REENCODE = os.environ.get("CROWD_REENCODE", "0") == "1"   '
                                           '# pipeline đã ghi H.264, mặc định không mã hoá lại\n', s, count=1)
    old = "                else:\n                    path = to_h264(path)"
    n4 = s.count(old)
    s = s.replace(old, "                elif REENCODE:\n                    path = to_h264(path)")
    return s, (n1, n2, n3, n4) == (1, 1, 1, 1), f"RESULT_PATTERNS={n1}, collect_results={n2}, KEEP_HOURS={n3}, to_h264={n4}"


def step_progress(s):
    watcher = '''def watch_status(job_id: str, out_dir: Path, stop: threading.Event):
    """Đọc status.json do pipeline ghi (progress 0..1, cập nhật mỗi 10 frame) và đẩy lên job."""
    sf = out_dir / "status.json"
    while not stop.wait(1.0):
        with jobs_lock:
            if jobs[job_id]["status"] != "running":
                return
        try:
            pr = float(json.loads(sf.read_text(encoding="utf-8")).get("progress", 0.0))
        except Exception:  # noqa: BLE001  (file chưa có / đang ghi dở)
            continue
        set_job(job_id, progress=round(min(max(pr, 0.0), 1.0) * 100, 1))


'''
    n1 = s.count("def run_job(")
    s = s.replace("def run_job(", watcher + "def run_job(", 1) if n1 == 1 else s
    a = "        for line in proc.stdout:\n"
    n2 = s.count(a)
    s = s.replace(a, "        stop = threading.Event()\n"
                     "        threading.Thread(target=watch_status, args=(job_id, out_dir, stop), daemon=True).start()\n" + a)
    b = "        if proc.wait() != 0:\n"
    n3 = s.count(b)
    s = s.replace(b, "        rc = proc.wait()\n        stop.set()\n        if rc != 0:\n")
    return s, (n1, n2, n3) == (1, 1, 1), f"run_job={n1}, for-stdout={n2}, proc.wait={n3}"


for name, marker, fn in (("results", "heatmap_white_video", step_results), ("progress", "def watch_status", step_progress)):
    if marker in s:
        print(f"[{name}] đã có, bỏ qua")
        continue
    s2, ok, info = fn(s)
    if not ok:
        sys.exit(f"[{name}] KHÔNG khớp mẫu ({info})"
                 "Chưa có thay đổi nào được ghi.")
    s = s2
    print(f"[{name}] đã patch")
p.write_text(s, encoding="utf-8")
print("Xong:", p)
