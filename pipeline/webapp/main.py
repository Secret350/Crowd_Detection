"""
Crowd Detection - web host
Chạy:  uvicorn webapp.main:app --host 0.0.0.0 --port 8000
(chạy từ thư mục gốc E:\\crowd_detection_project, trong venv .venv310)

url: http://localhost:8000
"""
import base64
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

# ----------------------------------------------------------------------------
# CẤU HÌNH  (chỉ cần sửa trong khối này)
# ----------------------------------------------------------------------------
WEB_DIR = Path(__file__).resolve().parent
ROOT = WEB_DIR.parent                      # thư mục gốc chứa `pipeline/` và `ByteTrack/`
UPLOADS = WEB_DIR / "storage" / "uploads"
OUTPUTS = WEB_DIR / "storage" / "outputs"
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_VIDEO_MB = 500
MAX_IMAGE_MB = 30
# Đăng nhập HTTP Basic cho cả team. Đặt biến môi trường trước khi chạy; để trống = KHÔNG khoá.
AUTH_USER = os.environ.get("CROWD_USER", "team")
AUTH_PASS = os.environ.get("CROWD_PASS", "")
KEEP_HOURS = 24                            # tự xoá job cũ hơn số giờ này
REENCODE = os.environ.get("CROWD_REENCODE", "0") == "1"   # pipeline đã ghi H.264, mặc định không mã hoá lại


def build_cmd(video: Path, out_dir: Path) -> list[str]:
    """Lệnh chạy pipeline. ĐIỀU CHỈNH cho khớp CLI thật của pipeline/run.py."""
    return [
        sys.executable, "-m", "pipeline.run",
        "--input", str(video),
        "--out", str(out_dir),
        "--device", os.environ.get("CROWD_DEVICE", "gpu"),
        "--skip", os.environ.get("CROWD_SKIP", "1"),
    ]


# Cách nhận diện file kết quả trong out_dir (glob, không phân biệt vị trí)
RESULT_PATTERNS = {
    "heatmap_white_video": ["*heatmap_white*.mp4", "*heat*white*.mp4"],
    "flow_video":          ["*flow*.mp4"],
    "bbox_video":          ["*bbox*.mp4", "*track*.mp4"],
    "heatmap_video":       ["*heat*.mp4", "*density*.mp4"],
    "stats":               ["crowd_stats.json"],
    "log":                 ["*gather*.json", "*gather*.csv", "*gather*.txt", "*.log"],
}
# ----------------------------------------------------------------------------

for d in (UPLOADS, OUTPUTS):
    d.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Crowd Detection")


@app.middleware("http")
async def basic_auth(request: Request, call_next):
    if not AUTH_PASS:
        return await call_next(request)
    ok = False
    header = request.headers.get("authorization", "")
    if header.startswith("Basic "):
        try:
            user, _, pw = base64.b64decode(header[6:]).decode("utf-8").partition(":")
            ok = (secrets.compare_digest(user.encode(), AUTH_USER.encode())
                  and secrets.compare_digest(pw.encode(), AUTH_PASS.encode()))
        except Exception:  # noqa: BLE001
            ok = False
    if not ok:
        return Response("Cần đăng nhập", status_code=401,
                        headers={"WWW-Authenticate": 'Basic realm="Crowd Detection"'})
    return await call_next(request)

jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()
executor = ThreadPoolExecutor(max_workers=1)   # 1 job/lần -> tránh tranh GPU/CPU, job sau tự xếp hàng

PROGRESS_RE = re.compile(r"(\d+)\s*/\s*(\d+)")  # bắt dòng log dạng "120/900"


def save_job(job_id: str):
    """Ghi trạng thái job ra đĩa (gọi khi đang giữ jobs_lock)."""
    d = OUTPUTS / job_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "job.json").write_text(json.dumps(jobs[job_id], ensure_ascii=False), encoding="utf-8")


def set_job(job_id: str, **kw):
    with jobs_lock:
        jobs[job_id].update(kw)
        save_job(job_id)


def load_jobs():
    """Nạp lại job từ đĩa sau khi server restart."""
    for jf in OUTPUTS.glob("*/job.json"):
        try:
            j = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if j.get("status") in ("queued", "running"):
            j.update(status="error", message="Server đã restart khi job đang chạy, hãy upload lại")
        jobs[j["id"]] = j


load_jobs()
if not AUTH_PASS:
    print("[CẢNH BÁO] CROWD_PASS chưa đặt -> web KHÔNG có mật khẩu.")


def ffmpeg_exe() -> str:
    """Tìm ffmpeg: ưu tiên PATH, không có thì dùng bản đi kèm gói pip imageio-ffmpeg."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise RuntimeError(
            "Không tìm thấy ffmpeg. Chạy `pip install imageio-ffmpeg` trong venv rồi khởi động lại server."
        ) from None


def to_h264(src: Path) -> Path:
    """OpenCV hay ghi mp4v -> trình duyệt không phát được. Chuyển sang H.264 + faststart."""
    dst = src.with_name("web_" + src.name)
    subprocess.run(
        [ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src),
         "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",   # H.264 yêu cầu chiều rộng/cao chẵn
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
         "-movflags", "+faststart", "-an", str(dst)],
        check=True,
    )
    return dst


def image_to_video(img: Path, dst: Path, frames: int = 3, fps: int = 3) -> Path:
    """Pipeline nhận video -> biến ảnh thành clip ngắn (vài khung giống nhau, cạnh chẵn, tối đa 1920 px)."""
    subprocess.run(
        [ffmpeg_exe(), "-y", "-loglevel", "error", "-loop", "1", "-framerate", str(fps),
         "-i", str(img), "-frames:v", str(frames),
         "-vf", "scale=w='min(1920,trunc(iw/2)*2)':h=-2",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(dst)],
        check=True,
    )
    return dst


def last_frame_jpg(video: Path) -> Path:
    """Lấy khung cuối của video kết quả làm ảnh kết quả."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    last = None
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        last = fr
    cap.release()
    if last is None:
        raise RuntimeError("Không đọc được khung hình kết quả")
    dst = video.with_name("web_" + video.stem + ".jpg")
    cv2.imwrite(str(dst), last, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return dst


def collect_results(out_dir: Path) -> dict:
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


def watch_status(job_id: str, out_dir: Path, stop: threading.Event):
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


def run_job(job_id: str, src: Path, kind: str):
    out_dir = OUTPUTS / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    set_job(job_id, status="running", message="Đang chạy pipeline...", started=time.time())
    tail: list[str] = []
    try:
        video = src
        if kind == "image":
            video = image_to_video(src, UPLOADS / f"{job_id}_img.mp4")
        proc = subprocess.Popen(
            build_cmd(video, out_dir),
            cwd=ROOT,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"},
        )
        stop = threading.Event()
        threading.Thread(target=watch_status, args=(job_id, out_dir, stop), daemon=True).start()
        for line in proc.stdout:
            line = line.rstrip()
            tail = (tail + [line])[-30:]
            m = PROGRESS_RE.search(line)
            if m and int(m.group(2)) > 0:
                set_job(job_id, progress=round(100 * int(m.group(1)) / int(m.group(2)), 1))
        rc = proc.wait()
        stop.set()
        if rc != 0:
            raise RuntimeError("Pipeline trả về lỗi")

        set_job(job_id, message="Đang chuyển video sang định dạng web...")
        results = collect_results(out_dir)
        if not results:
            raise RuntimeError("Không tìm thấy file kết quả trong thư mục output")

        files = {}
        for key, path in results.items():
            if key.endswith("_video"):
                if kind == "image":
                    path = last_frame_jpg(path)
                    key = key.replace("_video", "_image")
                elif REENCODE:
                    path = to_h264(path)
            files[key] = f"/files/{job_id}/{path.name}"
        set_job(job_id, status="done", progress=100, message="Hoàn tất", files=files)
    except Exception as e:  # noqa: BLE001
        set_job(job_id, status="error", message=f"{type(e).__name__}: {e}", log_tail="\n".join(tail))


def cleanup_old():
    limit = time.time() - KEEP_HOURS * 3600
    for base in (UPLOADS, OUTPUTS):
        for p in base.iterdir():
            if p.stat().st_mtime < limit:
                shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)


@app.post("/api/jobs")
async def create_job(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext in VIDEO_EXT:
        kind = "video"
    elif ext in IMAGE_EXT:
        kind = "image"
    else:
        raise HTTPException(400, f"Chỉ nhận: {', '.join(sorted(VIDEO_EXT | IMAGE_EXT))}")
    cleanup_old()

    job_id = uuid.uuid4().hex[:12]
    dst = UPLOADS / f"{job_id}{ext}"
    size = 0
    limit = (MAX_IMAGE_MB if kind == "image" else MAX_VIDEO_MB) * 1024 * 1024
    with open(dst, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > limit:
                f.close()
                dst.unlink(missing_ok=True)
                raise HTTPException(413, f"File vượt quá {limit // 1048576} MB")
            f.write(chunk)

    with jobs_lock:
        jobs[job_id] = {"id": job_id, "status": "queued", "progress": 0,
                        "message": "Đang xếp hàng...", "files": {},
                        "filename": file.filename, "kind": kind, "created": time.time()}
        save_job(job_id)
    executor.submit(run_job, job_id, dst, kind)
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if job and job["status"] == "queued":
            ahead = sum(1 for j in jobs.values()
                        if j["status"] in ("queued", "running")
                        and j.get("created", 0) < job.get("created", 0))
            job = {**job, "ahead": ahead}
    if not job:
        raise HTTPException(404, "Không có job này (có thể server đã restart hoặc job đã bị xoá)")
    return job


app.mount("/files", StaticFiles(directory=OUTPUTS), name="files")   # hỗ trợ Range -> tua video được


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "static" / "index.html")