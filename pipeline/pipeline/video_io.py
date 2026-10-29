import shutil
import subprocess

import cv2

def _ffmpeg_exe():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise RuntimeError("Không tìm thấy ffmpeg: cài vào PATH hoặc `pip install imageio-ffmpeg`")

def open_reader(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {path}")
    meta = {
        "w": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "h": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS) or 30.0,
        "total": max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1),   # đôi khi lệch, chỉ dùng cho progress
    }
    return cap, meta


def open_writer(path, w, h, fps):
    """Pipe frame BGR vào ffmpeg -> H.264 yuv420p (+faststart) để trình duyệt phát/tua được."""
    cmd = [_ffmpeg_exe(), "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
           "-an", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",          # yuv420p cần cạnh chẵn
           "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def close_writer(proc):
    proc.stdin.close()
    return proc.wait()
