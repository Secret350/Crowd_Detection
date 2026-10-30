"""Bổ sung cho pipeline: heatmap nền trắng, bản đồ dòng người (flow) và số người theo thời gian.

Gắn vào run.py (3 chỗ):
    from .extras import CrowdExtras                      # hoặc: from pipeline.extras import CrowdExtras
    extras = CrowdExtras(analyzer, (w, h), out_fps, out_dir)   # sau khi tạo analyzer; out_fps = fps đang dùng cho writer chính
    ...
    extras.update(tlwhs, ids, t_sec)    # trong vòng lặp, NGAY SAU analyzer.update(...), cùng tlwhs/ids/t_sec
    ...
    extras.finish()                     # cuối cùng, cạnh chỗ ghi gathering_log.json

Đầu ra trong out_dir:  output_heatmap_white.mp4, output_flow.mp4, crowd_stats.json
"""
import json
import math
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np

from .video_io import close_writer, open_writer

FLOW_DEFAULTS = dict(
    flow_cell_frac=1 / 24,       # cạnh ô lưới vector, theo bề rộng frame
    flow_tau_s=2.0,              # hằng số thời gian suy giảm của trường vận tốc (giây)
    flow_vel_window_s=1.0,       # vận tốc = độ dời trong cửa sổ này / thời gian (chia cho khoảng dài => triệt nhiễu rung bbox)
    flow_vel_min_span_s=0.4,     # track cần có ít nhất khoảng thời gian này trong cửa sổ mới tính vận tốc
    flow_min_speed_frac=0.01,    # (bề rộng frame / giây) dưới ngưỡng này coi là đứng yên
    flow_min_support=1.5,        # trọng số tối thiểu trong ô để vẽ mũi tên
    flow_trail_s=3.0,            # độ dài vệt quỹ đạo (giây)
    flow_ref_speed_frac=0.10,    # tốc độ ứng với mũi tên dài nhất (bề rộng frame / giây)
    flow_max_gap_s=1.0,          # track mất quá lâu rồi xuất hiện lại thì reset vận tốc
)
# toạ độ ảnh: y hướng xuống => góc 90° = "down"
DIRS = ["right", "down-right", "down", "down-left", "left", "up-left", "up", "up-right"]
FONT = cv2.FONT_HERSHEY_SIMPLEX


class FlowAnalyzer:
    """Vận tốc điểm chân từng track -> trường vận tốc theo ô lưới (mũi tên) + vệt quỹ đạo, vẽ trên nền trắng."""

    def __init__(self, cfg, size, fps):
        self.w, self.h = size
        self.p = {k: getattr(cfg, k, v) for k, v in FLOW_DEFAULTS.items()}
        self.cell = max(int(self.w * self.p["flow_cell_frac"]), 12)
        self.gw, self.gh = math.ceil(self.w / self.cell), math.ceil(self.h / self.cell)
        self.F = np.zeros((self.gh, self.gw, 2), np.float32)    # tổng vận tốc (có suy giảm)
        self.W = np.zeros((self.gh, self.gw), np.float32)       # tổng trọng số (có suy giảm)
        self.tracks, self.t_prev = {}, None
        self.min_speed = self.p["flow_min_speed_frac"] * self.w
        self.ref_speed = self.p["flow_ref_speed_frac"] * self.w
        self._lut = self._build_lut()
        self._wheel = self._build_wheel()
        self.stat = dict(moving=0, speed=0.0, dir=None)

    # ------------------------------------------------------------ màu theo hướng
    @staticmethod
    def _build_lut():
        hsv = np.zeros((1, 360, 3), np.uint8)
        hsv[0, :, 0] = (np.arange(360) // 2) % 180
        hsv[0, :, 1], hsv[0, :, 2] = 230, 210
        return [tuple(int(c) for c in px) for px in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0]]

    def _build_wheel(self, r=22):
        img = np.full((2 * r + 1, 2 * r + 1, 3), 255, np.uint8)
        mask = np.zeros((2 * r + 1, 2 * r + 1), np.uint8)
        for a in range(0, 360, 4):
            cv2.ellipse(img, (r, r), (r, r), 0, a, a + 4, self._lut[a], -1, cv2.LINE_AA)
            cv2.ellipse(mask, (r, r), (r, r), 0, a, a + 4, 255, -1)
        return img, mask.astype(bool)

    def _color(self, vx, vy):
        return self._lut[int(math.degrees(math.atan2(vy, vx))) % 360]

    @staticmethod
    def _dir_name(vx, vy):
        ang = math.degrees(math.atan2(vy, vx))
        return DIRS[int(((ang + 22.5) % 360) // 45)]

    def _is_moving(self, tr):
        return tr["ok"] and math.hypot(*tr["v"]) >= self.min_speed

    # ------------------------------------------------------------------ update
    def update(self, tlwhs, ids, t_sec):
        p = self.p
        if self.t_prev is not None and t_sec > self.t_prev:
            k = math.exp(-(t_sec - self.t_prev) / p["flow_tau_s"])
            self.F *= k
            self.W *= k
        self.t_prev = t_sec

        vs = []
        for (x, y, w, h), i in zip(tlwhs, ids):
            i = int(i)
            fx, fy = float(x + w / 2), float(y + h)             # điểm chân, giống analyzer
            tr = self.tracks.get(i)
            if tr is None:
                tr = self.tracks[i] = {"t": t_sec, "v": (0.0, 0.0), "ok": False, "trail": deque()}
            elif t_sec - tr["t"] > p["flow_max_gap_s"]:
                tr["trail"].clear()                              # mất dấu quá lâu -> bắt đầu lại, tránh vận tốc ảo
            tr["t"] = t_sec
            tr["trail"].append((t_sec, fx, fy))
            while tr["trail"] and t_sec - tr["trail"][0][0] > p["flow_trail_s"]:
                tr["trail"].popleft()
            base = next((e for e in tr["trail"] if e[0] >= t_sec - p["flow_vel_window_s"]), None)
            span = t_sec - base[0] if base else 0.0
            if span >= p["flow_vel_min_span_s"]:
                tr["v"], tr["ok"] = ((fx - base[1]) / span, (fy - base[2]) / span), True
            else:
                tr["v"], tr["ok"] = (0.0, 0.0), False
            if self._is_moving(tr):
                gx = min(max(int(fx / self.cell), 0), self.gw - 1)
                gy = min(max(int(fy / self.cell), 0), self.gh - 1)
                self.F[gy, gx] += tr["v"]
                self.W[gy, gx] += 1.0
                vs.append(tr["v"])

        for i in [i for i, tr in self.tracks.items() if t_sec - tr["t"] > p["flow_trail_s"]]:
            del self.tracks[i]                                   # dọn track đã biến mất

        if vs:                                                   # hướng chủ đạo = hướng có nhiều người nhất (không lấy trung bình vector
            names = Counter(self._dir_name(*v) for v in vs)      # vì 2 dòng ngược chiều sẽ triệt tiêu nhau)
            self.stat = dict(moving=len(vs), speed=float(np.mean([math.hypot(*v) for v in vs])),
                             dir=names.most_common(1)[0][0])
        else:
            self.stat = dict(moving=0, speed=0.0, dir=None)

    # ------------------------------------------------------------------ render
    def render(self, t_sec):
        """Frame nền trắng: vệt quỹ đạo + mũi tên vận tốc trung bình theo ô. Gọi ngay sau update() cùng t_sec."""
        img = np.full((self.h, self.w, 3), 255, np.uint8)
        ink = (40, 40, 40)

        for tr in self.tracks.values():
            if tr["t"] != t_sec:                                 # chỉ vẽ track đang hiện diện ở frame này
                continue
            col = self._color(*tr["v"]) if self._is_moving(tr) else (170, 170, 170)
            pts = np.array([[x, y] for _, x, y in tr["trail"]], np.int32)
            if len(pts) >= 2:
                cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, col, 1, cv2.LINE_AA)
            cv2.circle(img, (int(pts[-1][0]), int(pts[-1][1])), 3, col, -1, cv2.LINE_AA)

        ys, xs = np.nonzero(self.W >= self.p["flow_min_support"])
        for gy, gx in zip(ys, xs):
            vx, vy = self.F[gy, gx] / self.W[gy, gx]
            sp = math.hypot(vx, vy)
            if sp < self.min_speed:
                continue
            ux, uy = vx / sp, vy / sp
            length = self.cell * (0.3 + 0.5 * min(sp / self.ref_speed, 1.0))     # tối đa 0.8 ô -> các mũi tên không dính nhau
            cx, cy = (gx + 0.5) * self.cell, (gy + 0.5) * self.cell
            p0 = (int(cx - ux * length / 2), int(cy - uy * length / 2))
            p1 = (int(cx + ux * length / 2), int(cy + uy * length / 2))
            cv2.arrowedLine(img, p0, p1, self._color(ux, uy), 2, cv2.LINE_AA, tipLength=0.45)

        s = self.stat
        cv2.putText(img, f"t={t_sec:.1f}s  moving={s['moving']}  speed={s['speed']:.0f}px/s  dir={s['dir'] or '-'}",
                    (10, 24), FONT, 0.55, ink, 1, cv2.LINE_AA)
        wheel, mask = self._wheel
        d = wheel.shape[0]
        x0, y0 = self.w - d - 14, self.h - d - 26
        if x0 > 0 and y0 > 30:                                   # bánh xe màu = quy ước màu theo hướng
            roi = img[y0:y0 + d, x0:x0 + d]
            roi[mask] = wheel[mask]
            cv2.putText(img, "direction", (x0 - 12, y0 + d + 16), FONT, 0.4, ink, 1, cv2.LINE_AA)
        return img


class CrowdExtras:
    def __init__(self, analyzer, size, fps, out_dir, cfg=None):
        w, h = size
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.an, self.fps = analyzer, float(fps)
        self.flow = FlowAnalyzer(cfg, size, fps)
        self.wr_white = open_writer(out / "output_heatmap_white.mp4", w, h, fps)
        self.wr_flow = open_writer(out / "output_flow.mp4", w, h, fps)
        self.stats_path = out / "crowd_stats.json"
        self.s = {"t": [], "count": [], "moving": [], "speed": [], "dir": []}

    def update(self, tlwhs, ids, t_sec):
        self.flow.update(tlwhs, ids, t_sec)
        st, s = self.flow.stat, self.s
        s["t"].append(round(float(t_sec), 3))
        s["count"].append(len(ids))                              # số track đang hiện trong frame (sau lọc của pipeline)
        s["moving"].append(st["moving"])
        s["speed"].append(round(st["speed"], 1))
        s["dir"].append(st["dir"])
        self.wr_white.stdin.write(np.ascontiguousarray(self.an.render_white(t_sec)).tobytes())
        self.wr_flow.stdin.write(np.ascontiguousarray(self.flow.render(t_sec)).tobytes())

    def finish(self):
        for wr in (self.wr_white, self.wr_flow):
            try:
                close_writer(wr)
            except Exception:                                    # noqa: BLE001
                pass
        s, n = self.s, len(self.s["count"])
        summary = {}
        if n:
            mx = int(np.argmax(s["count"]))
            summary = {"max_count": int(s["count"][mx]), "max_count_t": s["t"][mx],
                       "mean_count": round(float(np.mean(s["count"])), 1), "last_count": int(s["count"][-1])}
        data = {"fps": self.fps, "n_samples": n, **s, "summary": summary}
        self.stats_path.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        return data
