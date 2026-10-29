"""Heatmap mật độ từ điểm chân + phát hiện tụ tập bất thường.

Giao diện cố định:
    update(frame, tlwhs, ids, t_sec) -> frame heatmap overlay (BGR)
    finish()                         -> list sự kiện để ghi gathering_log.json

  * Mỗi người = 1 điểm chân (x + w/2, y + h). Mật độ lân cận tại mỗi điểm của lưới:
        L(p) = sum_i exp(-|p - f_i|^2 / (2 sigma^2))
    (= số người "tương đương" quanh p; 1 người đứng riêng lẻ cho L = 1 tại chỗ họ đứng).
  * L được làm mượt theo thời gian (EMA), tính trên lưới thô (frame / heat_scale) cho nhanh.
  * Vùng nóng = L >= gather_min_density. Mỗi vùng được theo dõi qua các frame thành 1 "sự kiện ứng viên".
  * Xác nhận tụ tập khi: số track_id trong vùng >= N, lưu trú trung bình >= T giây,
    và tâm vùng không trôi quá xa (loại nhóm người đi cùng nhau).
"""
import cv2
import numpy as np

DEFAULTS = dict(
    heat_scale=4,                # lưới mật độ = frame / heat_scale
    heat_ema=0.3,                # alpha EMA theo thời gian
    heat_alpha=0.6,              # độ đậm overlay
    gather_sigma_frac=0.04,      # sigma cửa sổ lân cận, theo bề rộng frame
    gather_min_density=4.0,      # ngưỡng L để thành vùng nóng
    gather_min_area_frac=0.0005, # bỏ vùng quá nhỏ (theo diện tích lưới)
    gather_min_ids=5,            # N: số track_id tối thiểu trong vùng
    gather_min_dwell_s=3.0,      # T: lưu trú trung bình tối thiểu (giây)
    gather_gap_s=1.0,            # vùng biến mất quá lâu thì đóng sự kiện
    gather_match_frac=0.12,      # khoảng cách tối đa (theo bề rộng frame) để coi là cùng 1 vùng
    gather_max_drift_frac=0.15,  # tâm vùng trôi quá mức này => nhóm đang di chuyển, không phải tụ tập
)


def _fmt(t):
    return f"{int(t // 60):02d}:{t % 60:05.2f}"


class CrowdAnalyzer:
    enabled = True

    def __init__(self, cfg, size, fps):
        self.w, self.h = size
        self.fps = fps
        self.p = {k: getattr(cfg, k, v) for k, v in DEFAULTS.items()}   # ghi đè được từ Config
        s = self.p["heat_scale"]
        self.wl, self.hl = max(self.w // s, 1), max(self.h // s, 1)
        self.sig_px = self.p["gather_sigma_frac"] * self.w
        self.sig_low = max(self.sig_px / s, 0.8)
        self.L_ema = np.zeros((self.hl, self.wl), np.float32)
        self.active, self.done = [], []
        self._jet = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(256, 1),
                                      cv2.COLORMAP_JET).reshape(256, 3)   # bảng màu dựng 1 lần
        self.fi, self.t_prev = -1, None

    # ------------------------------------------------------------------ main
    def update(self, frame, tlwhs, ids, t_sec):
        self.fi += 1
        dt = (t_sec - self.t_prev) if self.t_prev is not None else 1.0 / self.fps
        self.t_prev = t_sec
        s, p = self.p["heat_scale"], self.p

        feet = np.array([[x + w / 2, y + h] for x, y, w, h in tlwhs], np.float32).reshape(-1, 2)
        ids = [int(i) for i in ids]

        # 1) bản đồ mật độ lân cận + EMA
        imp = np.zeros((self.hl, self.wl), np.float32)
        if len(feet):
            gx = np.clip((feet[:, 0] / s).astype(int), 0, self.wl - 1)
            gy = np.clip((feet[:, 1] / s).astype(int), 0, self.hl - 1)
            np.add.at(imp, (gy, gx), 1.0)
        L = cv2.GaussianBlur(imp, (0, 0), self.sig_low, borderType=cv2.BORDER_CONSTANT)
        L *= 2 * np.pi * self.sig_low ** 2
        a = p["heat_ema"]
        self.L_ema = L if self.fi == 0 else a * L + (1 - a) * self.L_ema

        # 2) vùng nóng -> sự kiện ứng viên
        regions = self._regions(feet, ids)
        match_d = p["gather_match_frac"] * self.w
        free = list(range(len(self.active)))
        for r in regions:
            best, bd = None, match_d
            for ei in free:
                c = self.active[ei]["center"]
                d = float(np.hypot(r["center"][0] - c[0], r["center"][1] - c[1]))
                if d < bd:
                    best, bd = ei, d
            if best is None:
                ev = self._new_event(r, t_sec)
                self.active.append(ev)
            else:
                ev = self.active[best]
                free.remove(best)
            self._update_event(ev, r, t_sec, dt)
        for ev in list(self.active):
            if t_sec - ev["last_seen"] > p["gather_gap_s"]:
                self._close(ev)

        return self._render(frame, t_sec)

    def finish(self):
        for ev in list(self.active):
            self._close(ev)
        out = []
        for i, ev in enumerate(sorted(self.done, key=lambda e: e["start_t"]), 1):
            x0, y0, x1, y1 = ev["peak_bbox"]
            n = ev["max_people"]
            N = self.p["gather_min_ids"]
            out.append({
                "event_id": i,
                "start_sec": round(ev["start_t"], 2), "end_sec": round(ev["last_seen"], 2),
                "start": _fmt(ev["start_t"]), "end": _fmt(ev["last_seen"]),
                "frame_start": ev["start_f"], "frame_end": ev["last_f"],
                "center_xy": [int(ev["peak_center"][0]), int(ev["peak_center"][1])],
                "region_bbox": [int(x0), int(y0), int(x1), int(y1)],
                "people_count": n,
                "peak_density": round(ev["peak_density"], 2),
                "avg_dwell_s": round(float(np.mean(list(ev["dwell"].values()))), 1),
                "track_ids": sorted(ev["dwell"]),
                "severity": "high" if n >= 2 * N else "medium" if n >= 1.5 * N else "low",
            })
        return out

    # --------------------------------------------------------------- helpers
    def _regions(self, feet, ids):
        s, p = self.p["heat_scale"], self.p
        mask = (self.L_ema >= p["gather_min_density"]).astype(np.uint8)
        n, lab, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
        min_area = p["gather_min_area_frac"] * self.wl * self.hl
        m = self.sig_px                                         # nới bbox 1 sigma để gom người ở rìa vùng
        regions = []
        for k in range(1, n):
            x, y, bw, bh, area = stats[k]
            if area < min_area:
                continue
            x0, y0 = max(x * s - m, 0), max(y * s - m, 0)
            x1, y1 = min((x + bw) * s + m, self.w), min((y + bh) * s + m, self.h)
            inside = []
            if len(feet):
                sel = (feet[:, 0] >= x0) & (feet[:, 0] <= x1) & (feet[:, 1] >= y0) & (feet[:, 1] <= y1)
                inside = [ids[i] for i in np.flatnonzero(sel)]
            regions.append({"bbox": [x0, y0, x1, y1], "center": (cents[k][0] * s, cents[k][1] * s),
                            "peak": float(self.L_ema[lab == k].max()), "ids": inside})
        return regions

    def _new_event(self, r, t):
        return {"start_t": t, "start_f": self.fi, "last_seen": t, "last_f": self.fi,
                "center": r["center"], "center0": r["center"], "dwell": {}, "confirmed": False,
                "max_people": 0, "peak_bbox": r["bbox"], "peak_center": r["center"],
                "peak_density": 0.0, "bbox_now": r["bbox"], "ids_now": [], "avg_now": 0.0}

    def _update_event(self, ev, r, t, dt):
        p = self.p
        ev.update(last_seen=t, last_f=self.fi, center=r["center"], bbox_now=r["bbox"], ids_now=r["ids"])
        for i in r["ids"]:
            ev["dwell"][i] = ev["dwell"].get(i, 0.0) + dt
        ev["peak_density"] = max(ev["peak_density"], r["peak"])
        n = len(r["ids"])
        if n > ev["max_people"]:
            ev.update(max_people=n, peak_bbox=r["bbox"], peak_center=r["center"])
        ev["avg_now"] = float(np.mean([ev["dwell"][i] for i in r["ids"]])) if r["ids"] else 0.0
        drift = float(np.hypot(r["center"][0] - ev["center0"][0], r["center"][1] - ev["center0"][1]))
        if (not ev["confirmed"] and n >= p["gather_min_ids"] and ev["avg_now"] >= p["gather_min_dwell_s"]
                and drift <= p["gather_max_drift_frac"] * self.w):
            ev["confirmed"] = True

    def _close(self, ev):
        self.active.remove(ev)
        if ev["confirmed"]:
            self.done.append(ev)

    def _render(self, frame, t_sec):
        thr = self.p["gather_min_density"]
        norm = np.clip(self.L_ema / (thr * 1.5), 0.0, 1.0)                # đỏ khi vượt ngưỡng tụ tập
        size = (self.w, self.h)
        # tô màu + tính alpha trên lưới thô rồi mới phóng to, blend bằng phép uint8 (nhanh hơn float toàn khung ~nhiều lần)
        color = cv2.resize(self._jet[(norm * 255).astype(np.uint8)], size, interpolation=cv2.INTER_LINEAR)
        a8 = (np.clip(norm * 3, 0, 1) * self.p["heat_alpha"] * 255).astype(np.uint8)
        a8 = cv2.resize(a8, size, interpolation=cv2.INTER_LINEAR)
        a3 = cv2.merge([a8, a8, a8])
        out = cv2.add(cv2.multiply(frame, cv2.bitwise_not(a3), scale=1 / 255.0),
                      cv2.multiply(color, a3, scale=1 / 255.0))
        for ev in self.active:
            if ev["last_seen"] != t_sec:
                continue
            x0, y0, x1, y1 = [int(v) for v in ev["bbox_now"]]
            if ev["confirmed"]:
                col, txt = (0, 0, 255), f"GATHERING n={len(ev['ids_now'])} dwell={ev['avg_now']:.0f}s"
            else:
                col, txt = (0, 220, 255), f"candidate n={len(ev['ids_now'])} dwell={ev['avg_now']:.0f}s"
            cv2.rectangle(out, (x0, y0), (x1, y1), col, 2)
            cv2.putText(out, txt, (x0, max(y0 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
        cv2.putText(out, f"t={t_sec:.1f}s", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return out