"""Пайплайн: анализ кромок -> гомография -> ресемплинг -> кроп."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import geometry as g
from .imageops import resize, smooth, to_gray
from .lines import LineFit, detect_lines, dominant_angle, edge_responses
from .warp import warp_perspective


@dataclass
class Analysis:
    theta_h: float                     # угол кромок полок, градусы
    theta_v: float                     # угол вертикальных кромок, градусы
    rails: list                       # LineFit: y = a x + b (рабочий кадр)
    verticals: list                    # LineFit: x = a y + b
    scale: float                       # work_w / full_w
    work_size: tuple

    @property
    def angle_deviation(self) -> float:
        """Насколько сетка кромок не прямоугольна (0 — идеал)."""
        return float(self.theta_v - self.theta_h - 90.0)


@dataclass
class RectifyResult:
    image: np.ndarray
    H: np.ndarray           # исходный кадр -> выходной кадр
    out_w: int
    out_h: int
    analysis: Analysis
    rails_out: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)


def analyze(rgb, work_width=900, n_strips=24, max_rails=10):
    """Измеряет наклон и положение кромок полок и вертикалей."""
    h, w = rgb.shape[:2]
    scale = min(1.0, work_width / float(w))
    work = resize(rgb, int(round(w * scale)), int(round(h * scale))) if scale < 1.0 else rgb

    gray = smooth(to_gray(work), sigma=1.0)
    r_h, r_v = edge_responses(gray)

    rails = detect_lines(r_h, "h", n_strips=n_strips, min_dist=6, max_peaks=24,
                         tol=2.0, max_lines=max_rails)
    verticals = detect_lines(r_v, "v", n_strips=max(12, n_strips // 2), min_dist=6,
                             max_peaks=24, tol=2.0, max_lines=8)

    if rails:
        s_max = max(l.support for l in rails)
        rails = [l for l in rails if l.support >= 0.18 * s_max]
    if verticals:
        s_max = max(l.support for l in verticals)
        verticals = [l for l in verticals if l.support >= 0.18 * s_max]

    return Analysis(
        theta_h=dominant_angle(rails, 0.0),
        theta_v=dominant_angle(verticals, 90.0),
        rails=rails,
        verticals=verticals,
        scale=scale,
        work_size=(work.shape[1], work.shape[0]),
    )


def rails_in_source(an):
    """Прямые полок в координатах исходного кадра (y = a x + b)."""
    return [(l.a, l.b / an.scale) for l in an.rails]


def rails_out_y(H, rails, w):
    ys = []
    for a, b in rails:
        xs = np.array([0.1 * w, 0.9 * w], dtype=np.float64)
        pts = np.column_stack([xs, a * xs + b])
        ys.append(float(g.apply_h(H, pts)[:, 1].mean()))
    return ys


def rectify(rgb, mode="directions", crop="none", work_width=900, max_side=2400, analysis=None):
    """Ректификация кадра.

    mode: "directions" — по кромкам полок и вертикалям (аффинно);
          "roll"       — только поворот по кромкам полок.
    crop: "none" — весь кадр; "shelf" — обрезать по найденным полкам.
    """
    h, w = rgb.shape[:2]
    an = analysis if analysis is not None else analyze(rgb, work_width=work_width)

    info = {}
    if mode == "vp":
        if len(an.rails) < 2:
            raise ValueError("need at least 2 rail lines for mode='vp'")
        H_lin, info = g.rectifying_homography(
            an.rails, an.verticals, an.work_size[0], an.work_size[1]
        )
    elif mode == "directions":
        H_lin = g.affine_from_directions(an.theta_h, an.theta_v)
    elif mode == "roll":
        H_lin = g.rotation(-an.theta_h)
    else:
        raise ValueError("unknown mode: %s" % mode)

    cx, cy = (an.work_size[0] - 1) * 0.5, (an.work_size[1] - 1) * 0.5
    H_work = g.recenter(H_lin, cx, cy)
    H_src = g.scaling(1.0 / an.scale) @ H_work @ g.scaling(an.scale)
    H, out_w, out_h = g.fit_frame(H_src, w, h, max_side=max_side)

    rails = rails_in_source(an)
    rails_out = rails_out_y(H, rails, w) if rails else []

    cropped = False
    if crop in ("valid", "auto"):
        x0, y0, x1, y1 = g.largest_inscribed_rect(H, w, h, out_w, out_h)
        if x1 - x0 > 64 and y1 - y0 > 64:
            H = g.translation(-x0, -y0) @ H
            out_w, out_h = int(round(x1 - x0)), int(round(y1 - y0))
            rails_out = [y - y0 for y in rails_out]
            cropped = True

    if crop in ("shelf", "auto") and len(rails_out) >= 2:
        ys = sorted(rails_out)
        pitch = float(np.median(np.diff(ys)))
        margin = 0.45 * pitch
        y0 = max(0, int(round(ys[0] - margin)))
        y1 = min(out_h, int(round(ys[-1] + margin)))
        if y1 - y0 > 32:
            H = g.translation(0.0, -y0) @ H
            out_h = y1 - y0
            rails_out = [y - y0 for y in rails_out]
            cropped = True

    img = warp_perspective(rgb, H, out_w, out_h)

    resid = [g.line_angle_after(H, a, b, w, "h") for a, b in rails] if rails else []
    resid = [((r + 90.0) % 180.0) - 90.0 for r in resid]

    rms = float(np.sqrt(np.mean(np.square(resid)))) if resid else None
    spread = float(max(resid) - min(resid)) if resid else None

    meta = {
        "quality": {
            "rails": len(rails),
            "rms_deg": rms,
            "spread_deg": spread,
            "score": float(max(0.0, min(1.0, 1.0 - (rms or 0.0) / 3.0))),
            "ok": bool(len(rails) >= 2 and rms is not None and rms < 1.5),
        },
        "theta_h_deg": an.theta_h,
        "rails_angle_after_deg": resid,
        "rails_angle_rms_deg": float(np.sqrt(np.mean(np.square(resid)))) if resid else None,
        "rails_angle_spread_deg": float(max(resid) - min(resid)) if resid else None,
        "vp_h": (info["vp_h"] / info["vp_h"][2]).tolist() if info else None,
        "vp_v": info["vp_v"].tolist() if info else [0.0, 1.0, 0.0],
        "used_vertical_vp": info.get("used_vertical_vp") if info else None,
        "line_at_infinity": info["line_at_infinity"].tolist() if info else None,
        "theta_v_deg": an.theta_v,
        "angle_deviation_deg": an.angle_deviation,
        "rails": [{"angle_deg": l.angle_deg, "a": l.a, "b": l.b,
                   "support": l.support, "inliers": l.inliers} for l in an.rails],
        "verticals": [{"angle_deg": l.angle_deg, "a": l.a, "b": l.b,
                       "support": l.support, "inliers": l.inliers} for l in an.verticals],
        "rails_in_output": rails_out,
        "mode": mode,
        "crop": crop,
        "cropped": cropped,
        "out_size": [out_w, out_h],
        "source_size": [w, h],
        "H": np.asarray(H, dtype=float).round(9).tolist(),
        "affine": g.decompose_affine(H),
    }
    return RectifyResult(img, H, out_w, out_h, an, rails_out, meta)


def crop_cells(image, rails_out, pad_frac=0.04, max_cells=10):
    """Полосы (ячейки) между соседними кромками уже выпрямленного кадра."""
    h = image.shape[0]
    ys = sorted(float(y) for y in rails_out)
    cells = []
    for y0, y1 in zip(ys[:-1], ys[1:]):
        if y1 - y0 < 0.02 * h:
            continue
        m = pad_frac * (y1 - y0)
        a = max(0, int(round(y0 - m)))
        b = min(h, int(round(y1 + m)))
        if b - a > 8:
            cells.append(image[a:b].copy())
    return cells[:max_cells]
