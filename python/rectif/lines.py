"""Детектор линий полок и вертикалей: без CV-библиотек и без Hough.

  1. Отклик на горизонтальные кромки: r_h = max(0, |gy| - k|gx|) — давим вертикальные кромки товаров.
  2. Кадр режется на полосы; в каждой — профиль отклика, берутся сильные пики (кромки полок).
  3. Пики -> точки (x полосы, y пика); RANSAC связывает их в прямые y = a x + b.
  4. Отбраковка: линия должна занимать >= половины полос и >= половины ширины кадра.
  5. Аналогично для вертикальных кромок (x = a y + b).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .imageops import box_blur, gradients


@dataclass
class LineFit:
    a: float
    b: float
    angle_deg: float
    support: float
    inliers: int
    coverage: float = 0.0
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))

    def at(self, x):
        return self.a * np.asarray(x, dtype=np.float64) + self.b


def edge_responses(gray, cross_suppression=0.5):
    """Отклики на горизонтальные и вертикальные кромки."""
    gx, gy = gradients(gray)
    ax, ay = np.abs(gx), np.abs(gy)
    return (np.maximum(0.0, ay - cross_suppression * ax),
            np.maximum(0.0, ax - cross_suppression * ay))


def _peaks_1d(prof, min_dist, max_peaks, z_min=3.0):
    """Пики 1D-профиля с субпиксельной параболой; возвращает (позиции, z)."""
    p = box_blur(np.asarray(prof, dtype=np.float32)[None, :], 1)[0]
    if p.size < 3:
        return [], []
    med = float(np.median(p))
    mad = float(np.median(np.abs(p - med))) + 1e-6
    z = (p - med) / mad
    is_max = (z >= np.roll(z, 1)) & (z >= np.roll(z, -1))
    idx = np.nonzero(is_max & (z > z_min))[0]
    if idx.size == 0:
        return [], []
    keep = []
    for i in idx[np.argsort(-z[idx])]:
        if all(abs(int(i) - j) >= min_dist for j in keep):
            keep.append(int(i))
            if len(keep) >= max_peaks:
                break
    keep.sort()
    pos, strg = [], []
    for i in keep:
        d = 0.0
        if 0 < i < len(z) - 1:
            y0, y1, y2 = float(z[i - 1]), float(z[i]), float(z[i + 1])
            den = y0 - 2.0 * y1 + y2
            if abs(den) > 1e-9:
                d = float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5))
        pos.append(float(i) + d)
        strg.append(float(z[i]))
    return pos, strg


def _collect_points(resp, n_strips=24, min_dist=8, max_peaks=6, z_min=3.0):
    """Кандидаты (x, y, вес, номер полосы)."""
    h, w = resp.shape
    edges = np.linspace(0, w, n_strips + 1).astype(int)
    out = []
    for s in range(n_strips):
        x0, x1 = int(edges[s]), int(edges[s + 1])
        if x1 - x0 < 3:
            continue
        prof = resp[:, x0:x1].mean(axis=1)
        pos, strength = _peaks_1d(prof, min_dist, max_peaks, z_min)
        xc = 0.5 * (x0 + x1)
        out.extend((xc, p, st, s) for p, st in zip(pos, strength))
    if not out:
        return np.zeros((0, 4))
    return np.asarray(out, dtype=np.float64)


def _ransac_lines(pts, span, tol=2.0, slope_limit=0.6, iters=1200,
                  max_lines=8, seed=0, min_strip_frac=0.5, min_span_frac=0.5):
    """Прямые y = a x + b по точкам (x, y, вес, strip)."""
    if len(pts) < 4:
        return []
    rng = np.random.default_rng(seed)
    alive = np.ones(len(pts), dtype=bool)
    lines = []
    for _ in range(max_lines):
        idx = np.nonzero(alive)[0]
        if idx.size < 4:
            break
        P = pts[idx]
        n = len(P)
        # все гипотезы сразу: (n x iters) расстояний и один argmax вместо цикла
        i1 = rng.integers(0, n, iters)
        i2 = rng.integers(0, n, iters)
        dx = P[i2, 0] - P[i1, 0]
        ok = (np.abs(dx) > 0.25 * span) & (i1 != i2)
        if not ok.any():
            break
        a_all = (P[i2, 1] - P[i1, 1]) / np.where(ok, dx, 1.0)
        ok = ok & (np.abs(a_all) <= slope_limit)
        a_all = np.where(ok, a_all, 0.0)
        b_all = P[i1, 1] - a_all * P[i1, 0]
        d_all = np.abs(P[:, 1][:, None] - (P[:, 0][:, None] * a_all[None, :] + b_all[None, :]))
        inl_all = (d_all < tol) & ok[None, :]
        scores = inl_all.T.astype(np.float32) @ P[:, 2].astype(np.float32)
        k = int(np.argmax(scores))
        if not ok[k]:
            break
        a, b = float(a_all[k]), float(b_all[k])
        inl = inl_all[:, k].copy()
        for _ in range(3):
            n_in = int(inl.sum())
            if n_in < 2:
                break
            X = np.column_stack([P[inl, 0], np.ones(n_in)])
            W = P[inl, 2]
            coef, *_ = np.linalg.lstsq(X * W[:, None], P[inl, 1] * W, rcond=None)
            a, b = float(coef[0]), float(coef[1])
            inl = np.abs(P[:, 1] - (a * P[:, 0] + b)) < tol

        # отбраковка «случайных» прямых: покрытие полос и ширина
        if int(inl.sum()) < 4:
            break
        covered = np.unique(P[inl, 3])
        cov = len(covered) / float(len(np.unique(pts[:, 3])))
        width = float(P[inl, 0].max() - P[inl, 0].min())
        if cov < min_strip_frac or width < min_span_frac * span:
            alive[idx[inl]] = False
            continue
        lines.append(LineFit(a=a, b=b, angle_deg=float(np.degrees(np.arctan(a))),
                             support=float(P[inl, 2].sum()), inliers=int(inl.sum()),
                             coverage=cov, points=P[inl, :2].copy()))
        alive[idx[inl]] = False
    return lines


def detect_lines(resp, orientation="h", **kw):
    """'h' — полки (y = a x + b, angle_deg = atan a), 'v' — вертикали (x = a y + b)."""
    h, w = resp.shape
    pts_kw = {k: v for k, v in kw.items() if k in ("n_strips", "min_dist", "max_peaks", "z_min")}
    ran_kw = {k: v for k, v in kw.items()
              if k in ("tol", "slope_limit", "max_lines", "seed", "min_strip_frac", "min_span_frac")}
    if orientation == "h":
        return sorted(_ransac_lines(_collect_points(resp, **pts_kw), span=float(w), **ran_kw),
                      key=lambda l: l.b)
    pts = _collect_points(resp.T, **pts_kw)
    if len(pts) == 0:
        return []
    # в транспонированном кадре строка = исходный столбец: модель x = a*y + b
    out = []
    for ln in _ransac_lines(pts, span=float(h), **ran_kw):
        out.append(LineFit(a=ln.a, b=ln.b,
                           angle_deg=float(np.degrees(np.arctan2(1.0, ln.a))),
                           support=ln.support, inliers=ln.inliers, coverage=ln.coverage,
                           points=ln.points[:, ::-1].copy()))
    return sorted(out, key=lambda l: l.b)


def dominant_angle(lines, fallback=0.0):
    """Взвешенный по поддержке угол (устойчиво к одной-двум ошибочным линиям)."""
    if not lines:
        return float(fallback)
    ang = np.array([l.angle_deg for l in lines])
    wgt = np.array([max(l.support, 1e-6) for l in lines])
    ref = float(np.median(ang))
    d = (ang - ref + 90.0) % 180.0 - 90.0
    keep = np.abs(d) < 15.0
    if not keep.any():
        return ref
    return float(ref + float((d[keep] * wgt[keep]).sum() / wgt[keep].sum()))
