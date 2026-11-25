"""Собственный ресемплер: обратное отображение, билинейная интерполяция, антиалиасинг."""

from __future__ import annotations

import numpy as np

from .imageops import box_blur


def _bilinear(img: np.ndarray, sx: np.ndarray, sy: np.ndarray) -> np.ndarray:
    """Билинейная выборка через плоские индексы; вне кадра — 0."""
    h, w, c = img.shape
    inside = (sx >= -0.5) & (sx <= w - 0.5) & (sy >= -0.5) & (sy <= h - 0.5)
    x0 = np.floor(sx).astype(np.int32)
    y0 = np.floor(sy).astype(np.int32)
    ax = (sx - x0)[..., None].astype(np.float32)
    ay = (sy - y0)[..., None].astype(np.float32)
    np.clip(x0, 0, w - 1, out=x0)
    np.clip(y0, 0, h - 1, out=y0)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)

    flat = img.reshape(-1, c)
    top = flat[y0 * w + x0] * (1.0 - ax) + flat[y0 * w + x1] * ax
    bot = flat[y1 * w + x0] * (1.0 - ax) + flat[y1 * w + x1] * ax
    out = top * (1.0 - ay) + bot * ay
    if not inside.all():
        out[~inside] = 0.0
    return out


def minification_factor(H_inv: np.ndarray, out_w: int, out_h: int) -> float:
    """Во сколько раз в среднем сжимается изображение (>=1 — уменьшение)."""
    pts = np.array([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]], dtype=np.float64)
    e = 4.0
    dets = []
    for px, py in pts:
        q = np.array([[px + e, py], [px, py + e]], dtype=np.float64)
        qh = np.hstack([q, np.ones((2, 1))]) @ H_inv.T
        qs = qh[:, :2] / qh[:, 2:3]
        ph = np.array([px, py, 1.0]) @ H_inv.T
        p0 = ph[:2] / ph[2]
        J = np.column_stack([(qs[0] - p0) / e, (qs[1] - p0) / e])
        dets.append(abs(np.linalg.det(J)))
    return float(np.sqrt(max(dets)))


def prefilter(img: np.ndarray, radius: int) -> np.ndarray:
    """Префильтрация перед уменьшением (box-фильтр по каналам)."""
    if radius < 1:
        return img
    if img.ndim == 2:
        return box_blur(img, radius)
    return np.stack([box_blur(img[..., i], radius) for i in range(img.shape[2])], axis=-1)


def warp_perspective(img, H, out_w, out_h, supersample="auto", tile_rows: int = 256):
    """Перенос изображения гомографией H (src -> dst) в растр out_w x out_h."""
    img = np.asarray(img, dtype=np.float32)
    squeeze = img.ndim == 2
    if squeeze:
        img = img[..., None]
    H = np.asarray(H, dtype=np.float64)
    H_inv = np.linalg.inv(H)

    if supersample == "auto":
        f = minification_factor(H_inv, out_w, out_h)
        img = prefilter(img, int(round(0.5 * (f - 1.0))) if f > 1.4 else 0)
        k = 1
    else:
        k = max(1, int(supersample))

    c = img.shape[2]
    out = np.zeros((out_h, out_w, c), dtype=np.float32)
    xs = np.arange(out_w, dtype=np.float32)[None, :]
    inv = float(k * k)
    for y0 in range(0, out_h, tile_rows):
        y1 = min(out_h, y0 + tile_rows)
        ys = np.arange(y0, y1, dtype=np.float32)[:, None]
        acc = np.zeros((y1 - y0, out_w, c), dtype=np.float32)
        for dy in range(k):
            for dx in range(k):
                X = xs + ((dx + 0.5) / k - 0.5)
                Y = ys + ((dy + 0.5) / k - 0.5)
                den = H_inv[2, 0] * X + H_inv[2, 1] * Y + H_inv[2, 2]
                small = np.abs(den) < 1e-9
                if small.any():
                    den = np.where(small, 1e-9, den)
                sx = (H_inv[0, 0] * X + H_inv[0, 1] * Y + H_inv[0, 2]) / den
                sy = (H_inv[1, 0] * X + H_inv[1, 1] * Y + H_inv[1, 2]) / den
                acc += _bilinear(img, sx, sy)
        out[y0:y1] = acc / inv if k > 1 else acc
    return out[..., 0] if squeeze else out
