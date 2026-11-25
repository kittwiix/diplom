"""Базовые операции над изображением (только numpy)."""

from __future__ import annotations

import numpy as np


def to_gray(rgb: np.ndarray) -> np.ndarray:
    a = np.asarray(rgb, dtype=np.float32)
    if a.ndim == 2:
        return a
    return a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114


def box_blur(a: np.ndarray, r: int) -> np.ndarray:
    """Box-фильтр через интегральные суммы, окно 2r+1."""
    a = np.asarray(a, dtype=np.float32)
    if r < 1:
        return a
    k = 2 * r + 1
    p = np.pad(a, ((0, 0), (r + 1, r)), mode="edge")
    c = np.cumsum(p, axis=1, dtype=np.float64)
    a = ((c[:, k:] - c[:, :-k]) / k).astype(np.float32)
    p = np.pad(a, ((r + 1, r), (0, 0)), mode="edge")
    c = np.cumsum(p, axis=0, dtype=np.float64)
    return ((c[k:, :] - c[:-k, :]) / k).astype(np.float32)


def smooth(a: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Дешёвое приближение гаусса: два box-прохода."""
    r = max(1, int(round(sigma)))
    return box_blur(box_blur(a, r), r)


def gradients(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gx = np.zeros_like(a, dtype=np.float32)
    gy = np.zeros_like(a, dtype=np.float32)
    gx[:, 1:-1] = (a[:, 2:] - a[:, :-2]) * 0.5
    gy[1:-1, :] = (a[2:, :] - a[:-2, :]) * 0.5
    return gx, gy


def gradient_magnitude(a: np.ndarray) -> np.ndarray:
    gx, gy = gradients(np.asarray(a, dtype=np.float32))
    return np.sqrt(gx * gx + gy * gy)


def resize(img: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    """Билинейное изменение размера (для анализа, без префильтрации)."""
    img = np.asarray(img, dtype=np.float32)
    squeeze = img.ndim == 2
    if squeeze:
        img = img[..., None]
    h, w = img.shape[:2]
    if (w, h) == (new_w, new_h):
        out = img.copy()
    else:
        xs = (np.arange(new_w, dtype=np.float32) + 0.5) * (w / new_w) - 0.5
        ys = (np.arange(new_h, dtype=np.float32) + 0.5) * (h / new_h) - 0.5
        xs = np.clip(xs, 0, w - 1)
        ys = np.clip(ys, 0, h - 1)
        x0 = np.floor(xs).astype(np.int32)
        x1 = np.minimum(x0 + 1, w - 1)
        y0 = np.floor(ys).astype(np.int32)
        y1 = np.minimum(y0 + 1, h - 1)
        ax = (xs - x0)[None, :, None]
        ay = (ys - y0)[:, None, None]
        top = img[y0][:, x0] * (1.0 - ax) + img[y0][:, x1] * ax
        bot = img[y1][:, x0] * (1.0 - ax) + img[y1][:, x1] * ax
        out = top * (1.0 - ay) + bot * ay
    return out[..., 0] if squeeze else out


def to_uint8(a: np.ndarray) -> np.ndarray:
    return (np.clip(a, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
