"""Отладочная отрисовка (в рантайме ректификации не участвует)."""

from __future__ import annotations

import numpy as np


def draw_line(img, p0, p1, color=(1.0, 0.0, 0.0), width=2):
    h, w = img.shape[:2]
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    n = int(max(abs(p1[0] - p0[0]), abs(p1[1] - p0[1]))) + 1
    if n <= 1:
        return
    xs = np.linspace(p0[0], p1[0], n)
    ys = np.linspace(p0[1], p1[1], n)
    r = max(0, int(width) // 2)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            xi = np.clip(np.round(xs).astype(int) + dx, 0, w - 1)
            yi = np.clip(np.round(ys).astype(int) + dy, 0, h - 1)
            img[yi, xi] = color


def draw_rails(img, rails, color=(1.0, 0.15, 0.15), width=3):
    """rails: (a, b) для y = a*x + b в координатах этого изображения."""
    h, w = img.shape[:2]
    for a, b in rails:
        draw_line(img, (0, b), (w - 1, a * (w - 1) + b), color=color, width=width)


def draw_verticals(img, verticals, color=(0.2, 0.5, 1.0), width=3):
    """verticals: (a, b) для x = a*y + b."""
    h, w = img.shape[:2]
    for a, b in verticals:
        draw_line(img, (b, 0), (a * (h - 1) + b, h - 1), color=color, width=width)


def draw_hlines(img, ys, color=(0.0, 0.9, 0.3), width=3):
    w = img.shape[1]
    for y in ys:
        draw_line(img, (0, y), (w - 1, y), color=color, width=width)
