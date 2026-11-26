"""Схема логики поворота: кромки -> точка схода -> параллельность -> горизонталь."""

import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from rectif import geometry as g
from rectif.debug import draw_line
from rectif.imageops import resize, to_uint8
from rectif.io import load_image
from rectif.pipeline import analyze, rectify
from rectif.warp import warp_perspective

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
SRC = os.path.join(ROOT, "data", "samples", "photo_2.jpeg")
OUT = os.path.join(ROOT, "out", "logic")

FONTS = (r"C:/Windows/Fonts/arial.ttf", r"C:/Windows/Fonts/segoeui.ttf")


def font(size):
    for p in FONTS:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                pass
    return ImageFont.load_default()


def caption(img, text, width, size=26):
    pil = Image.fromarray(to_uint8(img))
    band = 52
    canvas = Image.new("RGB", (width, pil.height + band), (255, 255, 255))
    canvas.paste(pil, (0, band))
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, 0, width, band], fill=(30, 30, 34))
    d.text((14, 10), text, font=font(size), fill=(245, 245, 245))
    return canvas


def main():
    rgb = load_image(SRC)
    h, w = rgb.shape[:2]
    an = analyze(rgb)
    s = an.scale
    rails = [(l.a, l.b / s) for l in an.rails]
    vp = g.vp_from_lines(an.rails)
    vp = vp / vp[2]

    # --- 1. кромки и точка схода (точка вне кадра -> расширяем холст)
    k = 0.5
    vx, vy = vp[0] / s, vp[1] / s
    cw = int(max(w * k + 20, abs(vx) * k + 40))
    chh = int(max(h * k + 20, abs(vy) * k + 40))
    canvas = np.ones((chh, cw, 3), np.float32)
    small = resize(rgb, int(w * k), int(h * k))
    canvas[0:small.shape[0], 0:small.shape[1]] = small
    for a, b in rails:
        draw_line(canvas, (0, b * k), (vx * k, vy * k), color=(1.0, 0.1, 0.1), width=2)
    px, py = int(vx * k), int(vy * k)
    draw_line(canvas, (px - 14, py), (px + 14, py), color=(1.0, 0.85, 0.0), width=5)
    draw_line(canvas, (px, py - 14), (px, py + 14), color=(1.0, 0.85, 0.0), width=5)

    # --- 2. аффинная ректификация: веер -> параллельные
    l = np.cross(vp * 1.0, np.array([0.0, 1.0, 0.0]))
    l = l / np.linalg.norm(l)
    if l[2] < 0:
        l = -l
    ha = np.eye(3)
    ha[2] = l
    ha_src = g.scaling(1.0 / s) @ ha @ g.scaling(s)
    h2, ow, oh = g.fit_frame(ha_src, w, h, max_side=1400)
    img2 = warp_perspective(rgb, h2, ow, oh)
    for a, b in rails:
        p = g.apply_h(h2, np.array([[0.0, b], [w * 1.0, a * w + b]]))
        draw_line(img2, tuple(p[0]), tuple(p[1]), color=(1.0, 0.1, 0.1), width=3)

    # --- 3. доворот до горизонтали
    res = rectify(rgb, mode="vp", crop="valid", analysis=an)
    img3 = res.image.copy()
    for y in res.rails_out:
        draw_line(img3, (0, y), (img3.shape[1] - 1, y), color=(0.1, 0.9, 0.3), width=3)

    # --- сборка
    W = 1400
    p1 = caption(resize(canvas, W, max(1, int(canvas.shape[0] * W / canvas.shape[1]))),
                 "1. Нашли кромки полок (красные). Они сходятся в точке схода — она далеко вне кадра",
                 W)
    p2 = caption(resize(img2, W, max(1, int(img2.shape[0] * W / img2.shape[1]))),
                 "2. Аффинная ректификация: точку схода увели на бесконечность — кромки стали параллельны",
                 W)
    p3 = caption(resize(img3, W, max(1, int(img3.shape[0] * W / img3.shape[1]))),
                 "3. Доворот до горизонтали: остаточный угол %.2f град" % res.meta["quality"]["rms_deg"],
                 W)
    total = p1.height + p2.height + p3.height + 40
    out = Image.new("RGB", (W, total), (255, 255, 255))
    y = 0
    for p in (p1, p2, p3):
        out.paste(p, (0, y))
        y += p.height + 20
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "rotation_logic.png")
    out.save(path)
    print("saved", path, out.size)


if __name__ == "__main__":
    main()
