"""Демо стадии 1: кромки полок -> точки схода -> гомография -> полосы ячеек."""

import json
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from rectif.debug import draw_hlines, draw_rails, draw_verticals
from rectif.imageops import resize, to_uint8
from rectif.io import load_image
from rectif.pipeline import analyze, crop_cells, rectify

SRC = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "..", "..", "data", "samples"))
OUT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "..", "..", "out", "stage1"))


def save(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(to_uint8(img)).save(path, quality=92, subsampling=1)


def montage(strips, gap=12):
    wmax = max(s.shape[1] for s in strips)
    total = sum(s.shape[0] for s in strips) + gap * (len(strips) - 1)
    canvas = np.ones((total, wmax, 3), np.float32)
    y = 0
    for s in strips:
        canvas[y:y + s.shape[0], :s.shape[1]] = s
        y += s.shape[0] + gap
    return canvas


def side_by_side(a, b, gap=20):
    h = min(a.shape[0], b.shape[0])
    a = resize(a, int(round(a.shape[1] * h / a.shape[0])), h)
    b = resize(b, int(round(b.shape[1] * h / b.shape[0])), h)
    c = np.ones((h, a.shape[1] + b.shape[1] + gap, 3), np.float32)
    c[:, :a.shape[1]] = a
    c[:, a.shape[1] + gap:] = b
    return c


def run(name):
    rgb = load_image(os.path.join(SRC, name))
    t0 = time.time()
    an = analyze(rgb)
    res = rectify(rgb, mode="vp", crop="valid", analysis=an)
    cells = crop_cells(res.image, res.rails_out)
    dt = time.time() - t0

    stem = os.path.splitext(name)[0]
    save(os.path.join(OUT, stem + "_rectified.jpg"), res.image)
    if cells:
        save(os.path.join(OUT, stem + "_cells.jpg"), montage(cells))

    s = an.scale
    before = rgb.copy()
    draw_rails(before, [(l.a, l.b / s) for l in an.rails])
    draw_verticals(before, [(l.a, l.b / s) for l in an.verticals])
    draw_hlines(before, [l.a * 0.5 * rgb.shape[1] + l.b / an.scale for l in an.rails],
                color=(0.0, 0.85, 0.25))
    after = res.image.copy()
    draw_hlines(after, res.rails_out)
    save(os.path.join(OUT, stem + "_compare.jpg"), side_by_side(before, after))

    with open(os.path.join(OUT, stem + "_meta.json"), "w", encoding="utf-8") as f:
        json.dump(res.meta, f, indent=2, ensure_ascii=False)

    m = res.meta
    print("%-22s rails=%2d cells=%d  residual RMS=%.2f deg spread=%.2f deg  %.1fs" % (
        name, len(an.rails), len(cells), m["rails_angle_rms_deg"], m["rails_angle_spread_deg"], dt))


if __name__ == "__main__":
    for n in ("photo.jpeg", "photo_2.jpeg", "photo_warped.jpeg", "photo_2_warped.jpeg"):
        run(n)
    print("saved:", OUT)
