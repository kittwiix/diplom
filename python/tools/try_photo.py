"""Ручной прогон: python tools/try_photo.py <файл|папка> [--mode vp|roll] [--crop valid|none|shelf|auto]

Для каждого кадра сохраняет в out/try/<имя>/:
  <имя>_rectified.jpg  — выпрямленный кадр
  <имя>_overlay.jpg    — исходник: найденные кромки (красные), вертикали (синие), горизонтали (зелёные)
  <имя>_cells.jpg      — полосы полок (то, что уходит в VLM)
  <имя>_meta.json      — H, точки схода, углы, качество
и печатает сводную таблицу.
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))

from rectif.debug import draw_hlines, draw_rails, draw_verticals
from rectif.imageops import to_uint8
from rectif.io import load_image
from rectif.pipeline import analyze, crop_cells, rectify

OUT = os.path.join(ROOT, "out", "try")
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def save(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(to_uint8(img)).save(path, quality=92, subsampling=1)


def montage(strips, gap=10):
    w = max(s.shape[1] for s in strips)
    total = sum(s.shape[0] for s in strips) + gap * (len(strips) - 1)
    canvas = np.ones((total, w, 3), np.float32)
    y = 0
    for s in strips:
        canvas[y:y + s.shape[0], :s.shape[1]] = s
        y += s.shape[0] + gap
    return canvas


def run_one(path, mode, crop, max_side):
    t0 = time.time()
    rgb = load_image(path)
    an = analyze(rgb)
    used_mode = mode
    if mode == "vp" and len(an.rails) < 2:
        used_mode = "roll"
    res = rectify(rgb, mode=used_mode, crop=crop, max_side=max_side, analysis=an)
    dt = time.time() - t0

    stem = os.path.splitext(os.path.basename(path))[0]
    dst = os.path.join(OUT, stem)
    save(os.path.join(dst, stem + "_rectified.jpg"), res.image)

    s = an.scale
    ov = rgb.copy()
    draw_rails(ov, [(l.a, l.b / s) for l in an.rails], width=4)
    draw_verticals(ov, [(l.a, l.b / s) for l in an.verticals], width=3)
    draw_hlines(ov, [l.a * 0.5 * rgb.shape[1] + l.b / s for l in an.rails])
    save(os.path.join(dst, stem + "_overlay.jpg"), ov)

    cells = crop_cells(res.image, res.rails_out)
    if cells:
        save(os.path.join(dst, stem + "_cells.jpg"), montage(cells))

    with open(os.path.join(dst, stem + "_meta.json"), "w", encoding="utf-8") as f:
        json.dump(res.meta, f, indent=2, ensure_ascii=False)

    return {
        "name": os.path.basename(path),
        "rails": len(an.rails),
        "cells": len(cells),
        "rms": res.meta["quality"]["rms_deg"] or 0.0,
        "score": res.meta["quality"]["score"],
        "ok": res.meta["quality"]["ok"],
        "mode": used_mode,
        "out": "%dx%d" % (res.out_w, res.out_h),
        "sec": dt,
    }


def collect(target):
    if os.path.isdir(target):
        files = []
        for e in EXTS:
            files += glob.glob(os.path.join(target, "*" + e))
            files += glob.glob(os.path.join(target, "*" + e.upper()))
        return sorted(set(files))
    return [target]


def main(argv=None):
    p = argparse.ArgumentParser(description="Manual rectification run")
    p.add_argument("target", help="file or folder with photos")
    p.add_argument("--mode", default="vp", choices=("vp", "directions", "roll"))
    p.add_argument("--crop", default="valid", choices=("none", "valid", "shelf", "auto"))
    p.add_argument("--max-side", type=int, default=2000)
    args = p.parse_args(argv)

    files = collect(args.target)
    if not files:
        print("no images found:", args.target)
        return 1

    rows = []
    for f in files:
        try:
            r = run_one(f, args.mode, args.crop, args.max_side)
            rows.append(r)
            print("  %-28s rails=%2d cells=%d rms=%5.2f score=%.2f %s %s %.2fs"
                  % (r["name"], r["rails"], r["cells"], r["rms"], r["score"],
                     "OK " if r["ok"] else "!! ", r["out"], r["sec"]))
        except Exception as exc:
            print("  %-28s ERROR: %s: %s" % (os.path.basename(f), type(exc).__name__, exc))

    print("")
    print("%-28s %5s %5s %7s %6s %4s" % ("file", "rails", "cells", "rms", "score", "ok"))
    for r in rows:
        print("%-28s %5d %5d %7.2f %6.2f %4s"
              % (r["name"], r["rails"], r["cells"], r["rms"], r["score"],
                 "yes" if r["ok"] else "NO"))
    good = sum(1 for r in rows if r["ok"])
    print("")
    print("total: %d/%d OK, %.1f s" % (good, len(rows), sum(r["sec"] for r in rows)))
    print("results:", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
