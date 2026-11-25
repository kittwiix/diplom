"""Наглядный отчёт: исходник -> результат -> зум ценников. Один PNG на фото."""

import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from rectif.debug import draw_hlines, draw_rails, draw_verticals
from rectif.imageops import resize, to_uint8
from rectif.io import load_image
from rectif.pipeline import analyze, rectify

SRC = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "..", "..", "data", "samples"))
OUT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "..", "..", "out", "report"))

FONT_CANDIDATES = (
    r"C:/Windows/Fonts/arial.ttf",
    r"C:/Windows/Fonts/segoeui.ttf",
    r"C:/Windows/Fonts/tahoma.ttf",
)


def font(size):
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                pass
    return ImageFont.load_default()


def fit(img, W, H):
    canvas = np.ones((H, W, 3), np.float32)
    s = min(W / float(img.shape[1]), H / float(img.shape[0]))
    r = resize(img, max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s)))
    y = (H - r.shape[0]) // 2
    x = (W - r.shape[1]) // 2
    canvas[y:y + r.shape[0], x:x + r.shape[1]] = r
    return canvas


def panel(img, W, H, title, footer=""):
    canvas = Image.new("RGB", (W, H + 60), (255, 255, 255))
    canvas.paste(Image.fromarray(to_uint8(fit(img, W, H))), (0, 32))
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, 0, W, 31], fill=(30, 30, 32))
    d.text((10, 5), title, font=font(18), fill=(240, 240, 240))
    if footer:
        d.text((10, H + 36), footer, font=font(16), fill=(45, 45, 45))
    return canvas


def band(img, y_center, height, x_frac=(0.26, 0.74), width=720):
    h, w = img.shape[:2]
    hh = int(max(16, min(height, h)))
    y0 = int(np.clip(y_center - hh / 2.0, 0, h - hh))
    x0, x1 = int(x_frac[0] * w), int(x_frac[1] * w)
    crop = img[y0:y0 + hh, x0:x1]
    return resize(crop, width, max(16, int(round(crop.shape[0] * width / crop.shape[1]))))


def build(name):
    rgb = load_image(os.path.join(SRC, name))
    an = analyze(rgb)
    res = rectify(rgb, mode="vp", crop="valid", analysis=an)
    h, w = rgb.shape[:2]
    s = an.scale

    rails_src = [(l.a, l.b / s) for l in an.rails]
    ys_center = [l.a * 0.5 * w + l.b / s for l in an.rails]

    before = rgb.copy()
    draw_rails(before, rails_src, color=(1.0, 0.12, 0.12), width=4)
    draw_verticals(before, [(l.a, l.b / s) for l in an.verticals], color=(0.15, 0.5, 1.0), width=3)
    draw_hlines(before, ys_center, color=(0.0, 0.85, 0.2), width=3)

    after = res.image.copy()
    draw_hlines(after, res.rails_out, color=(0.0, 0.85, 0.2), width=3)

    # самая «сильная» полка -> зум ценников
    k = int(np.argmax([l.support for l in an.rails])) if an.rails else 0
    ys_out = sorted(res.rails_out)
    pitch = float(np.median(np.diff(sorted(ys_center)))) if len(ys_center) > 1 else 0.1 * h
    zh = 0.55 * pitch + 0.03 * h
    zb = band(before, ys_center[k], zh)
    ya = res.rails_out[k] if k < len(res.rails_out) else 0.5 * res.out_h
    za = band(after, ya, zh)

    q = res.meta["quality"]
    t1 = "1. Исходник: %d кромок полок, разброс %.1f°" % (
        len(an.rails), max(l.angle_deg for l in an.rails) - min(l.angle_deg for l in an.rails))
    t2 = "2. После ректификации: остаток %.2f°" % (q["rms_deg"] or 0.0)
    t3 = "3. Ценники: до"
    t4 = "4. Ценники: после"

    p1 = panel(before, 720, 930, t1, "красные — найденные кромки полок, зелёные — горизонталь")
    p2 = panel(after, 720, 930, t2, "зелёные — те же кромки после гомографии")
    p3 = panel(zb, 720, 200, t3, "полка №%d, центральная часть кадра" % (k + 1))
    p4 = panel(za, 720, 200, t4, "та же полка после выпрямления")

    cap_h = 96
    W = 720 * 2 + 48
    H = 990 + 260 + cap_h
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    canvas.paste(p1, (0, 0))
    canvas.paste(p2, (720 + 48, 0))
    canvas.paste(p3, (0, 990))
    canvas.paste(p4, (720 + 48, 990))

    d = ImageDraw.Draw(canvas)
    vp = res.meta.get("vp_h")
    vp_txt = ""
    if vp:
        vx, vy = vp[0] / s, vp[1] / s
        inside = 0 <= vx < w and 0 <= vy < h
        vp_txt = "ТС полок: (%d, %d) px%s" % (vx, vy, "" if inside else " — вне кадра")
    lines = [
        "%s   %dx%d -> %dx%d   rail-кромок: %d   RMS: %.2f°   разброс: %.2f°   score: %.2f" % (
            name, w, h, res.out_w, res.out_h, len(an.rails),
            q["rms_deg"] or 0.0, q["spread_deg"] or 0.0, q["score"]),
        "гомография по точкам схода, %s;  %s" % (res.meta["mode"], vp_txt),
    ]
    y = 990 + 260 + 12
    for ln in lines:
        d.text((16, y), ln, font=font(19), fill=(25, 25, 25))
        y += 34

    path = os.path.join(OUT, os.path.splitext(name)[0] + "_report.png")
    os.makedirs(OUT, exist_ok=True)
    canvas.save(path)
    print("saved", path, canvas.size, "| rms=%.2f" % (q["rms_deg"] or 0.0))


if __name__ == "__main__":
    for n in ("photo.jpeg", "photo_2.jpeg"):
        build(n)
