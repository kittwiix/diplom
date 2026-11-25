"""CLI ректификации: python -m rectif.cli -i photo.jpeg -o out.jpg"""

from __future__ import annotations

import argparse
import json
import os

from PIL import Image

from .debug import draw_hlines, draw_rails, draw_verticals
from .imageops import to_uint8
from .io import load_image
from .pipeline import crop_cells, rectify


def _dump(path: str, img) -> None:
    folder = os.path.dirname(os.path.abspath(path))
    if folder:
        os.makedirs(folder, exist_ok=True)
    Image.fromarray(to_uint8(img)).save(path, quality=92, subsampling=1)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Ректификация фото стеллажа по гомографии")
    p.add_argument("-i", "--input", required=True, help="исходное фото")
    p.add_argument("-o", "--output", required=True, help="куда сохранить выпрямленный кадр")
    p.add_argument("--mode", default="vp", choices=("vp", "directions", "roll"),
                   help="vp — гомография по точкам схода (основной), directions — аффинно, roll — только поворот")
    p.add_argument("--crop", default="valid", choices=("none", "valid", "shelf", "auto"),
                   help="none — весь кадр; valid — убрать чёрные углы; shelf — по полкам; auto — valid+shelf")
    p.add_argument("--max-side", type=int, default=2000)
    p.add_argument("--cells-dir", default=None, help="сюда сохранить полосы полок (кропы для VLM)")
    p.add_argument("--json", default=None, help="метаданные (H, ТС, углы, качество)")
    p.add_argument("--debug-dir", default=None, help="отладочные картинки (линии на исходнике)")
    args = p.parse_args(argv)

    rgb = load_image(args.input)
    res = rectify(rgb, mode=args.mode, crop=args.crop, max_side=args.max_side)
    _dump(args.output, res.image)

    stem = os.path.splitext(os.path.basename(args.output))[0]

    if args.cells_dir:
        cells = crop_cells(res.image, res.rails_out)
        os.makedirs(args.cells_dir, exist_ok=True)
        for i, cell in enumerate(cells):
            _dump(os.path.join(args.cells_dir, "%s_cell%02d.jpg" % (stem, i)), cell)

    if args.json:
        folder = os.path.dirname(os.path.abspath(args.json))
        if folder:
            os.makedirs(folder, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(res.meta, f, indent=2, ensure_ascii=False)

    if args.debug_dir:
        os.makedirs(args.debug_dir, exist_ok=True)
        s = res.analysis.scale
        dbg = rgb.copy()
        draw_rails(dbg, [(l.a, l.b / s) for l in res.analysis.rails])
        draw_verticals(dbg, [(l.a, l.b / s) for l in res.analysis.verticals])
        _dump(os.path.join(args.debug_dir, stem + "_source_lines.jpg"), dbg)
        out_dbg = res.image.copy()
        draw_hlines(out_dbg, res.rails_out)
        _dump(os.path.join(args.debug_dir, stem + "_result_lines.jpg"), out_dbg)

    q = res.meta["quality"]
    print("quality: rails=%d rms=%.2f deg spread=%.2f score=%.2f ok=%s"
          % (q["rails"], q["rms_deg"] or 0.0, q["spread_deg"] or 0.0, q["score"], q["ok"]))
    print("mode=%s crop=%s out=%dx%d rails=%d verticals=%d"
          % (args.mode, args.crop, res.out_w, res.out_h,
             len(res.analysis.rails), len(res.analysis.verticals)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
