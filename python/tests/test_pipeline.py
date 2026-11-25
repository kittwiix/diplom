"""Сквозной тест: синтетическая полка с перекосом -> ректификация."""

import numpy as np

from rectif import geometry as g
from rectif.pipeline import analyze, rectify
from rectif.warp import warp_perspective


def make_shelf(w=600, h=450):
    """Идеальная полка: горизонтальные кромки + вертикальные стойки."""
    img = np.ones((h, w, 3), np.float32) * 0.85
    for y in (80, 190, 300, 410):
        img[y - 3:y + 3, :, :] = 0.15
        img[y + 6:y + 26, 20:w - 20, :] = np.array([1.0, 0.85, 0.2])
    for x in (60, 200, 340, 480, 580):
        img[:, x - 2:x + 2, :] = 0.3
    return img


def make_photo(img, strength=1.2e-3):
    """Снимаем «под углом»: проективная деформация с точкой схода."""
    h, w = img.shape[:2]
    H = np.eye(3)
    H[2, 0] = -strength          # веер: точки схода по горизонтали
    H2, ow, oh = g.fit_frame(H, w, h, max_side=1200)
    return warp_perspective(img, H2, ow, oh)


def test_pipeline_straightens_synthetic_shelf():
    photo = make_photo(make_shelf())
    an = analyze(photo, work_width=600)
    assert len(an.rails) >= 3
    res = rectify(photo, mode="vp", analysis=an)
    assert res.meta["rails_angle_rms_deg"] < 1.0
    assert res.meta["rails_angle_spread_deg"] < 2.0
    assert res.image.shape[0] > 100
