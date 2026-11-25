"""Тесты геометрии: DLT, сдвиг выхода, ректификация по точке схода."""

import numpy as np

from rectif import geometry as g
from rectif.lines import LineFit


def _wrap90(a):
    return ((a + 90.0) % 180.0) - 90.0


def test_dlt_roundtrip():
    src = np.array([[0, 0], [1000, 0], [1000, 800], [0, 800]], float)
    dst = np.array([[10, 20], [1110, 25], [1105, 1120], [12, 1118]], float)
    H = g.homography_from_points(src, dst)
    assert np.allclose(g.apply_h(H, src), dst, atol=1e-9)


def test_recenter_keeps_point():
    H = np.array([[1.0, 0, 0], [0.16, 1, 0], [4.7e-4, 0, 1]])
    H2 = g.recenter(H, 449.5, 599.5)
    assert np.allclose(g.apply_h(H2, [[449.5, 599.5]])[0], [449.5, 599.5], atol=1e-9)


def _fan_lines(vp_x, vp_y, offsets):
    lines = []
    for b in offsets:
        a = (vp_y - b) / vp_x
        lines.append(LineFit(a=a, b=float(b), angle_deg=float(np.degrees(np.arctan(a))),
                             support=10.0, inliers=10, coverage=1.0))
    return lines


def test_rectifying_homography_straightens_fan():
    lines = _fan_lines(2000.0, 500.0, (100, 300, 500, 700, 900))
    H, info = g.rectifying_homography(lines, [], 1000, 800, use_vertical_vp=False)
    assert info["used_vertical_vp"] is False
    angs = [_wrap90(g.line_angle_after(H, l.a, l.b, 1000, "h")) for l in lines]
    assert max(abs(a) for a in angs) < 0.05, angs


def test_affine_from_directions():
    H = g.affine_from_directions(-7.0, 95.0)
    uh = np.array([[np.cos(np.deg2rad(-7.0)), np.sin(np.deg2rad(-7.0))]])
    uv = np.array([[np.cos(np.deg2rad(95.0)), np.sin(np.deg2rad(95.0))]])
    a = g.apply_h(H, uh)[0]
    b = g.apply_h(H, uv)[0]
    assert abs(np.degrees(np.arctan2(a[1], a[0]))) < 1e-6
    assert abs(np.degrees(np.arctan2(b[1], b[0])) - 90.0) < 1e-6
