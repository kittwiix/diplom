"""Проективная геометрия: гомографии 3x3, DLT, базис по двум направлениям.

Соглашение о координатах: пиксельные индексы, (x, y) = (столбец, строка),
начало в левом верхнем углу, H отображает исходный кадр в выходной.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------- примитивы
def apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Перенос точек (N,2) гомографией H."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    ph = np.hstack([pts, np.ones((len(pts), 1))]) @ np.asarray(H, dtype=np.float64).T
    return ph[:, :2] / ph[:, 2:3]


def translation(tx: float, ty: float) -> np.ndarray:
    H = np.eye(3)
    H[0, 2], H[1, 2] = tx, ty
    return H


def scaling(s: float) -> np.ndarray:
    H = np.eye(3)
    H[0, 0] = H[1, 1] = s
    return H


def rotation(theta_deg: float) -> np.ndarray:
    t = np.deg2rad(theta_deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def about_point(H: np.ndarray, cx: float, cy: float) -> np.ndarray:
    """Сопряжение гомографии центром (cx, cy).

    Корректно только для аффинных H: для проективных сопряжение трансляцией меняет
    перспективу. Для общих H используйте recenter().
    """
    return translation(cx, cy) @ H @ translation(-cx, -cy)


def recenter(H: np.ndarray, cx: float, cy: float) -> np.ndarray:
    """Сдвиг выхода так, чтобы точка (cx, cy) осталась на месте (не меняет направлений)."""
    p = apply_h(H, np.array([[cx, cy]], dtype=np.float64))[0]
    return translation(cx - p[0], cy - p[1]) @ H


def affine_from_directions(theta_h_deg: float, theta_v_deg: float) -> np.ndarray:
    """Аффинное преобразование, переводящее направление theta_h в ось X, theta_v в ось Y.

    Оба угла заданы как угол вектора направления линии от оси +X (в градусах).
    Именно это нужно по схеме «взяли кромки полок и кромки вертикалей — выпрямили кадр».
    """
    th = np.deg2rad(theta_h_deg)
    tv = np.deg2rad(theta_v_deg)
    u_h = np.array([np.cos(th), np.sin(th)], dtype=np.float64)
    u_v = np.array([np.cos(tv), np.sin(tv)], dtype=np.float64)
    M = np.column_stack([u_h, u_v])
    if abs(np.linalg.det(M)) < 0.2:  # направления почти параллельны — вырождение
        raise ValueError("directions are nearly parallel")
    A = np.linalg.inv(M)
    H = np.eye(3)
    H[:2, :2] = A
    return H


def line_endpoints(theta_deg: float, offset: float, cx: float, cy: float, half_len: float):
    """Точки прямой n·(p - c) = offset (n — нормаль к направлению theta)."""
    th = np.deg2rad(theta_deg)
    n = np.array([-np.sin(th), np.cos(th)], dtype=np.float64)
    u = np.array([np.cos(th), np.sin(th)], dtype=np.float64)
    p0 = np.array([cx, cy], dtype=np.float64) + offset * n
    return p0 - half_len * u, p0 + half_len * u


# ----------------------------------------------------------------------- DLT
def normalize_pts(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Нормализация Хартли: центроид в 0, средний радиус sqrt(2)."""
    p = np.asarray(p, dtype=np.float64)
    c = p.mean(0)
    d = np.sqrt(((p - c) ** 2).sum(1)).mean()
    s = np.sqrt(2.0) / max(d, 1e-12)
    T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]], dtype=np.float64)
    ph = np.hstack([p, np.ones((len(p), 1))]) @ T.T
    return ph[:, :2] / ph[:, 2:], T


def dlt(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Прямой линейный метод: H по N>=4 соответствиям (без нормализации координат)."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = len(src)
    A = np.zeros((2 * n, 9), dtype=np.float64)
    for i in range(n):
        x, y = src[i]
        u, v = dst[i]
        A[2 * i] = [-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u]
        A[2 * i + 1] = [0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v]
    # full_matrices=True обязательно: у матрицы 2N x 9 вектор нуль-пространства —
    # последняя строка Vt, при full_matrices=False она отбрасывается для N=4
    _, _, vt = np.linalg.svd(A, full_matrices=True)
    H = vt[-1].reshape(3, 3)
    norm = float(H[2, 2]) if abs(H[2, 2]) > 1e-12 else float(np.abs(H).max())
    return H / norm


def homography_from_points(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """DLT с нормализацией Хартли (устойчиво при больших координатах)."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if len(src) < 4:
        raise ValueError("need at least 4 point correspondences")
    ps, Ts = normalize_pts(src)
    pd, Td = normalize_pts(dst)
    Hn = dlt(ps, pd)
    H = np.linalg.inv(Td) @ Hn @ Ts
    return H / H[2, 2]


def refine_lm(src: np.ndarray, dst: np.ndarray, H: np.ndarray, iters: int = 50) -> np.ndarray:
    """Уточнение H по минимуму ошибки репроекции (Левенберг—Марквардт)."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)

    def resid(h: np.ndarray) -> np.ndarray:
        M = np.append(h, 1.0).reshape(3, 3)
        return (apply_h(M, src) - dst).ravel()

    h = (H / H[2, 2]).ravel()[:8].astype(np.float64)
    r = resid(h)
    cost = float(r @ r)
    lam = 1e-3
    for _ in range(iters):
        J = np.zeros((len(r), 8), dtype=np.float64)
        for j in range(8):
            hp = h.copy()
            hp[j] += 1e-6
            J[:, j] = (resid(hp) - r) / 1e-6
        A = J.T @ J
        g = J.T @ r
        improved = False
        for _ in range(8):
            try:
                delta = np.linalg.solve(A + lam * np.eye(8), -g)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            hn = h + delta
            rn = resid(hn)
            cn = float(rn @ rn)
            if cn < cost:
                h, r, cost, lam, improved = hn, rn, cn, max(lam / 3.0, 1e-7), True
                break
            lam *= 4.0
        if not improved:
            break
    return np.append(h, 1.0).reshape(3, 3)


# ---------------------------------------------------------------- утилиты
def transformed_bbox(w: int, h: int, H: np.ndarray) -> tuple[float, float, float, float]:
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    q = apply_h(H, corners)
    return float(q[:, 0].min()), float(q[:, 1].min()), float(q[:, 0].max()), float(q[:, 1].max())


def fit_frame(H: np.ndarray, w: int, h: int, max_side: int = 2400) -> tuple[np.ndarray, int, int]:
    """Сдвигает/масштабирует H так, чтобы всё содержимое попало в кадр (0,0)-(out_w,out_h)."""
    x0, y0, x1, y1 = transformed_bbox(w, h, H)
    scale = 1.0
    span = max(x1 - x0, y1 - y0)
    if max_side and span > max_side:
        scale = max_side / span
    H2 = scaling(scale) @ translation(-x0, -y0) @ H
    out_w = max(1, int(round((x1 - x0) * scale)))
    out_h = max(1, int(round((y1 - y0) * scale)))
    return H2, out_w, out_h


def line_angle_after(H: np.ndarray, a: float, b: float, extent: float, kind: str = "h") -> float:
    """Угол прямой после преобразования H (в градусах, mod 180)."""
    if kind == "h":
        p = np.array([[0.0, b], [extent, a * extent + b]], dtype=np.float64)
    else:
        p = np.array([[b, 0.0], [a * extent + b, extent]], dtype=np.float64)
    q = apply_h(H, p)
    v = q[1] - q[0]
    return float(np.degrees(np.arctan2(v[1], v[0])) % 180.0)


def vp_from_lines(lines, kind: str = "h") -> np.ndarray:
    """Точка схода семейства прямых (взвешенный МНК: l_i . v = 0)."""
    A = []
    for ln in lines:
        if kind == "h":
            v = np.array([ln.a, -1.0, ln.b], dtype=np.float64)
        else:
            v = np.array([1.0, -ln.a, -ln.b], dtype=np.float64)
        A.append(v / np.linalg.norm(v) * np.sqrt(max(float(ln.support), 1e-6)))
    A = np.asarray(A)
    if len(A) < 2:
        return np.array([0.0, 0.0, 1.0])
    _, _, vt = np.linalg.svd(A, full_matrices=False)
    return vt[-1]


def circular_mean_mod180(angles_deg) -> float:
    a = np.deg2rad(np.asarray(angles_deg, dtype=np.float64) * 2.0)
    return float(np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) / 2.0)


def rectifying_homography(rails, verticals, width: float, height: float, use_vertical_vp=None):
    """Гомография ректификации полки по точкам схода.

    1. ТС горизонталей (полки) vx;  ТС вертикалей vy (или допущение «вертикали вертикальны»).
    2. Линия бесконечности l = vx x vy -> аффинная ректификация (Ha с третьей строкой l).
    3. В выправленном кадре полки уже параллельны; аффинное преобразование по двум направлениям
       доводит их до осей (убирает остаточный скос/анизотропию).
    """
    vx = vp_from_lines(rails, "h")
    if use_vertical_vp is None:
        # ТС вертикалей используем только при достаточной поддержке:
        # две короткие вертикальные линии дают численно неустойчивую ТС
        use_vertical_vp = len(verticals) >= 4 and max(
            [getattr(v, "coverage", 0.0) for v in verticals] + [0.0]
        ) >= 0.4
    vy = vp_from_lines(verticals, "v") if use_vertical_vp else np.array([0.0, 1.0, 0.0])

    l = np.cross(vx, vy)
    l = l / np.linalg.norm(l)
    if l[2] < 0:  # фиксируем знак, чтобы не получить зеркальный кадр
        l = -l
    if abs(l[2]) < 1e-12:
        l[2] = 1e-12
    Ha = np.eye(3)
    Ha[2, :] = l

    ang_h = circular_mean_mod180([line_angle_after(Ha, r.a, r.b, width, "h") for r in rails])
    if use_vertical_vp:
        ang_v = circular_mean_mod180([line_angle_after(Ha, r.a, r.b, height, "v") for r in verticals])
    else:
        ang_v = 90.0

    try:
        H = affine_from_directions(ang_h, ang_v) @ Ha
    except ValueError:  # направления вырождены — только поворот
        H = rotation(-ang_h) @ Ha
        ang_v = 90.0
    info = {
        "vp_h": vx,
        "vp_v": vy,
        "line_at_infinity": l,
        "angle_h_after_affine": ang_h,
        "angle_v_after_affine": ang_v,
        "used_vertical_vp": bool(use_vertical_vp),
    }
    return H, info


def largest_inscribed_rect(H, w, h, out_w, out_h, steps: int = 120):
    """Наибольший прямоугольник выходного кадра, который целиком попадает в исходный.

    Нужен, чтобы убрать чёрные углы после перспективного варпа.
    """
    Hin = np.linalg.inv(np.asarray(H, dtype=np.float64))
    ys = np.linspace(0.0, out_h - 1.0, steps)
    xs = np.linspace(0.0, out_w - 1.0, 2 * steps)
    X, Y = np.meshgrid(xs, ys)
    den = Hin[2, 0] * X + Hin[2, 1] * Y + Hin[2, 2]
    den = np.where(np.abs(den) < 1e-12, 1e-12, den)
    sx = (Hin[0, 0] * X + Hin[0, 1] * Y + Hin[0, 2]) / den
    sy = (Hin[1, 0] * X + Hin[1, 1] * Y + Hin[1, 2]) / den
    ok = (sx >= -0.5) & (sx <= w - 0.5) & (sy >= -0.5) & (sy <= h - 0.5)

    lo = np.full(len(ys), -1, dtype=np.int64)
    hi = np.full(len(ys), -1, dtype=np.int64)
    for i in range(len(ys)):
        idx = np.flatnonzero(ok[i])
        if idx.size == 0:
            continue
        runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
        run = max(runs, key=len)
        lo[i], hi[i] = int(run[0]), int(run[-1])

    best = (0.0, 0, 0, 0.0, 0.0)
    for i0 in range(len(ys)):
        if lo[i0] < 0:
            continue
        x_lo, x_hi = -1e18, 1e18
        for i1 in range(i0, len(ys)):
            if lo[i1] < 0:
                break
            x_lo = max(x_lo, float(lo[i1]))
            x_hi = min(x_hi, float(hi[i1]))
            if x_hi <= x_lo:
                break
            area = (x_hi - x_lo) * (i1 - i0)
            if area > best[0]:
                best = (area, i0, i1, x_lo, x_hi)

    if best[0] <= 0:
        return 0.0, 0.0, float(out_w), float(out_h)
    _, i0, i1, x_lo, x_hi = best
    return (float(xs[int(x_lo)]), float(ys[i0]),
            float(xs[int(x_hi)]), float(ys[i1]))


def decompose_affine(H: np.ndarray) -> dict:
    """Полярное разложение аффинной части: поворот, масштабы, скос."""
    A = np.asarray(H, dtype=np.float64)[:2, :2]
    U, S, Vt = np.linalg.svd(A)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = -R
        S = S[::-1]
    return {
        "rotation_deg": float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))),
        "scale": [float(S[0]), float(S[1])],
        "shear": float(A[0, 1] + A[1, 0]),
    }
