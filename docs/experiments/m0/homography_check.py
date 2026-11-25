"""M0 sanity check.

Question: is photo*_warped.jpeg a pure projective transform (homography) of photo*.jpeg?
Method (numpy + PIL only, no OpenCV):
  Harris corners -> normalized patch descriptors (2 scales) -> ratio-test matching
  -> RANSAC DLT (Hartley normalization) -> Levenberg-Marquardt refinement.
Writes: H matrices + overlay/check figures.
"""
import json
import sys

import numpy as np
from PIL import Image, ImageFilter, ImageDraw


# ---------------------------------------------------------------- image utils
def load_gray(path, max_side=1100):
    im = Image.open(path).convert("L")
    w, h = im.size
    s = min(1.0, max_side / max(w, h))
    if s < 1:
        im = im.resize((round(w * s), round(h * s)), Image.LANCZOS)
    return np.asarray(im, np.float32) / 255.0, s


def blur(a, sigma):
    im = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
    return np.asarray(im.filter(ImageFilter.GaussianBlur(sigma)), np.float32) / 255.0


def downsample(a, factor=2):
    im = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
    w, h = im.size
    im = im.resize((max(1, w // factor), max(1, h // factor)), Image.LANCZOS)
    return np.asarray(im, np.float32) / 255.0


def harris(g, n=1600, cell=6, k=0.05):
    ix = np.zeros_like(g)
    iy = np.zeros_like(g)
    ix[:, 1:-1] = (g[:, 2:] - g[:, :-2]) * 0.5
    iy[1:-1, :] = (g[2:, :] - g[:-2, :]) * 0.5
    ixx, iyy, ixy = blur(ix * ix, 1.5), blur(iy * iy, 1.5), blur(ix * iy, 1.5)
    r = ixx * iyy - ixy ** 2 - k * (ixx + iyy) ** 2

    # non-max suppression over 3x3 (rolled, edges ignored afterwards)
    m = r.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            m = np.maximum(m, np.roll(np.roll(r, dy, 0), dx, 1))
    peaks = (r >= m) & (r > 0)

    # one best corner per cell -> spatial spread, then keep top-n
    ys, xs = np.nonzero(peaks)
    score = r[ys, xs]
    order = np.argsort(-score)
    ys, xs, score = ys[order], xs[order], score[order]
    h, w = g.shape
    seen = np.zeros((h // cell + 2, w // cell + 2), bool)
    keep_y, keep_x = [], []
    for y, x, _ in zip(ys, xs, score):
        cy, cx = y // cell, x // cell
        if seen[cy, cx]:
            continue
        seen[cy, cx] = True
        keep_y.append(y)
        keep_x.append(x)
        if len(keep_y) >= n:
            break
    return np.stack([keep_x, keep_y], 1).astype(np.float32)


def descriptors(g, pts, half):
    """11x11 patches at two scales, mean-removed and unit-normalized."""
    gs = blur(g, 1.0)
    gh = blur(half, 1.5)
    pad = 6
    gp = np.pad(gs, pad, mode="edge")
    hp = np.pad(gh, pad, mode="edge")
    out = np.zeros((len(pts), 2 * 121), np.float32)
    for i, (x, y) in enumerate(pts):
        x, y = int(x), int(y)
        p1 = gp[y:y + 11, x:x + 11].ravel()
        # same physical window at half resolution (~2x context)
        hx, hy = x // 2, y // 2
        p2 = hp[hy:hy + 11, hx:hx + 11].ravel()
        v = np.concatenate([p1 - p1.mean(), p2 - p2.mean()])
        nv = np.linalg.norm(v)
        out[i] = v / (nv + 1e-8)
    return out


# ------------------------------------------------------------------ geometry
def normalize_pts(p):
    c = p.mean(0)
    d = np.sqrt(((p - c) ** 2).sum(1)).mean()
    s = np.sqrt(2) / max(d, 1e-9)
    T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]], np.float64)
    ph = np.hstack([p, np.ones((len(p), 1))]) @ T.T
    return ph[:, :2] / ph[:, 2:], T


def dlt(src, dst):
    ps, Ts = normalize_pts(src)
    pd, Td = normalize_pts(dst)
    n = len(ps)
    A = np.zeros((2 * n, 9), np.float64)
    for i in range(n):
        x, y = ps[i]
        u, v = pd[i]
        A[2 * i] = [-x, -y, -1, 0, 0, 0, u * x, u * y, u]
        A[2 * i + 1] = [0, 0, 0, -x, -y, -1, v * x, v * y, v]
    _, _, vt = np.linalg.svd(A, full_matrices=False)
    Hn = vt[-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ Hn @ Ts
    return H / H[2, 2]


def apply_h(H, p):
    ph = np.hstack([p, np.ones((len(p), 1))]) @ H.T
    return ph[:, :2] / ph[:, 2:3]


def ransac_dlt(src, dst, thresh=2.5, iters=4000, seed=0):
    rng = np.random.default_rng(seed)
    best = (-1, None)
    n = len(src)
    for _ in range(iters):
        idx = rng.choice(n, 4, replace=False)
        try:
            H = dlt(src[idx], dst[idx])
        except np.linalg.LinAlgError:
            continue
        if not np.all(np.isfinite(H)):
            continue
        err = np.linalg.norm(apply_h(H, src) - dst, axis=1)
        inl = err < thresh
        score = int(inl.sum())
        if score > best[0]:
            best = (score, inl)
    return best[1]


def refine_lm(src, dst, H, iters=60):
    def resid(h):
        M = np.append(h, 1.0).reshape(3, 3)
        e = apply_h(M, src) - dst
        return e.ravel()

    h = (H / H[2, 2]).ravel()[:8].astype(np.float64)
    r = resid(h)
    cost = float(r @ r)
    lam = 1e-3
    for _ in range(iters):
        J = np.zeros((len(r), 8))
        for j in range(8):
            hp = h.copy()
            hp[j] += 1e-6
            J[:, j] = (resid(hp) - r) / 1e-6
        A = J.T @ J
        g = J.T @ r
        improved = False
        for _ in range(10):
            try:
                delta = np.linalg.solve(A + lam * np.eye(8), -g)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            hn = h + delta
            rn = resid(hn)
            cn = float(rn @ rn)
            if cn < cost:
                h, r, cost, lam = hn, rn, cn, max(lam / 3, 1e-7)
                improved = True
                break
            lam *= 4
        if not improved:
            break
    return np.append(h, 1.0).reshape(3, 3)


def warp_bilinear(img, H, out_w, out_h):
    """H maps src pixel coords -> dst pixel coords."""
    Hinv = np.linalg.inv(H)
    ys, xs = np.mgrid[0:out_h, 0:out_w].astype(np.float32)
    ph = np.stack([xs, ys, np.ones_like(xs)], -1) @ Hinv.T
    sx = ph[..., 0] / ph[..., 2]
    sy = ph[..., 1] / ph[..., 2]
    h, w = img.shape[:2]
    valid = (sx >= 0) & (sx <= w - 1) & (sy >= 0) & (sy <= h - 1)
    sx = np.clip(sx, 0, w - 1)
    sy = np.clip(sy, 0, h - 1)
    x0 = np.floor(sx).astype(np.int32)
    y0 = np.floor(sy).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    ax = (sx - x0)[..., None]
    ay = (sy - y0)[..., None]
    out = (img[y0, x0] * (1 - ax) * (1 - ay) + img[y0, x1] * ax * (1 - ay) +
           img[y1, x0] * (1 - ax) * ay + img[y1, x1] * ax * ay)
    out = np.clip(out, 0, 1)
    out[~valid] = 0
    return (out * 255).astype(np.uint8), valid


# ---------------------------------------------------------------------- main
def run(a_path, b_path, tag, scale=1.0):
    ga, sa = load_gray(a_path)
    gb, sb = load_gray(b_path)
    print(f"[{tag}] working res: {ga.shape[::-1]} / {gb.shape[::-1]} (scales {sa:.4f}, {sb:.4f})")

    pa = harris(ga, n=1800)
    pb = harris(gb, n=1800)
    da = descriptors(ga, pa, downsample(ga))
    db = descriptors(gb, pb, downsample(gb))
    sim = da @ db.T
    best_b = sim.argmax(1)
    best_a = sim.argmax(0)
    best1 = sim[np.arange(len(pa)), best_b]
    sim_masked = sim.copy()
    sim_masked[np.arange(len(pa)), best_b] = -9
    best2 = sim_masked.max(1)
    ok = (best1 > 0.80) & (best1 > 0.90 * best2)
    ok &= best_a[best_b] == np.arange(len(pa))
    src = pa[ok]
    dst = pb[best_b[ok]]
    print(f"[{tag}] keypoints {len(pa)}/{len(pb)}  mutual+ratio matches: {len(src)}")

    inl = ransac_dlt(src, dst, thresh=2.5)
    print(f"[{tag}] ransac inliers: {int(inl.sum())}/{len(src)}")
    H = dlt(src[inl], dst[inl])
    H = refine_lm(src[inl], dst[inl], H)
    # re-select inliers after refinement
    err = np.linalg.norm(apply_h(H, src) - dst, axis=1)
    inl = err < 2.5
    H = refine_lm(src[inl], dst[inl], H)
    err = np.linalg.norm(apply_h(H, src) - dst, axis=1)
    print(f"[{tag}] final inliers: {int(inl.sum())}, median reproj err {np.median(err[inl]):.2f}px, "
          f"p90 {np.percentile(err[inl], 90):.2f}px")

    # ---- check: warp A into B frame, compare
    A = np.asarray(Image.open(a_path).convert("RGB"), np.float32) / 255.0
    B = np.asarray(Image.open(b_path).convert("RGB"), np.float32) / 255.0
    # H is in working-resolution coordinates of A/B; build full-res versions
    Sa = np.diag([sa, sa, 1.0])
    Sb = np.diag([sb, sb, 1.0])
    H_full = np.linalg.inv(Sb) @ H @ Sa

    out_w, out_h = B.shape[1], B.shape[0]
    Hs = np.diag([0.25, 0.25, 1.0])
    Hq = Hs @ H_full
    warpA, valid = warp_bilinear(A, Hq, B.shape[1] // 4, B.shape[0] // 4)
    Bsmall = np.asarray(Image.fromarray((B * 255).astype(np.uint8)).resize(
        (B.shape[1] // 4, B.shape[0] // 4), Image.LANCZOS), np.float32) / 255.0

    if valid.sum() > 0:
        diff = np.abs(warpA.astype(np.float32) / 255.0 - Bsmall).mean(2)
        diff[~valid] = 0
        print(f"[{tag}] mean |diff| inside overlap: {diff[valid].mean()*255:.1f}/255, "
              f"coverage {valid.mean()*100:.1f}%")

    # overlay: green = warped A, magenta = B
    ov = np.zeros((Bsmall.shape[0], Bsmall.shape[1], 3), np.uint8)
    ov[..., 1] = warpA[..., 1]
    ov[..., 0] = Bsmall[..., 0] * 255
    ov[..., 2] = Bsmall[..., 2] * 255
    Image.fromarray(ov).save(f"out_{tag}_overlay.png")

    # quad of B's frame in A's coordinates + H warp of A alone
    corners = np.array([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]], np.float64)
    quad = apply_h(np.linalg.inv(H_full), corners)
    vis = Image.open(a_path).convert("RGB").resize((A.shape[1] // 3, A.shape[0] // 3))
    dr = ImageDraw.Draw(vis)
    poly = [tuple(p) for p in (quad / 3).tolist()] + [tuple((quad[0] / 3).tolist())]
    dr.line(poly, fill=(0, 255, 0), width=4)
    vis.save(f"out_{tag}_quad_on_source.png")

    warped_full, _ = warp_bilinear(A, H_full, out_w, out_h)
    Image.fromarray(warped_full).resize((out_w // 2, out_h // 2)).save(f"out_{tag}_warpA.png")

    return {
        "tag": tag,
        "pairs": int(len(src)),
        "inliers": int(inl.sum()),
        "median_err_px": float(np.median(err[inl])),
        "H_full": H_full.tolist(),
        "quad_in_source": quad.tolist(),
        "out_size": [out_w, out_h],
        "src_size": [A.shape[1], A.shape[0]],
    }


if __name__ == "__main__":
    base = r"C:/Users/Panart/.lmstudio/scratchpads/of"
    res = []
    res.append(run(f"{base}/photo.jpeg", f"{base}/photo_warped.jpeg", "p1"))
    res.append(run(f"{base}/photo_2.jpeg", f"{base}/photo_2_warped.jpeg", "p2"))
    with open(f"{base}/m0/homography_m0.json", "w") as f:
        json.dump(res, f, indent=2)
    for r in res:
        print("\n", r["tag"], "quad in source:", np.round(np.array(r["quad_in_source"]), 1).tolist())
