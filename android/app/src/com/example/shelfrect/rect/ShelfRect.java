package com.example.shelfrect.rect;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Random;

/**
 * Ректификация фото полки: кромки полок -> точка схода -> гомография -> варп.
 * Чистый Java без сторонних библиотек; на входе и выходе массив ARGB.
 */
public final class ShelfRect {

    /** Результат обработки кадра. */
    public static final class Result {
        public int[] argb;
        public int width;
        public int height;
        public double[] h = Geo.eye();
        public int rails;
        public int cells;
        public double rmsDeg;
        public double score;
        public double angleDeg;
        public double spreadDeg;
        public double[][] railsSource = new double[0][];
        public boolean ok;
        public String note = "";
    }

    private static final class Line {
        double a;
        double b;
        double angle;
        double support;
        int inliers;
        double cov;
    }

    private static final class Pt {
        double x;
        double y;
        double z;
        int strip;
    }

    private static final int N_STRIPS = 24;
    private static final int MAX_LINES = 8;
    private static final int RANSAC_ITERS = 1200;
    private static final double TOL = 2.0;

    private ShelfRect() {
    }

    public static Result rectify(int[] src, int w, int h, int maxSide) {
        Result res = new Result();
        res.argb = src;
        res.width = w;
        res.height = h;
        if (w < 32 || h < 32) {
            res.note = "too small";
            return res;
        }

        double sw = Math.min(1.0, 900.0 / w);
        int ww = Math.max(32, (int) Math.round(w * sw));
        int wh = Math.max(32, (int) Math.round(h * sw));

        float[] gray = resizeGray(src, w, h, ww, wh);
        float[] resp = horizontalResponse(gray, ww, wh);
        List<Line> rails = detectLines(resp, ww, wh);
        res.rails = rails.size();
        res.cells = Math.max(0, rails.size() - 1);
        double minAng = 1e9;
        double maxAng = -1e9;
        for (Line ln : rails) {
            minAng = Math.min(minAng, ln.angle);
            maxAng = Math.max(maxAng, ln.angle);
        }
        res.spreadDeg = maxAng - minAng;
        res.railsSource = new double[rails.size()][2];
        for (int i = 0; i < rails.size(); i++) {
            res.railsSource[i][0] = rails.get(i).a;
            res.railsSource[i][1] = rails.get(i).b / sw;
        }
        if (rails.size() < 2) {
            res.note = "rails < 2";
            return res;
        }

        double[] ha = Geo.eye();
        double vx = vpX(rails);
        if (!Double.isNaN(vx)) {
            double[] l = new double[]{-1.0, 0.0, vx};
            double nl = Math.sqrt(l[0] * l[0] + l[2] * l[2]);
            l[0] /= nl;
            l[2] /= nl;
            if (l[2] < 0) {
                l[0] = -l[0];
                l[2] = -l[2];
            }
            if (Math.abs(l[2]) < 1e-12) {
                l[2] = 1e-12;
            }
            ha[6] = l[0];
            ha[8] = l[2];
        }

        double[] angs = new double[rails.size()];
        for (int i = 0; i < rails.size(); i++) {
            angs[i] = Geo.lineAngle(ha, rails.get(i).a, rails.get(i).b, ww);
        }
        double angH = Geo.circMean180(angs);
        res.angleDeg = angH;
        double th = Math.toRadians(angH);
        double m00 = Math.cos(th);
        double m10 = Math.sin(th);
        double det = m00;
        double[] hd;
        if (Math.abs(det) > 0.2) {
            hd = new double[]{1.0 / m00, 0, 0, -m10 / m00, 1.0, 0, 0, 0, 1};
        } else {
            hd = Geo.rotation(-angH);
        }
        // H найдена в координатах рабочего кадра -> переводим в координаты исходного
        double[] hh = Geo.mul(Geo.scaling(1.0 / sw), Geo.mul(Geo.mul(hd, ha), Geo.scaling(sw)));

        double minX = 1e18, minY = 1e18, maxX = -1e18, maxY = -1e18;
        double[][] corners = {{0, 0}, {w, 0}, {w, h}, {0, h}};
        for (double[] c : corners) {
            double[] q = Geo.apply(hh, c[0], c[1]);
            minX = Math.min(minX, q[0]);
            maxX = Math.max(maxX, q[0]);
            minY = Math.min(minY, q[1]);
            maxY = Math.max(maxY, q[1]);
        }
        double spanX = maxX - minX;
        double spanY = maxY - minY;
        double sc = Math.min(1.0, (double) maxSide / Math.max(spanX, spanY));
        double[] hf = Geo.mul(Geo.scaling(sc), Geo.mul(Geo.translation(-minX, -minY), hh));
        int outW = Math.max(1, (int) Math.round(spanX * sc));
        int outH = Math.max(1, (int) Math.round(spanY * sc));

        double acc = 0.0;
        for (Line ln : rails) {
            double ang = Geo.lineAngle(hf, ln.a, ln.b / sw, w);
            ang = ((ang + 90.0) % 180.0) - 90.0;
            acc += ang * ang;
        }
        double rms = Math.sqrt(acc / rails.size());
        res.rmsDeg = rms;
        res.score = Math.max(0.0, Math.min(1.0,
                (1.0 - rms / 3.0) * Math.min(1.0, rails.size() / 3.0)));
        res.ok = rms < 1.5;

        res.argb = warp(src, w, h, hf, outW, outH);
        res.width = outW;
        res.height = outH;
        res.h = hf;
        if (!res.ok) {
            res.note = "low confidence";
        }
        return res;
    }

    // ---------------------------------------------------------------- вывод
    private static int[] warp(int[] src, int w, int h, double[] hf, int ow, int oh) {
        double[] hi = Geo.inv(hf);
        int[] out = new int[ow * oh];
        for (int y = 0; y < oh; y++) {
            double Y = y + 0.5;
            for (int x = 0; x < ow; x++) {
                double X = x + 0.5;
                double W = hi[6] * X + hi[7] * Y + hi[8];
                if (Math.abs(W) < 1e-12) {
                    out[y * ow + x] = 0xFF000000;
                    continue;
                }
                double sx = (hi[0] * X + hi[1] * Y + hi[2]) / W - 0.5;
                double sy = (hi[3] * X + hi[4] * Y + hi[5]) / W - 0.5;
                if (sx < 0 || sy < 0 || sx > w - 1 || sy > h - 1) {
                    out[y * ow + x] = 0xFF000000;
                    continue;
                }
                int x0 = (int) sx;
                int y0 = (int) sy;
                int x1 = Math.min(x0 + 1, w - 1);
                int y1 = Math.min(y0 + 1, h - 1);
                double ax = sx - x0;
                double ay = sy - y0;
                int p00 = src[y0 * w + x0];
                int p01 = src[y0 * w + x1];
                int p10 = src[y1 * w + x0];
                int p11 = src[y1 * w + x1];
                int r = bil((p00 >> 16) & 255, (p01 >> 16) & 255, (p10 >> 16) & 255, (p11 >> 16) & 255, ax, ay);
                int g = bil((p00 >> 8) & 255, (p01 >> 8) & 255, (p10 >> 8) & 255, (p11 >> 8) & 255, ax, ay);
                int b = bil(p00 & 255, p01 & 255, p10 & 255, p11 & 255, ax, ay);
                out[y * ow + x] = 0xFF000000 | (r << 16) | (g << 8) | b;
            }
        }
        return out;
    }

    private static int bil(int a, int b, int c, int d, double ax, double ay) {
        double top = a + (b - a) * ax;
        double bot = c + (d - c) * ax;
        int v = (int) Math.round(top + (bot - top) * ay);
        return v < 0 ? 0 : (v > 255 ? 255 : v);
    }

    // ------------------------------------------------------------- анализ
    private static float[] resizeGray(int[] argb, int w, int h, int nw, int nh) {
        float[] g = new float[nw * nh];
        for (int y = 0; y < nh; y++) {
            double sy = (y + 0.5) * h / nh - 0.5;
            int y0 = (int) Math.floor(sy);
            double ay = sy - y0;
            y0 = Math.max(0, Math.min(y0, h - 1));
            int y1 = Math.min(y0 + 1, h - 1);
            for (int x = 0; x < nw; x++) {
                double sx = (x + 0.5) * w / nw - 0.5;
                int x0 = (int) Math.floor(sx);
                double ax = sx - x0;
                x0 = Math.max(0, Math.min(x0, w - 1));
                int x1 = Math.min(x0 + 1, w - 1);
                double v = lum(argb[y0 * w + x0]) * (1 - ax) * (1 - ay)
                        + lum(argb[y0 * w + x1]) * ax * (1 - ay)
                        + lum(argb[y1 * w + x0]) * (1 - ax) * ay
                        + lum(argb[y1 * w + x1]) * ax * ay;
                g[y * nw + x] = (float) v;
            }
        }
        return g;
    }

    private static double lum(int c) {
        int r = (c >> 16) & 255;
        int g = (c >> 8) & 255;
        int b = c & 255;
        return (0.299 * r + 0.587 * g + 0.114 * b) / 255.0;
    }

    private static float[] boxBlur(float[] a, int w, int h) {
        float[] out = new float[a.length];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                float s = 0f;
                for (int dy = -1; dy <= 1; dy++) {
                    int yy = Math.min(h - 1, Math.max(0, y + dy));
                    for (int dx = -1; dx <= 1; dx++) {
                        int xx = Math.min(w - 1, Math.max(0, x + dx));
                        s += a[yy * w + xx];
                    }
                }
                out[y * w + x] = s / 9f;
            }
        }
        return out;
    }

    private static float[] horizontalResponse(float[] gray, int w, int h) {
        float[] b = boxBlur(boxBlur(gray, w, h), w, h);
        float[] r = new float[w * h];
        for (int y = 1; y < h - 1; y++) {
            for (int x = 1; x < w - 1; x++) {
                float gx = (b[y * w + x + 1] - b[y * w + x - 1]) * 0.5f;
                float gy = (b[(y + 1) * w + x] - b[(y - 1) * w + x]) * 0.5f;
                r[y * w + x] = Math.max(0f, Math.abs(gy) - 0.5f * Math.abs(gx));
            }
        }
        return r;
    }

    private static List<Line> detectLines(float[] r, int w, int h) {
        List<Pt> pts = new ArrayList<>();
        int stripsUsed = 0;
        for (int s = 0; s < N_STRIPS; s++) {
            int x0 = (int) Math.round((double) w * s / N_STRIPS);
            int x1 = (int) Math.round((double) w * (s + 1) / N_STRIPS);
            if (x1 - x0 < 3) {
                continue;
            }
            double[] prof = new double[h];
            double mx = 0;
            for (int y = 0; y < h; y++) {
                double sum = 0;
                int row = y * w;
                for (int x = x0; x < x1; x++) {
                    sum += r[row + x];
                }
                prof[y] = sum / (x1 - x0);
                if (prof[y] > mx) {
                    mx = prof[y];
                }
            }
            if (mx < 0.01) {
                continue;
            }
            double[] p = new double[h];
            for (int y = 0; y < h; y++) {
                double acc2 = 0;
                int cnt = 0;
                for (int d = -1; d <= 1; d++) {
                    int yy = y + d;
                    if (yy >= 0 && yy < h) {
                        acc2 += prof[yy];
                        cnt++;
                    }
                }
                p[y] = acc2 / cnt;
            }
            double[] sorted = p.clone();
            Arrays.sort(sorted);
            double med = sorted[h / 2];
            double[] dev = new double[h];
            for (int i = 0; i < h; i++) {
                dev[i] = Math.abs(p[i] - med);
            }
            Arrays.sort(dev);
            double mad = Math.max(dev[h / 2], 1e-3) + 1e-6;

            List<double[]> cand = new ArrayList<>();
            for (int y = 4; y < h - 4; y++) {
                if (p[y] >= p[y - 1] && p[y] >= p[y + 1]) {
                    double z = (p[y] - med) / mad;
                    if (z > 3.0) {
                        cand.add(new double[]{y, z});
                    }
                }
            }
            if (cand.isEmpty()) {
                continue;
            }
            cand.sort((u, v) -> Double.compare(v[1], u[1]));
            List<Integer> keep = new ArrayList<>();
            for (double[] c : cand) {
                int y = (int) c[0];
                boolean far = true;
                for (int k : keep) {
                    if (Math.abs(k - y) < 6) {
                        far = false;
                        break;
                    }
                }
                if (far) {
                    keep.add(y);
                    if (keep.size() >= 24) {
                        break;
                    }
                }
            }
            if (keep.isEmpty()) {
                continue;
            }
            double xc = 0.5 * (x0 + x1);
            for (int y : keep) {
                Pt pt = new Pt();
                pt.x = xc;
                pt.y = y;
                pt.z = (p[y] - med) / mad;
                pt.strip = stripsUsed;
                pts.add(pt);
            }
            stripsUsed++;
        }
        return ransac(pts, w, stripsUsed);
    }

    private static List<Line> ransac(List<Pt> pts, int w, int stripsUsed) {
        List<Line> out = new ArrayList<>();
        int n = pts.size();
        if (n < 4 || stripsUsed < 2) {
            return out;
        }
        boolean[] alive = new boolean[n];
        Arrays.fill(alive, true);
        Random rnd = new Random(7);
        int[] idx = new int[n];
        boolean[] inl = new boolean[n];

        for (int k = 0; k < MAX_LINES; k++) {
            int m = 0;
            for (int i = 0; i < n; i++) {
                if (alive[i]) {
                    idx[m++] = i;
                }
            }
            if (m < 4) {
                break;
            }
            double bestScore = -1;
            double ba = 0;
            double bb = 0;
            boolean[] bestMask = null;
            for (int it = 0; it < RANSAC_ITERS; it++) {
                Pt p1 = pts.get(idx[rnd.nextInt(m)]);
                Pt p2 = pts.get(idx[rnd.nextInt(m)]);
                double dx = p2.x - p1.x;
                if (Math.abs(dx) < 0.25 * w) {
                    continue;
                }
                double a = (p2.y - p1.y) / dx;
                if (Math.abs(a) > 0.6) {
                    continue;
                }
                double b = p1.y - a * p1.x;
                double score = 0;
                for (int j = 0; j < m; j++) {
                    Pt pt = pts.get(idx[j]);
                    boolean in = Math.abs(pt.y - (a * pt.x + b)) < TOL;
                    inl[j] = in;
                    if (in) {
                        score += pt.z;
                    }
                }
                if (score > bestScore) {
                    bestScore = score;
                    ba = a;
                    bb = b;
                    bestMask = inl.clone();
                }
            }
            if (bestMask == null) {
                break;
            }
            double a = ba;
            double b = bb;
            boolean[] cur = bestMask;
            for (int it = 0; it < 3; it++) {
                double sw = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
                for (int j = 0; j < m; j++) {
                    if (!cur[j]) {
                        continue;
                    }
                    Pt pt = pts.get(idx[j]);
                    sw += pt.z;
                    sx += pt.z * pt.x;
                    sy += pt.z * pt.y;
                    sxx += pt.z * pt.x * pt.x;
                    sxy += pt.z * pt.x * pt.y;
                }
                double det = sw * sxx - sx * sx;
                if (Math.abs(det) > 1e-9) {
                    a = (sw * sxy - sx * sy) / det;
                    b = (sy - a * sx) / sw;
                }
                for (int j = 0; j < m; j++) {
                    Pt pt = pts.get(idx[j]);
                    cur[j] = Math.abs(pt.y - (a * pt.x + b)) < TOL;
                }
            }
            int cnt = 0;
            double support = 0;
            double xmin = 1e18;
            double xmax = -1e18;
            HashSet<Integer> strips = new HashSet<>();
            for (int j = 0; j < m; j++) {
                if (!cur[j]) {
                    continue;
                }
                Pt pt = pts.get(idx[j]);
                cnt++;
                support += pt.z;
                strips.add(pt.strip);
                xmin = Math.min(xmin, pt.x);
                xmax = Math.max(xmax, pt.x);
            }
            double cov = strips.size() / (double) stripsUsed;
            if (cnt >= 4 && cov >= 0.5 && (xmax - xmin) >= 0.5 * w) {
                Line ln = new Line();
                ln.a = a;
                ln.b = b;
                ln.angle = Math.toDegrees(Math.atan(a));
                ln.support = support;
                ln.inliers = cnt;
                ln.cov = cov;
                out.add(ln);
            }
            for (int j = 0; j < m; j++) {
                if (cur[j]) {
                    alive[idx[j]] = false;
                }
            }
        }
        out.sort((u, v) -> Double.compare(u.b, v.b));
        return out;
    }

    private static double vpX(List<Line> rails) {
        double a00 = 0, a01 = 0, a11 = 0, b0 = 0, b1 = 0;
        for (Line ln : rails) {
            double l0 = ln.a;
            double l1 = -1.0;
            double l2 = ln.b;
            double nl = Math.sqrt(l0 * l0 + l1 * l1);
            double wt = Math.sqrt(Math.max(ln.support, 1e-6));
            double p = wt * l0 / nl;
            double q = wt * l1 / nl;
            double rr = wt * l2 / nl;
            a00 += p * p;
            a01 += p * q;
            a11 += q * q;
            b0 -= p * rr;
            b1 -= q * rr;
        }
        double det = a00 * a11 - a01 * a01;
        if (Math.abs(det) < 1e-9) {
            return Double.NaN;
        }
        double x = (b0 * a11 - b1 * a01) / det;
        if (!Double.isFinite(x) || Math.abs(x) > 1e7) {
            return Double.NaN;
        }
        return x;
    }

    /**
     * Разметка кадра: красные — найденные кромки полок, зелёные — горизонтали
     * на уровне кромки в центре кадра (видно, насколько кромки «веером»).
     */
    public static int[] drawOverlay(int[] src, int w, int h, double[][] rails) {
        int[] out = src.clone();
        if (rails == null || rails.length == 0) {
            return out;
        }
        int lw = Math.max(4, (int) Math.round(Math.min(w, h) / 300.0));
        for (double[] ln : rails) {
            int x1 = w - 1;
            int y0 = (int) Math.round(ln[1]);
            int y1 = (int) Math.round(ln[0] * (w - 1) + ln[1]);
            drawLine(out, w, h, 0, y0, x1, y1, 0xFFFF2A2A, lw);
        }
        for (double[] ln : rails) {
            int yc = (int) Math.round(ln[0] * (w * 0.5) + ln[1]);
            drawLine(out, w, h, 0, yc, w - 1, yc, 0xFF22DD55, lw);
        }
        return out;
    }

    private static void drawLine(int[] px, int w, int h, int x0, int y0, int x1, int y1,
                                 int color, int width) {
        int steps = Math.max(Math.abs(x1 - x0), Math.abs(y1 - y0)) + 1;
        int rad = Math.max(1, width / 2);
        for (int i = 0; i <= steps; i++) {
            double t = steps == 0 ? 0.0 : (double) i / steps;
            int x = (int) Math.round(x0 + (x1 - x0) * t);
            int y = (int) Math.round(y0 + (y1 - y0) * t);
            for (int dy = -rad; dy <= rad; dy++) {
                int yy = y + dy;
                if (yy < 0 || yy >= h) {
                    continue;
                }
                for (int dx = -rad; dx <= rad; dx++) {
                    int xx = x + dx;
                    if (xx >= 0 && xx < w) {
                        px[yy * w + xx] = color;
                    }
                }
            }
        }
    }
}
