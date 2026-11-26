package com.example.shelfrect.rect;

/** Минимальная линейная алгебра 3x3 (row-major), собственные реализации. */
public final class Geo {

    private Geo() {
    }

    public static double[] eye() {
        return new double[]{1, 0, 0, 0, 1, 0, 0, 0, 1};
    }

    public static double[] mul(double[] a, double[] b) {
        double[] r = new double[9];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                r[i * 3 + j] = a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
            }
        }
        return r;
    }

    /** p' = H p с делением на однородную координату. */
    public static double[] apply(double[] h, double x, double y) {
        double X = h[0] * x + h[1] * y + h[2];
        double Y = h[3] * x + h[4] * y + h[5];
        double W = h[6] * x + h[7] * y + h[8];
        if (Math.abs(W) < 1e-12) {
            W = 1e-12;
        }
        return new double[]{X / W, Y / W};
    }

    public static double[] inv(double[] m) {
        double a = m[0], b = m[1], c = m[2];
        double d = m[3], e = m[4], f = m[5];
        double g = m[6], h = m[7], i = m[8];
        double det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
        if (Math.abs(det) < 1e-12) {
            return eye();
        }
        double id = 1.0 / det;
        return new double[]{
                (e * i - f * h) * id, (c * h - b * i) * id, (b * f - c * e) * id,
                (f * g - d * i) * id, (a * i - c * g) * id, (c * d - a * f) * id,
                (d * h - e * g) * id, (b * g - a * h) * id, (a * e - b * d) * id
        };
    }

    public static double[] translation(double tx, double ty) {
        return new double[]{1, 0, tx, 0, 1, ty, 0, 0, 1};
    }

    public static double[] scaling(double s) {
        return new double[]{s, 0, 0, 0, s, 0, 0, 0, 1};
    }

    public static double[] rotation(double deg) {
        double t = Math.toRadians(deg);
        double c = Math.cos(t), s = Math.sin(t);
        return new double[]{c, -s, 0, s, c, 0, 0, 0, 1};
    }

    /** Угол прямой y = a x + b после преобразования H, градусы 0..180. */
    public static double lineAngle(double[] h, double a, double b, double extent) {
        double[] p1 = apply(h, 0, b);
        double[] p2 = apply(h, extent, a * extent + b);
        double ang = Math.toDegrees(Math.atan2(p2[1] - p1[1], p2[0] - p1[0]));
        return ((ang % 180.0) + 180.0) % 180.0;
    }

    /** Круговое среднее углов, определённых с точностью до 180 градусов. */
    public static double circMean180(double[] angles) {
        double sx = 0, sy = 0;
        for (double v : angles) {
            double t = Math.toRadians(v * 2.0);
            sx += Math.cos(t);
            sy += Math.sin(t);
        }
        return Math.toDegrees(Math.atan2(sy, sx)) / 2.0;
    }
}
