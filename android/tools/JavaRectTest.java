import com.example.shelfrect.rect.ShelfRect;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;

/** Десктопная проверка ядра и разметки: те же классы, что и в приложении. */
public final class JavaRectTest {
    public static void main(String[] args) throws Exception {
        for (String path : args) {
            BufferedImage img = ImageIO.read(new File(path));
            int w = img.getWidth();
            int h = img.getHeight();
            int[] px = img.getRGB(0, 0, w, h, null, 0, w);
            long t0 = System.currentTimeMillis();
            ShelfRect.Result r = ShelfRect.rectify(px, w, h, 1600);
            long dt = System.currentTimeMillis() - t0;
            String base = new File(path).getName().replace('.', '_');

            BufferedImage out = new BufferedImage(r.width, r.height, BufferedImage.TYPE_INT_RGB);
            out.setRGB(0, 0, r.width, r.height, r.argb, 0, r.width);
            ImageIO.write(out, "jpg", new File("out_" + base + ".jpg"));

            int[] ov = ShelfRect.drawOverlay(px, w, h, r.railsSource);
            BufferedImage ovImg = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
            ovImg.setRGB(0, 0, w, h, ov, 0, w);
            ImageIO.write(ovImg, "jpg", new File("overlay_" + base + ".jpg"));

            System.out.printf("%-22s rails=%2d cells=%d spread=%5.1f rms=%5.2f ok=%-5s %dx%d %dms%n",
                    new File(path).getName(), r.rails, r.cells, r.spreadDeg, r.rmsDeg,
                    String.valueOf(r.ok), r.width, r.height, dt);
        }
    }
}
