import com.example.shelfrect.rect.ShelfRect;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;

/** Десктопная проверка ядра: те же классы, что и в приложении. */
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
            BufferedImage out = new BufferedImage(r.width, r.height, BufferedImage.TYPE_INT_RGB);
            out.setRGB(0, 0, r.width, r.height, r.argb, 0, r.width);
            String base = new File(path).getName().replace('.', '_');
            ImageIO.write(out, "jpg", new File("out_" + base + ".jpg"));
            System.out.printf("%-22s rails=%2d cells=%d rms=%5.2f score=%.2f ok=%-5s %dx%d %dms%n",
                    new File(path).getName(), r.rails, r.cells, r.rmsDeg, r.score, String.valueOf(r.ok),
                    r.width, r.height, dt);
        }
    }
}
