package com.example.shelfrect;

import android.app.Activity;
import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.graphics.drawable.BitmapDrawable;
import android.graphics.drawable.Drawable;
import android.media.ExifInterface;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import java.io.OutputStream;

import com.example.shelfrect.rect.ShelfRect;

import java.io.InputStream;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/** Простое приложение: фото из галереи -> выпрямленный кадр. */
public class MainActivity extends Activity {

    private static final int REQ_PICK = 1001;
    private static final int MAX_SIDE = 1400;

    private final ExecutorService pool = Executors.newSingleThreadExecutor();

    private ImageView imgBefore;
    private ImageView imgAfter;
    private TextView tvInfo;
    private Button btnSave;
    private Bitmap outBitmap;
    private Bitmap markupBitmap;

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        setContentView(R.layout.main);

        imgBefore = findViewById(R.id.imgBefore);
        imgAfter = findViewById(R.id.imgAfter);
        tvInfo = findViewById(R.id.tvInfo);
        btnSave = findViewById(R.id.btnSave);

        findViewById(R.id.btnPick).setOnClickListener(v -> pick());
        btnSave.setOnClickListener(v -> saveBitmap(outBitmap, "shelf_"));

        String dbgPath = getIntent() != null ? getIntent().getStringExtra("image_path") : null;
        if (dbgPath != null && dbgPath.length() > 0) {
            tvInfo.setText("Загрузка…");
            pool.execute(() -> processFile(dbgPath));
        }
        findViewById(R.id.btnSaveMarkup).setOnClickListener(
                v -> saveBitmap(markupBitmap, "shelf_markup_"));
        btnSave.setEnabled(false);
    }

    @Override
    protected void onNewIntent(Intent intent) {
        super.onNewIntent(intent);
        setIntent(intent);
        String p = intent != null ? intent.getStringExtra("image_path") : null;
        if (p != null && p.length() > 0) {
            tvInfo.setText("Загрузка…");
            pool.execute(() -> processFile(p));
        }
    }

    private void pick() {
        Intent i = new Intent(Intent.ACTION_GET_CONTENT);
        i.setType("image/*");
        i.addCategory(Intent.CATEGORY_OPENABLE);
        startActivityForResult(Intent.createChooser(i, "Фото полки"), REQ_PICK);
    }


    /** Освобождает битмап, который сейчас показан в ImageView. */
    private static void releaseImage(ImageView view) {
        Drawable d = view.getDrawable();
        view.setImageDrawable(null);
        if (d instanceof BitmapDrawable) {
            Bitmap b = ((BitmapDrawable) d).getBitmap();
            if (b != null && !b.isRecycled()) {
                b.recycle();
            }
        }
    }

    private void ui(Runnable r) {
        runOnUiThread(r);
    }

    private static String fmt(double v) {
        return String.format(Locale.US, "%.2f", v);
    }

    @Override
    protected void onActivityResult(int req, int res, Intent data) {
        super.onActivityResult(req, res, data);
        if (req != REQ_PICK || res != RESULT_OK || data == null || data.getData() == null) {
            return;
        }
        final Uri uri = data.getData();
        tvInfo.setText("Загрузка…");
        pool.execute(() -> process(uri));
    }

    private void process(Uri uri) {
        Bitmap bmp;
        try {
            bmp = load(uri);
        } catch (Exception e) {
            bmp = null;
        }
        handle(bmp);
    }

    private void processFile(String path) {
        Bitmap bmp;
        try {
            bmp = loadFile(path);
        } catch (Exception e) {
            bmp = null;
        }
        handle(bmp);
    }

    private void handle(Bitmap loaded) {
        try {
            final Bitmap src = loaded;
            if (src == null) {
                ui(() -> tvInfo.setText("Не удалось открыть фото"));
                return;
            }
            ui(() -> {
                releaseImage(imgBefore);
                releaseImage(imgAfter);
                markupBitmap = null;
                outBitmap = null;
                btnSave.setEnabled(false);
                imgBefore.setImageBitmap(src);
                tvInfo.setText("Обработка…");
            });

            int w = src.getWidth();
            int h = src.getHeight();
            int[] px = new int[w * h];
            src.getPixels(px, 0, w, 0, 0, w, h);

            android.util.Log.i("ShelfRect", "img " + w + "x" + h + " cfg=" + src.getConfig());
            long t0 = System.currentTimeMillis();
            ShelfRect.Result r = ShelfRect.rectify(px, w, h, MAX_SIDE);
            long dt = System.currentTimeMillis() - t0;
            StringBuilder dbg = new StringBuilder();
            for (double[] ln : r.railsSource) {
                dbg.append(String.format(Locale.US, " (%.4f,%.1f)", ln[0], ln[1]));
            }
            android.util.Log.i("ShelfRect", "rails=" + r.rails + " rms=" + r.rmsDeg
                    + " note=" + r.note + dbg);

            if (r.rails < 2) {
                ui(() -> tvInfo.setText("Полки не найдены — нужно фото, где видны линии ценников."));
                return;
            }
            final Bitmap out = Bitmap.createBitmap(r.argb, r.width, r.height, Bitmap.Config.ARGB_8888);
            final Bitmap markup = Bitmap.createBitmap(
                    ShelfRect.drawOverlay(px, w, h, r.railsSource), w, h, Bitmap.Config.ARGB_8888);
            markupBitmap = markup;
            final String msg = "Кромок полок: " + r.rails + "   ·   полос: " + r.cells
                    + "\nРазброс углов до: " + fmt(r.spreadDeg) + "°  →  после: " + fmt(r.rmsDeg) + "°"
                    + "\n" + dt + " мс" + (r.ok ? "" : "   ·   низкая уверенность");
            outBitmap = out;
            ui(() -> {
                releaseImage(imgBefore);
                releaseImage(imgAfter);
                imgBefore.setImageBitmap(markup);
                imgAfter.setImageBitmap(out);
                tvInfo.setText(msg);
                btnSave.setEnabled(true);
            });
        } catch (Throwable t) {
            final String m = t.getClass().getSimpleName() + ": " + t.getMessage();
            ui(() -> tvInfo.setText("Ошибка: " + m));
        }
    }

    private Bitmap loadFile(String path) {
        BitmapFactory.Options bounds = new BitmapFactory.Options();
        bounds.inJustDecodeBounds = true;
        BitmapFactory.decodeFile(path, bounds);
        int sample = 1;
        while (bounds.outWidth / sample > 1800 || bounds.outHeight / sample > 1800) {
            sample *= 2;
        }
        BitmapFactory.Options opts = new BitmapFactory.Options();
        opts.inSampleSize = sample;
        Bitmap bmp = BitmapFactory.decodeFile(path, opts);
        if (bmp == null) {
            return null;
        }
        try {
            ExifInterface ex = new ExifInterface(path);
            int ori = ex.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL);
            int rot = 0;
            if (ori == ExifInterface.ORIENTATION_ROTATE_90) {
                rot = 90;
            } else if (ori == ExifInterface.ORIENTATION_ROTATE_180) {
                rot = 180;
            } else if (ori == ExifInterface.ORIENTATION_ROTATE_270) {
                rot = 270;
            }
            if (rot != 0) {
                Matrix m = new Matrix();
                m.postRotate(rot);
                Bitmap rotated = Bitmap.createBitmap(bmp, 0, 0, bmp.getWidth(), bmp.getHeight(), m, true);
                if (rotated != bmp) {
                    bmp.recycle();
                }
                bmp = rotated;
            }
        } catch (Exception ignored) {
            // EXIF не обязателен
        }
        return bmp;
    }

    private Bitmap load(Uri uri) throws Exception {
        ContentResolver cr = getContentResolver();
        BitmapFactory.Options bounds = new BitmapFactory.Options();
        bounds.inJustDecodeBounds = true;
        try (InputStream in = cr.openInputStream(uri)) {
            BitmapFactory.decodeStream(in, null, bounds);
        }
        int sample = 1;
        while (bounds.outWidth / sample > 1800 || bounds.outHeight / sample > 1800) {
            sample *= 2;
        }
        BitmapFactory.Options opts = new BitmapFactory.Options();
        opts.inSampleSize = sample;
        Bitmap bmp;
        try (InputStream in = cr.openInputStream(uri)) {
            bmp = BitmapFactory.decodeStream(in, null, opts);
        }
        if (bmp == null) {
            return null;
        }
        int rot = 0;
        try (InputStream in = cr.openInputStream(uri)) {
            ExifInterface ex = new ExifInterface(in);
            int ori = ex.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL);
            if (ori == ExifInterface.ORIENTATION_ROTATE_90) {
                rot = 90;
            } else if (ori == ExifInterface.ORIENTATION_ROTATE_180) {
                rot = 180;
            } else if (ori == ExifInterface.ORIENTATION_ROTATE_270) {
                rot = 270;
            }
        } catch (Exception ignored) {
            // EXIF не обязателен
        }
        if (rot != 0) {
            Matrix m = new Matrix();
            m.postRotate(rot);
            bmp = Bitmap.createBitmap(bmp, 0, 0, bmp.getWidth(), bmp.getHeight(), m, true);
        }
        return bmp;
    }

    private void saveBitmap(Bitmap bmp, String prefix) {
        if (bmp == null) {
            return;
        }
        if (Build.VERSION.SDK_INT <= 28
                && checkSelfPermission(android.Manifest.permission.WRITE_EXTERNAL_STORAGE)
                != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{android.Manifest.permission.WRITE_EXTERNAL_STORAGE}, 42);
            Toast.makeText(this, "Разрешите доступ к памяти и нажмите ещё раз", Toast.LENGTH_LONG).show();
            return;
        }
        try {
            String name = prefix + System.currentTimeMillis() + ".jpg";
            ContentValues cv = new ContentValues();
            cv.put(MediaStore.Images.Media.DISPLAY_NAME, name);
            cv.put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg");
            if (Build.VERSION.SDK_INT >= 29) {
                cv.put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/ShelfRect");
                cv.put(MediaStore.Images.Media.IS_PENDING, 1);
            }
            Uri uri = getContentResolver().insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, cv);
            if (uri == null) {
                Toast.makeText(this, "Не удалось сохранить", Toast.LENGTH_LONG).show();
                return;
            }
            try (OutputStream os = getContentResolver().openOutputStream(uri)) {
                bmp.compress(Bitmap.CompressFormat.JPEG, 95, os);
            }
            if (Build.VERSION.SDK_INT >= 29) {
                cv.clear();
                cv.put(MediaStore.Images.Media.IS_PENDING, 0);
                getContentResolver().update(uri, cv, null, null);
            }
            Toast.makeText(this, "Сохранено: Pictures/ShelfRect", Toast.LENGTH_LONG).show();
        } catch (Exception e) {
            Toast.makeText(this, "Ошибка сохранения: " + e.getMessage(), Toast.LENGTH_LONG).show();
        }
    }
}
