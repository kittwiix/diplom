package com.example.shelfrect;

import android.app.Activity;
import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
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
    private static final int MAX_SIDE = 1600;

    private final ExecutorService pool = Executors.newSingleThreadExecutor();

    private ImageView imgBefore;
    private ImageView imgAfter;
    private TextView tvInfo;
    private Button btnSave;
    private Bitmap outBitmap;

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        setContentView(R.layout.main);

        imgBefore = findViewById(R.id.imgBefore);
        imgAfter = findViewById(R.id.imgAfter);
        tvInfo = findViewById(R.id.tvInfo);
        btnSave = findViewById(R.id.btnSave);

        findViewById(R.id.btnPick).setOnClickListener(v -> pick());
        btnSave.setOnClickListener(v -> saveToGallery());
        btnSave.setEnabled(false);
    }

    private void pick() {
        Intent i = new Intent(Intent.ACTION_GET_CONTENT);
        i.setType("image/*");
        i.addCategory(Intent.CATEGORY_OPENABLE);
        startActivityForResult(Intent.createChooser(i, "Фото полки"), REQ_PICK);
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
        try {
            final Bitmap src = load(uri);
            if (src == null) {
                ui(() -> tvInfo.setText("Не удалось открыть фото"));
                return;
            }
            ui(() -> {
                imgBefore.setImageBitmap(src);
                imgAfter.setImageBitmap(null);
                tvInfo.setText("Обработка…");
                btnSave.setEnabled(false);
            });

            int w = src.getWidth();
            int h = src.getHeight();
            int[] px = new int[w * h];
            src.getPixels(px, 0, w, 0, 0, w, h);

            long t0 = System.currentTimeMillis();
            ShelfRect.Result r = ShelfRect.rectify(px, w, h, MAX_SIDE);
            long dt = System.currentTimeMillis() - t0;

            if (r.rails < 2) {
                ui(() -> tvInfo.setText("Полки не найдены — нужно фото, где видны линии ценников."));
                return;
            }
            final Bitmap out = Bitmap.createBitmap(r.argb, r.width, r.height, Bitmap.Config.ARGB_8888);
            final String msg = "Кромок полок: " + r.rails + "   ·   полос: " + r.cells
                    + "\nОстаточный наклон: " + fmt(r.rmsDeg) + "°   ·   " + dt + " мс"
                    + (r.ok ? "" : "\nНизкая уверенность — проверьте результат");
            outBitmap = out;
            ui(() -> {
                imgAfter.setImageBitmap(out);
                tvInfo.setText(msg);
                btnSave.setEnabled(true);
            });
        } catch (Throwable t) {
            final String m = t.getClass().getSimpleName() + ": " + t.getMessage();
            ui(() -> tvInfo.setText("Ошибка: " + m));
        }
    }

    private Bitmap load(Uri uri) throws Exception {
        ContentResolver cr = getContentResolver();
        BitmapFactory.Options bounds = new BitmapFactory.Options();
        bounds.inJustDecodeBounds = true;
        try (InputStream in = cr.openInputStream(uri)) {
            BitmapFactory.decodeStream(in, null, bounds);
        }
        int sample = 1;
        while (bounds.outWidth / sample > 2200 || bounds.outHeight / sample > 2200) {
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

    private void saveToGallery() {
        if (outBitmap == null) {
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
            String name = "shelf_" + System.currentTimeMillis() + ".jpg";
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
                outBitmap.compress(Bitmap.CompressFormat.JPEG, 95, os);
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
