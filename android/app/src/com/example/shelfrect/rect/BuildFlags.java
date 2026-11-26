package com.example.shelfrect.rect;

/** Флаги сборки. build_apk.sh генерирует свою версию под вариант (прототип / с обрезкой). */
public final class BuildFlags {

    /** true — обрезать чёрные углы после поворота. */
    public static final boolean CROP_VALID = false;

    private BuildFlags() {
    }
}
