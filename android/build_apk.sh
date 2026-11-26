#!/usr/bin/env bash
# Сборка APK без Gradle: aapt2 -> javac -> d8 -> zipalign -> apksigner
# Использование: ./build_apk.sh [proto|crop]
set -euo pipefail

VARIANT="${1:-proto}"
SDK="${ANDROID_SDK:-C:/Users/Panart/Android/Sdk}"
BT="$SDK/build-tools/34.0.0"
JAR="$SDK/platforms/android-34/android.jar"
ROOT="$(cd "$(dirname "$0")" && pwd)"
APP="$ROOT/app"
B="$ROOT/build/apk"

if [ "$VARIANT" = "crop" ]; then
    MANIFEST="$APP/AndroidManifest-crop.xml"
    OUT="$B/ShelfRect_crop.apk"
    CROP=true
else
    MANIFEST="$APP/AndroidManifest.xml"
    OUT="$B/ShelfRect.apk"
    CROP=false
fi
WORK="$ROOT/build/$VARIANT"

rm -rf "$WORK"
mkdir -p "$WORK/gen" "$WORK/classes" "$WORK/dex" "$B" "$WORK/flags/com/example/shelfrect/rect"

cat > "$WORK/flags/com/example/shelfrect/rect/BuildFlags.java" <<JEOF
package com.example.shelfrect.rect;

public final class BuildFlags {
    public static final boolean CROP_VALID = $CROP;

    private BuildFlags() {
    }
}
JEOF

"$BT/aapt2.exe" compile --dir "$APP/res" -o "$WORK/res.zip"
"$BT/aapt2.exe" link -o "$WORK/app-unsigned.apk" -I "$JAR" \
    --manifest "$MANIFEST" -R "$WORK/res.zip" \
    --java "$WORK/gen" --custom-package com.example.shelfrect --auto-add-overlay

SRC=$(find "$WORK/gen" "$APP/src" -name "*.java" | grep -v "BuildFlags.java")
javac --release 8 -encoding UTF-8 -classpath "$JAR" -d "$WORK/classes" $SRC \
    "$WORK/flags/com/example/shelfrect/rect/BuildFlags.java"

"$BT/d8.bat" --release --lib "$JAR" --min-api 24 --output "$WORK/dex" \
    $(find "$WORK/classes" -name "*.class")

python - "$WORK" <<'PY'
import shutil, sys, zipfile
work = sys.argv[1]
shutil.copy(work + "/app-unsigned.apk", work + "/app-dex.apk")
with zipfile.ZipFile(work + "/app-dex.apk", "a", zipfile.ZIP_DEFLATED) as z:
    z.write(work + "/dex/classes.dex", "classes.dex")
PY

"$BT/zipalign.exe" -f 4 "$WORK/app-dex.apk" "$WORK/app-aligned.apk"

KS="$ROOT/keystore/debug.keystore"
mkdir -p "$ROOT/keystore"
if [ ! -f "$KS" ]; then
    keytool -genkeypair -noprompt -keystore "$KS" -storepass android -keypass android \
        -alias androiddebugkey -keyalg RSA -keysize 2048 -validity 10000 \
        -dname "CN=Android Debug,O=Android,C=US" >/dev/null 2>&1
fi

"$BT/apksigner.bat" sign --ks "$KS" --ks-pass pass:android --key-pass pass:android \
    --out "$OUT" "$WORK/app-aligned.apk"
echo "APK[$VARIANT]: $OUT"
