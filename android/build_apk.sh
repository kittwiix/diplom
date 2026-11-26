#!/usr/bin/env bash
# Сборка APK без Gradle: aapt2 -> javac -> d8 -> zipalign -> apksigner
set -euo pipefail

SDK="${ANDROID_SDK:-C:/Users/Panart/Android/Sdk}"
BT="$SDK/build-tools/34.0.0"
JAR="$SDK/platforms/android-34/android.jar"
ROOT="$(cd "$(dirname "$0")" && pwd)"
APP="$ROOT/app"
B="$ROOT/build/apk"

rm -rf "$B" "$ROOT/build/gen" "$ROOT/build/classes-android" "$ROOT/build/dex"
mkdir -p "$B" "$ROOT/build/gen" "$ROOT/build/classes-android" "$ROOT/build/dex"

"$BT/aapt2.exe" compile --dir "$APP/res" -o "$B/res.zip"
"$BT/aapt2.exe" link -o "$B/app-unsigned.apk" -I "$JAR" \
    --manifest "$APP/AndroidManifest.xml" -R "$B/res.zip" \
    --java "$ROOT/build/gen" --auto-add-overlay

javac --release 8 -encoding UTF-8 -classpath "$JAR" -d "$ROOT/build/classes-android" \
    $(find "$ROOT/build/gen" "$APP/src" -name "*.java")

"$BT/d8.bat" --release --lib "$JAR" --min-api 24 --output "$ROOT/build/dex" \
    $(find "$ROOT/build/classes-android" -name "*.class")

python - "$B" "$ROOT" <<'PY'
import shutil, sys, zipfile
b, root = sys.argv[1], sys.argv[2]
shutil.copy(b + "/app-unsigned.apk", b + "/app-dex.apk")
with zipfile.ZipFile(b + "/app-dex.apk", "a", zipfile.ZIP_DEFLATED) as z:
    z.write(root + "/build/dex/classes.dex", "classes.dex")
PY

"$BT/zipalign.exe" -f 4 "$B/app-dex.apk" "$B/app-aligned.apk"

KS="$ROOT/keystore/debug.keystore"
mkdir -p "$ROOT/keystore"
if [ ! -f "$KS" ]; then
    keytool -genkeypair -noprompt -keystore "$KS" -storepass android -keypass android \
        -alias androiddebugkey -keyalg RSA -keysize 2048 -validity 10000 \
        -dname "CN=Android Debug,O=Android,C=US" >/dev/null 2>&1
fi

"$BT/apksigner.bat" sign --ks "$KS" --ks-pass pass:android --key-pass pass:android \
    --out "$B/ShelfRect.apk" "$B/app-aligned.apk"
"$BT/apksigner.bat" verify --print-certs "$B/ShelfRect.apk" | head -2
echo "APK: $B/ShelfRect.apk"
