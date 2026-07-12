#!/usr/bin/env bash
# Package sapient-ffi for Android: a drop-in Gradle library module.
#
#   scripts/package-android.sh [--emulator] [--api N] [--out DIR]
#
# Produces (default out dir: dist/mobile):
#   sapient-android/            — a complete `com.android.library` module:
#                                 build.gradle.kts (JNA dep included),
#                                 src/main/jniLibs/arm64-v8a/libsapient_ffi.so,
#                                 src/main/java/uniffi/sapient_ffi/sapient_ffi.kt
#                                 Consume by copying next to your app and adding
#                                 `include(":sapient-android")` to settings.gradle.
#   sapient-android.zip         — the same module, zipped for distribution
#
# --emulator additionally builds the x86_64 ABI (Android emulator on
# x86 hosts; ARM-Mac emulators run the arm64 image and don't need it).
#
# This is a source-module bundle, not a published Maven AAR — building an
# AAR requires the consumer's Gradle/AGP anyway, and a module keeps the
# generated Kotlin readable/debuggable. Maven publishing is a later rung.
#
# Requirements: Android NDK r26+ (auto-located from $ANDROID_NDK_HOME,
# $ANDROID_NDK_LATEST_HOME, or the newest install under the SDK dir);
# rust target(s) auto-added. See docs/MOBILE.md for context + the
# device-testing safety ladder.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="dist/mobile"
API=24
EMULATOR=0
# GPU (wgpu→Vulkan) is ON by default: `Auto` probes for an adapter at load and
# falls back to CPU, so the GPU-featured library is safe on Vulkan-less
# devices and emulators.
FEATURES="--features wgpu"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --emulator) EMULATOR=1; shift ;;
    --api) API="$2"; shift 2 ;;
    --out) OUT_DIR="$2"; shift 2 ;;
    --cpu-only) FEATURES=""; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

# ── Locate the NDK ────────────────────────────────────────────────────────────
find_newest_ndk() {
  local base="$1"
  [[ -d "$base" ]] || return 1
  ls -1 "$base" | sort -V | tail -1 | sed "s|^|$base/|"
}
NDK="${ANDROID_NDK_HOME:-${ANDROID_NDK_LATEST_HOME:-}}"
if [[ -z "$NDK" ]]; then
  for base in "$HOME/Library/Android/sdk/ndk" "$HOME/Android/Sdk/ndk" \
              "/usr/local/lib/android/sdk/ndk"; do
    if NDK=$(find_newest_ndk "$base"); then break; fi
  done
fi
[[ -n "${NDK:-}" && -d "$NDK" ]] || {
  echo "error: Android NDK not found — set ANDROID_NDK_HOME" >&2; exit 1;
}

case "$(uname -s)" in
  Darwin) HOST_TAG=darwin-x86_64 ;;  # NDK ships x86_64-named universal binaries on ARM Macs too
  Linux)  HOST_TAG=linux-x86_64 ;;
  *) echo "error: unsupported host $(uname -s)" >&2; exit 1 ;;
esac
NDK_BIN="$NDK/toolchains/llvm/prebuilt/$HOST_TAG/bin"
[[ -x "$NDK_BIN/aarch64-linux-android$API-clang" ]] || {
  echo "error: clang for API $API not found in $NDK_BIN" >&2; exit 1;
}
echo "==> NDK: $NDK (API $API)"

# ── Build the .so per ABI ─────────────────────────────────────────────────────
build_abi() { # rust-target abi-dir clang-prefix
  local target="$1" abi="$2" prefix="$3"
  rustup target list --installed | grep -q "^$target\$" || rustup target add "$target"
  local tu; tu=$(echo "$target" | tr '-' '_')
  echo "==> Building $target"
  env "CC_${tu}=$NDK_BIN/${prefix}${API}-clang" \
      "CXX_${tu}=$NDK_BIN/${prefix}${API}-clang++" \
      "AR_${tu}=$NDK_BIN/llvm-ar" \
      "CARGO_TARGET_$(echo "$tu" | tr '[:lower:]' '[:upper:]')_LINKER=$NDK_BIN/${prefix}${API}-clang" \
      cargo build -p sapient-ffi --release --target "$target" $FEATURES
  JNILIBS_SRC+=("target/$target/release/libsapient_ffi.so:$abi")
}
JNILIBS_SRC=()
build_abi aarch64-linux-android arm64-v8a aarch64-linux-android
if [[ "$EMULATOR" == "1" ]]; then
  build_abi x86_64-linux-android x86_64 x86_64-linux-android
fi

# ── Generate Kotlin bindings from the host library ───────────────────────────
# ($FEATURES adds no FFI surface — kept here only so back-to-back runs of the
# two packaging scripts don't churn a host rebuild over feature unification.)
echo "==> Generating Kotlin bindings"
cargo build -p sapient-ffi --release $FEATURES
HOST_LIB="target/release/libsapient_ffi.dylib"
[[ -f "$HOST_LIB" ]] || HOST_LIB="target/release/libsapient_ffi.so"
GEN_DIR="$OUT_DIR/generated-kotlin"
rm -rf "$GEN_DIR" && mkdir -p "$GEN_DIR"
cargo run -p sapient-ffi --features bindgen --bin uniffi-bindgen --quiet -- \
  generate --library "$HOST_LIB" --language kotlin --no-format --out-dir "$GEN_DIR"

# ── Verify the .so actually exports the FFI surface ──────────────────────────
echo "==> Verifying exported symbols"
SYMS=$("$NDK_BIN/llvm-nm" -D --defined-only "target/aarch64-linux-android/release/libsapient_ffi.so" | grep -c "uniffi_sapient_ffi_fn" || true)
[[ "$SYMS" -gt 0 ]] || { echo "error: no uniffi exports in the .so" >&2; exit 1; }
echo "    $SYMS uniffi entry points exported"

# ── Assemble the Gradle module ────────────────────────────────────────────────
MOD="$OUT_DIR/sapient-android"
rm -rf "$MOD"
mkdir -p "$MOD/src/main/java/uniffi/sapient_ffi"
cp "$GEN_DIR/uniffi/sapient_ffi/sapient_ffi.kt" "$MOD/src/main/java/uniffi/sapient_ffi/"
for entry in "${JNILIBS_SRC[@]}"; do
  src="${entry%%:*}"; abi="${entry##*:}"
  mkdir -p "$MOD/src/main/jniLibs/$abi"
  cp "$src" "$MOD/src/main/jniLibs/$abi/"
done
cat > "$MOD/build.gradle.kts" <<EOF
// Generated by scripts/package-android.sh — do not edit by hand.
plugins {
    id("com.android.library")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "so.openhorizon.sapient"
    compileSdk = 34
    defaultConfig { minSdk = $API }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
}

dependencies {
    // UniFFI's Kotlin bindings load the native lib through JNA.
    implementation("net.java.dev.jna:jna:5.14.0@aar")
    // The async FFI exports (load_session, chat_async, chat_stream_async,
    // chat_messages_stream) generate suspend functions —
    // suspendCancellableCoroutine needs kotlinx-coroutines.
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.9.0")
}
EOF
cat > "$MOD/README.md" <<'EOF'
# Sapient — Android library module (generated)

Generated by `scripts/package-android.sh` from the SAPIENT repo. Contains the
UniFFI-generated Kotlin bindings + the prebuilt `libsapient_ffi.so`
(arm64-v8a; `--emulator` adds x86_64).

Consume as a local Gradle module:

1. Copy this directory next to your app module.
2. `settings.gradle.kts`: `include(":sapient-android")`
3. App `build.gradle.kts`: `implementation(project(":sapient-android"))`

```kotlin
import uniffi.sapient_ffi.*

// Call from Dispatchers.IO — load() blocks (and downloads on first run).
val session = LlmSession.load(
    "qwen2.5-0.5b",
    GenerationOptions(maxTokens = 256u))
val reply = session.chat("Hi!")
```

Point the model cache at app storage before the first `load()`:
`Os.setenv("HF_HOME", context.cacheDir.resolve("sapient").path, true)`.

BEFORE running on a personal device, read `docs/MOBILE.md` §5 (the
safe-testing ladder) in the SAPIENT repo.
EOF

(cd "$OUT_DIR" && rm -f sapient-android.zip && zip -qry sapient-android.zip sapient-android)

echo "==> Done:"
du -sh "$MOD" "$OUT_DIR/sapient-android.zip" 2>/dev/null || true
