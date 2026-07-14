#!/usr/bin/env bash
# Package sapient-ffi for Apple platforms: XCFramework + Swift Package.
#
#   scripts/package-swift.sh [--smoke] [--out DIR]
#
# Produces (default out dir: dist/mobile):
#   SapientFFI.xcframework      — static libs for iOS device, iOS simulator,
#                                 and macOS (arm64), with the generated C
#                                 header + modulemap per slice
#   sapient-swift/              — a local Swift Package: the generated
#                                 sapient_ffi.swift source + the XCFramework
#                                 as a binaryTarget. Drag into Xcode or
#                                 depend on it with a path dependency.
#   sapient-swift.zip           — the same package, zipped for distribution
#   SapientFFI.xcframework.zip  — the bare XCFramework (framework at zip root,
#                                 as SwiftPM requires) + .sha256 — the release
#                                 asset that openhorizon-labs/sapient-swift's
#                                 remote `.binaryTarget(url:checksum:)` points
#                                 at, so consumers add the package by URL.
#
# --smoke additionally compiles and RUNS a small macOS executable against
# the packaged static lib (version + catalog + alias resolution) — the
# end-to-end proof that the bindings, header, and link flags are coherent.
#
# Requirements: macOS with Xcode (xcodebuild, swiftc), rustup targets
# aarch64-apple-ios, aarch64-apple-ios-sim (auto-added if missing).
# See docs/MOBILE.md for the wider context and the device-testing ladder.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="dist/mobile"
SMOKE=0
# GPU (wgpu→Metal) is ON by default: `Auto` probes for an adapter at load and
# falls back to CPU, so the GPU-featured library is safe everywhere.
FEATURES="--features wgpu"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) SMOKE=1; shift ;;
    --out) OUT_DIR="$2"; shift 2 ;;
    --cpu-only) FEATURES=""; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

# The C deps (onig_sys) must target the same min-iOS as the Rust link, or
# the link dies on ___chkstk_darwin — see docs/MOBILE.md troubleshooting.
# The macOS floor matches Package.swift's .macOS(.v12).
export IPHONEOS_DEPLOYMENT_TARGET="${IPHONEOS_DEPLOYMENT_TARGET:-14.0}"
export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-12.0}"

IOS_TARGET=aarch64-apple-ios
SIM_TARGET=aarch64-apple-ios-sim
SIM_X86_TARGET=x86_64-apple-ios   # x86_64 slice IS the Intel simulator
MAC_TARGET=aarch64-apple-darwin

echo "==> Ensuring rust targets"
for t in "$IOS_TARGET" "$SIM_TARGET" "$SIM_X86_TARGET"; do
  rustup target list --installed | grep -q "^$t\$" || rustup target add "$t"
done

echo "==> Building sapient-ffi static libs (this is the slow part)${FEATURES:+ [gpu: wgpu]}"
cargo build -p sapient-ffi --release --target "$IOS_TARGET" $FEATURES
cargo build -p sapient-ffi --release --target "$SIM_TARGET" $FEATURES
cargo build -p sapient-ffi --release --target "$SIM_X86_TARGET" $FEATURES
cargo build -p sapient-ffi --release --target "$MAC_TARGET" $FEATURES
# Host dylib for the bindings generator (reads exported metadata from it).
cargo build -p sapient-ffi --release $FEATURES

# Universal simulator library — Xcode's generic iOS Simulator destination
# links BOTH arm64 and x86_64; an arm64-only slice fails with
# "symbol(s) not found for architecture x86_64" (and Intel-Mac devs need it).
echo "==> Creating universal simulator library (arm64 + x86_64)"
SIM_UNIVERSAL_DIR="target/ios-sim-universal/release"
mkdir -p "$SIM_UNIVERSAL_DIR"
lipo -create \
  "target/$SIM_TARGET/release/libsapient_ffi.a" \
  "target/$SIM_X86_TARGET/release/libsapient_ffi.a" \
  -output "$SIM_UNIVERSAL_DIR/libsapient_ffi.a"

echo "==> Generating Swift bindings"
GEN_DIR="$OUT_DIR/generated-swift"
rm -rf "$GEN_DIR" && mkdir -p "$GEN_DIR"
cargo run -p sapient-ffi --features bindgen --bin uniffi-bindgen --quiet -- \
  generate --library target/release/libsapient_ffi.dylib \
  --language swift --out-dir "$GEN_DIR"

# Per-slice headers dir: header + modulemap (renamed to module.modulemap so
# clang picks it up inside the framework Headers dir).
HEADERS_DIR="$OUT_DIR/headers"
rm -rf "$HEADERS_DIR" && mkdir -p "$HEADERS_DIR"
cp "$GEN_DIR/sapient_ffiFFI.h" "$HEADERS_DIR/"
cp "$GEN_DIR/sapient_ffiFFI.modulemap" "$HEADERS_DIR/module.modulemap"

echo "==> Assembling SapientFFI.xcframework"
XCF="$OUT_DIR/SapientFFI.xcframework"
rm -rf "$XCF"
xcodebuild -create-xcframework \
  -library "target/$IOS_TARGET/release/libsapient_ffi.a" -headers "$HEADERS_DIR" \
  -library "$SIM_UNIVERSAL_DIR/libsapient_ffi.a" -headers "$HEADERS_DIR" \
  -library "target/$MAC_TARGET/release/libsapient_ffi.a" -headers "$HEADERS_DIR" \
  -output "$XCF"

echo "==> Assembling Swift Package"
PKG="$OUT_DIR/sapient-swift"
rm -rf "$PKG"
mkdir -p "$PKG/Sources/Sapient" "$PKG/Frameworks"
cp "$GEN_DIR/sapient_ffi.swift" "$PKG/Sources/Sapient/"
cp -R "$XCF" "$PKG/Frameworks/"
cat > "$PKG/Package.swift" <<'EOF'
// swift-tools-version:5.9
// Generated by scripts/package-swift.sh — do not edit by hand.
import PackageDescription

let package = Package(
    name: "Sapient",
    platforms: [.iOS(.v14), .macOS(.v12)],
    products: [
        .library(name: "Sapient", targets: ["Sapient"])
    ],
    targets: [
        .binaryTarget(name: "SapientFFI", path: "Frameworks/SapientFFI.xcframework"),
        .target(
            name: "Sapient",
            dependencies: ["SapientFFI"],
            path: "Sources/Sapient",
            linkerSettings: [
                // The Rust staticlib carries C++ (esaxx) and iconv users.
                .linkedLibrary("c++"),
                .linkedLibrary("iconv"),
                // reqwest/hyper-util read system proxy settings on Apple.
                .linkedFramework("SystemConfiguration"),
                .linkedFramework("CoreFoundation"),
                // wgpu's Metal backend (GPU inference) — CAMetalLayer &co.
                .linkedFramework("Metal"),
                .linkedFramework("QuartzCore"),
            ]
        ),
    ]
)
EOF
cat > "$PKG/README.md" <<'EOF'
# Sapient — Swift Package (generated)

Generated by `scripts/package-swift.sh` from the SAPIENT repo. Contains the
UniFFI-generated Swift bindings + a static XCFramework (iOS device, iOS
simulator, macOS arm64).

Use it as a local package dependency (`Add Package Dependency → Add Local…`
in Xcode, or a `path:` dependency in Package.swift), then:

```swift
import Sapient

let session = try LlmSession.load(
    model: "qwen2.5-0.5b",
    options: GenerationOptions(maxTokens: 256))
print(try session.chat(userMessage: "Hi!"))
```

Call from a background queue — `load` blocks and downloads on first run.
BEFORE running on a personal device, read `docs/MOBILE.md` §5 (the
safe-testing ladder) in the SAPIENT repo.
EOF

(cd "$OUT_DIR" && rm -f sapient-swift.zip && zip -qry sapient-swift.zip sapient-swift)

# Bare-XCFramework zip for SwiftPM's remote binaryTarget (framework must sit
# at the zip root). Its SHA-256 doubles as the SwiftPM checksum — verified
# against `swift package compute-checksum` so the Package.swift the dist-swift
# release job writes can never carry a mismatched value.
echo "==> Zipping bare XCFramework for the SwiftPM binaryTarget"
(cd "$OUT_DIR" && rm -f SapientFFI.xcframework.zip \
  && zip -qry SapientFFI.xcframework.zip SapientFFI.xcframework \
  && shasum -a 256 SapientFFI.xcframework.zip > SapientFFI.xcframework.zip.sha256)
# (run inside the generated package — the subcommand needs a Package.swift cwd
# on some toolchains; the zip path is absolutized first)
XCF_ZIP_ABS="$(cd "$OUT_DIR" && pwd)/SapientFFI.xcframework.zip"
SPM_CHECKSUM=$(cd "$PKG" && swift package compute-checksum "$XCF_ZIP_ABS")
SHA_CHECKSUM=$(cut -d' ' -f1 "$OUT_DIR/SapientFFI.xcframework.zip.sha256")
if [[ "$SPM_CHECKSUM" != "$SHA_CHECKSUM" ]]; then
  echo "error: SwiftPM checksum ($SPM_CHECKSUM) != sha256 ($SHA_CHECKSUM)" >&2
  exit 1
fi

if [[ "$SMOKE" == "1" ]]; then
  echo "==> Smoke test: compile & run a macOS binary against the package"
  SMOKE_DIR="$OUT_DIR/smoke-swift"
  rm -rf "$SMOKE_DIR" && mkdir -p "$SMOKE_DIR"
  cat > "$SMOKE_DIR/main.swift" <<'EOF'
// Links the packaged macOS slice and exercises the catalog surface —
// no model download, so it runs in CI.
let v = version()
let models = listModels()
let repo = try! resolveAlias(name: models[0].alias)
print("sapient-ffi \(v); catalog \(models.count) models; \(models[0].alias) -> \(repo)")
precondition(!models.isEmpty, "catalog must not be empty")
EOF
  MAC_SLICE_DIR=$(dirname "$(find "$XCF" -name 'libsapient_ffi.a' -path '*macos*' | head -1)")
  swiftc -O \
    "$SMOKE_DIR/main.swift" "$PKG/Sources/Sapient/sapient_ffi.swift" \
    -I "$MAC_SLICE_DIR/Headers" \
    -L "$MAC_SLICE_DIR" -lsapient_ffi -lc++ -liconv \
    -framework SystemConfiguration -framework CoreFoundation \
    -framework Metal -framework QuartzCore \
    -o "$SMOKE_DIR/smoke"
  "$SMOKE_DIR/smoke"
  echo "==> Smoke test PASSED"
fi

echo "==> Done:"
du -sh "$XCF" "$PKG" "$OUT_DIR/sapient-swift.zip" \
  "$OUT_DIR/SapientFFI.xcframework.zip" 2>/dev/null || true
