# SAPIENT Chat — Jetpack Compose example (Android)

Minimal streaming chat over the packaged Android module — fully on-device
inference through [`sapient-ffi`](../../crates/sapient-ffi).

```bash
# 0. Package the engine module (repo root, once — and after any FFI API change)
./scripts/package-android.sh            # add --emulator for x86_64 emulators

# Build the APK (JDK 17; ANDROID_HOME or local.properties pointing at the SDK)
cd examples/android-chat
./gradlew :app:assembleDebug            # → app/build/outputs/apk/debug/app-debug.apk
```

The engine comes in as a plain Gradle module — `settings.gradle.kts` includes
`:sapient-android` from `../../dist/mobile/sapient-android` (generated, not
committed).

Install on an emulator (`adb install app-debug.apk`) or, after reading
[`docs/MOBILE.md`](../../docs/MOBILE.md) §5, on a device with developer mode.
Defaults follow the safe-testing ladder: model field starts at
`smollm2-135m-q4`; models download into the app's cache dir (`HF_HOME` is set in
`MainActivity.onCreate` before the first load). **Stop** cancels generation
engine-side.

Integration note baked into this example: the FFI error enum's field is
`reason`, not `message` — UniFFI maps error enums to Kotlin exception
classes, and a `message` field collides with `Throwable.message` (the
generated Kotlin doesn't compile). This app is the regression gate for that.
