# 📱 Mobile & Embedding SDKs — build, use, and test safely

*The Phase-11 guide (repo roadmap Phase 5). Covers the `sapient-ffi` crate
(Swift / Kotlin via UniFFI), the TypeScript SDK (Node.js / React Native), and
— read this before touching a phone — the **safe-testing ladder for personal
hardware**.*

---

## 1. Architecture

One Rust core, three ways in. The chat/generate engine (`sapient-generate`'s
`Pipeline`) is wrapped once by a small, stable, blocking API and surfaced per
ecosystem:

```
                         ┌────────────────────────────┐
                         │       your application      │
                         └──┬─────────┬───────────┬───┘
                   Swift ───┘   Kotlin┘           └─ TypeScript
              (iOS/macOS)   (Android/JVM)      (Node.js / React Native)
                    │             │                   │
              UniFFI-generated bindings         @openhorizon/sapient
                    │             │                   │
                    ▼             ▼                   ▼
              ┌──────────────────────────┐   ┌──────────────────────┐
              │ crates/sapient-ffi       │   │ HTTP → sapient serve │  (today)
              │ staticlib / cdylib       │   │ napi / RN-JSI → ffi  │  (next rung)
              └────────────┬─────────────┘   └──────────────────────┘
                           ▼
              sapient-generate → sapient-models → sapient-backends-cpu
              (Pipeline, chat, streaming, prefix cache — the same engine
               the CLI and server use; CPU-only on mobile today)
```

Key design decisions (why it looks like this):

- **Blocking API, internal runtime.** `sapient-ffi` exposes synchronous
  calls; a private tokio runtime drives the async internals. Mobile apps call
  from a background queue/coroutine — never the UI thread. This keeps the
  binding surface trivial in every language.
- **Streaming is a foreign callback.** `TokenListener.on_token(token) -> bool`
  receives text fragments as they decode; returning `false` cancels
  generation (it drops the internal channel, which halts the engine at its
  next emit — no new engine API needed).
- **Sessions own the conversation.** `LlmSession` keeps chat history and
  enables the engine's prefix cache, so multi-turn chats only prefill the new
  turn — same trick `sapient serve` uses.
- **The TS SDK is transport-pluggable.** Today it speaks OpenAI-compatible
  HTTP to `sapient serve` (works on Node ≥ 18 and React Native immediately,
  and keeps inference *off* your phone during UI development). A native
  napi/JSI transport over `sapient-ffi` is the next rung — same API.

## 2. Status matrix

| Surface | Tech | Status |
|---|---|---|
| Rust host apps | `sapient-ffi` crate (or `sapient-generate` directly) | ✅ shipped, unit + e2e tested |
| Swift (iOS/macOS) | UniFFI bindgen → `sapient_ffi.swift` + C header/modulemap | ✅ generation validated, `swiftc -parse` clean; XCFramework packaging = next rung |
| Kotlin (Android/JVM) | UniFFI bindgen → `sapient_ffi.kt` (JNA) | ✅ generation validated; AAR packaging = next rung |
| iOS device build | `aarch64-apple-ios` staticlib | ✅ cross-compiles (needs `IPHONEOS_DEPLOYMENT_TARGET=14.0`) |
| iOS simulator build | `aarch64-apple-ios-sim` staticlib | ✅ cross-compiles |
| Android build | `aarch64-linux-android` cdylib via NDK ≥ 26 | ✅ cross-compiles (~11 MB `.so`, API 24+) |
| Node.js | `@openhorizon/sapient` → `sapient serve` HTTP | ✅ shipped; 11 tests + live-serve verified |
| React Native | same TS SDK (`fetch` injectable; `expo/fetch` for streaming) | ✅ non-streamed + streamed via expo/fetch; on-device native module = later rung |
| Sample apps (SwiftUI / Compose) | demo chat apps | ⬜ next rung |
| On-device thermal/battery-aware scheduling | iOS/Android governor (the CPU `ThermalGovernor` reads Linux sysfs only) | ⬜ later rung |

## 3. The FFI API in one glance

Generated names are idiomatic per language (`chat_stream` → `chatStream`).

- `version() -> String` — engine version.
- `list_models() -> [ModelEntry]` — the curated catalog (alias, repo, family,
  params, category, gated).
- `resolve_alias(name) -> String` — alias → HF repo id (fuzzy-matched; errors
  list the catalog).
- `LlmSession.load(model, options)` — download (first time) + load, blocking.
  `GenerationOptions`: `maxTokens` (512), `temperature`/`topP`/`topK`/
  `repetitionPenalty` (all unset → greedy), `systemPrompt`, `backend`
  (`auto`|`cpu`|`metal`|`wgpu`; mobile static libs are CPU-only, `auto` = CPU).
- `session.chat(userMessage) -> String` — one blocking turn, history-managed.
- `session.chatStream(userMessage, listener) -> String` — streamed turn;
  `listener.onToken(token) -> Bool`, return `false` to cancel. Returns the
  full (possibly partial-on-cancel) reply.
- `session.reset()` / `session.transcript()` / `session.model()` /
  `session.backendLabel()` / `session.isMmap()`.

## 4. Build & bindings generation

All commands run from the repo root. The bindings generator reads the
compiled library, so build first:

```bash
# 1. Build the library for your host (dev loop)
cargo build -p sapient-ffi --release

# 2. Generate Swift + Kotlin sources from it
cargo run -p sapient-ffi --features bindgen --bin uniffi-bindgen -- \
  generate --library target/release/libsapient_ffi.dylib \
  --language swift --language kotlin --out-dir bindings/generated
```

Outputs: `sapient_ffi.swift` + `sapient_ffiFFI.h` + `sapient_ffiFFI.modulemap`
(Swift) and `uniffi/sapient_ffi/sapient_ffi.kt` (Kotlin, needs the
[JNA](https://github.com/java-native-access/jna) jar at runtime). Generated
sources are **not** committed — regenerate whenever the crate's exported API
changes.

### iOS (validated on this repo)

```bash
rustup target add aarch64-apple-ios aarch64-apple-ios-sim

# IMPORTANT: without a modern deployment target the C deps (onig_sys) compile
# against the SDK default and the link fails on ___chkstk_darwin.
IPHONEOS_DEPLOYMENT_TARGET=14.0 cargo build -p sapient-ffi --release --target aarch64-apple-ios
IPHONEOS_DEPLOYMENT_TARGET=14.0 cargo build -p sapient-ffi --release --target aarch64-apple-ios-sim
```

Package the two `libsapient_ffi.a` slices + the generated header/modulemap as
an XCFramework (rename the modulemap to `module.modulemap` inside each
Headers dir):

```bash
xcodebuild -create-xcframework \
  -library target/aarch64-apple-ios/release/libsapient_ffi.a     -headers bindings/generated/ios-headers \
  -library target/aarch64-apple-ios-sim/release/libsapient_ffi.a -headers bindings/generated/ios-headers \
  -output SapientFFI.xcframework
```

Drop `SapientFFI.xcframework` + `sapient_ffi.swift` into an Xcode target (or
wrap in a Swift Package). The static archive is large before linking; the
app slice dead-strips to a small fraction.

```swift
// Call from a background queue — load() blocks (and downloads on first run).
let session = try LlmSession.load(
    model: "qwen2.5-0.5b",
    options: GenerationOptions(maxTokens: 256, systemPrompt: "Be concise."))

final class Printer: TokenListener {
    func onToken(token: String) -> Bool {
        DispatchQueue.main.async { /* append token to your UI */ }
        return true   // false = cancel generation
    }
}
let reply = try session.chatStream(userMessage: "Hi!", listener: Printer())
```

### Android (validated on this repo, NDK 26)

```bash
rustup target add aarch64-linux-android
NDKB=$HOME/Library/Android/sdk/ndk/26.1.10909125/toolchains/llvm/prebuilt/darwin-x86_64/bin

# esaxx-rs is C++ → CXX must be set too, not just CC.
CC_aarch64_linux_android=$NDKB/aarch64-linux-android24-clang \
CXX_aarch64_linux_android=$NDKB/aarch64-linux-android24-clang++ \
AR_aarch64_linux_android=$NDKB/llvm-ar \
CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER=$NDKB/aarch64-linux-android24-clang \
cargo build -p sapient-ffi --release --target aarch64-linux-android
```

(Or install [`cargo-ndk`](https://github.com/bbqsrc/cargo-ndk), which sets all
of that up: `cargo ndk -t arm64-v8a build -p sapient-ffi --release`.)

Ship `libsapient_ffi.so` in `src/main/jniLibs/arm64-v8a/`, add the generated
`sapient_ffi.kt` and a `net.java.dev.jna:jna:5.14.0@aar` dependency:

```kotlin
// Call from Dispatchers.IO — load() blocks (and downloads on first run).
val session = LlmSession.load(
    "qwen2.5-0.5b",
    GenerationOptions(maxTokens = 256u, systemPrompt = "Be concise."))

val reply = session.chatStream("Hi!", object : TokenListener {
    override fun onToken(token: String): Boolean {
        runOnUiThread { /* append token */ }
        return true   // false = cancel
    }
})
```

**Model storage:** the engine caches weights under the process home dir (HF
cache). On mobile, set `HF_HOME` to an app-writable path (e.g. iOS
`Library/Caches`, Android `context.cacheDir`) via `setenv` **before** the
first `load()` call.

### Node.js / React Native (TypeScript SDK)

See [`sdks/typescript/README.md`](../sdks/typescript/README.md). Short version:

```bash
sapient serve                       # on your Mac / server / Pi
npm install @openhorizon/sapient    # in your app
```

```ts
import { SapientClient } from '@openhorizon/sapient';
const client = new SapientClient();            // http://127.0.0.1:11435
for await (const tok of client.chatStream(
  [{ role: 'user', content: 'Tell me a haiku.' }], 'qwen2.5-0.5b'))
  process.stdout.write(tok);
```

React Native: `new SapientClient({ baseUrl: 'http://<your-mac-lan-ip>:11435' })`
works out of the box for `chat()`; for `chatStream()` pass `fetch` from
`expo/fetch` (RN's built-in fetch can't stream bodies). The native on-device
module (napi/JSI over `sapient-ffi`) is the next rung — the client API will
not change.

## 5. 🔒 Testing during development WITHOUT risking your hardware

These are **personal devices** — a bricked phone or a cooked battery is not
an acceptable dev cost. The rules below are ordered; each rung must pass
before you climb to the next. Nothing in this project ever requires
jailbreak, root, sideloaded provisioning hacks, or disabling OS protections
— if a step seems to need one, the step is wrong.

### 5.1 The testing ladder

| Rung | Where | What it validates | Model |
|---|---|---|---|
| 1 | **Mac, Rust tests** | API logic, error paths (`cargo test -p sapient-ffi`), real inference (`cargo test -p sapient-ffi --release -- --ignored`) | smollm2-135m |
| 2 | **Mac, host bindings** | the generated Swift/Kotlin actually drives the dylib (macOS Swift target / JVM + JNA — no device involved) | smollm2-135m |
| 3 | **iOS Simulator / Android emulator** | packaging, sandbox paths, `HF_HOME`, UI wiring. *Correctness only — perf numbers here are meaningless.* | smollm2-135m |
| 4 | **Real device, short runs** | memory ceiling, real tok/s, first-token latency — runs of **≤ 60 s**, device on a desk, not in a case | smollm2-135m → qwen2.5-0.5b |
| 5 | **Real device, longer soak** | sustained decode + thermal behavior — only after rung 4 is boring, and still supervised (never overnight, never in a pocket/bag) | qwen2.5-0.5b |

For React Native there's a rung 0 that skips the device entirely: point the
TS SDK at `sapient serve` on your Mac's LAN IP and build the whole UI with
zero on-device inference.

### 5.2 Model size rules (memory is the thing that kills apps)

iOS enforces per-app memory limits (jetsam): a phone app that allocates more
than roughly **50–60 % of device RAM** is killed on the spot, and the limit is
lower in the background. Android's LMK behaves similarly under pressure.

- **Dev default: `smollm2-135m`** (~100 MB Q8). It loads in seconds and
  exercises every code path. Only move up once the plumbing works.
- **Phone ceiling: ~1B Q4** (e.g. `llama3.2-1b-q4`, ~0.8 GB) on a 6–8 GB
  device. `qwen2.5-0.5b` (~0.4 GB) is the comfortable middle.
- **Never ship a 3B+ file to a phone "to see what happens"** — you'll spend
  ten minutes downloading on the device's flash and then get jetsam-killed at
  load. Do the math first: model file size + ~1 GB working set must stay
  under half the device's RAM.
- The engine's own guards help: GGUF loads mmap-backed when the file is large
  relative to free RAM (weights stay evictable page-cache, not heap), and the
  KV cache is capped at 8192 positions. **Set `SAPIENT_CTX=1024` (env) on
  phones** to shrink the KV allocation further — long contexts are a desktop
  luxury.
- Watch real memory in Xcode's memory gauge / Instruments (Allocations) or
  `adb shell dumpsys meminfo <pkg>`. If RSS approaches half of RAM, stop and
  shrink the model or context — don't "try once more".

### 5.3 Thermal & battery discipline (phones are fanless)

Sustained decode is the hottest workload a phone CPU can run. The engine's
`ThermalGovernor` only reads Linux sysfs thermal zones — **it is inert on iOS
and Android today**, so during development *you* are the governor:

- **Short bursts.** Cap dev runs at ~60 s of continuous decode, then let the
  device return to ambient temperature. A soak test is a deliberate,
  supervised event — not a loop you walk away from.
- **Stop conditions.** Device uncomfortably warm to hold → stop. OS thermal
  warning (iOS `ProcessInfo.thermalState` ≥ `.serious`, Android
  `PowerManager.getThermalStatus()` ≥ `THERMAL_STATUS_MODERATE`) → stop.
  Sudden tok/s collapse mid-run usually *is* throttling — treat it as the
  signal that the run has ended, not something to push through.
- **Monitor, don't guess.** iOS: Xcode Organizer/Instruments thermal state;
  Android: `adb shell dumpsys thermalservice`. Log
  `thermalState`/`getThermalStatus` next to your tok/s numbers — a benchmark
  without thermal context is noise anyway.
- **Battery:** bench plugged in (decode at 100 % battery draw drains fast),
  but avoid long hot runs while fast-charging — heat is cumulative, and heat
  is what actually ages the battery. Never run benchmarks with the phone in
  a case, in a pocket, or on a soft surface (blocks the chassis, which *is*
  the heatsink).
- **In-app hygiene for demo apps:** keep the screen-idle timer enabled, pause
  generation when the app backgrounds (iOS will kill a hot background app
  anyway), and never auto-restart generation loops.

### 5.4 Storage hygiene

- Models download to the HF cache (`HF_HOME`) — on device, point it at the
  app **Caches** directory so the OS can reclaim it and uninstall removes it.
- A phone's flash filling up degrades the whole device. Budget: dev models
  ≤ 1 GB total; delete swapped-out models (the cache is just files — remove
  the repo dir).
- Don't re-download on every debug session: keep the cache dir stable across
  installs while iterating (on iOS use an app group or keep reinstalls to
  the same bundle id; on Android avoid `adb uninstall` when you don't need it).

### 5.5 Emulator/simulator caveats

- Rung-3 targets validate **correctness only**. The iOS Simulator runs
  arm64-sim binaries on the Mac's cores with macOS's scheduler; Android
  emulators translate or virtualize. Tok/s, TTFT, and thermal behavior
  measured there are fiction — never record them as benchmarks.
- The simulator has the Mac's RAM — a model that loads fine there can still
  be jetsam-killed on the real phone. Rung 4 exists precisely for this.

### 5.6 What "safe" rules out entirely

- No jailbreak / root / bootloader unlocks — standard dev deployment (free
  Apple personal team certificates, Android developer mode + `adb`) covers
  everything this project needs.
- No overnight/unattended on-device loops. (The Pi/Jetson soak rigs exist
  for that — they have heatsinks and are expendable in a way your phone
  isn't.)
- No disabling of OS memory/thermal safeguards, no `adb shell` tinkering
  with thermal profiles, no forced-performance governors.
- If a device ever behaves oddly after a run (heat, battery drain, UI lag),
  stop testing on it, let it cool, and move that day's work back down the
  ladder.

## 6. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| iOS link fails: `___chkstk_darwin` undefined | Missing `IPHONEOS_DEPLOYMENT_TARGET=14.0` — C deps compiled against the SDK default while rustc linked at iOS 10. |
| Android build: `esaxx-rs … ToolNotFound clang++` | Set `CXX_aarch64_linux_android` (the C++ compiler), not just `CC` — or use `cargo-ndk`. |
| First `load()` extremely slow | It's downloading the model. Ship progress UI; pre-warm on Wi-Fi; cache per §5.4. |
| App killed during `load()` on device | Jetsam/LMK — model too big for the device (§5.2). Smaller quant, `SAPIENT_CTX=1024`. |
| `chatStream()` throws "not streamable" in React Native | RN's fetch can't stream — pass `expo/fetch` in `ClientOptions.fetch`, or use `chat()`. |
| UI freezes during generation | You called the blocking API on the main thread. Background queue / `Dispatchers.IO` / worker. |
| Kotlin bindgen warning "ktlint not found" | Cosmetic — the generated `.kt` is valid, just unformatted. |

## 7. Remaining rungs (tracked in ROADMAP.md Phase 5 / Notion Phase 11)

1. **Packaging:** XCFramework build script + Swift Package; Android AAR
   (bundled `.so` + generated Kotlin + JNA) — turn the validated commands
   above into CI artifacts.
2. **Sample apps:** minimal SwiftUI + Jetpack Compose chat apps (the
   phase's success metric: a 1B Q4 model on-device in a demo app).
3. **Native TS transport:** napi module for Node, JSI/TurboModule for React
   Native over `sapient-ffi` — same `SapientClient` API, no server.
4. **On-device niceties:** iOS/Android thermal governor hooks
   (`ProcessInfo.thermalState` / `PowerManager` → the existing
   effective-threads mechanism), download progress callbacks, background-safe
   model eviction.
5. **CI:** cross-compile jobs for the three mobile targets + TS SDK
   build/test job.
