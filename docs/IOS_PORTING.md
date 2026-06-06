# iOS Porting Plan (Phase 5 — iPhone/iPad)

> Status: **planning only — no code written yet.** This document is the working
> plan for bringing SAPIENT's on-device inference (LLM + Whisper STT + Kokoro/
> Orpheus TTS) to iOS/iPadOS. It is grounded in (a) a scan of the v0.4.4 codebase
> and (b) fact-checked external research (Nov 2025–2026 sources). Pick this up
> later and execute top-to-bottom; each phase has concrete file targets.

## TL;DR verdict

Porting is **feasible and moderate effort** (~2–4 weeks for a CPU-only 0.5–1.5B
chat MVP; +1–2 weeks for audio STT/TTS). The core inference is pure Rust + NEON
with no desktop OS dependency, so it compiles for `aarch64-apple-ios` largely
unchanged. The work is **packaging + a handful of platform shims**, not a rewrite.

- **Backend choice:** ship **CPU-NEON first** (it is *faster* than the iPhone GPU
  for ≤1.5B models), add **wgpu→Metal** later for larger models, **drop MLX** on
  iOS (`mlx-rs` iOS support unconfirmed; MLX has no Neural Engine path anyway).
- **Distribution:** new `staticlib` crate + C FFI → XCFramework linked into a
  Swift app. SAPIENT has **no FFI surface today** — this is the biggest new piece.
- **Main constraint:** iOS per-app memory budget (jetsam) → favor 4-bit
  0.5–1.5B models; the increased-memory-limit entitlement helps but is best-effort.
- **App Store:** downloading HF weights at runtime is *likely* allowed (Guideline
  2.5.2 forbids executable *code*, not ML *data*) — not guaranteed; plan a
  bundled-model fallback.

---

## Current state (from the codebase scan, v0.4.4)

### What already works toward iOS
- **Pure-Rust CPU inference**, NEON-vectorized (incl. K-quant kernels), `rayon` +
  `tokio::spawn_blocking`. No desktop OS dependency in the hot path.
- **`dirs` crate** (`sapient-hub/src/cache.rs`) already resolves to the iOS app
  container `Caches` dir in a library build.
- **Audio permissions are already AVFoundation FFI** (`sapient-audio/src/permissions.rs`)
  — iOS uses the *same* `AVCaptureDevice` API; just gated `cfg(target_os = "macos")`.
- **cpal 0.15** (`MicCapture`/`SpeakerPlayback`) — cpal officially supports iOS
  (CoreAudio/RemoteIO AudioUnit), so the converse/speak/transcribe audio paths port.

### Blockers / gaps (all solvable)
| # | Issue | Where (file) | Fix |
|---|---|---|---|
| 1 | CLI-only, **no library/FFI** | no `crate-type=staticlib/cdylib`; only `[[bin]] sapient`; no `extern "C"` exports | new `sapient-ffi` staticlib crate + C API |
| 2 | **No iOS build targets** | `.cargo/config.toml`, `.github/workflows/release.yml` (desktop-only) | add `aarch64-apple-ios[-sim]` + XCFramework build |
| 3 | **MLX hard-gated to macOS** | `sapient-models/src/forward/mlx_engine.rs:1` `#![cfg(all(target_os="macos", feature="mlx"))]`; `backend.rs` ("only available on macOS") | leave gated; **do not** use MLX on iOS |
| 4 | **RAM detection returns 0 on iOS** | `sapient-generate/src/pipeline.rs` (`/proc/meminfo`+`sysctl` branches); `backend.rs::total_system_ram_bytes()` | iOS branch via FFI `ProcessInfo.physicalMemory` / `os_proc_available_memory()` |
| 5 | **`std::process::Command("sysctl")` forbidden on iOS** | `sapient-generate/src/device.rs:~564`, `pipeline.rs` | iOS stub returning sane defaults |
| 6 | **Auto-mmap of large GGUF** risky under jetsam | `sapient-generate/src/pipeline.rs` (mmap when file >80% RAM); `sapient-models/src/gguf_weights.rs` | prefer small in-RAM models on iOS; guard/disable auto-mmap |
| 7 | **`.cache` CWD fallback** is sandbox-hostile | `sapient-hub/src/cache.rs` | add `set_cache_dir()`; Swift passes container path |
| 8 | **Audio permission gated to macOS only** | `sapient-audio/src/permissions.rs` (`cfg(target_os="macos")`) | widen to `any(macos, ios)`; stub `open_privacy_settings()` on iOS |
| 9 | `sapient stats` uses `sysinfo` (limited on iOS) | `sapient-cli/src/stats.rs` | CLI-only, not in inference path — exclude from iOS lib |

> Note: `no-JIT` is a **non-issue** — SAPIENT is AOT-compiled Rust.

---

## Backend decision (validated)

| Backend | iOS viability | Decision |
|---|---|---|
| **CPU (NEON, pure Rust)** | ✅ Proven; **~1.31× (Q4) / 1.33× (F16) faster than Metal GPU** for ≤1.5B on iPhone 15 Pro | **MVP primary** |
| **wgpu→Metal** | ✅ Runs on real device via `Surface` from `CAMetalLayer` (`SurfaceTargetUnsafe::CoreAnimationLayer`), no winit dep; **Simulator has Metal gaps** | **Phase 3 (>1.5B / newer GPUs)** |
| **MLX (`mlx-rs`)** | ⚠️ MLX *Swift* runs on iOS, but the **Rust binding `mlx-rs` iOS build is unconfirmed**; MLX has **no ANE path** | **Skip on iOS** |

Performance context: GPUs only outpace CPU **above 1.5B** params (iPhone 15 Pro,
arXiv 2505.06461). A Kokoro-TTS MLX-Swift port runs **~3.3× real-time on iPhone
13 Pro**, so SAPIENT's pure-Rust Kokoro (RTF ≈0.79 on M4) should be real-time on
modern A-series CPUs. iPhone 17 Pro GPU is up to 3.1× faster than 16 Pro for
large Transformers (NPU only +25%) — irrelevant until >1.5B.

---

## Phased plan

### Phase 0 — Spikes / de-risk (before committing) ~2–3 days
Resolve the open questions that could change the architecture:
1. **`mlx-rs`/`mlx-c` for `aarch64-apple-ios`?** Try `cargo build --target aarch64-apple-ios -p sapient-backends-metal`. Expected: fails/unsupported → confirms "wgpu-only GPU path." (If it works, revisit.)
2. **Jetsam headroom + background budget** on a real device: how much RAM before termination; does a multi-second decode survive backgrounding or trip the watchdog.
3. **App Store reality**: confirm multi-GB HF download at first run is acceptable, or design a bundled small-model path / On-Demand Resources.

### Phase 1 — FFI + build foundation ~1–2 weeks
- **New crate `crates/sapient-ffi`** (`crate-type = ["staticlib"]`), depends on
  `sapient-generate` with iOS-appropriate features (CPU only; no `mlx`, audio-io optional).
  - C API surface, e.g.:
    - `sapient_pipeline_load(model_path, cache_dir, *out_handle) -> i32`
    - `sapient_generate_next(handle, ...) -> i32` / streaming callback
    - `sapient_chat(handle, prompt, *out) -> i32`
    - `sapient_free(handle)`
  - Keep it `#[no_mangle] extern "C"`, `#[repr(C)]` structs, opaque handles.
    Consider **UniFFI** or **swift-bridge** to auto-generate the Swift layer
    (evaluate vs. a hand-written C header).
- **Cache dir injection:** add `set_cache_dir()` to `sapient-hub` (gap #7); FFI
  `load` accepts the container path from Swift.
- **Platform shims** (gaps #4, #5, #6, #8) behind `#[cfg(target_os = "ios")]`:
  - RAM: FFI to `ProcessInfo.physicalMemory` / `os_proc_available_memory()`.
  - `sysctl`/`Command`: stub with conservative defaults.
  - mmap: guard auto-mmap off on iOS; in-RAM load for small models.
  - audio permissions: widen `cfg` to include `ios`; stub settings launch.
- **Targets/CI:** add `aarch64-apple-ios` (+ `-sim` for development) to
  `.cargo/config.toml`; new CI job to build the staticlib and assemble an
  **XCFramework** (`lipo`/`xcodebuild -create-xcframework`).

### Phase 2 — Swift demo app (CPU-only) ~1 week
- Minimal SwiftUI app links the XCFramework.
- Flow: download a 4-bit 0.5–1.5B model from HF into the app container → load →
  chat (token stream to UI). Mirrors Apple's MLX-Swift `LLMEval` pattern.
- Add `NSMicrophoneUsageDescription` to Info.plist (for the audio phase).
- Test on a **physical iPhone (A14+)** — not the Simulator.
- **Success metric (matches ROADMAP):** a 1–3B Q4 model running on-device.

### Phase 3 — Audio (STT/TTS) ~1–2 weeks
- Enable `audio-io` for iOS; wire `AVAudioSession` setup from Swift.
- Whisper STT (`transcribe`), Kokoro TTS (`speak`), and the `converse` loop
  through the FFI. (Mic enumeration on iOS is finicky — cpal #842; budget time.)
- Confirm Kokoro real-time on-device; Whisper-tiny latency unmeasured — benchmark.

### Phase 4 — GPU + polish (optional)
- Add **wgpu→Metal** path (`Surface` from a `CAMetalLayer` handed in by Swift)
  for >1.5B models; gate behind a runtime/device check. Real device only.
- Thermal/throttle-aware scheduling; background-task handling; memory-pressure
  callbacks (`os_proc_available_memory()` polling → evict/stop).

---

## Open questions to answer during execution
1. Does `mlx-rs`/`mlx-c` build & link for `aarch64-apple-ios`, or is the MLX engine effectively macOS-only on Apple platforms (→ rely on wgpu→Metal)?
2. Concrete iOS background-execution / watchdog budget for multi-second inference (LLM decode + Whisper encode + Kokoro synth) — survive backgrounding?
3. Actual measured tok/s + memory footprint for 4-bit Qwen2.5-0.5B/1.5B + Whisper-tiny + Kokoro-82M on a real A-series iPhone, SAPIENT's own NEON kernels vs. wgpu→Metal (vs. llama.cpp reference).
4. Will App Store review accept first-run multi-GB GGUF/safetensors downloads from HF? Any download-size / On-Demand-Resources / background-download policy beyond 2.5.2?

## Risks & caveats
- App Store 2.5.2 compliance rests on the *absence* of "model weights" in the guideline text, not an affirmative ruling — "likely permissible," not guaranteed. Never download executable code/JIT.
- All verified iOS-MLX evidence is for MLX **Swift**, not `mlx-rs` — treat MLX-on-iOS-via-Rust as unproven.
- Perf numbers are from one academic study (iPhone 15 Pro, llama.cpp) — the CPU-vs-GPU-by-size *trend* transfers; absolute tok/s for SAPIENT will differ.
- iOS **Simulator** lacks the Metal features wgpu/MLX need — all GPU testing on physical devices.

## Key references
- Rust static-lib + FFI on iOS: `github.com/jinleili/wgpu-in-app`, `github.com/RustAudio/cpal` (`examples/ios-feedback`)
- wgpu on iOS (Metal via CAMetalLayer): `github.com/gfx-rs/wgpu`, `jinleili/wgpu-in-app`
- MLX on iOS (Swift): `github.com/ml-explore/mlx-swift-examples`, Awni Hannun iPhone MLX gist; `github.com/mlalma/kokoro-ios`
- MLX no-ANE / device support: `github.com/ml-explore/mlx`
- cpal iOS support: `RustAudio/cpal` PR #485 (CoreAudio/RemoteIO, Jan 2021)
- Memory entitlement: `developer.apple.com/documentation/bundleresources/entitlements/com.apple.developer.kernel.increased-memory-limit`
- App Store Guideline 2.5.2: `developer.apple.com/app-store/review/guidelines`
- On-device perf: arXiv 2505.06461 (iPhone 15 Pro), Argmax iPhone-17 benchmarks
- SAPIENT roadmap: `docs/ROADMAP.md` Phase 5
