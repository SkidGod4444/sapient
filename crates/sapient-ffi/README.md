# sapient-ffi

The stable FFI surface for embedding SAPIENT in host applications — the
Phase-11 (mobile & SDKs) boundary layer.

- **Swift (iOS/macOS)** and **Kotlin (Android/JVM)** bindings are generated
  from this crate with [UniFFI](https://mozilla.github.io/uniffi-rs/).
- **Node.js / React Native** go through the first-party TypeScript SDK in
  `sdks/typescript`, which talks to `sapient serve` today and will bind this
  crate natively (napi / JSI) next.

API in one glance: `version()`, `list_models()`, `resolve_alias()`, and
`LlmSession` (`load` → `chat` / `chat_stream(listener)` / `reset` /
`transcript`). Streaming uses a foreign `TokenListener` callback whose return
value cancels generation.

Build recipes, per-platform packaging (XCFramework / AAR), and the
**safe-testing-on-personal-hardware guide** live in
[`docs/MOBILE.md`](../../docs/MOBILE.md). Not published to crates.io — this
repo ships binaries only.
