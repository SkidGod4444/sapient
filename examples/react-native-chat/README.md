# SAPIENT Chat — React Native example (Expo)

Streaming chat UI over the [TypeScript SDK](../../sdks/typescript) with **two
transports**, toggled at runtime in the header:

- **on-device (default)** — the engine runs inside the app via
  [`@openhorizon/sapient-react-native`](../../sdks/react-native)
  (`sapient-ffi` → UniFFI → JSI TurboModule). GPU (wgpu→Metal/Vulkan) with
  automatic CPU fallback; model downloads land in the app's Caches.
- **server** — HTTP to `sapient serve` (the rung-0 dev loop from
  [`docs/MOBILE.md`](../../docs/MOBILE.md): inference stays off the phone
  while you iterate on UI; streaming via `expo/fetch`).

The `SapientClient` API is identical over both — the toggle swaps the
transport, nothing else.

```bash
# 1. Build the native RN library once (repo root; rebuilds after FFI changes)
cd sdks/react-native && npm install && \
  IPHONEOS_DEPLOYMENT_TARGET=14.0 npm run ubrn:ios   # + ubrn:android for Android

# 2. This app — native code means a DEVELOPMENT BUILD (Expo Go can't run it)
cd examples/react-native-chat
npm install
npx expo prebuild -p ios          # generates ios/ (CNG — not committed)
cd ios && pod install && cd ..
npx expo run:ios                  # or open ios/*.xcworkspace in Xcode
```

Server mode only (no native build needed — works in Expo Go):
`sapient serve` binds 0.0.0.0:11435. In the app, set **Base URL**:
- iOS simulator (same machine): `http://127.0.0.1:11435`
- Android emulator (same machine): `http://10.0.2.2:11435` — the emulator's
  alias for the host's loopback; `127.0.0.1` is the emulator itself
- Physical phone (Expo Go, same Wi-Fi): `http://<your-dev-machine-lan-ip>:11435`

Checks (no device needed): `npm run typecheck` and `npm run bundle:check`
(headless Metro bundle — CI runs both).

Dependency security: `package.json` carries npm `overrides` pinning patched
versions of Expo 52's vulnerable transitive dev-deps (`@xmldom/xmldom`,
`postcss`, `uuid`). The remaining `tar` advisories have **no fix on the 6.x
line** Expo 52 requires (tar 7 breaks `expo prebuild`'s template extraction —
tested); exposure is dev-time-only (extracting Expo's own template tarball),
and the real fix is the deliberate Expo 57 migration tracked alongside the
ubrn 0.31 upgrade in ROADMAP Phase 5.

Integration notes baked into this example:
- `metro.config.js`: the SDK and the native package are `file:` deps outside
  the app root → both need `watchFolders` + the app's `node_modules` in
  `nodeModulesPaths`; AND the native package's own `node_modules/react-native`
  (newer, Flow `match` syntax) must be block-listed or Metro tries to parse it.
- The native package's `index.tsx` installs the Rust crate into Hermes at
  import time — importing it in Expo Go throws; keep server mode for Go.
- **Stop** aborts the stream, which cancels generation engine-side over both
  transports (HTTP reader cancel / `TokenListener` returning `false`).
- One model resident at a time on-device (phone RAM — MOBILE.md §5.2); the
  transport survives mode flips so the model stays warm.
