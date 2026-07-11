# SAPIENT Chat — React Native example (Expo)

Streaming chat UI over the [TypeScript SDK](../../sdks/typescript) — the
**rung-0 dev loop** from [`docs/MOBILE.md`](../../docs/MOBILE.md): inference
runs on `sapient serve` (your dev machine, a server, or a Pi); the phone only
renders. Zero on-device model risk while you build UI.

```bash
# 1. Engine — on your dev machine
sapient serve                              # binds 0.0.0.0:11435 — reachable
                                           # from a phone on the same Wi-Fi

# 2. SDK — build once
cd sdks/typescript && npm install && npm run build

# 3. App
cd examples/react-native-chat
npm install
npm start                                  # Expo — press i for iOS simulator, a for Android
```

In the app, set **Base URL**:
- iOS simulator (same machine): `http://127.0.0.1:11435`
- Android emulator (same machine): `http://10.0.2.2:11435` — the emulator's
  alias for the host's loopback; `127.0.0.1` is the emulator itself
- Physical phone (Expo Go, same Wi-Fi): `http://<your-dev-machine-lan-ip>:11435`

Streaming uses `expo/fetch` (RN's built-in fetch can't stream bodies) —
already wired in `App.tsx`. **Stop** aborts the request, which cancels
generation server-side.

Checks (no device needed): `npm run typecheck` and `npm run bundle:check`
(headless Metro bundle — this is what CI runs).

Note `metro.config.js`: the SDK is a `file:` dependency outside the app
root, so Metro needs it in `watchFolders` and needs `nodeModulesPaths`
pointed at the app's `node_modules`.

The fully on-device RN path (JSI/TurboModule over `sapient-ffi`) is a later
Phase-11 rung — the `SapientClient` API will not change.
