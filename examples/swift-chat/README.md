# SAPIENT Chat — SwiftUI example (macOS + iOS)

Minimal streaming chat over the packaged Swift bindings — fully on-device
inference through [`sapient-ffi`](../../crates/sapient-ffi). One shared
SwiftUI view (`Sources/SapientChatUI`), two thin app entries.

```bash
# 0. Package the bindings (repo root, once — and after any FFI API change)
./scripts/package-swift.sh

# macOS app
cd examples/swift-chat
swift run SapientChatMac

# iOS app (project is generated, not committed)
xcodegen                        # brew install xcodegen
open SapientChat.xcodeproj      # scheme: SapientChatApp
```

Headless iOS build (what CI runs — no signing needed for the simulator):

```bash
xcodebuild -project SapientChat.xcodeproj -scheme SapientChatApp \
  -destination 'generic/platform=iOS Simulator' build CODE_SIGNING_ALLOWED=NO
```

Defaults are deliberately conservative per the
[safe-testing ladder](../../docs/MOBILE.md): the model field starts at
`smollm2-135m-q4` (~100 MB — plumbing-validation size); switch to
`llama3.2-1b-q4` once things are boring. First send downloads the model into
the app's caches (`HF_HOME` is pointed there by `ChatViewModel`). **Stop**
cancels generation engine-side (the token listener returns `false`).

For a personal device: Xcode → Signing & Capabilities → your (free) personal
team — and read `docs/MOBILE.md` §5 first; it's a project rule.

Demo/testing hook: launching with `-autosend "<prompt>"` sends one message on
appear — `xcrun simctl launch <sim> so.openhorizon.sapient.chat -autosend
"Hi"` drives a real end-to-end turn on a simulator with no UI scripting
(this is how the in-app inference gate was verified).

Two integration notes baked into this example:
- The SwiftPM package is named `SapientChatKit`, NOT `SapientChat` — a name
  collision with the Xcode project binds the app scheme to the package and
  destination resolution fails with "supported platforms is empty".
- The FFI is blocking: all engine calls run on a dedicated serial
  `DispatchQueue`, never the main thread or the Swift cooperative pool.
