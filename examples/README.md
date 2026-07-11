# SAPIENT examples

Sample apps for the Phase-11 SDK surfaces. Read
[`docs/MOBILE.md`](../docs/MOBILE.md) first — especially §5, the
safe-testing ladder for personal hardware.

| Example | Stack | Inference | Prereq |
|---|---|---|---|
| [`swift-chat`](swift-chat) | SwiftUI — macOS app + iOS app (XcodeGen) | **on-device** via `sapient-ffi` | `./scripts/package-swift.sh` |
| [`android-chat`](android-chat) | Jetpack Compose | **on-device** via `sapient-ffi` | `./scripts/package-android.sh` |
| [`react-native-chat`](react-native-chat) | Expo / React Native + TypeScript SDK | on your dev machine via `sapient serve` (rung-0 loop; native on-device transport is a later rung) | `sdks/typescript` built |

All three share the same shape: streaming token-by-token chat, Stop
(cancels generation engine/server-side), a model field defaulting to
`smollm2-135m-q4` (dev size — see the ladder before going bigger).
