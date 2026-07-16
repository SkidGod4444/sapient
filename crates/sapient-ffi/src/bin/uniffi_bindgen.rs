// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)

//! UniFFI bindings generator for `sapient-ffi`. Usage (from the repo root):
//!
//! ```bash
//! cargo build -p sapient-ffi --release
//! cargo run -p sapient-ffi --features bindgen --bin uniffi-bindgen -- \
//!   generate --library target/release/libsapient_ffi.dylib \
//!   --language swift --language kotlin --out-dir bindings/generated
//! ```
//!
//! See `docs/MOBILE.md` for the full per-platform build recipes.

fn main() {
    uniffi::uniffi_bindgen_main()
}
