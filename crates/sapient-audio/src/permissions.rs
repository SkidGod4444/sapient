//! Microphone-capture permission handling (behind the `audio-io` feature).
//!
//! Permission models differ sharply by OS, so this exposes one honest API:
//!
//! - **macOS** has a real per-app consent API (TCC, via AVFoundation).
//!   [`request_microphone`] reads the current status and, when it is
//!   *undetermined*, triggers the system consent prompt and blocks until the
//!   user responds.
//! - **Windows** desktop (unpackaged) apps have **no** per-app runtime prompt —
//!   the mic is gated by a single global toggle (Settings ▸ Privacy ▸
//!   Microphone ▸ "Let desktop apps access your microphone"). There is no API
//!   to raise a consent dialog, so we report [`MicPermission::Unknown`] and rely
//!   on the device-open + live level meter to reveal a muted/blocked mic.
//! - **Linux** has no OS permission model — capture is governed by device/group
//!   permissions (the `audio` group) or, in sandboxes, the PipeWire/xdg portal.
//!   We report [`MicPermission::Unknown`].
//!
//! Audio **output** (speakers) requires no permission on any supported OS, so
//! there is deliberately no "request speaker access" — there is nothing to ask.
//!
//! [`open_privacy_settings`] best-effort opens the relevant settings pane, and
//! [`microphone_guidance`] returns an OS-specific one-liner for when capture
//! looks silent/blocked.

/// Outcome of a microphone-permission check/request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MicPermission {
    /// The OS granted (or already had) microphone access.
    Granted,
    /// The OS explicitly denied access (user declined, or it's blocked).
    Denied,
    /// No per-app runtime consent API on this OS (Windows desktop / Linux), or
    /// the request couldn't be resolved — proceed and let the live level meter
    /// reveal a silent device.
    Unknown,
}

/// Request microphone access from the OS.
///
/// On macOS this triggers the system consent prompt when access is undetermined
/// and blocks (up to ~2 min) until the user responds; if already decided it
/// returns immediately. On Windows/Linux it returns [`MicPermission::Unknown`]
/// (no per-app prompt exists — see the module docs).
pub fn request_microphone() -> MicPermission {
    #[cfg(target_os = "macos")]
    {
        macos::request()
    }
    #[cfg(not(target_os = "macos"))]
    {
        MicPermission::Unknown
    }
}

/// Best-effort open the OS privacy/microphone settings pane. Returns `true` if a
/// settings command was launched (no guarantee the pane opened).
pub fn open_privacy_settings() -> bool {
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("open")
            .arg("x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone")
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }
    #[cfg(target_os = "windows")]
    {
        std::process::Command::new("cmd")
            .args(["/C", "start", "ms-settings:privacy-microphone"])
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        false
    }
}

/// An OS-specific one-line hint for when the microphone delivers no signal.
pub fn microphone_guidance() -> &'static str {
    #[cfg(target_os = "macos")]
    {
        "macOS: grant your terminal Microphone access in System Settings ▸ Privacy & Security ▸ Microphone, then re-run `sapient converse`."
    }
    #[cfg(target_os = "windows")]
    {
        "Windows: enable Settings ▸ Privacy & security ▸ Microphone ▸ \"Let desktop apps access your microphone\", then re-run `sapient converse`."
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        "Linux: check your user is in the `audio` group and the input isn't muted (try `pavucontrol`/`alsamixer`); ensure PulseAudio/PipeWire is running."
    }
}

#[cfg(target_os = "macos")]
mod macos {
    use std::sync::mpsc;
    use std::time::Duration;

    use objc::runtime::{Object, BOOL, YES};
    use objc::{class, msg_send, sel, sel_impl};

    #[link(name = "AVFoundation", kind = "framework")]
    extern "C" {
        /// `AVMediaTypeAudio` (an `NSString *`) — the media type we request.
        static AVMediaTypeAudio: *const Object;
    }

    // AVAuthorizationStatus values (AVFoundation).
    const NOT_DETERMINED: isize = 0;
    const RESTRICTED: isize = 1;
    const DENIED: isize = 2;
    const AUTHORIZED: isize = 3;

    pub fn request() -> super::MicPermission {
        use super::MicPermission;
        unsafe {
            let cls = class!(AVCaptureDevice);
            let status: isize = msg_send![cls, authorizationStatusForMediaType: AVMediaTypeAudio];
            match status {
                AUTHORIZED => return MicPermission::Granted,
                DENIED | RESTRICTED => return MicPermission::Denied,
                NOT_DETERMINED => {}
                _ => return MicPermission::Unknown,
            }

            // Undetermined → raise the system prompt and wait for the user.
            let (tx, rx) = mpsc::channel::<bool>();
            let handler = block::ConcreteBlock::new(move |granted: BOOL| {
                let _ = tx.send(granted == YES);
            });
            let handler = handler.copy();
            let _: () = msg_send![cls,
                requestAccessForMediaType: AVMediaTypeAudio
                completionHandler: &*handler];

            match rx.recv_timeout(Duration::from_secs(120)) {
                Ok(true) => MicPermission::Granted,
                Ok(false) => MicPermission::Denied,
                Err(_) => MicPermission::Unknown,
            }
        }
    }
}
