//! Self-update: download the latest release binary from GitHub.

use std::fs;
use std::io::{IsTerminal, Read};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

const REPO: &str = "openstackhq/sapient";
const BINARY: &str = "sapient";
/// MLX's compiled Metal shader library. The Metal build ships it next to the
/// binary because MLX loads it from the executable's directory at runtime.
const METALLIB: &str = "mlx.metallib";

/// Files staged in a temp dir, ready to install over the current binary.
struct Staged {
    binary: PathBuf,
    metallib: Option<PathBuf>,
}

struct PlatformAsset {
    triple: &'static str,
    archive: ArchiveKind,
}

/// Which build of SAPIENT to install.
#[derive(Copy, Clone, PartialEq, Eq)]
pub enum Variant {
    Cpu,
    Metal,
}

impl Variant {
    /// Filename suffix on the release archive (`-metal` for the GPU build).
    fn suffix(self) -> &'static str {
        match self {
            Variant::Cpu => "",
            Variant::Metal => "-metal",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Variant::Cpu => "CPU",
            Variant::Metal => "Metal (GPU)",
        }
    }
}

/// The variant this binary was compiled as. A binary built with the `mlx`
/// feature is the Apple Silicon Metal build; everything else is CPU.
const fn current_variant() -> Variant {
    if cfg!(feature = "mlx") {
        Variant::Metal
    } else {
        Variant::Cpu
    }
}

/// Whether this machine can run the Metal build: Apple Silicon (macOS aarch64),
/// the only platform we publish a `-metal` artifact for.
fn metal_capable() -> bool {
    std::env::consts::OS == "macos" && std::env::consts::ARCH == "aarch64"
}

#[derive(Copy, Clone)]
enum ArchiveKind {
    TarGz,
    Zip,
}

fn platform_asset() -> Result<PlatformAsset> {
    let asset = match std::env::consts::OS {
        "macos" => match std::env::consts::ARCH {
            "aarch64" => PlatformAsset {
                triple: "aarch64-apple-darwin",
                archive: ArchiveKind::TarGz,
            },
            "x86_64" => PlatformAsset {
                triple: "x86_64-apple-darwin",
                archive: ArchiveKind::TarGz,
            },
            other => bail!("unsupported macOS architecture: {other}"),
        },
        "linux" => match std::env::consts::ARCH {
            "x86_64" => PlatformAsset {
                triple: "x86_64-unknown-linux-gnu",
                archive: ArchiveKind::TarGz,
            },
            "aarch64" => PlatformAsset {
                triple: "aarch64-unknown-linux-gnu",
                archive: ArchiveKind::TarGz,
            },
            other => bail!("unsupported Linux architecture: {other}"),
        },
        "windows" => match std::env::consts::ARCH {
            "x86_64" => PlatformAsset {
                triple: "x86_64-pc-windows-msvc",
                archive: ArchiveKind::Zip,
            },
            "aarch64" => PlatformAsset {
                triple: "aarch64-pc-windows-msvc",
                archive: ArchiveKind::Zip,
            },
            other => bail!("unsupported Windows architecture: {other}"),
        },
        other => bail!("self-update is not supported on {other}"),
    };
    Ok(asset)
}

fn parse_version(tag: &str) -> Option<(u32, u32, u32)> {
    let tag = tag.strip_prefix('v').unwrap_or(tag);
    let mut parts = tag.split('.');
    Some((
        parts.next()?.parse().ok()?,
        parts.next()?.parse().ok()?,
        parts.next()?.parse().ok()?,
    ))
}

fn version_lt(a: &str, b: &str) -> bool {
    match (parse_version(a), parse_version(b)) {
        (Some(av), Some(bv)) => av < bv,
        _ => a != b,
    }
}

fn http_get_bytes(url: &str) -> Result<Vec<u8>> {
    let resp = ureq::get(url)
        .set("User-Agent", "sapient-cli")
        .call()
        .with_context(|| format!("GET {url}"))?;
    let mut buf = Vec::new();
    resp.into_reader().read_to_end(&mut buf)?;
    Ok(buf)
}

fn fetch_latest_tag() -> Result<String> {
    let api_url = format!("https://api.github.com/repos/{REPO}/releases/latest");
    match ureq::get(&api_url)
        .set("User-Agent", "sapient-cli")
        .call()
    {
        Ok(resp) => {
            let body = resp.into_string()?;
            let json: serde_json::Value = serde_json::from_str(&body)?;
            json["tag_name"]
                .as_str()
                .map(String::from)
                .context("releases/latest response missing tag_name")
        }
        // GitHub API rate limit (403) or secondary limit (429) — fall back to
        // reading the redirect Location header from the releases page, which is
        // not subject to the same IP-based rate limits as the REST API.
        Err(ureq::Error::Status(403 | 429, _)) => fetch_latest_tag_via_redirect(),
        Err(e) => Err(anyhow::anyhow!(
            "failed to check for updates: {e}\n\
             Tip: if you are behind a corporate proxy, set GITHUB_TOKEN \
             in your environment to avoid API rate limits."
        )),
    }
}

/// Fallback version check that reads the redirect from the GitHub releases page.
/// `https://github.com/{REPO}/releases/latest` issues a 302 to
/// `.../releases/tag/vX.Y.Z` — the tag is in the Location header.
/// This path does not consume the unauthenticated API rate limit.
fn fetch_latest_tag_via_redirect() -> Result<String> {
    let url = format!("https://github.com/{REPO}/releases/latest");
    // Build a one-shot agent that does NOT follow redirects.
    let agent = ureq::builder().redirects(0).build();
    let location = match agent.get(&url).set("User-Agent", "sapient-cli").call() {
        Ok(r) => r.header("location").map(str::to_owned),
        Err(ureq::Error::Status(_, r)) => r.header("location").map(str::to_owned),
        Err(e) => anyhow::bail!("update check failed: {e}"),
    };
    location
        .as_deref()
        .and_then(|loc| loc.rsplit('/').next())
        .map(str::to_owned)
        .context(
            "GitHub is rate-limiting anonymous requests from your IP.\n\
             Set GITHUB_TOKEN in your environment for a higher limit, \
             or try again later.",
        )
}

fn verify_sha256(data: &[u8], expected_hex: &str) -> Result<()> {
    use sha2::{Digest, Sha256};
    let hash = Sha256::digest(data);
    let actual = hash.iter().map(|b| format!("{b:02x}")).collect::<String>();
    if actual != expected_hex.trim().to_ascii_lowercase() {
        bail!("checksum mismatch: expected {expected_hex}, got {actual}");
    }
    Ok(())
}

fn extract_tar_gz(data: &[u8], dest: &Path) -> Result<PathBuf> {
    let decoder = flate2::read::GzDecoder::new(data);
    let mut archive = tar::Archive::new(decoder);
    archive.unpack(dest).context("failed to extract tar.gz")?;

    let binary = dest.join(BINARY);
    if binary.is_file() {
        return Ok(binary);
    }
    bail!("binary '{BINARY}' not found in archive")
}

fn extract_zip(data: &[u8], dest: &Path) -> Result<PathBuf> {
    let reader = std::io::Cursor::new(data);
    let mut archive = zip::ZipArchive::new(reader).context("failed to read zip")?;
    archive.extract(dest).context("failed to extract zip")?;
    let binary = dest.join(format!("{BINARY}.exe"));
    if binary.is_file() {
        return Ok(binary);
    }
    bail!("binary '{BINARY}.exe' not found in archive")
}

fn download_release(asset: &PlatformAsset, tag: &str, variant: Variant) -> Result<Staged> {
    let suffix = variant.suffix();
    let filename = match asset.archive {
        ArchiveKind::TarGz => format!("{BINARY}-{}{suffix}.tar.gz", asset.triple),
        ArchiveKind::Zip => format!("{BINARY}-{}{suffix}.zip", asset.triple),
    };
    let url = format!("https://github.com/{REPO}/releases/download/{tag}/{filename}");
    println!(
        "Downloading {tag} ({}, {})...",
        asset.triple,
        variant.label()
    );

    let archive_bytes = http_get_bytes(&url)?;

    if let Ok(checksum_text) = http_get_bytes(&format!("{url}.sha256")) {
        let expected = String::from_utf8_lossy(&checksum_text)
            .split_whitespace()
            .next()
            .unwrap_or("")
            .to_string();
        if !expected.is_empty() {
            verify_sha256(&archive_bytes, &expected)?;
            println!("Checksum verified.");
        }
    }

    let tmp = tempfile::tempdir().context("failed to create temp dir")?;
    let extracted = match asset.archive {
        ArchiveKind::TarGz => extract_tar_gz(&archive_bytes, tmp.path())?,
        ArchiveKind::Zip => extract_zip(&archive_bytes, tmp.path())?,
    };

    // Stage into a fresh directory so we can carry sidecar files (the Metal
    // build ships `mlx.metallib`, which MLX loads from next to the executable).
    let staged_dir = std::env::temp_dir().join(format!("sapient-update-{tag}{suffix}"));
    let _ = fs::remove_dir_all(&staged_dir);
    fs::create_dir_all(&staged_dir).context("failed to create staging dir")?;

    let staged_bin = staged_dir.join(extracted.file_name().unwrap());
    fs::copy(&extracted, &staged_bin)
        .with_context(|| format!("failed to stage {}", staged_bin.display()))?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&staged_bin, fs::Permissions::from_mode(0o755))?;
    }

    // Carry the Metal shader library if the archive bundled one.
    let staged_metallib = {
        let src = tmp.path().join(METALLIB);
        if src.is_file() {
            let dst = staged_dir.join(METALLIB);
            fs::copy(&src, &dst).with_context(|| format!("failed to stage {METALLIB}"))?;
            Some(dst)
        } else {
            None
        }
    };

    Ok(Staged {
        binary: staged_bin,
        metallib: staged_metallib,
    })
}

fn replace_current_binary(staged: &Staged) -> Result<()> {
    let current = std::env::current_exe().context("could not locate current sapient binary")?;
    let new_binary = &staged.binary;

    #[cfg(windows)]
    {
        let backup = current.with_extension("exe.old");
        let _ = fs::remove_file(&backup);
        if current.exists() {
            fs::rename(&current, &backup).ok();
        }
        fs::copy(new_binary, &current).with_context(|| {
            format!(
                "failed to install update to {}. \
                 Try: irm https://raw.githubusercontent.com/{REPO}/main/install.ps1 | iex",
                current.display()
            )
        })?;
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        // On Linux, `cp` into a running binary fails with ETXTBSY (os error 26).
        // The fix: write the new binary to a temp file in the *same directory* as
        // the target (guarantees same filesystem), set permissions, then `rename()`
        // it into place. `rename()` is atomic and replaces the directory entry while
        // the kernel keeps the old inode alive for any process still executing it.
        let dir = current
            .parent()
            .context("binary path has no parent directory")?;
        let tmp = dir.join(format!(".sapient-update-{}", std::process::id()));

        fs::copy(new_binary, &tmp)
            .with_context(|| format!("failed to stage update binary to {}", tmp.display()))?;
        fs::set_permissions(&tmp, fs::Permissions::from_mode(0o755))?;
        fs::rename(&tmp, &current).with_context(|| {
            // Clean up the temp file if rename fails (different filesystem, etc.)
            let _ = fs::remove_file(&tmp);
            format!("failed to install update to {}", current.display())
        })?;
    }

    // Install the Metal shader library next to the binary so MLX can find it.
    if let Some(metallib) = &staged.metallib {
        if let Some(dir) = current.parent() {
            let dst = dir.join(METALLIB);
            fs::copy(metallib, &dst)
                .with_context(|| format!("failed to install {} to {}", METALLIB, dst.display()))?;
        }
    }

    Ok(())
}

/// Decide which build variant to install.
///
/// Order of precedence: explicit `--metal`/`--cpu` flag → interactive prompt on
/// Apple Silicon → the variant this binary was built as. On machines without a
/// Metal artifact, always CPU (and reject an explicit `--metal`).
fn resolve_variant(explicit: Option<Variant>) -> Result<Variant> {
    if !metal_capable() {
        if explicit == Some(Variant::Metal) {
            bail!("the Metal build is only available on Apple Silicon (macOS arm64)");
        }
        return Ok(Variant::Cpu);
    }

    if let Some(v) = explicit {
        return Ok(v);
    }

    // Apple Silicon, no explicit choice: ask if we're attached to a terminal,
    // otherwise keep whatever this binary already is.
    if std::io::stdin().is_terminal() {
        Ok(prompt_variant(current_variant()))
    } else {
        Ok(current_variant())
    }
}

/// Ask the user which build to install, defaulting to `default` on empty input.
fn prompt_variant(default: Variant) -> Variant {
    use std::io::Write;
    println!("You're on Apple Silicon — choose a build to install:");
    println!("  1) Metal (GPU) — faster on Apple Silicon");
    println!("  2) CPU         — maximum compatibility");
    let default_num = match default {
        Variant::Metal => 1,
        Variant::Cpu => 2,
    };
    print!(
        "Build [1/2] (default {default_num}, currently {}): ",
        default.label()
    );
    let _ = std::io::stdout().flush();

    let mut buf = String::new();
    if std::io::stdin().read_line(&mut buf).is_err() {
        return default;
    }
    match buf.trim() {
        "1" => Variant::Metal,
        "2" => Variant::Cpu,
        _ => default,
    }
}

/// Check for updates and install the latest release if available.
pub fn run_update(force: bool, variant: Option<Variant>) -> Result<()> {
    let current = env!("CARGO_PKG_VERSION");
    let latest_tag = fetch_latest_tag()?;
    let latest = latest_tag.strip_prefix('v').unwrap_or(&latest_tag);

    let target_variant = resolve_variant(variant)?;
    let switching = target_variant != current_variant();

    // If already current AND not switching build variant, nothing to do.
    if !force && !switching && !version_lt(current, latest) {
        println!(
            "sapient {current} ({}) is already up to date.",
            current_variant().label()
        );
        return Ok(());
    }

    if switching {
        println!(
            "Switching build: {} → {}",
            current_variant().label(),
            target_variant.label()
        );
    }
    println!("Updating sapient {current} → {latest}...");
    let asset = platform_asset()?;
    let staged = download_release(&asset, &latest_tag, target_variant)?;
    replace_current_binary(&staged)?;
    if let Some(dir) = staged.binary.parent() {
        let _ = fs::remove_dir_all(dir);
    }

    println!("✓ Updated to sapient {latest} ({})", target_variant.label());
    println!("  Run: sapient --version");
    Ok(())
}
