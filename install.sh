#!/usr/bin/env sh
# SAPIENT Installer
# Usage: curl -fsSL https://github.com/SkidGod4444/sapient/releases/latest/download/install.sh | sh
#
# Supported platforms:
#   macOS  — Apple Silicon (arm64) + Intel (x86_64)
#   Linux  — x86_64, aarch64 64-bit (e.g. Pi 4/5 with 64-bit OS — 32-bit armv7 is not supported)
#   WSL    — treated as Linux

set -e

REPO="SkidGod4444/sapient"
BINARY_NAME="sapient"
INSTALL_DIR="${SAPIENT_INSTALL_DIR:-/usr/local/bin}"

# ── Colours (stderr so `BINARY=$(download)` only captures the path) ───────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

print_banner() {
  printf "\n${BOLD}${CYAN}  ⚡ SAPIENT${RESET} ${BOLD}— edge inference engine${RESET}\n" >&2
  printf "  Run small language models locally. No Python. No Docker. No GPU required.\n\n" >&2
}

info()    { printf "${CYAN}  →${RESET} %s\n" "$1" >&2; }
success() { printf "${GREEN}  ✓${RESET} %s\n" "$1" >&2; }
warn()    { printf "${YELLOW}  ⚠${RESET} %s\n" "$1" >&2; }
error()   { printf "${RED}  ✗ ERROR:${RESET} %s\n" "$1" >&2; exit 1; }

is_interactive() {
  [ -t 0 ] && [ -t 1 ]
}

# ── Detect OS and arch ───────────────────────────────────────────────────────
detect_platform() {
  OS="$(uname -s)"
  ARCH="$(uname -m)"

  case "$OS" in
    Darwin)
      case "$ARCH" in
        arm64)  PLATFORM="aarch64-apple-darwin" ;;
        x86_64) PLATFORM="x86_64-apple-darwin" ;;
        *)      error "Unsupported macOS architecture: $ARCH" ;;
      esac
      EXT="tar.gz"
      ;;
    Linux)
      case "$ARCH" in
        x86_64)  PLATFORM="x86_64-unknown-linux-gnu" ;;
        aarch64) PLATFORM="aarch64-unknown-linux-gnu" ;;
        armv7l)
          error "32-bit ARM (e.g. Raspberry Pi 3) is not supported. Use 64-bit Raspberry Pi OS on Pi 4 or Pi 5." ;;
        *)       error "Unsupported Linux architecture: $ARCH" ;;
      esac
      EXT="tar.gz"
      ;;
    MINGW*|MSYS*|CYGWIN*)
      error "Please use the PowerShell installer on Windows:\n  irm https://raw.githubusercontent.com/SkidGod4444/sapient/main/install.ps1 | iex"
      ;;
    *)
      error "Unsupported OS: $OS"
      ;;
  esac
}

# ── Get latest release version ───────────────────────────────────────────────
get_latest_version() {
  fetch_release() {
    if command -v curl > /dev/null 2>&1; then
      curl -fsSL -H "User-Agent: sapient-installer" \
        "https://api.github.com/repos/${REPO}/releases/latest"
    elif command -v wget > /dev/null 2>&1; then
      wget -qO- --header="User-Agent: sapient-installer" \
        "https://api.github.com/repos/${REPO}/releases/latest"
    else
      error "Neither curl nor wget found. Please install one and retry."
    fi
  }

  # GitHub returns compact JSON; never grep the whole line — extract tag_name only.
  VERSION=$(fetch_release \
    | grep -o '"tag_name"[[:space:]]*:[[:space:]]*"[^"]*"' \
    | head -1 \
    | sed 's/.*"\([^"]*\)"$/\1/')

  case "$VERSION" in
    v[0-9]*.[0-9]*.[0-9]*) ;;
    *)
      error "Could not determine latest version (got: ${VERSION:-empty}). Check your internet connection."
      ;;
  esac
}

hash_file() {
  FILE="$1"
  if command -v sha256sum > /dev/null 2>&1; then
    sha256sum "$FILE" | awk '{print $1}'
  elif command -v shasum > /dev/null 2>&1; then
    shasum -a 256 "$FILE" | awk '{print $1}'
  else
    echo ""
  fi
}

# ── Download ─────────────────────────────────────────────────────────────────
download() {
  FILENAME="${BINARY_NAME}-${PLATFORM}.${EXT}"
  URL="https://github.com/${REPO}/releases/download/${VERSION}/${FILENAME}"
  TMPDIR="$(mktemp -d)"
  TMPFILE="${TMPDIR}/${FILENAME}"

  info "Downloading ${BINARY_NAME} ${VERSION} for ${PLATFORM}..."

  if command -v curl > /dev/null 2>&1; then
    curl -fsSL --progress-bar "$URL" -o "$TMPFILE" || error "Download failed. URL: $URL"
  else
    wget -q --show-progress "$URL" -O "$TMPFILE" || error "Download failed. URL: $URL"
  fi

  # Verify checksum of the downloaded archive (matches release .sha256 files)
  CHECKSUM_URL="${URL}.sha256"
  if command -v curl > /dev/null 2>&1; then
    EXPECTED=$(curl -fsSL "$CHECKSUM_URL" 2>/dev/null | awk '{print $1}')
  else
    EXPECTED=$(wget -qO- "$CHECKSUM_URL" 2>/dev/null | awk '{print $1}')
  fi

  if [ -n "$EXPECTED" ]; then
    ACTUAL=$(hash_file "$TMPFILE")
    if [ -n "$ACTUAL" ] && [ "$ACTUAL" != "$EXPECTED" ]; then
      error "Checksum mismatch! Expected: $EXPECTED  Got: $ACTUAL\nDownload may be corrupted."
    fi
    success "Checksum verified"
  fi

  info "Extracting..."
  case "$EXT" in
    tar.gz) tar -xzf "$TMPFILE" -C "$TMPDIR" ;;
    zip)    unzip -q "$TMPFILE" -d "$TMPDIR" ;;
  esac

  EXTRACTED_BINARY="${TMPDIR}/${BINARY_NAME}"
  if [ ! -f "$EXTRACTED_BINARY" ]; then
    EXTRACTED_BINARY="$(find "$TMPDIR" -name "${BINARY_NAME}" -type f | head -1)"
  fi

  [ -f "$EXTRACTED_BINARY" ] || error "Binary not found after extraction"

  chmod +x "$EXTRACTED_BINARY"
  echo "$EXTRACTED_BINARY"
}

# ── Install ───────────────────────────────────────────────────────────────────
install_binary() {
  EXTRACTED_BINARY="$1"
  FINAL_PATH=""

  if [ -n "${SAPIENT_INSTALL_DIR:-}" ]; then
    mkdir -p "$INSTALL_DIR"
  fi

  if [ -w "$INSTALL_DIR" ]; then
    cp "$EXTRACTED_BINARY" "${INSTALL_DIR}/${BINARY_NAME}"
    FINAL_PATH="${INSTALL_DIR}/${BINARY_NAME}"
  elif is_interactive && command -v sudo > /dev/null 2>&1; then
    info "Requesting sudo to install to ${INSTALL_DIR}..."
    sudo cp "$EXTRACTED_BINARY" "${INSTALL_DIR}/${BINARY_NAME}"
    sudo chmod +x "${INSTALL_DIR}/${BINARY_NAME}"
    FINAL_PATH="${INSTALL_DIR}/${BINARY_NAME}"
  else
    LOCAL_BIN="${HOME}/.local/bin"
    mkdir -p "$LOCAL_BIN"
    cp "$EXTRACTED_BINARY" "${LOCAL_BIN}/${BINARY_NAME}"
    FINAL_PATH="${LOCAL_BIN}/${BINARY_NAME}"
    if ! is_interactive; then
      warn "Non-interactive install — using ${LOCAL_BIN} (sudo skipped)"
    else
      warn "Installed to ${LOCAL_BIN} (no write access to ${INSTALL_DIR})"
    fi
    case ":${PATH}:" in
      *":${LOCAL_BIN}:"*) ;;
      *)
        warn "Add to your PATH: export PATH=\"\$HOME/.local/bin:\$PATH\""
        warn "Then restart your terminal, or run: hash -r"
        ;;
    esac
  fi

  success "Installed to ${FINAL_PATH}"
  INSTALLED_PATH="$FINAL_PATH"
}

# ── Post-install ──────────────────────────────────────────────────────────────
post_install() {
  printf "\n${BOLD}${GREEN}✓ SAPIENT ${VERSION} installed!${RESET}\n\n"

  # PATH hint — only shown when the binary isn't reachable yet
  if ! command -v "${BINARY_NAME}" > /dev/null 2>&1; then
    printf "  ${YELLOW}⚠  Add sapient to your PATH:${RESET}\n"
    printf "     ${BOLD}export PATH=\"\$HOME/.local/bin:\$PATH\"${RESET}\n"
    printf "  Then reload your shell or run: ${BOLD}hash -r${RESET}\n\n"
  fi

  printf "  ${BOLD}See what models you can run:${RESET}\n\n"
  printf "    ${CYAN}sapient models${RESET}\n\n"

  printf "  ${BOLD}Start chatting (downloads the model on first use):${RESET}\n\n"
  printf "    ${BOLD}sapient chat openhorizon/phi-2${RESET}             # 2.7B · Phi\n"
  printf "    ${BOLD}sapient chat openhorizon/qwen2.5-0.5b${RESET}      # 0.5B · Qwen2.5\n"
  printf "    ${BOLD}sapient chat openhorizon/qwen2.5-0.5b-q4${RESET}   # 0.5B · GGUF Q8 (~640 MB)\n\n"

  printf "  ${BOLD}Other commands:${RESET}\n"
  printf "    ${CYAN}sapient pull   openhorizon/phi-2${RESET}    # Pre-download a model\n"
  printf "    ${CYAN}sapient list${RESET}                         # List downloaded models\n"
  printf "    ${CYAN}sapient rm     openhorizon/phi-2${RESET}    # Remove a model\n"
  printf "    ${CYAN}sapient update${RESET}                       # Update sapient\n"
  printf "    ${CYAN}sapient --help${RESET}                       # Full command reference\n\n"

  printf "  Docs & source: ${BOLD}https://github.com/${REPO}${RESET}\n\n"
}

# ── Main ──────────────────────────────────────────────────────────────────────
main() {
  print_banner
  detect_platform
  get_latest_version
  BINARY=$(download)
  install_binary "$BINARY"
  post_install

  rm -rf "$(dirname "$BINARY")"
}

main "$@"
