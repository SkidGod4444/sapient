#!/usr/bin/env bash
# update-homebrew-formula.sh
# Run after publishing a GitHub release to update sha256 checksums in Formula/sapient.rb
#
# Usage: ./scripts/update-homebrew-formula.sh v0.2.0

set -e

VERSION="${1:?Usage: $0 <version-tag, e.g. v0.1.0>}"
REPO="SkidGod4444/sapient"
FORMULA="Formula/sapient.rb"

fetch_sha256() {
  local target="$1"
  local filename="sapient-${target}.tar.gz"
  local url="https://github.com/${REPO}/releases/download/${VERSION}/${filename}.sha256"
  curl -fsSL "$url" | awk '{print $1}'
}

echo "→ Fetching checksums for ${VERSION}..."

SHA_AARCH64_DARWIN=$(fetch_sha256 "aarch64-apple-darwin")
SHA_X86_64_DARWIN=$(fetch_sha256 "x86_64-apple-darwin")
SHA_AARCH64_LINUX=$(fetch_sha256 "aarch64-unknown-linux-gnu")
SHA_X86_64_LINUX=$(fetch_sha256 "x86_64-unknown-linux-gnu")

echo "  aarch64-apple-darwin:      ${SHA_AARCH64_DARWIN}"
echo "  x86_64-apple-darwin:       ${SHA_X86_64_DARWIN}"
echo "  aarch64-unknown-linux-gnu: ${SHA_AARCH64_LINUX}"
echo "  x86_64-unknown-linux-gnu:  ${SHA_X86_64_LINUX}"

# Strip the 'v' prefix for the formula version field
VER="${VERSION#v}"

sed -i '' \
  -e "s/version \".*\"/version \"${VER}\"/" \
  -e "s/REPLACE_WITH_SHA256_AARCH64_APPLE_DARWIN/${SHA_AARCH64_DARWIN}/" \
  -e "s/REPLACE_WITH_SHA256_X86_64_APPLE_DARWIN/${SHA_X86_64_DARWIN}/" \
  -e "s/REPLACE_WITH_SHA256_AARCH64_LINUX/${SHA_AARCH64_LINUX}/" \
  -e "s/REPLACE_WITH_SHA256_X86_64_LINUX/${SHA_X86_64_LINUX}/" \
  "$FORMULA"

echo "✓ Updated ${FORMULA} to ${VER}"
echo ""
echo "  Next steps:"
echo "    1. Copy Formula/sapient.rb to your homebrew-tap repo"
echo "    2. git add Formula/sapient.rb && git commit -m 'sapient ${VER}' && git push"
echo "    3. Users can now: brew install skidgod4444/tap/sapient"
