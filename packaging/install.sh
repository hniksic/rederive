#!/bin/sh
# Install Rederive on Linux or macOS. Downloads the archive for this machine,
# unpacks it under ~/.local/share and links the launcher into ~/.local/bin.
#
#   curl -LsSf https://github.com/hniksic/rederive/releases/latest/download/install.sh | sh
#
# Options have to be passed through sh when the script is piped into it:
#
#   curl -LsSf .../install.sh | sh -s -- --prefix /opt --bin /usr/local/bin
#
#   --prefix DIR    where the program directory goes (default ~/.local/share)
#   --bin DIR       where the link goes (default ~/.local/bin)
#   --archive FILE  install from an archive already downloaded
#   --uninstall     remove both again
#
# Windows has an installer of its own; see the README.

set -eu

BASE_URL="${REDERIVE_BASE_URL:-https://github.com/hniksic/rederive/releases/latest/download}"
PREFIX="${HOME}/.local/share"
BIN="${HOME}/.local/bin"
ARCHIVE=""
UNINSTALL=""

fail() {
    printf 'rederive: %s\n' "$1" >&2
    exit 1
}

while [ $# -gt 0 ]; do
    case "$1" in
        --prefix) PREFIX="${2:-}"; shift 2 || fail "--prefix needs a directory" ;;
        --bin) BIN="${2:-}"; shift 2 || fail "--bin needs a directory" ;;
        --archive) ARCHIVE="${2:-}"; shift 2 || fail "--archive needs a file" ;;
        --uninstall) UNINSTALL=yes; shift ;;
        -h|--help) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) fail "unknown option: $1" ;;
    esac
done

TARGET="${PREFIX}/rederive"
LINK="${BIN}/rederive"

if [ -n "$UNINSTALL" ]; then
    rm -rf "$TARGET"
    rm -f "$LINK"
    printf 'Removed %s and %s\n' "$TARGET" "$LINK"
    exit 0
fi

# Which archive this machine wants. Only the two combinations a release builds are
# named here; anything else is told where to go rather than given a wrong file.
system="$(uname -s)"
machine="$(uname -m)"
case "${system}/${machine}" in
    Linux/x86_64) platform="linux-x86_64" ;;
    Darwin/arm64) platform="macos-arm64" ;;
    Darwin/x86_64) fail "no build for Intel Macs; run from source (see the README)" ;;
    *) fail "no build for ${system} ${machine}; run from source (see the README)" ;;
esac

work=""
cleanup() { [ -n "$work" ] && rm -rf "$work"; }
trap cleanup EXIT INT TERM

if [ -z "$ARCHIVE" ]; then
    command -v curl >/dev/null 2>&1 || fail "curl is needed to download the archive"
    work="$(mktemp -d)"
    ARCHIVE="${work}/rederive-${platform}.tar.gz"
    printf 'Downloading rederive-%s...\n' "$platform"
    curl -LsSf "${BASE_URL}/rederive-${platform}.tar.gz" -o "$ARCHIVE" ||
        fail "could not download the archive from ${BASE_URL}"

    # Check it against the release's own list, when there is one to check against.
    # A download that arrives corrupt is worth catching before it is unpacked over
    # a working install.
    if curl -LsSf "${BASE_URL}/SHA256SUMS" -o "${work}/SHA256SUMS" 2>/dev/null; then
        if command -v sha256sum >/dev/null 2>&1; then
            sum="$(sha256sum "$ARCHIVE" | cut -d' ' -f1)"
        elif command -v shasum >/dev/null 2>&1; then
            sum="$(shasum -a 256 "$ARCHIVE" | cut -d' ' -f1)"
        else
            sum=""
        fi
        expected="$(grep " rederive-${platform}.tar.gz\$" "${work}/SHA256SUMS" |
            cut -d' ' -f1 || true)"
        if [ -n "$sum" ] && [ -n "$expected" ] && [ "$sum" != "$expected" ]; then
            fail "the downloaded archive does not match its published checksum"
        fi
    fi
else
    [ -f "$ARCHIVE" ] || fail "${ARCHIVE}: no such file"
fi

mkdir -p "$PREFIX" "$BIN"
# The old install goes before the new one is unpacked, so that a file dropped from
# a later release cannot survive into it.
rm -rf "$TARGET"
tar -xzf "$ARCHIVE" -C "$PREFIX" || fail "could not unpack ${ARCHIVE}"
[ -x "${TARGET}/rederive" ] || fail "${ARCHIVE} did not hold a rederive directory"
ln -sf "${TARGET}/rederive" "$LINK"

printf 'Installed %s\n' "$TARGET"
case ":${PATH}:" in
    *":${BIN}:"*) printf 'Run it with: rederive\n' ;;
    *)
        printf 'Run it with: %s\n\n' "$LINK"
        printf '%s is not on your PATH. To put it there, add this to your shell\n' "$BIN"
        printf 'profile (~/.zshrc, ~/.bashrc or equivalent):\n\n'
        printf '    export PATH="%s:$PATH"\n' "$BIN"
        ;;
esac
