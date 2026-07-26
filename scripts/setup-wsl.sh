#!/usr/bin/env bash
#
# One-time development environment setup for WSL2 + Ubuntu.
#
#   ./scripts/setup-wsl.sh
#
# Run it as a normal user. sudo is used only for the apt step and will prompt once.
#
# The layout it establishes — source on /mnt/c, build output and datasets on ext4 — is
# explained in docs/decisions/0001-language-and-environment.md. Under WSL2 the Windows
# filesystem is markedly slower than ext4, and leaving compile artefacts there costs minutes
# on every build.

set -euo pipefail

CARGO_TARGET="${CARGO_TARGET_DIR:-$HOME/.cache/anka-target}"
DATASETS="${ANKA_DATASETS:-$HOME/anka-datasets}"
VENV="$HOME/.venvs/anka"
PROFILE="$HOME/.bashrc"

log() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }

log "installing system packages (sudo will prompt for your password)"
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
  build-essential \
  pkg-config \
  python3-dev \
  python3-venv \
  valgrind \
  shellcheck

# Rust is a user-level install; nothing here needs root. build-essential above matters for
# more than C code: rustc drives the system linker, so `cc` has to exist.
if command -v rustup >/dev/null 2>&1; then
  log "updating the existing Rust toolchain"
  rustup update stable
else
  log "installing the Rust toolchain"
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain stable --profile default
fi
# shellcheck disable=SC1091
. "$HOME/.cargo/env"

log "creating the Python environment (dataset conversion + hnswlib baseline)"
python3 -m venv "$VENV"
"$VENV/bin/pip" install --quiet --upgrade pip wheel
# hnswlib has no wheels and is compiled from source here — hence build-essential and
# python3-dev above. It is the reference implementation for the comparison in phase 2, and
# never a runtime dependency of Anka itself.
"$VENV/bin/pip" install --quiet numpy h5py hnswlib

log "recording environment variables in $PROFILE"
mkdir -p "$CARGO_TARGET" "$DATASETS"
append_once() {
  grep -qxF -- "$1" "$PROFILE" 2>/dev/null || printf '%s\n' "$1" >>"$PROFILE"
}
append_once '# --- anka ---'
append_once "export CARGO_TARGET_DIR=\"$CARGO_TARGET\""
append_once "export ANKA_DATASETS=\"$DATASETS\""
append_once "export ANKA_VENV=\"$VENV\""

log "installed:"
printf '  %-10s %s\n' \
  rustc   "$(rustc --version)" \
  cargo   "$(cargo --version)" \
  gcc     "$(gcc --version | head -1)" \
  python  "$("$VENV/bin/python" --version)" \
  numpy   "$("$VENV/bin/python" -c 'import numpy; print(numpy.__version__)')" \
  h5py    "$("$VENV/bin/python" -c 'import h5py; print(h5py.__version__)')" \
  hnswlib "$("$VENV/bin/python" -c 'import hnswlib; print("ok")')"

cat <<EOF

Setup complete. Open a new shell (or run 'source $PROFILE') so the environment variables take
effect, then:

  cd /mnt/c/Users/pc/Desktop/projeler/Anka
  ./scripts/download_datasets.sh siftsmall
  cargo test --workspace

EOF
