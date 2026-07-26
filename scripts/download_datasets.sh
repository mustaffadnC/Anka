#!/usr/bin/env bash
#
# Download and prepare the benchmark datasets.
#
#   ./scripts/download_datasets.sh [siftsmall|sift1m|glove|all]
#
# Datasets total roughly 1.3 GB and are never committed. They deliberately live outside the
# repository: under WSL2 the working tree sits on /mnt/c, where I/O is markedly slower than
# ext4 and would distort load-time and mmap measurements. Override the location with
# $ANKA_DATASETS.
#
# Everything here is idempotent — an already-downloaded file with a matching checksum is
# left alone.

set -euo pipefail

DATASETS_DIR="${ANKA_DATASETS:-$HOME/anka-datasets}"
VENV_DIR="${ANKA_VENV:-$HOME/.venvs/anka}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKSUMS="$SCRIPT_DIR/checksums.txt"

SIFT_BASE_URL="ftp://ftp.irisa.fr/local/texmex/corpus"
GLOVE_URL="http://ann-benchmarks.com/glove-100-angular.hdf5"

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m warn\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31merror\033[0m %s\n' "$*" >&2; exit 1; }

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "'$1' not found. Install it and re-run."
}

# Verify against a pinned checksum when we have one; record it on first download so later
# runs are verified. A recorded-but-unreviewed entry is called out, because a checksum that
# was generated from whatever we happened to download proves nothing on its own — it only
# starts proving something once it has been eyeballed and committed.
verify_checksum() {
  local file="$1" name expected actual
  name="$(basename "$file")"
  actual="$(sha256sum "$file" | cut -d' ' -f1)"

  expected=""
  if [[ -f "$CHECKSUMS" ]]; then
    expected="$(awk -v n="$name" '$2 == n { print $1 }' "$CHECKSUMS" || true)"
  fi

  if [[ -n "$expected" ]]; then
    [[ "$expected" == "$actual" ]] || die "checksum mismatch for $name
  expected $expected
  actual   $actual
Delete the file and re-run, or investigate before trusting any measurement built on it."
    log "$name checksum verified"
  else
    printf '%s  %s\n' "$actual" "$name" >>"$CHECKSUMS"
    warn "$name had no pinned checksum; recorded $actual"
    warn "review scripts/checksums.txt and commit it so future runs are verified"
  fi
}

fetch() {
  local url="$1" dest="$2"
  if [[ -f "$dest" ]]; then
    log "$(basename "$dest") already present, skipping download"
  else
    log "downloading $(basename "$dest")"
    # --continue-at resumes a partial transfer; write to .part so an interrupted download
    # is never mistaken for a complete file on the next run.
    curl --fail --location --continue-at - --progress-bar -o "$dest.part" "$url" \
      || die "download failed: $url
If FTP is blocked on this network, see the note at the bottom of this script for the
HTTP fallback."
    mv "$dest.part" "$dest"
  fi
  verify_checksum "$dest"
}

extract_tar() {
  local archive="$1" marker="$2"
  if [[ -f "$marker" ]]; then
    log "$(basename "$archive") already extracted, skipping"
  else
    log "extracting $(basename "$archive")"
    tar -xzf "$archive" -C "$DATASETS_DIR"
  fi
}

do_siftsmall() {
  fetch "$SIFT_BASE_URL/siftsmall.tar.gz" "$DATASETS_DIR/siftsmall.tar.gz"
  extract_tar "$DATASETS_DIR/siftsmall.tar.gz" "$DATASETS_DIR/siftsmall/siftsmall_base.fvecs"
  log "siftsmall ready (10k vectors — this is the CI recall-regression fixture)"
}

do_sift1m() {
  fetch "$SIFT_BASE_URL/sift.tar.gz" "$DATASETS_DIR/sift.tar.gz"
  extract_tar "$DATASETS_DIR/sift.tar.gz" "$DATASETS_DIR/sift/sift_base.fvecs"
  log "sift1m ready (1M x 128, L2, official .ivecs ground truth)"
}

# GloVe ships as HDF5 from ann-benchmarks, not as .fvecs. The reader in anka-core only
# speaks .fvecs/.ivecs, so conversion happens here rather than in Rust: pulling in an HDF5
# C dependency to read one file once would be a poor trade.
# setup-wsl.sh puts numpy and h5py in a virtualenv, so the system interpreter is the wrong
# thing to test — prefer the venv and fall back to a system Python that happens to have them.
resolve_python() {
  local candidate
  for candidate in "$VENV_DIR/bin/python" python3; do
    if command -v "$candidate" >/dev/null 2>&1 &&
      "$candidate" -c 'import h5py, numpy' >/dev/null 2>&1; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

do_glove() {
  local python
  python="$(resolve_python)" || die "no Python with numpy and h5py available.
Run ./scripts/setup-wsl.sh, or build one by hand:
  python3 -m venv $VENV_DIR && $VENV_DIR/bin/pip install numpy h5py"
  log "converting with $python"

  fetch "$GLOVE_URL" "$DATASETS_DIR/glove-100-angular.hdf5"

  local out="$DATASETS_DIR/glove100"
  if [[ -f "$out/glove100_base.fvecs" ]]; then
    log "glove100 already converted, skipping"
  else
    log "converting glove-100-angular.hdf5 to .fvecs/.ivecs"
    mkdir -p "$out"
    "$python" "$SCRIPT_DIR/hdf5_to_fvecs.py" \
      --input "$DATASETS_DIR/glove-100-angular.hdf5" \
      --outdir "$out" \
      --prefix glove100
  fi
  log "glove100 ready (1,183,514 x 100, cosine)"
  warn "GloVe vectors are NOT pre-normalised — cosine requires normalising at insert time"
}

main() {
  need_cmd curl
  need_cmd tar
  need_cmd sha256sum

  local target="${1:-all}"
  mkdir -p "$DATASETS_DIR"
  log "dataset directory: $DATASETS_DIR"

  case "$target" in
    siftsmall) do_siftsmall ;;
    sift1m)    do_sift1m ;;
    glove)     do_glove ;;
    all)       do_siftsmall; do_sift1m; do_glove ;;
    *)         die "unknown target '$target' (expected: siftsmall | sift1m | glove | all)" ;;
  esac

  log "done. Point the CLI at this directory with ANKA_DATASETS=$DATASETS_DIR"
}

main "$@"

# ------------------------------------------------------------------------------------------
# HTTP fallback if ftp.irisa.fr is unreachable
#
# ann-benchmarks mirrors SIFT1M over HTTP as an HDF5 file, and hdf5_to_fvecs.py reads it:
#
#   curl -fLO http://ann-benchmarks.com/sift-128-euclidean.hdf5
#   python3 scripts/hdf5_to_fvecs.py -i sift-128-euclidean.hdf5 -o sift --prefix sift
#
# Prefer the FTP source when it works: phase 1 verifies our generated ground truth against
# the *official* .ivecs file, and ann-benchmarks ships its own recomputed neighbours. They
# should agree, but "should" is exactly the kind of assumption this project exists to check.
# ------------------------------------------------------------------------------------------
