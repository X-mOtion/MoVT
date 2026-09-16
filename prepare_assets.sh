#!/usr/bin/env bash
# Link (or copy) the four checkpoints that text-to-motion inference needs.
#
# Usage:
#   ./prepare_assets.sh                                  # symlink (default)
#   ./prepare_assets.sh --copy                           # copy instead
#   MOVT_SOURCE_ROOT=/path/to/checkout ./prepare_assets.sh
#
# Checkpoints are weights, not code, so they are kept out of git. This script
# is for people who already have a MoVT training checkout and would rather link
# its weights into ./checkpoints than download a second copy.
#
# Without MOVT_SOURCE_ROOT the source defaults to this repository, so the script
# is a no-op when ./checkpoints is already populated.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_ROOT="${MOVT_SOURCE_ROOT:-$HERE}"
SOURCE_CKPT="$SOURCE_ROOT/checkpoints/t2m"
TARGET_CKPT="$HERE/checkpoints/t2m"

NEEDED=(
  "rvq_name"
  "rvq_hml3d_xy_2d_supervised_align66_m4096e3_ce2"
  "mtrans_rvq_name_66d_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns"
  "rtrans_rvq_name_66d_nlayer8_nhead6_ld384_ff1024_cdp0.2_sw"
)

mode="link"
if [[ "${1:-}" == "--copy" ]]; then
  mode="copy"
elif [[ -n "${1:-}" ]]; then
  echo "unknown option: $1" >&2
  exit 2
fi

if [[ ! -d "$SOURCE_CKPT" ]]; then
  echo "checkpoint source not found: $SOURCE_CKPT" >&2
  echo >&2
  echo "This script links checkpoints from an existing MoVT training checkout." >&2
  echo "If you do not have one, download the inference checkpoints (774 MB):" >&2
  echo "  https://pan.baidu.com/s/1Zn3fAO1a8A3Xj1-nkVfSWw?pwd=pejb  (code: pejb)" >&2
  echo "and unpack MoVT-inference-ckpt.zip into this directory; then you can" >&2
  echo "run infer_text_2d3d.py directly and this script is not needed." >&2
  echo >&2
  echo "Otherwise set MOVT_SOURCE_ROOT to the checkout that holds checkpoints/." >&2
  exit 1
fi

mkdir -p "$TARGET_CKPT"

for name in "${NEEDED[@]}"; do
  src="$SOURCE_CKPT/$name"
  dst="$TARGET_CKPT/$name"
  if [[ ! -d "$src" ]]; then
    echo "missing checkpoint directory: $src" >&2
    exit 1
  fi
  if [[ -e "$dst" || -L "$dst" ]]; then
    echo "already present, leaving untouched: $dst"
    continue
  fi
  if [[ "$mode" == "copy" ]]; then
    cp -r "$src" "$dst"
    echo "copied $name"
  else
    ln -s "$src" "$dst"
    echo "linked $name"
  fi
done

echo
echo "assets ready under $HERE/checkpoints"
