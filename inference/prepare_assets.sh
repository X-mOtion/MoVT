#!/usr/bin/env bash
# Link (or copy) the four checkpoints that text-to-motion inference needs.
#
# Usage:
#   ./prepare_assets.sh            # symlink the checkpoint tree (default)
#   ./prepare_assets.sh --copy     # copy the four needed directories instead
#
# Checkpoints are weights, not code, so they are kept out of this folder. Both
# modes need pre-trained weights to already exist in the parent MoVT checkout.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_ROOT="${MOVT_SOURCE_ROOT:-$(dirname "$HERE")}"
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
  echo "set MOVT_SOURCE_ROOT to the MoVT checkout that holds checkpoints/." >&2
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
