#!/usr/bin/env bash
# Fetch Orient Anything V2 (Wang et al., NeurIPS 2025) at a pinned commit.
# The code is CC-BY-4.0 (VGGT parts Apache-2.0 / Meta license), so it is fetched
# rather than vendored. Checkpoint goes to the normal Hugging Face cache.
set -euo pipefail
COMMIT=73b11c9dc83e84daeb563d0c766831f2c66b0a18
DEST=${1:-third_party/Orient-Anything-V2}
if [ ! -d "$DEST/.git" ]; then
  git clone --quiet https://github.com/SpatialVision/Orient-Anything-V2.git "$DEST"
fi
git -C "$DEST" fetch --quiet --depth 1 origin "$COMMIT"
git -C "$DEST" checkout --quiet "$COMMIT"
PY=${PY:-.venv-eval/bin/python}
"$PY" -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('Viglong/OriAnyV2_ckpt', 'demo_ckpts/rotmod_realrotaug_best.pt'))"
