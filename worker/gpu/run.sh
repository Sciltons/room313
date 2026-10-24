#!/bin/bash
# runs inside pkg/ on the GPU node. Installs deps if missing, runs the worker, tars outputs for the stash.
set -o pipefail
# keep our pip installs OUT of ~/.local (persistent home; a stray upgrade there can break the node's Jupyter at startup)
export PYTHONUSERBASE="$(cd .. && pwd)/pyenv"; mkdir -p "$PYTHONUSERBASE"; export PATH="$PYTHONUSERBASE/bin:$PATH"; export PIP_USER=1
export PYTHONPATH="$PYTHONUSERBASE/lib/python$(python3 -c 'import sys;print(f"{sys.version_info[0]}.{sys.version_info[1]}")')/site-packages:$PYTHONPATH"
export TRANSFORMERS_NO_TORCHAUDIO=1; python3 - <<'PY'
# neutralise the broken torchaudio import without uninstalling system packages
import importlib.util, pathlib, os
spec = importlib.util.find_spec("torchaudio")
PY
pip uninstall -y -q torchaudio 2>/dev/null || true
python3 -c "import diffusers, transformers, imageio" 2>/dev/null || pip install -q -U "diffusers>=0.33" "transformers>=4.46" accelerate safetensors sentencepiece protobuf "imageio[ffmpeg]" pillow 2>&1 | tail -2
python3 -c "import torch; assert torch.cuda.is_available()" || echo "WARN: no cuda torch"
export HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DISABLE_XET=1
# --- smoke test FIRST (fail fast, never burn GPU hours on a broken environment)
pip install -q -U brotli zstandard "bitsandbytes>=0.45" 2>&1 | tail -1
python3 -c "import bitsandbytes" || { echo "RUN_FAILED bitsandbytes"; tar czf out.tar.gz out; exit 1; }
free -g | sed -n 2p
smoke() { python3 - <<'PY'
import sys
from huggingface_hub import hf_hub_download
hf_hub_download("YuCollection/FLUX.1-schnell-Diffusers", "model_index.json"); hf_hub_download("Lightricks/LTX-Video-0.9.7-distilled", "model_index.json")
import torch, diffusers, transformers; assert torch.cuda.is_available(); print("SMOKE_OK", torch.cuda.get_device_name(0), diffusers.__version__)
PY
}
smoke || { echo "smoke failed -> removing brotli and retrying"; pip uninstall -y -q brotli brotlicffi 2>/dev/null; smoke; } || { echo "RUN_FAILED smoke"; tar czf out.tar.gz out; exit 1; }
export ASSETS_DIR="$(cd .. && pwd)/assets"; mkdir -p "$ASSETS_DIR"
PHASE=images python3 gpu/worker.py 2>&1 | tee out/worker.log; rc=${PIPESTATUS[0]}
cp -f out/gpu_status.json out/gpu_status_images.json 2>/dev/null
PHASE=videos python3 gpu/worker.py 2>&1 | tee -a out/worker.log; rc2=${PIPESTATUS[0]}; [ "$rc" = "0" ] && rc=$rc2
tar czf out.tar.gz out && ls -la out.tar.gz
[ "$rc" = "0" ] && echo "RUN_DONE" || echo "RUN_FAILED rc=$rc"
