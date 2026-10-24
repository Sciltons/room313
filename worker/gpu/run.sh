#!/bin/bash
# runs inside pkg/ on the GPU node. Installs deps if missing, runs the worker, tars outputs for the stash.
set -o pipefail
pip uninstall -y -q torchaudio 2>/dev/null || true   # broken ABI on the Camber image; transformers imports it if present
python3 -c "import diffusers, transformers, imageio" 2>/dev/null || pip install -q -U "diffusers>=0.33" "transformers>=4.46" accelerate safetensors sentencepiece protobuf "imageio[ffmpeg]" pillow 2>&1 | tail -2
python3 -c "import torch; assert torch.cuda.is_available()" || echo "WARN: no cuda torch"
export HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DISABLE_XET=1
python3 gpu/worker.py 2>&1 | tee out/worker.log; rc=${PIPESTATUS[0]}
tar czf out.tar.gz out && ls -la out.tar.gz
[ "$rc" = "0" ] && echo "RUN_DONE" || echo "RUN_FAILED rc=$rc"
