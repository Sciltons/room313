#!/bin/bash
# Camber probe: what does a GPU job see?
echo "== host"; hostname; nproc; free -g | sed -n 2p; df -h . | tail -1; pwd; ls -la
echo "== gpu"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>&1 || echo "no nvidia-smi"
echo "== python"; which python3; python3 --version; python3 -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-', 'bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else '-')" 2>&1
python3 -c "import diffusers,transformers;print('diffusers',diffusers.__version__,'transformers',transformers.__version__)" 2>&1 | tail -1
echo "== internet"; for u in https://huggingface.co/api/models/Lightricks/LTX-Video https://pypi.org/simple/ https://github.com; do printf "%-55s " "$u"; curl -sS -o /dev/null -w "%{http_code}\n" --max-time 15 "$u" || echo "FAIL"; done
echo "== pip"; pip install -q --dry-run diffusers 2>&1 | tail -2 || true
echo "== ffmpeg"; which ffmpeg ffprobe || echo "no ffmpeg"
echo "== env"; env | grep -iE "camber|cuda|home|stash" | head
echo "== done"
