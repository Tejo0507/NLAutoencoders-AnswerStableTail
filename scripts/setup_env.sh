#!/usr/bin/env bash
set -euo pipefail

TORCH_VERSION="${TORCH_VERSION:-2.9.1}"
CUDA_TAG="${CUDA_TAG:-cu128}"
VENV_PATH="${VENV_PATH:-.venv}"
HF_HOME="${HF_HOME:-nla_models/hf_cache}"
NLAAST_QUANT_DIR="${NLAAST_QUANT_DIR:-nla_models/nf4}"

if [ ! -d "$VENV_PATH" ]; then
  echo "Creating virtualenv at $VENV_PATH ..."
  python3 -m venv "$VENV_PATH"
fi

PY="$VENV_PATH/bin/python"
PIP="$VENV_PATH/bin/pip"

echo "Upgrading pip / setuptools / wheel ..."
"$PY" -m pip install --upgrade pip setuptools wheel

OS_NAME="$(uname -s)"
if [ "$OS_NAME" = "Linux" ] && command -v nvidia-smi >/dev/null 2>&1 && [ "$CUDA_TAG" != "cpu" ]; then
  echo "Installing torch==$TORCH_VERSION ($CUDA_TAG) ..."
  "$PIP" install "torch==$TORCH_VERSION" --index-url "https://download.pytorch.org/whl/$CUDA_TAG"
else
  echo "Installing torch==$TORCH_VERSION (CPU/default index) ..."
  "$PIP" install "torch==$TORCH_VERSION"
fi

echo "Installing package (editable + dev extras) ..."
"$PIP" install -e ".[dev]"

export HF_HOME
export NLAAST_QUANT_DIR
mkdir -p "$HF_HOME" "$NLAAST_QUANT_DIR"

echo "Verifying installation ..."
"$PY" - <<'EOF'
import torch
import transformers
from math_verify import parse, verify

print("torch:", torch.__version__, "| cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    import bitsandbytes as bnb
    print("device:", torch.cuda.get_device_name(0), "| compute_cap:", torch.cuda.get_device_capability(0))
    print("bitsandbytes:", bnb.__version__)
else:
    print("CUDA not available; skipping CUDA and bitsandbytes checks.")
print("transformers:", transformers.__version__)
assert verify(parse("1/2"), parse("0.5"))
print("math_verify OK")
EOF

echo ""
echo "Environment ready. Set for current shell (add to shell profile if desired):"
echo "  HF_HOME=$HF_HOME"
echo "  NLAAST_QUANT_DIR=$NLAAST_QUANT_DIR"
echo "Next:"
echo "  source $VENV_PATH/bin/activate"
echo "  pytest -m \"not slow\""
