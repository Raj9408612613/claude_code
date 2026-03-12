#!/bin/bash
# Download NVIDIA GR00T N1 model weights from HuggingFace
# Weights are ~4-8 GB and require a HuggingFace account
#
# Usage:
#   bash download_weights.sh              # Download default N1-2B model
#   bash download_weights.sh N1-2B        # Download specific variant
#
# Prerequisites:
#   pip install huggingface_hub
#   huggingface-cli login  (if gated model)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/model_weights"
VARIANT="${1:-N1-2B}"
MODEL_ID="nvidia/GR00T-${VARIANT}"

echo "=== GR00T N1 Weight Downloader ==="
echo "Model:      ${MODEL_ID}"
echo "Target dir: ${MODEL_DIR}"
echo ""

# Install huggingface_hub if not present
if ! python -c "import huggingface_hub" 2>/dev/null; then
    echo "Installing huggingface_hub..."
    pip install huggingface_hub
fi

# Create output directory
mkdir -p "${MODEL_DIR}"

# Download
echo "Downloading weights (this may take 10-30 minutes)..."
huggingface-cli download "${MODEL_ID}" --local-dir "${MODEL_DIR}/${VARIANT}"

echo ""
echo "Done! Weights saved to: ${MODEL_DIR}/${VARIANT}"
echo "Total size: $(du -sh "${MODEL_DIR}/${VARIANT}" | cut -f1)"
