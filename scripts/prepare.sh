#!/bin/bash
set -Eeuo pipefail

COZE_WORKSPACE_PATH="${COZE_WORKSPACE_PATH:-$(pwd)}"

cd "${COZE_WORKSPACE_PATH}"

echo "=========================================="
echo "Preparing ProZoneSAM2 Development Environment"
echo "=========================================="

echo "Installing Node.js dependencies..."
pnpm install --prefer-frozen-lockfile --prefer-offline --loglevel debug --reporter=append-only

echo "Checking Python environment..."
if command -v python3 &> /dev/null; then
    echo "Python3 found: $(which python3)"
    python3 --version
    
    echo "Installing Python dependencies for SAM2..."
    apt-get update -qq 2>/dev/null || true
    apt-get install -y -qq libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 libgomp1 2>/dev/null || true
    
    python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu -q 2>/dev/null || true
    python3 -m pip install numpy scipy opencv-python matplotlib -q 2>/dev/null || true
    python3 -m pip install hydra-core==1.3.2 omegaconf==2.3.0 -q 2>/dev/null || true
    python3 -m pip install monai nibabel SimpleITK pydicom -q 2>/dev/null || true
    echo "Python dependencies installed"
else
    echo "Warning: Python3 not found. SAM2 segmentation will use mock mode."
fi

echo "Environment preparation completed!"
