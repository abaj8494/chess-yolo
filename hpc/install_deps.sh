#!/bin/bash
# Install Chess YOLO dependencies into existing environment (kits)
# Usage: source ~/.venvs/kits/bin/activate && bash install_deps.sh

set -e

echo "==========================================="
echo "Installing Chess YOLO dependencies"
echo "==========================================="

# Check if we're in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Activating kits environment..."
    source ~/.venvs/kits/bin/activate
fi

echo "Using environment: $VIRTUAL_ENV"

# Use scratch for pip cache to avoid quota issues
export PIP_CACHE_DIR="/srv/scratch/$USER/pip-cache"
mkdir -p "$PIP_CACHE_DIR"
echo "Pip cache: $PIP_CACHE_DIR"
echo ""

# Install Ultralytics (YOLOv8) - the main dependency
pip install ultralytics

# Chess logic
pip install chess

# Data/tracking
pip install roboflow
pip install supervision
pip install albumentations
pip install lapx

# Utilities (some may already be installed)
pip install rich typer python-dotenv

# Verify installation
echo ""
echo "Verifying installation..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

from ultralytics import YOLO
print('Ultralytics: OK')

import chess
print('python-chess: OK')

import supervision
print('supervision: OK')

import albumentations
print('albumentations: OK')
"

echo ""
echo "==========================================="
echo "Done! Chess YOLO dependencies installed."
echo "==========================================="
