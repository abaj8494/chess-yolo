#!/bin/bash
# Quick test to verify Chess YOLO setup before submitting jobs
# Run this interactively on a GPU node: qsub -I -l select=1:ncpus=6:ngpus=1:mem=32gb,walltime=1:00:00

set -e

echo "==========================================="
echo "Chess YOLO Setup Test"
echo "==========================================="

# Configuration
VENV_DIR="${VENV_DIR:-$HOME/.venvs/kits}"
PROJECT_DIR="${PROJECT_DIR:-$HOME/chess-yolo}"
SCRATCH_BASE="${SCRATCH_BASE:-/srv/scratch/$USER}"

echo ""
echo "Configuration:"
echo "  VENV_DIR: $VENV_DIR"
echo "  PROJECT_DIR: $PROJECT_DIR"
echo "  SCRATCH_BASE: $SCRATCH_BASE"
echo ""

# Check venv exists
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "ERROR: Virtual environment not found at $VENV_DIR"
    echo "Run: bash hpc/environment.sh"
    exit 1
fi
echo "✓ Virtual environment exists"

# Activate
source "$VENV_DIR/bin/activate"
echo "✓ Virtual environment activated"

# Set scratch cache
export TORCH_HOME="$SCRATCH_BASE/chess-yolo-cache/torch"
export YOLO_CONFIG_DIR="$SCRATCH_BASE/chess-yolo-cache/yolo"
mkdir -p "$TORCH_HOME" "$YOLO_CONFIG_DIR"

# Test Python imports
echo ""
echo "Testing Python imports..."
python -c "
import sys
print(f'Python: {sys.version}')

import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')

    # Quick CUDA test
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.matmul(x, x)
    print(f'CUDA tensor test: OK')

from ultralytics import YOLO
print('Ultralytics: OK')

import chess
print('python-chess: OK')

import cv2
print(f'OpenCV: {cv2.__version__}')

import albumentations
print('albumentations: OK')

print()
print('All imports successful!')
"

# Test YOLO model download (small model)
echo ""
echo "Testing YOLO model download..."
python -c "
from ultralytics import YOLO
import os

# This will download yolov8n if not cached
model = YOLO('yolov8n.pt')
print(f'YOLOv8n loaded successfully')
print(f'Model cached at: {os.environ.get(\"YOLO_CONFIG_DIR\", \"default\")}')
"

echo ""
echo "==========================================="
echo "All tests passed! Ready for training."
echo "==========================================="
echo ""
echo "Next steps:"
echo "  1. Download/prepare dataset"
echo "  2. Submit training job: sbatch hpc/submit_training.sbatch"
