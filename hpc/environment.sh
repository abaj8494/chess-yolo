#!/bin/bash
# Environment setup script for Chess YOLO on HPC (using venv)

set -e

VENV_DIR="${1:-$HOME/chess-yolo-venv}"

echo "Setting up Chess YOLO environment..."
echo "Virtual environment location: $VENV_DIR"

# Create virtual environment
python3 -m venv "$VENV_DIR"

# Activate it
source "$VENV_DIR/bin/activate"

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support
# For CUDA 12.x (H200 compatible) - using CUDA 12.1 wheels:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install Ultralytics (YOLOv8)
pip install ultralytics

# Install other dependencies
pip install opencv-python-headless  # headless for HPC (no display)
pip install albumentations
pip install chess
pip install roboflow
pip install supervision
pip install rich tqdm pyyaml python-dotenv typer lapx

# Training extras
pip install tensorboard wandb matplotlib

# Verify installation
echo ""
echo "Verifying installation..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')

from ultralytics import YOLO
print(f'Ultralytics installed successfully')

import chess
print(f'python-chess installed successfully')
"

echo ""
echo "=============================================="
echo "Environment setup complete!"
echo "=============================================="
echo ""
echo "To activate the environment, run:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "To deactivate, run:"
echo "  deactivate"
