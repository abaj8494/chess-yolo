#!/bin/bash
#SBATCH --job-name=chess-yolo
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --signal=B:SIGUSR1@300
#SBATCH --requeue

# Chess YOLO Training Script for HPC
# Uses scratch directory to avoid quota issues

set -e

echo "================================================"
echo "Chess YOLO Training Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "================================================"

# ============================================
# CONFIGURATION - Edit these paths as needed
# ============================================
VENV_DIR="${VENV_DIR:-$HOME/.venvs/kits}"
PROJECT_DIR="${PROJECT_DIR:-$HOME/chess-yolo}"
SCRATCH_BASE="${SCRATCH_BASE:-/srv/scratch/$USER}"

# Scratch subdirectories
SCRATCH_CACHE="$SCRATCH_BASE/chess-yolo-cache"
SCRATCH_EXPERIMENTS="$SCRATCH_BASE/chess-yolo-experiments"
SCRATCH_DATA="$SCRATCH_BASE/chess-yolo-data"

# ============================================
# SETUP
# ============================================

# Create directories
mkdir -p "$SCRATCH_CACHE" "$SCRATCH_EXPERIMENTS" "$SCRATCH_DATA"
mkdir -p "$PROJECT_DIR/logs"

# Activate virtual environment
echo "Activating: $VENV_DIR"
source "$VENV_DIR/bin/activate" || {
    echo "Error: virtual environment not found at $VENV_DIR"
    exit 1
}

# Set cache directories to scratch (avoids home quota)
export TORCH_HOME="$SCRATCH_CACHE/torch"
export HF_HOME="$SCRATCH_CACHE/huggingface"
export YOLO_CONFIG_DIR="$SCRATCH_CACHE/yolo"
export PIP_CACHE_DIR="$SCRATCH_CACHE/pip"
mkdir -p "$TORCH_HOME" "$HF_HOME" "$YOLO_CONFIG_DIR" "$PIP_CACHE_DIR"

echo ""
echo "Cache directories (on scratch):"
echo "  TORCH_HOME: $TORCH_HOME"
echo "  YOLO_CONFIG_DIR: $YOLO_CONFIG_DIR"
echo "  Experiments: $SCRATCH_EXPERIMENTS"
echo ""

# Print GPU info
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo ""

# Print Python/PyTorch info
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"
echo ""

# Change to project directory
cd "$PROJECT_DIR"
echo "Working directory: $(pwd)"
echo ""

# ============================================
# TRAINING
# ============================================
echo "Starting training..."
echo "Experiments will be saved to: $SCRATCH_EXPERIMENTS"
echo ""

python scripts/train.py \
    --config configs/training/yolov8m_pieces.yaml \
    --data configs/data/chess_pieces.yaml \
    --epochs 300 \
    --batch-size 64 \
    --project "$SCRATCH_EXPERIMENTS" \
    --name chess-pieces-v1 \
    --resume

echo ""
echo "================================================"
echo "Job completed: $(date)"
echo "================================================"
echo ""
echo "Results saved to: $SCRATCH_EXPERIMENTS/chess-pieces-v1"
