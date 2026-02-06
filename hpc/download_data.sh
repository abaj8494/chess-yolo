#!/bin/bash
# Download chess piece datasets to scratch directory
# Run this after setting up the environment

set -e

echo "==========================================="
echo "Chess YOLO Dataset Download"
echo "==========================================="

# Configuration
VENV_DIR="${VENV_DIR:-$HOME/.venvs/kits}"
PROJECT_DIR="${PROJECT_DIR:-$HOME/chess-yolo}"
SCRATCH_BASE="${SCRATCH_BASE:-/srv/scratch/$USER}"
DATA_DIR="$SCRATCH_BASE/chess-yolo-data"

echo ""
echo "Data will be downloaded to: $DATA_DIR"
echo ""

# Check for API key
if [ -z "$ROBOFLOW_API_KEY" ]; then
    echo "ERROR: ROBOFLOW_API_KEY not set"
    echo ""
    echo "Get your API key from: https://app.roboflow.com/settings/api"
    echo "Then run: export ROBOFLOW_API_KEY=your_key_here"
    exit 1
fi

# Activate environment
source "$VENV_DIR/bin/activate"

# Create directories
mkdir -p "$DATA_DIR/raw"

# Change to project dir for imports
cd "$PROJECT_DIR"

echo "Downloading datasets from Roboflow..."
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
export DATA_DIR="$DATA_DIR"

python << 'PYEOF'
import os
from pathlib import Path
from roboflow import Roboflow

DATA_DIR = os.environ['DATA_DIR']
output_dir = Path(DATA_DIR) / 'raw'
output_dir.mkdir(parents=True, exist_ok=True)

api_key = os.environ.get('ROBOFLOW_API_KEY')
if not api_key:
    print("ERROR: ROBOFLOW_API_KEY not set")
    exit(1)

rf = Roboflow(api_key=api_key)

# Chess piece datasets to download
datasets = [
    ("joseph-nelson", "chess-pieces-new", 1),
    ("roboflow-100", "chess-pieces-mjzgj", 2),
]

for workspace, project, version in datasets:
    print(f"\nDownloading {workspace}/{project} v{version}...")
    try:
        proj = rf.workspace(workspace).project(project)
        ds = proj.version(version).download("yolov8", location=str(output_dir / f"{project}-v{version}"))
        print(f"  Downloaded to: {ds.location}")
    except Exception as e:
        print(f"  Failed: {e}")

print("\nDownload complete!")
PYEOF

echo ""
echo "Merging and normalizing datasets..."
python << 'PYEOF'
import os
from pathlib import Path
import shutil
import random
import yaml

DATA_DIR = os.environ['DATA_DIR']
input_dir = Path(DATA_DIR) / 'raw'
output_dir = Path(DATA_DIR) / 'splits'

# Standard class names
STANDARD_CLASSES = [
    "white-king", "white-queen", "white-rook", "white-bishop", "white-knight", "white-pawn",
    "black-king", "black-queen", "black-rook", "black-bishop", "black-knight", "black-pawn",
]

# Find all dataset directories
dataset_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
print(f"Found {len(dataset_dirs)} datasets")

# Collect all images and labels
all_samples = []

for ds_dir in dataset_dirs:
    for split in ['train', 'valid', 'test']:
        images_dir = ds_dir / split / 'images'
        labels_dir = ds_dir / split / 'labels'

        if not images_dir.exists():
            continue

        for img_path in images_dir.glob('*'):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                label_path = labels_dir / f"{img_path.stem}.txt"
                if label_path.exists():
                    all_samples.append((img_path, label_path))

print(f"Total samples: {len(all_samples)}")

# Shuffle and split
random.seed(42)
random.shuffle(all_samples)

train_end = int(len(all_samples) * 0.8)
val_end = train_end + int(len(all_samples) * 0.1)

splits = {
    'train': all_samples[:train_end],
    'val': all_samples[train_end:val_end],
    'test': all_samples[val_end:],
}

# Create output directories and copy files
for split_name, samples in splits.items():
    img_dir = output_dir / split_name / 'images'
    lbl_dir = output_dir / split_name / 'labels'
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for i, (img_path, lbl_path) in enumerate(samples):
        new_name = f"{split_name}_{i:06d}"
        shutil.copy2(img_path, img_dir / f"{new_name}{img_path.suffix}")
        shutil.copy2(lbl_path, lbl_dir / f"{new_name}.txt")

    print(f"{split_name}: {len(samples)} samples")

# Create dataset.yaml
dataset_yaml = {
    'path': str(output_dir),
    'train': 'train/images',
    'val': 'val/images',
    'test': 'test/images',
    'nc': 12,
    'names': STANDARD_CLASSES,
}

with open(output_dir / 'dataset.yaml', 'w') as f:
    yaml.dump(dataset_yaml, f, default_flow_style=False)

print(f"\nDataset created at: {output_dir}")
PYEOF

# Update the data config to point to scratch
echo ""
echo "Creating data config pointing to scratch..."
cat > "$PROJECT_DIR/configs/data/chess_pieces.yaml" << EOF
# Chess Pieces Dataset Configuration for YOLOv8
# Data stored on scratch: $DATA_DIR

path: $DATA_DIR/splits
train: train/images
val: val/images
test: test/images

names:
  0: white-king
  1: white-queen
  2: white-rook
  3: white-bishop
  4: white-knight
  5: white-pawn
  6: black-king
  7: black-queen
  8: black-rook
  9: black-bishop
  10: black-knight
  11: black-pawn

nc: 12
EOF

echo ""
echo "==========================================="
echo "Dataset download complete!"
echo "==========================================="
echo ""
echo "Data location: $DATA_DIR/splits"
echo "Config updated: $PROJECT_DIR/configs/data/chess_pieces.yaml"
echo ""
echo "You can now submit training: sbatch hpc/submit_training.sbatch"
