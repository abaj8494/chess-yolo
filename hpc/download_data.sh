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

# Chess piece datasets to download - multiple for robust training
# Including varied piece styles, angles, and lighting conditions
datasets = [
    # Main datasets with good variety
    ("joseph-nelson", "chess-pieces-new", 1),           # ~292 images, varied angles
    ("roboflow-100", "chess-pieces-mjzgj", 2),          # Roboflow 100 benchmark
    ("block", "chess-pieces-wrdbb", 1),                 # ~628 images
    ("fyp-pbsgo", "chess-piece-detection-fyp", 1),      # ~505 images
    ("roboflow-jvuqo", "chess-pieces-zixng", 1),        # Additional variety
    ("yolo-ja0zo", "chess-pieces-kaid5", 1),            # 2025 dataset
    ("robotics-project-02zwj", "chess-piece-normal", 1), # Different piece styles
    ("chess-lpnmj", "chess-piece-detection-7alyg", 1),  # More variety
    ("chess-piece", "chess-pieces-pm2qa", 1),           # Additional images
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
import re

DATA_DIR = os.environ['DATA_DIR']
input_dir = Path(DATA_DIR) / 'raw'
output_dir = Path(DATA_DIR) / 'splits'

# Standard class names (our target format)
STANDARD_CLASSES = [
    "white-king", "white-queen", "white-rook", "white-bishop", "white-knight", "white-pawn",
    "black-king", "black-queen", "black-rook", "black-bishop", "black-knight", "black-pawn",
]

# Common name variations to normalize
NAME_MAPPING = {
    # White pieces
    "white-king": 0, "white_king": 0, "whiteking": 0, "white king": 0, "w-king": 0, "wk": 0,
    "white-queen": 1, "white_queen": 1, "whitequeen": 1, "white queen": 1, "w-queen": 1, "wq": 1,
    "white-rook": 2, "white_rook": 2, "whiterook": 2, "white rook": 2, "w-rook": 2, "wr": 2,
    "white-bishop": 3, "white_bishop": 3, "whitebishop": 3, "white bishop": 3, "w-bishop": 3, "wb": 3,
    "white-knight": 4, "white_knight": 4, "whiteknight": 4, "white knight": 4, "w-knight": 4, "wn": 4,
    "white-pawn": 5, "white_pawn": 5, "whitepawn": 5, "white pawn": 5, "w-pawn": 5, "wp": 5,
    # Black pieces
    "black-king": 6, "black_king": 6, "blackking": 6, "black king": 6, "b-king": 6, "bk": 6,
    "black-queen": 7, "black_queen": 7, "blackqueen": 7, "black queen": 7, "b-queen": 7, "bq": 7,
    "black-rook": 8, "black_rook": 8, "blackrook": 8, "black rook": 8, "b-rook": 8, "br": 8,
    "black-bishop": 9, "black_bishop": 9, "blackbishop": 9, "black bishop": 9, "b-bishop": 9, "bb": 9,
    "black-knight": 10, "black_knight": 10, "blackknight": 10, "black knight": 10, "b-knight": 10, "bn": 10,
    "black-pawn": 11, "black_pawn": 11, "blackpawn": 11, "black pawn": 11, "b-pawn": 11, "bp": 11,
}

def normalize_class_name(name):
    """Normalize a class name to our standard format."""
    name_lower = name.lower().strip()
    if name_lower in NAME_MAPPING:
        return NAME_MAPPING[name_lower]
    # Try removing special chars
    name_clean = re.sub(r'[^a-z]', '', name_lower)
    if name_clean in NAME_MAPPING:
        return NAME_MAPPING[name_clean]
    return None

def load_class_mapping(ds_dir):
    """Load class names from dataset's data.yaml and create mapping to standard classes."""
    yaml_paths = [
        ds_dir / 'data.yaml',
        ds_dir / 'dataset.yaml',
    ]

    for yaml_path in yaml_paths:
        if yaml_path.exists():
            with open(yaml_path) as f:
                data = yaml.safe_load(f)

            names = data.get('names', {})
            if isinstance(names, list):
                names = {i: n for i, n in enumerate(names)}

            mapping = {}
            unmapped = []
            for src_idx, name in names.items():
                std_idx = normalize_class_name(name)
                if std_idx is not None:
                    mapping[int(src_idx)] = std_idx
                else:
                    unmapped.append(f"{src_idx}:{name}")

            if unmapped:
                print(f"  Warning: Could not map classes: {unmapped}")

            return mapping

    print(f"  Warning: No data.yaml found in {ds_dir}")
    return None

def remap_label_file(src_path, dst_path, class_mapping):
    """Read label file, remap class indices, write to destination."""
    if class_mapping is None:
        # No mapping available, skip this file
        return False

    lines_out = []
    skipped = 0

    with open(src_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                src_class = int(parts[0])
                if src_class in class_mapping:
                    parts[0] = str(class_mapping[src_class])
                    lines_out.append(' '.join(parts))
                else:
                    skipped += 1

    if lines_out:
        with open(dst_path, 'w') as f:
            f.write('\n'.join(lines_out) + '\n')
        return True
    return False

# Find all dataset directories
dataset_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
print(f"Found {len(dataset_dirs)} datasets")

# Load class mappings for each dataset
ds_mappings = {}
for ds_dir in dataset_dirs:
    print(f"\nLoading class mapping for: {ds_dir.name}")
    mapping = load_class_mapping(ds_dir)
    if mapping:
        print(f"  Mapped {len(mapping)} classes")
        ds_mappings[ds_dir] = mapping

# Collect all images and labels with their dataset's mapping
all_samples = []

for ds_dir in dataset_dirs:
    mapping = ds_mappings.get(ds_dir)
    if not mapping:
        print(f"Skipping {ds_dir.name} - no class mapping")
        continue

    for split in ['train', 'valid', 'test']:
        images_dir = ds_dir / split / 'images'
        labels_dir = ds_dir / split / 'labels'

        if not images_dir.exists():
            continue

        for img_path in images_dir.glob('*'):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                label_path = labels_dir / f"{img_path.stem}.txt"
                if label_path.exists():
                    all_samples.append((img_path, label_path, mapping))

print(f"\nTotal samples: {len(all_samples)}")

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

# Create output directories and copy/remap files
total_skipped = 0
for split_name, samples in splits.items():
    img_dir = output_dir / split_name / 'images'
    lbl_dir = output_dir / split_name / 'labels'
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for i, (img_path, lbl_path, mapping) in enumerate(samples):
        new_name = f"{split_name}_{i:06d}"
        dst_lbl = lbl_dir / f"{new_name}.txt"

        # Remap labels to standard classes
        if remap_label_file(lbl_path, dst_lbl, mapping):
            shutil.copy2(img_path, img_dir / f"{new_name}{img_path.suffix}")
            written += 1
        else:
            total_skipped += 1

    print(f"{split_name}: {written} samples")

if total_skipped:
    print(f"Skipped {total_skipped} samples with unmappable labels")

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
print(f"All labels remapped to {len(STANDARD_CLASSES)} standard classes")
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
