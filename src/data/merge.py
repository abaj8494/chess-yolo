"""Merge multiple chess piece datasets and normalize labels."""

import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Optional

import yaml
from rich.console import Console
from rich.progress import track

console = Console()

# Standard class mapping - normalize all dataset labels to this format
STANDARD_CLASSES = {
    "white-king": 0,
    "white-queen": 1,
    "white-rook": 2,
    "white-bishop": 3,
    "white-knight": 4,
    "white-pawn": 5,
    "black-king": 6,
    "black-queen": 7,
    "black-rook": 8,
    "black-bishop": 9,
    "black-knight": 10,
    "black-pawn": 11,
}

# Common label variations found in different datasets
LABEL_MAPPINGS = {
    # White pieces
    "white_king": "white-king",
    "white king": "white-king",
    "wk": "white-king",
    "w-king": "white-king",
    "white_queen": "white-queen",
    "white queen": "white-queen",
    "wq": "white-queen",
    "w-queen": "white-queen",
    "white_rook": "white-rook",
    "white rook": "white-rook",
    "wr": "white-rook",
    "w-rook": "white-rook",
    "white_bishop": "white-bishop",
    "white bishop": "white-bishop",
    "wb": "white-bishop",
    "w-bishop": "white-bishop",
    "white_knight": "white-knight",
    "white knight": "white-knight",
    "wn": "white-knight",
    "w-knight": "white-knight",
    "white_pawn": "white-pawn",
    "white pawn": "white-pawn",
    "wp": "white-pawn",
    "w-pawn": "white-pawn",
    # Black pieces
    "black_king": "black-king",
    "black king": "black-king",
    "bk": "black-king",
    "b-king": "black-king",
    "black_queen": "black-queen",
    "black queen": "black-queen",
    "bq": "black-queen",
    "b-queen": "black-queen",
    "black_rook": "black-rook",
    "black rook": "black-rook",
    "br": "black-rook",
    "b-rook": "black-rook",
    "black_bishop": "black-bishop",
    "black bishop": "black-bishop",
    "bb": "black-bishop",
    "b-bishop": "black-bishop",
    "black_knight": "black-knight",
    "black knight": "black-knight",
    "bn": "black-knight",
    "b-knight": "black-knight",
    "black_pawn": "black-pawn",
    "black pawn": "black-pawn",
    "bp": "black-pawn",
    "b-pawn": "black-pawn",
}


def load_dataset_config(dataset_path: Path) -> dict:
    """Load dataset YAML config to get class mappings."""
    yaml_files = list(dataset_path.glob("*.yaml")) + list(dataset_path.glob("data.yaml"))
    if not yaml_files:
        raise FileNotFoundError(f"No YAML config found in {dataset_path}")

    with open(yaml_files[0]) as f:
        return yaml.safe_load(f)


def normalize_label(original_label: str) -> Optional[str]:
    """Normalize a label to standard format."""
    label_lower = original_label.lower().strip()

    # Direct match
    if label_lower in STANDARD_CLASSES:
        return label_lower

    # Check mappings
    if label_lower in LABEL_MAPPINGS:
        return LABEL_MAPPINGS[label_lower]

    # Try to infer from common patterns
    for piece in ["king", "queen", "rook", "bishop", "knight", "pawn"]:
        if piece in label_lower:
            if "white" in label_lower or label_lower.startswith("w"):
                return f"white-{piece}"
            elif "black" in label_lower or label_lower.startswith("b"):
                return f"black-{piece}"

    console.print(f"[yellow]Warning: Could not normalize label '{original_label}'")
    return None


def remap_annotation(
    label_path: Path,
    old_to_new_mapping: dict[int, int],
) -> list[str]:
    """Remap class IDs in a YOLO annotation file."""
    with open(label_path) as f:
        lines = f.readlines()

    remapped_lines = []
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue

        old_class_id = int(parts[0])
        if old_class_id in old_to_new_mapping:
            new_class_id = old_to_new_mapping[old_class_id]
            parts[0] = str(new_class_id)
            remapped_lines.append(" ".join(parts))

    return remapped_lines


def merge_datasets(
    input_dirs: list[Path],
    output_dir: Path,
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> dict:
    """Merge multiple datasets into train/val/test splits.

    Args:
        input_dirs: List of dataset directories (YOLO format)
        output_dir: Output directory for merged dataset
        split_ratios: Train/val/test ratios
        seed: Random seed for reproducibility

    Returns:
        Statistics about the merged dataset
    """
    random.seed(seed)

    output_dir = Path(output_dir)
    for split in ["train", "val", "test"]:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    all_samples = []  # (image_path, label_path, class_mapping)
    stats = defaultdict(int)

    for dataset_dir in input_dirs:
        dataset_dir = Path(dataset_dir)
        console.print(f"[blue]Processing {dataset_dir.name}...")

        try:
            config = load_dataset_config(dataset_dir)
        except FileNotFoundError:
            console.print(f"[yellow]Skipping {dataset_dir} - no config found")
            continue

        # Build class ID mapping
        old_names = config.get("names", {})
        if isinstance(old_names, list):
            old_names = {i: name for i, name in enumerate(old_names)}

        id_mapping = {}
        for old_id, old_name in old_names.items():
            normalized = normalize_label(old_name)
            if normalized:
                id_mapping[int(old_id)] = STANDARD_CLASSES[normalized]

        # Find all images
        for split in ["train", "valid", "val", "test"]:
            images_dir = dataset_dir / split / "images"
            labels_dir = dataset_dir / split / "labels"

            if not images_dir.exists():
                continue

            for img_path in images_dir.glob("*"):
                if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
                    continue

                label_path = labels_dir / f"{img_path.stem}.txt"
                if label_path.exists():
                    all_samples.append((img_path, label_path, id_mapping))
                    stats["total_images"] += 1

    console.print(f"[green]Found {len(all_samples)} total samples")

    # Shuffle and split
    random.shuffle(all_samples)

    train_end = int(len(all_samples) * split_ratios[0])
    val_end = train_end + int(len(all_samples) * split_ratios[1])

    splits = {
        "train": all_samples[:train_end],
        "val": all_samples[train_end:val_end],
        "test": all_samples[val_end:],
    }

    # Copy files with remapped annotations
    for split_name, samples in splits.items():
        console.print(f"[blue]Processing {split_name} split ({len(samples)} samples)...")

        for i, (img_path, label_path, id_mapping) in enumerate(
            track(samples, description=f"  {split_name}")
        ):
            # Generate unique filename
            new_name = f"{split_name}_{i:06d}"
            new_img_path = output_dir / split_name / "images" / f"{new_name}{img_path.suffix}"
            new_label_path = output_dir / split_name / "labels" / f"{new_name}.txt"

            # Copy image
            shutil.copy2(img_path, new_img_path)

            # Remap and save labels
            remapped_lines = remap_annotation(label_path, id_mapping)
            with open(new_label_path, "w") as f:
                f.write("\n".join(remapped_lines))

            stats[f"{split_name}_samples"] += 1

    # Create dataset YAML
    dataset_yaml = {
        "path": str(output_dir.absolute()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": 12,
        "names": list(STANDARD_CLASSES.keys()),
    }

    with open(output_dir / "dataset.yaml", "w") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False)

    console.print(f"\n[green]Merged dataset saved to {output_dir}")
    console.print(f"  Train: {stats['train_samples']} samples")
    console.print(f"  Val: {stats['val_samples']} samples")
    console.print(f"  Test: {stats['test_samples']} samples")

    return dict(stats)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Merge chess piece datasets")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing downloaded datasets",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/splits"),
        help="Output directory for merged dataset",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.8, help="Training set ratio"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.1, help="Validation set ratio"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Find all dataset directories
    input_dirs = [d for d in args.input_dir.iterdir() if d.is_dir()]

    merge_datasets(
        input_dirs=input_dirs,
        output_dir=args.output_dir,
        split_ratios=(args.train_ratio, args.val_ratio, 1 - args.train_ratio - args.val_ratio),
        seed=args.seed,
    )
