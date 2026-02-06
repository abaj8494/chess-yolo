#!/usr/bin/env python3
"""Training CLI for Chess YOLO."""

import argparse
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from src.training.trainer import ChessYOLOTrainer, TrainingConfig
from src.training.hpc_utils import (
    get_slurm_info,
    is_slurm_job,
    log_gpu_info,
    setup_cache_dirs,
)

console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 chess piece detector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to training config YAML",
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8m.pt",
        help="Base model to fine-tune",
    )

    # Data
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("configs/data/chess_pieces.yaml"),
        help="Path to dataset YAML",
    )

    # Training params
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size",
    )

    # Output
    parser.add_argument(
        "--project",
        type=str,
        default="experiments",
        help="Project directory for outputs",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="chess-pieces-v1",
        help="Experiment name",
    )

    # Resume
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from checkpoint if available",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh training (don't resume)",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device to train on (0, 1, cpu, etc.)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of data loader workers",
    )

    args = parser.parse_args()

    # Print header
    console.print("\n[bold blue]Chess YOLO Training[/bold blue]")
    console.print("=" * 50)

    # Check if running in SLURM
    if is_slurm_job():
        slurm_info = get_slurm_info()
        console.print(f"[green]Running as SLURM job: {slurm_info.get('SLURM_JOB_ID')}")
        setup_cache_dirs()

    # Log GPU info
    log_gpu_info()

    # Build config
    if args.config and args.config.exists():
        console.print(f"[blue]Loading config from: {args.config}")
        config = TrainingConfig.from_yaml(args.config)
    else:
        config = TrainingConfig(
            model=args.model,
            data=str(args.data),
            epochs=args.epochs,
            batch=args.batch_size,
            imgsz=args.imgsz,
            project=args.project,
            name=args.name,
            device=args.device,
            workers=args.workers,
        )

    # Print config
    console.print("\n[blue]Training Configuration:")
    console.print(f"  Model: {config.model}")
    console.print(f"  Data: {config.data}")
    console.print(f"  Epochs: {config.epochs}")
    console.print(f"  Batch size: {config.batch}")
    console.print(f"  Image size: {config.imgsz}")
    console.print(f"  Device: {config.device}")
    console.print(f"  Project: {config.project}/{config.name}")

    # Initialize trainer
    resume = args.resume and not args.no_resume
    console.print(f"  Resume: {resume}")

    trainer = ChessYOLOTrainer(config=config, resume=resume)

    # Train
    console.print("\n[green]Starting training...")
    try:
        results = trainer.train()
        console.print("\n[bold green]Training completed successfully!")

        # Print final metrics
        if hasattr(results, "results_dict"):
            console.print("\n[blue]Final Metrics:")
            for key, value in results.results_dict.items():
                if isinstance(value, float):
                    console.print(f"  {key}: {value:.4f}")

    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
