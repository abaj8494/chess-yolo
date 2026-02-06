"""YOLOv8 training wrapper with HPC/SLURM support."""

import signal
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml
from rich.console import Console
from ultralytics import YOLO

from .hpc_utils import find_latest_checkpoint, setup_slurm_signal_handler

console = Console()


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Model
    model: str = "yolov8m.pt"
    data: str = "configs/data/chess_pieces.yaml"

    # Training params
    epochs: int = 300
    batch: int = 64
    imgsz: int = 640
    patience: int = 50
    save_period: int = 10

    # Optimizer
    optimizer: str = "AdamW"
    lr0: float = 0.001
    lrf: float = 0.01

    # Device
    device: str = "0"
    workers: int = 16
    cache: str = "disk"
    amp: bool = True

    # Project
    project: str = "experiments"
    name: str = "chess-pieces-v1"
    exist_ok: bool = True

    @classmethod
    def from_yaml(cls, path: Path) -> "TrainingConfig":
        """Load config from YAML file."""
        with open(path) as f:
            config = yaml.safe_load(f)
        return cls(**{k: v for k, v in config.items() if hasattr(cls, k)})


class ChessYOLOTrainer:
    """YOLOv8 trainer with checkpoint resumption and signal handling."""

    def __init__(
        self,
        config: TrainingConfig,
        resume: bool = True,
    ):
        self.config = config
        self.resume = resume
        self.model: Optional[YOLO] = None
        self._shutdown_requested = False

        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Setup handlers for SLURM signals."""

        def signal_handler(signum, frame):
            console.print(f"\n[yellow]Received signal {signum}, initiating graceful shutdown...")
            self._shutdown_requested = True
            if self.model is not None:
                try:
                    # Save current model state
                    self.model.save(Path(self.config.project) / self.config.name / "weights" / "last.pt")
                    console.print("[green]Checkpoint saved successfully")
                except Exception as e:
                    console.print(f"[red]Failed to save checkpoint: {e}")
            sys.exit(0)

        # Handle common termination signals
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        # SIGUSR1 is commonly used by SLURM for preemption warning
        if hasattr(signal, "SIGUSR1"):
            signal.signal(signal.SIGUSR1, signal_handler)

    def _find_checkpoint(self) -> Optional[Path]:
        """Find latest checkpoint for resumption."""
        project_dir = Path(self.config.project) / self.config.name / "weights"
        return find_latest_checkpoint(project_dir)

    def train(self) -> dict:
        """Run training with automatic checkpoint resumption.

        Returns:
            Training results dictionary
        """
        # Check for existing checkpoint
        checkpoint = None
        if self.resume:
            checkpoint = self._find_checkpoint()
            if checkpoint:
                console.print(f"[green]Resuming from checkpoint: {checkpoint}")

        # Initialize model
        if checkpoint:
            self.model = YOLO(str(checkpoint))
        else:
            console.print(f"[blue]Starting fresh training with {self.config.model}")
            self.model = YOLO(self.config.model)

        # Run training
        console.print("[blue]Starting training...")
        results = self.model.train(
            data=self.config.data,
            epochs=self.config.epochs,
            batch=self.config.batch,
            imgsz=self.config.imgsz,
            patience=self.config.patience,
            save_period=self.config.save_period,
            optimizer=self.config.optimizer,
            lr0=self.config.lr0,
            lrf=self.config.lrf,
            device=self.config.device,
            workers=self.config.workers,
            cache=self.config.cache,
            amp=self.config.amp,
            project=self.config.project,
            name=self.config.name,
            exist_ok=self.config.exist_ok,
            resume=checkpoint is not None,
            verbose=True,
        )

        console.print("[green]Training completed!")
        return results

    def validate(self) -> dict:
        """Run validation on the trained model."""
        if self.model is None:
            checkpoint = self._find_checkpoint()
            if checkpoint:
                self.model = YOLO(str(checkpoint))
            else:
                raise ValueError("No trained model found")

        return self.model.val()

    def export(self, format: str = "coreml") -> Path:
        """Export the trained model.

        Args:
            format: Export format (coreml, onnx, etc.)

        Returns:
            Path to exported model
        """
        if self.model is None:
            best_pt = Path(self.config.project) / self.config.name / "weights" / "best.pt"
            if best_pt.exists():
                self.model = YOLO(str(best_pt))
            else:
                raise ValueError("No trained model found")

        return Path(self.model.export(format=format))
