"""HPC/SLURM utilities for distributed training and checkpointing."""

import os
import signal
from pathlib import Path
from typing import Callable, Optional

from rich.console import Console

console = Console()


def find_latest_checkpoint(weights_dir: Path) -> Optional[Path]:
    """Find the latest checkpoint in a weights directory.

    Checks for:
    1. last.pt (most recent checkpoint)
    2. best.pt (best validation checkpoint)
    3. Epoch-specific checkpoints (epoch*.pt)

    Args:
        weights_dir: Directory containing weight files

    Returns:
        Path to latest checkpoint or None if not found
    """
    weights_dir = Path(weights_dir)

    if not weights_dir.exists():
        return None

    # Priority 1: last.pt
    last_pt = weights_dir / "last.pt"
    if last_pt.exists():
        console.print(f"[green]Found last.pt checkpoint")
        return last_pt

    # Priority 2: epoch checkpoints (find highest epoch)
    epoch_checkpoints = sorted(
        weights_dir.glob("epoch*.pt"),
        key=lambda p: int(p.stem.replace("epoch", "")),
        reverse=True,
    )
    if epoch_checkpoints:
        console.print(f"[green]Found epoch checkpoint: {epoch_checkpoints[0].name}")
        return epoch_checkpoints[0]

    # Priority 3: best.pt (only if no other checkpoints)
    best_pt = weights_dir / "best.pt"
    if best_pt.exists():
        console.print(f"[yellow]Only best.pt found (training may have completed)")
        return best_pt

    return None


def get_slurm_info() -> dict:
    """Get SLURM job information from environment variables.

    Returns:
        Dictionary with SLURM job info or empty dict if not in SLURM
    """
    slurm_vars = [
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "SLURM_NODELIST",
        "SLURM_NNODES",
        "SLURM_NTASKS",
        "SLURM_CPUS_PER_TASK",
        "SLURM_GPUS_PER_NODE",
        "SLURM_JOB_PARTITION",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_ARRAY_JOB_ID",
    ]

    info = {}
    for var in slurm_vars:
        value = os.environ.get(var)
        if value:
            info[var] = value

    return info


def is_slurm_job() -> bool:
    """Check if running inside a SLURM job."""
    return "SLURM_JOB_ID" in os.environ


def setup_slurm_signal_handler(callback: Callable[[], None]) -> None:
    """Setup signal handler for SLURM preemption.

    SLURM sends SIGUSR1 (configurable) before job termination.
    This allows saving checkpoints before the job is killed.

    Args:
        callback: Function to call when preemption signal received
    """

    def handler(signum, frame):
        console.print(f"\n[yellow]SLURM preemption signal received (signal {signum})")
        console.print("[yellow]Saving checkpoint before termination...")
        callback()
        console.print("[green]Checkpoint saved, exiting gracefully")
        raise SystemExit(0)

    # SIGUSR1 is the default SLURM preemption signal
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, handler)

    # Also handle SIGTERM for clean shutdown
    signal.signal(signal.SIGTERM, handler)


def get_local_scratch() -> Optional[Path]:
    """Get local scratch directory for fast I/O.

    Many HPC systems provide fast local NVMe storage for jobs.
    Common environment variables used by different systems.

    Returns:
        Path to local scratch or None if not available
    """
    scratch_vars = [
        "TMPDIR",  # Common on many systems
        "LOCAL_SCRATCH",
        "SLURM_TMPDIR",
        "SCRATCH",
        "LSCRATCH",
    ]

    for var in scratch_vars:
        path = os.environ.get(var)
        if path and Path(path).exists():
            return Path(path)

    return None


def setup_cache_dirs(base_dir: Optional[Path] = None) -> dict:
    """Setup cache directories for training.

    Uses local scratch if available for faster I/O.

    Args:
        base_dir: Base directory for caches (defaults to local scratch or /tmp)

    Returns:
        Dictionary with paths for different cache types
    """
    if base_dir is None:
        base_dir = get_local_scratch() or Path("/tmp")

    user = os.environ.get("USER", "user")
    cache_base = base_dir / user / "chess-yolo-cache"
    cache_base.mkdir(parents=True, exist_ok=True)

    cache_dirs = {
        "data": cache_base / "data",
        "torch": cache_base / "torch",
        "yolo": cache_base / "yolo",
        "huggingface": cache_base / "huggingface",
    }

    for name, path in cache_dirs.items():
        path.mkdir(parents=True, exist_ok=True)

    # Set environment variables
    os.environ["TORCH_HOME"] = str(cache_dirs["torch"])
    os.environ["YOLO_CONFIG_DIR"] = str(cache_dirs["yolo"])
    os.environ["HF_HOME"] = str(cache_dirs["huggingface"])

    console.print(f"[blue]Cache directories set up at {cache_base}")
    return cache_dirs


def log_gpu_info():
    """Log GPU information for debugging."""
    try:
        import torch

        if torch.cuda.is_available():
            console.print(f"[green]CUDA available: {torch.cuda.is_available()}")
            console.print(f"[green]GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                console.print(
                    f"[green]  GPU {i}: {props.name} "
                    f"({props.total_memory / 1024**3:.1f} GB)"
                )
        else:
            console.print("[yellow]CUDA not available")
    except ImportError:
        console.print("[yellow]PyTorch not installed")
