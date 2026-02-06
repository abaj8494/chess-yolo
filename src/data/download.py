"""Download chess piece datasets from Roboflow."""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress
from roboflow import Roboflow

load_dotenv()

console = Console()

# Curated list of chess piece datasets from Roboflow
DATASETS = [
    {
        "workspace": "joseph-nelson",
        "project": "chess-pieces-new",
        "version": 1,
        "name": "chess-pieces-nelson",
    },
    {
        "workspace": "block",
        "project": "chess-pieces-wrdbb",
        "version": 1,
        "name": "chess-pieces-block",
    },
    {
        "workspace": "chess-lpnmj",
        "project": "chess-piece-detection-7alyg",
        "version": 1,
        "name": "chess-pieces-detection",
    },
]


def download_dataset(
    api_key: str,
    workspace: str,
    project: str,
    version: int,
    output_dir: Path,
    format: str = "yolov8",
) -> Path:
    """Download a single dataset from Roboflow.

    Args:
        api_key: Roboflow API key
        workspace: Roboflow workspace name
        project: Project name
        version: Dataset version number
        output_dir: Directory to save dataset
        format: Export format (default: yolov8)

    Returns:
        Path to downloaded dataset
    """
    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    dataset = proj.version(version).download(format, location=str(output_dir))
    return Path(dataset.location)


def download_all_datasets(
    output_dir: Path,
    api_key: Optional[str] = None,
) -> list[Path]:
    """Download all curated chess piece datasets.

    Args:
        output_dir: Base directory for downloads
        api_key: Roboflow API key (or set ROBOFLOW_API_KEY env var)

    Returns:
        List of paths to downloaded datasets
    """
    api_key = api_key or os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise ValueError(
            "Roboflow API key required. Set ROBOFLOW_API_KEY env var or pass api_key."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded = []

    with Progress() as progress:
        task = progress.add_task("[green]Downloading datasets...", total=len(DATASETS))

        for ds in DATASETS:
            console.print(f"[blue]Downloading {ds['name']}...")
            try:
                ds_path = download_dataset(
                    api_key=api_key,
                    workspace=ds["workspace"],
                    project=ds["project"],
                    version=ds["version"],
                    output_dir=output_dir / ds["name"],
                )
                downloaded.append(ds_path)
                console.print(f"[green]✓ Downloaded to {ds_path}")
            except Exception as e:
                console.print(f"[red]✗ Failed to download {ds['name']}: {e}")

            progress.advance(task)

    console.print(f"\n[green]Downloaded {len(downloaded)}/{len(DATASETS)} datasets")
    return downloaded


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download chess piece datasets")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Output directory for datasets",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Roboflow API key (or set ROBOFLOW_API_KEY)",
    )

    args = parser.parse_args()
    download_all_datasets(args.output_dir, args.api_key)
