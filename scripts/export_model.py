#!/usr/bin/env python3
"""Export CLI for Chess YOLO models."""

import argparse
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console

console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="Export Chess YOLO model to different formats",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "model",
        type=Path,
        help="Path to trained model (.pt)",
    )

    parser.add_argument(
        "-f", "--format",
        type=str,
        choices=["coreml", "onnx", "torchscript"],
        default="coreml",
        help="Export format",
    )

    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output directory",
    )

    parser.add_argument(
        "--half",
        action="store_true",
        default=True,
        help="Use FP16 quantization",
    )

    parser.add_argument(
        "--no-half",
        action="store_true",
        help="Disable FP16 quantization",
    )

    parser.add_argument(
        "--int8",
        action="store_true",
        help="Use INT8 quantization (more aggressive)",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size",
    )

    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify export with test inference",
    )

    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run inference benchmark",
    )

    args = parser.parse_args()

    console.print("\n[bold blue]Chess YOLO Model Export[/bold blue]")
    console.print("=" * 50)

    if not args.model.exists():
        console.print(f"[red]Model not found: {args.model}")
        sys.exit(1)

    half = args.half and not args.no_half

    from src.export.coreml_export import (
        export_to_coreml,
        export_to_onnx,
        verify_coreml_export,
        benchmark_coreml,
    )

    if args.format == "coreml":
        export_path = export_to_coreml(
            model_path=args.model,
            output_dir=args.output,
            half=half,
            int8=args.int8,
            imgsz=args.imgsz,
        )

        if args.verify:
            console.print("\n[blue]Verifying export...")
            verify_coreml_export(export_path)

        if args.benchmark:
            console.print("\n[blue]Running benchmark...")
            benchmark_coreml(export_path)

    elif args.format == "onnx":
        export_path = export_to_onnx(
            model_path=args.model,
            output_dir=args.output,
            half=half,
            imgsz=args.imgsz,
        )

    else:
        from ultralytics import YOLO
        model = YOLO(str(args.model))
        export_path = model.export(format=args.format, half=half, imgsz=args.imgsz)
        console.print(f"[green]Exported to: {export_path}")


if __name__ == "__main__":
    main()
