#!/usr/bin/env python3
"""
Export trained YOLO models to TFLite format for Flutter app.

Usage:
    python scripts/export_tflite.py --model runs/detect/chess_pieces/weights/best.pt --output exports/
    python scripts/export_tflite.py --model yolov8n.pt --output exports/  # For testing with pretrained
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def export_to_tflite(model_path: str, output_dir: str, quantize: bool = False) -> Path:
    """
    Export a YOLO model to TFLite format.

    Args:
        model_path: Path to the .pt model file
        output_dir: Directory to save exported model
        quantize: Whether to apply int8 quantization (smaller but slightly less accurate)

    Returns:
        Path to the exported TFLite model
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)

    # Get model name for output filename
    model_name = Path(model_path).stem

    # Export to TFLite
    print(f"Exporting to TFLite (quantize={quantize})...")
    export_path = model.export(
        format='tflite',
        imgsz=640,
        int8=quantize,
        # TFLite specific optimizations
        nms=True,  # Include NMS in model
    )

    # Move to output directory with proper name
    exported_file = Path(export_path)
    suffix = '_int8' if quantize else ''
    final_path = output_path / f"{model_name}_chess{suffix}.tflite"

    if exported_file.exists():
        exported_file.rename(final_path)
        print(f"Exported model saved to: {final_path}")
        print(f"Model size: {final_path.stat().st_size / (1024*1024):.2f} MB")
        return final_path
    else:
        print(f"Error: Export failed, file not found at {export_path}")
        return None


def export_all_sizes(base_model_dir: str, output_dir: str):
    """
    Export nano, small, and medium models if available.
    """
    models = {
        'nano': 'yolov8n',
        'small': 'yolov8s',
        'medium': 'yolov8m',
    }

    base_path = Path(base_model_dir)
    output_path = Path(output_dir)

    for size_name, model_base in models.items():
        # Look for trained model
        trained_path = base_path / f"{model_base}_chess/weights/best.pt"

        if trained_path.exists():
            print(f"\n{'='*50}")
            print(f"Exporting {size_name} model: {trained_path}")
            print('='*50)

            # Export FP32 version
            export_to_tflite(str(trained_path), str(output_path), quantize=False)

            # For nano model, also export quantized version (even smaller)
            if size_name == 'nano':
                export_to_tflite(str(trained_path), str(output_path), quantize=True)
        else:
            print(f"Skipping {size_name}: {trained_path} not found")


def main():
    parser = argparse.ArgumentParser(description='Export YOLO models to TFLite')
    parser.add_argument('--model', type=str, help='Path to single model to export')
    parser.add_argument('--output', type=str, default='exports/', help='Output directory')
    parser.add_argument('--quantize', action='store_true', help='Apply int8 quantization')
    parser.add_argument('--all', action='store_true', help='Export all available model sizes')
    parser.add_argument('--runs-dir', type=str, default='runs/detect', help='Base directory for trained models')

    args = parser.parse_args()

    if args.all:
        export_all_sizes(args.runs_dir, args.output)
    elif args.model:
        export_to_tflite(args.model, args.output, args.quantize)
    else:
        print("Please specify --model or --all")
        parser.print_help()


if __name__ == '__main__':
    main()
