#!/usr/bin/env python3
"""
Prepare a test model for the Flutter app.

This downloads a pre-trained YOLOv8n and exports it to TFLite format.
The model won't detect chess pieces specifically but can be used to test
the app's inference pipeline while waiting for custom training to complete.

Usage:
    python scripts/prepare_test_model.py

After running, copy the model to the Flutter app:
    cp exports/yolov8n_test.tflite ../chess_cam/assets/models/yolov8n_chess.tflite
"""

from pathlib import Path
from ultralytics import YOLO


def main():
    output_dir = Path('exports')
    output_dir.mkdir(exist_ok=True)

    print("Downloading YOLOv8n pre-trained model...")
    model = YOLO('yolov8n.pt')

    print("\nExporting to TFLite...")
    export_path = model.export(
        format='tflite',
        imgsz=640,
        int8=False,
    )

    exported_file = Path(export_path)
    if exported_file.exists():
        final_path = output_dir / 'yolov8n_test.tflite'
        exported_file.rename(final_path)
        print(f"\nTest model saved to: {final_path}")
        print(f"Model size: {final_path.stat().st_size / (1024*1024):.2f} MB")
        print("\nTo use in Flutter app:")
        print(f"  cp {final_path} ../chess_cam/assets/models/yolov8n_chess.tflite")
    else:
        print(f"Error: Export failed")


if __name__ == '__main__':
    main()
