#!/usr/bin/env python3
"""
Dataset Capture Tool for Custom Chess Pieces

Captures images from IP Webcam or webcam for training data collection.
Images are saved with perspective correction (warped to bird's eye view).

Usage:
    python scripts/capture_dataset.py --source http://192.168.1.101:8080 --output data/custom

Controls:
    SPACE: Capture current frame
    R: Toggle recording mode (auto-capture every 2 seconds)
    A: Auto-capture 20 frames with slight delays
    Q: Quit
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.aruco import ArUcoDetector
from src.inference.perspective import PerspectiveCorrector


class DatasetCapture:
    """Capture images for dataset creation."""

    def __init__(
        self,
        source: str,
        output_dir: str,
        orientation: str = "top",
        flip_board: bool = False,
    ):
        self.source = source
        self.output_dir = Path(output_dir)
        self.orientation = orientation
        self.flip_board = flip_board

        # Create output directories
        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.aruco = ArUcoDetector()
        self.perspective = PerspectiveCorrector((640, 640))
        self.calibrated = False

        # Rotation mapping
        self.rotation_map = {
            "right": cv2.ROTATE_90_COUNTERCLOCKWISE,
            "left": cv2.ROTATE_90_CLOCKWISE,
            "top": None,
            "bottom": cv2.ROTATE_180,
        }

        # Capture state
        self.frame_count = 0
        self.captured_count = 0
        self.recording = False
        self.last_capture_time = 0

    def _rotate_frame(self, frame: np.ndarray) -> np.ndarray:
        """Rotate frame based on orientation."""
        rotation = self.rotation_map.get(self.orientation)
        if rotation is not None:
            return cv2.rotate(frame, rotation)
        return frame

    def _is_ip_webcam_url(self, source: str) -> bool:
        """Check if source is an IP Webcam URL."""
        return source.startswith("http://") and ":8080" in source

    def _fetch_ip_webcam_frame(self, base_url: str):
        """Fetch a single frame from IP Webcam."""
        try:
            shot_url = base_url.rstrip('/').replace('/video', '').replace('/videofeed', '')
            if not shot_url.endswith('/shot.jpg'):
                shot_url = shot_url.rstrip('/') + '/shot.jpg'

            result = subprocess.run(
                ['curl', '-s', '-m', '2', shot_url],
                capture_output=True,
                timeout=3
            )
            if result.returncode == 0 and len(result.stdout) > 1000:
                img_array = np.asarray(bytearray(result.stdout), dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                return frame
            return None
        except Exception:
            return None

    def _save_frame(self, frame: np.ndarray, warped: np.ndarray):
        """Save both original and warped frames."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

        # Save warped (this is what we'll annotate)
        warped_path = self.images_dir / f"chess_{timestamp}.jpg"
        cv2.imwrite(str(warped_path), warped)

        self.captured_count += 1
        print(f"Captured: {warped_path.name} (Total: {self.captured_count})")

        return warped_path

    def run(self):
        """Main capture loop."""
        use_ip_webcam = self._is_ip_webcam_url(self.source)
        cap = None

        if use_ip_webcam:
            print(f"Connecting to IP Webcam: {self.source}")
            frame = self._fetch_ip_webcam_frame(self.source)
            if frame is None:
                print("Error: Could not connect to IP Webcam")
                return
            print("Connected!")
        else:
            if self.source.isdigit():
                cap = cv2.VideoCapture(int(self.source))
            else:
                cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                print(f"Error: Could not open video source: {self.source}")
                return

        print("\n=== Dataset Capture Tool ===")
        print(f"Output directory: {self.images_dir}")
        print("\nControls:")
        print("  SPACE: Capture current frame")
        print("  R: Toggle auto-recording (every 2s)")
        print("  A: Auto-capture 20 frames")
        print("  Q: Quit")
        print("\nTip: Move pieces between captures for variety!")
        print("=" * 40)

        cv2.namedWindow("Dataset Capture", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Dataset Capture", 1200, 700)

        while True:
            # Get frame
            if use_ip_webcam:
                frame = self._fetch_ip_webcam_frame(self.source)
                if frame is None:
                    time.sleep(0.1)
                    continue
            else:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.1)
                    continue

            self.frame_count += 1
            frame = self._rotate_frame(frame)

            # Detect ArUco markers
            corners = self.aruco.detect(frame)

            if corners and corners.all_found:
                self.perspective.calibrate(corners)
                self.calibrated = True

            # Create display
            if self.calibrated:
                warped = self.perspective.warp_frame(frame)
                if self.flip_board:
                    warped = cv2.flip(warped, 1)

                # Draw grid on warped
                display = self.perspective.draw_grid(warped.copy(), warped=True)

                # Auto-capture in recording mode
                if self.recording and time.time() - self.last_capture_time > 2.0:
                    self._save_frame(frame, warped)
                    self.last_capture_time = time.time()
            else:
                display = frame.copy()
                if corners:
                    display = self.aruco.draw_markers(display, corners)
                cv2.putText(display, "Looking for ArUco markers...", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Draw status
            status_color = (0, 255, 0) if self.calibrated else (0, 0, 255)
            cv2.putText(display, f"Calibrated: {self.calibrated}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            cv2.putText(display, f"Captured: {self.captured_count}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if self.recording:
                cv2.putText(display, "RECORDING", (10, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow("Dataset Capture", display)

            # Handle keys
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' ') and self.calibrated:
                warped = self.perspective.warp_frame(frame)
                if self.flip_board:
                    warped = cv2.flip(warped, 1)
                self._save_frame(frame, warped)
            elif key == ord('r'):
                self.recording = not self.recording
                print(f"Recording: {'ON' if self.recording else 'OFF'}")
            elif key == ord('a') and self.calibrated:
                print("Auto-capturing 20 frames...")
                for i in range(20):
                    if use_ip_webcam:
                        frame = self._fetch_ip_webcam_frame(self.source)
                    else:
                        ret, frame = cap.read()
                        frame = frame if ret else None

                    if frame is not None:
                        frame = self._rotate_frame(frame)
                        warped = self.perspective.warp_frame(frame)
                        if self.flip_board:
                            warped = cv2.flip(warped, 1)
                        self._save_frame(frame, warped)
                    time.sleep(0.3)
                print("Auto-capture complete!")

        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()

        print(f"\n=== Capture Complete ===")
        print(f"Total images captured: {self.captured_count}")
        print(f"Saved to: {self.images_dir}")
        print("\nNext steps:")
        print("1. Upload images to Roboflow for annotation")
        print("2. Or use: python scripts/auto_annotate.py --images", self.images_dir)


def main():
    parser = argparse.ArgumentParser(description="Dataset Capture Tool")
    parser.add_argument(
        "--source", "-s", default="0",
        help="Video source: camera index, IP Webcam URL, or video file"
    )
    parser.add_argument(
        "--output", "-o", default="data/custom",
        help="Output directory for captured images"
    )
    parser.add_argument(
        "--orientation", default="top",
        choices=["top", "bottom", "left", "right"],
        help="Camera orientation"
    )
    parser.add_argument(
        "--flip", "-f", action="store_true",
        help="Flip board horizontally"
    )

    args = parser.parse_args()

    capture = DatasetCapture(
        source=args.source,
        output_dir=args.output,
        orientation=args.orientation,
        flip_board=args.flip,
    )
    capture.run()


if __name__ == "__main__":
    main()
