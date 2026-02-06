#!/usr/bin/env python3
"""
Live Chess Detection using Frame Differencing

Much more reliable than YOLO-based detection for move tracking.
Detects changes in pixel content rather than trying to identify pieces.

Usage:
    python scripts/live_gui_framediff.py --source http://192.168.1.100:8080/video --flip

Controls:
    'r' - Reset (set current frame as reference, reset chess board)
    's' - Set reference frame (without resetting chess board)
    'q' - Quit
    'p' - Print current board state
"""

import argparse
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from collections import deque

import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.aruco import ArUcoDetector
from src.inference.perspective import PerspectiveCorrector
from src.inference.frame_diff_detector import FrameDiffMoveDetector


@dataclass
class GUIState:
    """Track GUI state."""
    moves: list = field(default_factory=list)
    fps: float = 0.0
    frame_count: int = 0
    aruco_calibrated: bool = False
    reference_set: bool = False
    status: str = "Waiting for ArUco markers..."


class FrameDiffGUI:
    """Live chess detection using frame differencing."""

    def __init__(
        self,
        source: str = "0",
        orientation: str = "top",
        flip_board: bool = False,
        rotate180: bool = False,
        window_width: int = 1200,
        window_height: int = 700,
        board_margin: int = 0,
    ):
        self.source = source
        self.orientation = orientation
        self.flip_board = flip_board
        self.rotate180 = rotate180  # Rotate 180 degrees (swaps a1<->h8)
        self.window_width = window_width
        self.window_height = window_height
        self.board_margin = board_margin  # Pixels to inset grid from edges

        # Video dimensions
        self.video_width = int(window_width * 0.7)
        self.video_height = window_height
        self.panel_width = window_width - self.video_width

        # State
        self.state = GUIState()
        self.fps_history = deque(maxlen=30)
        self.last_frame_time = time.time()

        # Rotation mapping
        self.rotation_map = {
            "right": cv2.ROTATE_90_COUNTERCLOCKWISE,
            "left": cv2.ROTATE_90_CLOCKWISE,
            "top": None,
            "bottom": cv2.ROTATE_180,
        }

        # Components
        print("Initializing...")
        self.aruco = ArUcoDetector()
        self.perspective = PerspectiveCorrector((640, 640))
        self.move_detector = FrameDiffMoveDetector(board_size=640, board_margin=board_margin)

        # Store warped frame
        self.last_warped = None

        print("Ready! Point camera at chess board with ArUco markers.")

    def _rotate_frame(self, frame: np.ndarray) -> np.ndarray:
        """Rotate frame based on orientation."""
        rotation = self.rotation_map.get(self.orientation)
        if rotation is not None:
            return cv2.rotate(frame, rotation)
        return frame

    def _draw_grid(self, frame: np.ndarray) -> np.ndarray:
        """Draw chess grid overlay."""
        viz = frame.copy()
        m = self.board_margin
        board_size = 640 - 2 * m  # Actual board area
        cell_size = board_size / 8

        # Draw grid lines
        for i in range(9):
            # Vertical lines
            x = int(m + i * cell_size)
            cv2.line(viz, (x, m), (x, 640 - m), (0, 255, 0), 1)
            # Horizontal lines
            y = int(m + i * cell_size)
            cv2.line(viz, (m, y), (640 - m, y), (0, 255, 0), 1)

        # Draw square labels
        files = "abcdefgh"
        ranks = "87654321"
        for r_idx, rank in enumerate(ranks):
            for f_idx, file in enumerate(files):
                x = int(m + f_idx * cell_size + 5)
                y = int(m + r_idx * cell_size + 15)
                label = f"{file}{rank}"
                cv2.putText(viz, label, (x, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 200, 0), 1)

        return viz

    def _draw_changes(self, frame: np.ndarray, changed: set) -> np.ndarray:
        """Highlight changed squares."""
        viz = frame.copy()
        m = self.board_margin
        board_size = 640 - 2 * m
        cell_size = board_size / 8
        files = "abcdefgh"
        ranks = "87654321"

        for square in changed:
            f_idx = files.index(square[0])
            r_idx = ranks.index(square[1])
            x1 = int(m + f_idx * cell_size)
            y1 = int(m + r_idx * cell_size)
            x2 = int(x1 + cell_size)
            y2 = int(y1 + cell_size)
            # Yellow overlay for changed
            overlay = viz.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), -1)
            cv2.addWeighted(overlay, 0.3, viz, 0.7, 0, viz)
            cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 255, 255), 2)

        return viz

    def _draw_panel(self, height: int) -> np.ndarray:
        """Draw info panel."""
        panel = np.zeros((height, self.panel_width, 3), dtype=np.uint8)
        panel[:] = (30, 30, 30)

        y = 30
        line_height = 25

        # Title
        cv2.putText(panel, "Chess Vision", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        y += line_height + 10

        # Status
        cv2.line(panel, (10, y), (self.panel_width - 10, y), (60, 60, 60), 1)
        y += 20

        # FPS
        fps_color = (0, 255, 0) if self.state.fps > 1 else (0, 255, 255)
        cv2.putText(panel, f"FPS: {self.state.fps:.1f}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, fps_color, 1)
        y += line_height

        # ArUco status
        aruco_color = (0, 255, 0) if self.state.aruco_calibrated else (0, 0, 255)
        aruco_text = "Board: OK" if self.state.aruco_calibrated else "Board: Looking..."
        cv2.putText(panel, aruco_text, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, aruco_color, 1)
        y += line_height

        # Reference status
        ref_color = (0, 255, 0) if self.state.reference_set else (0, 255, 255)
        ref_text = "Reference: SET" if self.state.reference_set else "Reference: Press 'r' to set"
        cv2.putText(panel, ref_text, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, ref_color, 1)
        y += line_height

        # Status message
        cv2.putText(panel, self.state.status, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        y += line_height + 10

        # Moves section
        cv2.line(panel, (10, y), (self.panel_width - 10, y), (60, 60, 60), 1)
        y += 25
        cv2.putText(panel, "Moves:", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        y += line_height + 5

        # Move list
        if self.state.moves:
            for i in range(0, len(self.state.moves), 2):
                move_num = i // 2 + 1
                white_move = self.state.moves[i] if i < len(self.state.moves) else ""
                black_move = self.state.moves[i + 1] if i + 1 < len(self.state.moves) else ""
                move_text = f"{move_num}. {white_move}  {black_move}"
                cv2.putText(panel, move_text, (25, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                y += line_height
                if y > height - 120:
                    cv2.putText(panel, "...", (25, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
                    break
        else:
            cv2.putText(panel, "(waiting for moves)", (25, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

        # Controls at bottom
        y = height - 80
        cv2.line(panel, (10, y), (self.panel_width - 10, y), (60, 60, 60), 1)
        y += 25
        cv2.putText(panel, "Controls:", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        y += line_height
        cv2.putText(panel, "R:Reset S:SetRef Q:Quit", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        y += 20
        cv2.putText(panel, "P:PrintBoard", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

        return panel

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process frame and return visualization."""
        # Rotate based on orientation
        frame = self._rotate_frame(frame)

        # Detect ArUco markers
        corners = self.aruco.detect(frame)

        # Only update calibration when all 4 markers are found with high confidence
        if corners and corners.all_found and corners.confidence >= 1.0:
            self.perspective.calibrate(corners)
            self.state.aruco_calibrated = True

        if self.state.aruco_calibrated and self.perspective.calibration is not None:
            # Warp to bird's eye view (uses last good calibration even if markers temporarily blocked)
            warped = self.perspective.warp_frame(frame)

            # Flip/rotate if needed
            if self.flip_board:
                warped = cv2.flip(warped, 1)
            if self.rotate180:
                warped = cv2.rotate(warped, cv2.ROTATE_180)

            self.last_warped = warped.copy()

            # Process with frame diff detector - only if all markers visible
            changed = set()
            markers_visible = corners and corners.all_found and len(corners.marker_ids) == 4

            if self.state.reference_set and markers_visible:
                result = self.move_detector.diff_detector.update(warped)
                if result:
                    changed = result.changed_squares

                # Try to detect move
                move_san = self.move_detector.update(warped)
                if move_san:
                    self.state.moves.append(move_san)
                    self.state.status = f"Move: {move_san}"
                    print(f"[GUI] Move confirmed: {move_san}")
            elif self.state.reference_set and not markers_visible:
                self.state.status = "Hand blocking markers..."

            # Draw visualization
            viz = self._draw_grid(warped)
            viz = self._draw_changes(viz, changed)

            # Add status text
            if not self.state.reference_set:
                cv2.putText(viz, "Press 'R' to set reference and start",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            return viz
        else:
            # Not calibrated - show original with marker overlay
            if corners:
                frame = self.aruco.draw_markers(frame, corners)

            h, w = frame.shape[:2]
            scale = min(self.video_width / w, self.video_height / h)
            new_w, new_h = int(w * scale), int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h))

            cv2.putText(frame, "Looking for ArUco markers...", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            return frame

    def _is_ip_webcam_url(self, source: str) -> bool:
        """Check if source is an IP Webcam URL."""
        return source.startswith("http://") and ":8080" in source

    def _fetch_ip_webcam_frame(self, base_url: str) -> Optional[np.ndarray]:
        """Fetch frame from IP Webcam."""
        import subprocess
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

    def run(self):
        """Main GUI loop."""
        use_ip_webcam = self._is_ip_webcam_url(self.source)
        cap = None

        if use_ip_webcam:
            print(f"Connecting to IP Webcam: {self.source}")
            frame = self._fetch_ip_webcam_frame(self.source)
            if frame is None:
                print(f"Error: Could not connect to IP Webcam")
                return
            print("IP Webcam connected!")
        else:
            if self.source.isdigit():
                cap = cv2.VideoCapture(int(self.source))
            else:
                cap = cv2.VideoCapture(self.source)

            if not cap.isOpened():
                print(f"Error: Could not open video source: {self.source}")
                return

        print(f"Connected to: {self.source}")
        print("Press 'R' to set reference frame and start detecting moves")
        print("Press 'Q' to quit")

        cv2.namedWindow("Chess Vision (Frame Diff)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Chess Vision (Frame Diff)", self.window_width, self.window_height)

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

            # Calculate FPS
            current_time = time.time()
            self.fps_history.append(1.0 / max(current_time - self.last_frame_time, 0.001))
            self.last_frame_time = current_time
            self.state.fps = sum(self.fps_history) / len(self.fps_history)
            self.state.frame_count += 1

            # Process frame
            viz = self._process_frame(frame)

            # Resize to fit
            h, w = viz.shape[:2]
            scale = min(self.video_width / w, self.video_height / h)
            new_w, new_h = int(w * scale), int(h * scale)
            viz = cv2.resize(viz, (new_w, new_h))

            # Create canvas
            canvas = np.zeros((self.window_height, self.window_width, 3), dtype=np.uint8)

            # Place video
            x_offset = (self.video_width - new_w) // 2
            y_offset = (self.window_height - new_h) // 2
            canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = viz

            # Draw panel
            panel = self._draw_panel(self.window_height)
            canvas[:, self.video_width:] = panel

            # Separator
            cv2.line(canvas, (self.video_width, 0), (self.video_width, self.window_height),
                     (60, 60, 60), 2)

            # Show
            cv2.imshow("Chess Vision (Frame Diff)", canvas)

            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Full reset
                if self.last_warped is not None:
                    self.move_detector.reset()
                    self.move_detector.set_reference(self.last_warped)
                    self.state.reference_set = True
                    self.state.moves = []
                    self.state.status = "Reference set - make your move!"
                    print("[GUI] Reference set, chess board reset")
            elif key == ord('s'):
                # Just update reference (don't reset chess board)
                if self.last_warped is not None:
                    self.move_detector.diff_detector.set_reference(self.last_warped)
                    self.state.reference_set = True
                    self.state.status = "Reference updated"
                    print("[GUI] Reference updated (board state preserved)")
            elif key == ord('p'):
                # Print board state
                print("\n" + "=" * 40)
                print("Current position:")
                print(self.move_detector.board)
                print(f"FEN: {self.move_detector.get_board_fen()}")
                print("=" * 40 + "\n")

        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Chess Vision with Frame Differencing")
    parser.add_argument(
        "--source", "-s", default="0",
        help="Video source: camera index, IP Webcam URL, or video file"
    )
    parser.add_argument(
        "--orientation", "-o", default="top",
        choices=["top", "bottom", "left", "right"],
        help="Camera orientation relative to board"
    )
    parser.add_argument(
        "--flip", "-f", action="store_true",
        help="Flip board image horizontally"
    )
    parser.add_argument(
        "--rotate180", action="store_true",
        help="Rotate board 180 degrees (a1 becomes h8)"
    )
    parser.add_argument(
        "--margin", "-m", type=int, default=0,
        help="Grid margin in pixels (positive = inset from edges, negative = extend beyond)"
    )

    args = parser.parse_args()

    gui = FrameDiffGUI(
        source=args.source,
        orientation=args.orientation,
        flip_board=args.flip,
        rotate180=args.rotate180,
        board_margin=args.margin,
    )
    gui.run()


if __name__ == "__main__":
    main()
