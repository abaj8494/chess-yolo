#!/usr/bin/env python3
"""
Live Chess Detection GUI

Streams from Android camera (via IP Webcam app) or webcam,
runs inference, and displays detections + moves side-by-side.

Setup for Android:
1. Install "IP Webcam" app from Play Store
2. Open app, scroll down, tap "Start server"
3. Note the IP address shown (e.g., http://192.168.1.100:8080)
4. Run: python scripts/live_gui.py --source http://192.168.1.100:8080/video

For webcam:
    python scripts/live_gui.py --source 0

For video file:
    python scripts/live_gui.py --source video/game.mp4
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


@dataclass
class GUIState:
    """Track GUI state."""
    moves: list = field(default_factory=list)
    current_fen: str = "startpos"
    fps: float = 0.0
    frame_count: int = 0
    calibrated: bool = False
    pieces_detected: int = 0
    last_board_state: Optional[dict] = None
    status: str = "Initializing..."


class ChessGUI:
    """Live chess detection GUI."""

    def __init__(
        self,
        model_path: str,
        source: str = "0",
        orientation: str = "top",
        confidence: float = 0.5,
        flip_board: bool = False,
        window_width: int = 1400,
        window_height: int = 800,
    ):
        self.source = source
        self.orientation = orientation
        self.confidence = confidence
        self.flip_board = flip_board
        self.window_width = window_width
        self.window_height = window_height

        # Video dimensions
        self.video_width = int(window_width * 0.65)
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

        # Load components
        print("Loading model...")
        from src.inference.detector import ChessPieceDetector
        from src.inference.aruco import ArUcoDetector
        from src.inference.perspective import PerspectiveCorrector
        from src.inference.board_detector import SquareMapper
        from src.inference.tracker import StateTracker
        from src.chess_logic.move_detector import MoveDetector
        from src.chess_logic.move_validator import MoveValidator

        self.detector = ChessPieceDetector(
            model_path=model_path,
            confidence_threshold=confidence,
        )
        self.aruco = ArUcoDetector()
        self.perspective = PerspectiveCorrector((640, 640))
        self.square_mapper = SquareMapper(640)
        self.state_tracker = StateTracker(stability_threshold=3)
        self.move_detector = MoveDetector()
        self.validator = MoveValidator()

        print("Ready!")

    def _rotate_frame(self, frame: np.ndarray) -> np.ndarray:
        """Rotate frame based on orientation."""
        rotation = self.rotation_map.get(self.orientation)
        if rotation is not None:
            return cv2.rotate(frame, rotation)
        return frame

    def _draw_panel(self, height: int) -> np.ndarray:
        """Draw the info panel with moves and status."""
        panel = np.zeros((height, self.panel_width, 3), dtype=np.uint8)
        panel[:] = (30, 30, 30)  # Dark gray background

        y = 30
        line_height = 25

        # Title
        cv2.putText(panel, "Chess Vision", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        y += line_height + 10

        # Status section
        cv2.line(panel, (10, y), (self.panel_width - 10, y), (60, 60, 60), 1)
        y += 20

        # FPS
        fps_color = (0, 255, 0) if self.state.fps > 15 else (0, 255, 255) if self.state.fps > 5 else (0, 0, 255)
        cv2.putText(panel, f"FPS: {self.state.fps:.1f}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, fps_color, 1)
        y += line_height

        # Calibration status
        cal_color = (0, 255, 0) if self.state.calibrated else (0, 0, 255)
        cal_text = "Calibrated" if self.state.calibrated else "Looking for markers..."
        cv2.putText(panel, cal_text, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, cal_color, 1)
        y += line_height

        # Pieces detected
        cv2.putText(panel, f"Pieces: {self.state.pieces_detected}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
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
                if y > height - 100:
                    cv2.putText(panel, "...", (25, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
                    break
        else:
            cv2.putText(panel, "(no moves yet)", (25, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

        # Controls at bottom
        y = height - 60
        cv2.line(panel, (10, y), (self.panel_width - 10, y), (60, 60, 60), 1)
        y += 25
        cv2.putText(panel, "Controls:", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        y += line_height
        cv2.putText(panel, "Q: Quit  R: Reset  S: Screenshot", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

        return panel

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process frame and return visualization."""
        # Rotate based on camera orientation
        frame = self._rotate_frame(frame)

        # Detect ArUco markers
        corners = self.aruco.detect(frame)

        if corners and corners.all_found:
            self.perspective.calibrate(corners)
            self.state.calibrated = True

        if self.state.calibrated:
            # Warp to bird's eye view
            warped = self.perspective.warp_frame(frame)

            # Flip horizontally if needed (corrects mirrored board)
            if self.flip_board:
                warped = cv2.flip(warped, 1)

            # Detect pieces
            detections = self.detector.detect(warped)
            self.state.pieces_detected = len(detections)

            # Debug: show raw detection count
            if self.state.frame_count % 60 == 0:
                print(f"[Debug] Raw detections: {len(detections)}")
                if detections:
                    for d in detections[:5]:
                        print(f"  - {d.class_name}: {d.confidence:.2f} at ({d.center[0]:.0f}, {d.center[1]:.0f})")

            # Update state tracker
            current_time = time.time()
            board_state, state_changed = self.state_tracker.update(detections, current_time)

            # Debug: show what we're detecting
            if self.state.frame_count % 30 == 0:  # Every ~1 second
                if board_state and board_state.squares:
                    pieces_str = ", ".join([f"{sq}:{p[:2]}" for sq, p in list(board_state.squares.items())[:8]])
                    print(f"[Debug] Pieces: {len(board_state.squares)} - {pieces_str}...")

            # Detect moves when state changes
            if state_changed and board_state:
                print(f"[Debug] State changed! Pieces on board: {len(board_state.squares) if board_state.squares else 0}")

                # Show previous vs current state for debugging
                if self.move_detector.previous_state:
                    prev_sq = set(self.move_detector.previous_state.squares.keys())
                    curr_sq = set(board_state.squares.keys())
                    disappeared = prev_sq - curr_sq
                    appeared = curr_sq - prev_sq
                    if disappeared or appeared:
                        print(f"[Debug] Disappeared: {disappeared}, Appeared: {appeared}")

                detected_move = self.move_detector.update(board_state)
                if detected_move:
                    print(f"[Debug] Detected move: {detected_move.from_square}->{detected_move.to_square} ({detected_move.piece})")
                    # Validate move
                    result = self.validator.validate(detected_move)
                    print(f"[Debug] Validation result: valid={result.is_valid}, move={result.validated_move}")
                    if result.is_valid and result.validated_move:
                        move_san = self.validator.board.san(result.validated_move)
                        self.validator.board.push(result.validated_move)
                        self.state.moves.append(move_san)
                        print(f"Move: {move_san}")
                    else:
                        print(f"[Debug] Move rejected: {result.error}")
                else:
                    print(f"[Debug] No move detected from state change")

            # Draw detections on warped frame
            viz = self.detector.draw_detections(warped, detections)

            # Draw grid
            viz = self.perspective.draw_grid(viz, warped=True)

            return viz
        else:
            # Not calibrated - show original with marker overlay
            if corners:
                frame = self.aruco.draw_markers(frame, corners)

            # Resize to fit
            h, w = frame.shape[:2]
            scale = min(self.video_width / w, self.video_height / h)
            new_w, new_h = int(w * scale), int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h))

            # Add "looking for markers" text
            cv2.putText(frame, "Looking for ArUco markers...", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            return frame

    def _is_ip_webcam_url(self, source: str) -> bool:
        """Check if source is an IP Webcam URL."""
        return source.startswith("http://") and ":8080" in source

    def _fetch_ip_webcam_frame(self, base_url: str) -> Optional[np.ndarray]:
        """Fetch a single frame from IP Webcam using shot.jpg endpoint."""
        import subprocess
        try:
            # Use shot.jpg endpoint which is more reliable
            shot_url = base_url.rstrip('/').replace('/video', '').replace('/videofeed', '')
            if not shot_url.endswith('/shot.jpg'):
                shot_url = shot_url.rstrip('/') + '/shot.jpg'

            # Use curl as workaround for Python networking issues
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
        except Exception as e:
            return None

    def run(self):
        """Main GUI loop."""
        use_ip_webcam = self._is_ip_webcam_url(self.source)
        cap = None

        if use_ip_webcam:
            # Test IP Webcam connection
            print(f"Connecting to IP Webcam: {self.source}")
            frame = self._fetch_ip_webcam_frame(self.source)
            if frame is None:
                print(f"Error: Could not connect to IP Webcam at {self.source}")
                print("Make sure the app is running and server is started.")
                return
            print("IP Webcam connected!")
        else:
            # Open video source
            if self.source.isdigit():
                cap = cv2.VideoCapture(int(self.source))
            else:
                cap = cv2.VideoCapture(self.source)

            if not cap.isOpened():
                print(f"Error: Could not open video source: {self.source}")
                return

        print(f"Connected to: {self.source}")
        print("Press 'q' to quit, 'r' to reset, 's' for screenshot")

        cv2.namedWindow("Chess Vision", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Chess Vision", self.window_width, self.window_height)

        while True:
            # Get frame from appropriate source
            if use_ip_webcam:
                frame = self._fetch_ip_webcam_frame(self.source)
                if frame is None:
                    print("Failed to fetch frame, retrying...")
                    time.sleep(0.1)
                    continue
            else:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame, retrying...")
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

            # Resize viz to fit video area
            h, w = viz.shape[:2]
            scale = min(self.video_width / w, self.video_height / h)
            new_w, new_h = int(w * scale), int(h * scale)
            viz = cv2.resize(viz, (new_w, new_h))

            # Create canvas
            canvas = np.zeros((self.window_height, self.window_width, 3), dtype=np.uint8)

            # Place video (centered in video area)
            x_offset = (self.video_width - new_w) // 2
            y_offset = (self.window_height - new_h) // 2
            canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = viz

            # Draw panel
            panel = self._draw_panel(self.window_height)
            canvas[:, self.video_width:] = panel

            # Vertical separator
            cv2.line(canvas, (self.video_width, 0), (self.video_width, self.window_height),
                     (60, 60, 60), 2)

            # Show
            cv2.imshow("Chess Vision", canvas)

            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                from src.chess_logic.move_validator import MoveValidator
                self.state = GUIState()
                self.state.status = "Reset"
                self.state_tracker.reset()
                self.move_detector.reset()
                self.validator = MoveValidator()  # Reset chess board
                print("Reset!")
            elif key == ord('s'):
                filename = f"screenshot_{int(time.time())}.png"
                cv2.imwrite(filename, canvas)
                print(f"Saved: {filename}")

        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Live Chess Detection GUI")
    parser.add_argument(
        "--source", "-s", default="0",
        help="Video source: camera index (0), IP Webcam URL, or video file"
    )
    parser.add_argument(
        "--model", "-m", type=str, default="models/best.pt",
        help="Path to trained model"
    )
    parser.add_argument(
        "--orientation", "-o", default="top",
        choices=["top", "bottom", "left", "right"],
        help="Camera orientation relative to board"
    )
    parser.add_argument(
        "--confidence", "-c", type=float, default=0.5,
        help="Detection confidence threshold"
    )
    parser.add_argument(
        "--flip", "-f", action="store_true",
        help="Flip board horizontally (if squares appear mirrored)"
    )

    args = parser.parse_args()

    gui = ChessGUI(
        model_path=args.model,
        source=args.source,
        orientation=args.orientation,
        confidence=args.confidence,
        flip_board=args.flip,
    )
    gui.run()


if __name__ == "__main__":
    main()
