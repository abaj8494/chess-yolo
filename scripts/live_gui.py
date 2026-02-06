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


from enum import Enum


class CalibrationState(Enum):
    WAITING = "waiting"      # Waiting for stable detection
    SAMPLING = "sampling"    # Sampling colors
    COMPLETE = "complete"    # Calibration done


@dataclass
class GUIState:
    """Track GUI state."""
    moves: list = field(default_factory=list)
    current_fen: str = "startpos"
    fps: float = 0.0
    frame_count: int = 0
    aruco_calibrated: bool = False  # ArUco markers found
    color_calibrated: bool = False  # Piece colors calibrated
    pieces_detected: int = 0
    white_pieces: int = 0
    black_pieces: int = 0
    last_board_state: Optional[dict] = None
    status: str = "Initializing..."
    calibration_state: str = "waiting"


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
        from src.inference.color_calibration import ColorCalibrator
        from src.inference.detection_smoother import DetectionSmoother
        from src.chess_logic.occupancy_move_detector import OccupancyMoveDetector

        self.detector = ChessPieceDetector(
            model_path=model_path,
            confidence_threshold=confidence,
        )

        # Detection smoother for temporal filtering
        # Use aggressive filtering due to noisy detection
        self.detection_smoother = DetectionSmoother(
            window_size=10,   # Look at last 10 frames
            min_detections=7, # Need 7/10 detections to add square
            min_empty=7,      # Need 7 empty frames to remove square
        )
        self.aruco = ArUcoDetector()
        self.perspective = PerspectiveCorrector((640, 640))
        self.square_mapper = SquareMapper(640)
        self.state_tracker = StateTracker(stability_threshold=3)

        # Color calibrator for white/black classification
        self.color_calibrator = ColorCalibrator(board_size=640)
        self.calibration_frames = 0
        self.calibration_threshold = 10  # Frames needed for calibration

        # Use occupancy-based detection (ignores piece classification)
        self.occupancy_detector = OccupancyMoveDetector()

        # Store warped frame for color sampling
        self.last_warped_frame = None

        print("Ready!")

    def _rotate_frame(self, frame: np.ndarray) -> np.ndarray:
        """Rotate frame based on orientation."""
        rotation = self.rotation_map.get(self.orientation)
        if rotation is not None:
            return cv2.rotate(frame, rotation)
        return frame

    def _draw_detections_with_color(
        self,
        frame: np.ndarray,
        detections,
        board_state
    ) -> np.ndarray:
        """Draw detections with color-coded bounding boxes."""
        viz = frame.copy()

        if not detections:
            return viz

        detected_squares = set(board_state.squares.keys()) if board_state else set()

        for det in detections:
            x1, y1, x2, y2 = map(int, det.bbox)
            cx, cy = det.center

            # Get square for this detection
            square = self.square_mapper.get_square(cx, cy)

            # Determine color based on calibration
            if self.state.color_calibrated and square in detected_squares:
                piece_color = self.color_calibrator.classify_piece(
                    self.last_warped_frame, square, flip_board=self.flip_board
                )
                if piece_color.value == "white":
                    box_color = (255, 255, 255)  # White box for white pieces
                    text_color = (0, 0, 0)
                elif piece_color.value == "black":
                    box_color = (50, 50, 50)  # Dark box for black pieces
                    text_color = (255, 255, 255)
                else:
                    box_color = (0, 255, 255)  # Yellow for unknown
                    text_color = (0, 0, 0)
            else:
                # Not calibrated yet - use detection's class
                if "white" in det.class_name:
                    box_color = (200, 200, 200)
                elif "black" in det.class_name:
                    box_color = (80, 80, 80)
                else:
                    box_color = (0, 255, 0)
                text_color = (0, 255, 0)

            # Draw bounding box
            cv2.rectangle(viz, (x1, y1), (x2, y2), box_color, 2)

            # Draw square label
            label = f"{square}"
            cv2.putText(viz, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)

        return viz

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

        # ArUco calibration status
        aruco_color = (0, 255, 0) if self.state.aruco_calibrated else (0, 0, 255)
        aruco_text = "Board: OK" if self.state.aruco_calibrated else "Board: Looking..."
        cv2.putText(panel, aruco_text, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, aruco_color, 1)
        y += line_height

        # Color calibration status
        color_color = (0, 255, 0) if self.state.color_calibrated else (0, 255, 255)
        color_text = "Colors: OK" if self.state.color_calibrated else f"Colors: {self.state.calibration_state}"
        cv2.putText(panel, color_text, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_color, 1)
        y += line_height

        # Pieces detected with color breakdown
        cv2.putText(panel, f"Pieces: {self.state.pieces_detected}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y += line_height

        if self.state.color_calibrated:
            cv2.putText(panel, f"  White: {self.state.white_pieces}  Black: {self.state.black_pieces}", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            y += line_height

        y += 5

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
        cv2.putText(panel, "Q:Quit R:Reset L:Lock S:Save", (20, y),
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
            self.state.aruco_calibrated = True

        if self.state.aruco_calibrated:
            # Warp to bird's eye view
            warped = self.perspective.warp_frame(frame)

            # Flip horizontally if needed (corrects mirrored board)
            if self.flip_board:
                warped = cv2.flip(warped, 1)

            # Store for color sampling
            self.last_warped_frame = warped.copy()

            # Detect pieces
            detections = self.detector.detect(warped)
            self.state.pieces_detected = len(detections)

            # Update state tracker
            current_time = time.time()
            board_state, state_changed = self.state_tracker.update(detections, current_time)

            # Get raw detected squares
            raw_squares = set(board_state.squares.keys()) if board_state else set()

            # Apply temporal smoothing to filter noise
            smoothed = self.detection_smoother.update(raw_squares)
            detected_squares = smoothed.occupied

            # Debug: show raw vs smoothed every 60 frames
            if self.state.frame_count % 60 == 0:
                stats = self.detection_smoother.get_stats()
                print(f"[Smoother] Raw: {stats['raw_detected']}, Smoothed: {stats['smoothed_detected']}", flush=True)

            # Color calibration phase
            if not self.state.color_calibrated:
                self.state.status = "Calibrating colors..."
                self.state.calibration_state = "sampling"

                # Try to calibrate if we have enough pieces
                if len(detected_squares) >= 20:  # Need most pieces visible
                    self.calibration_frames += 1

                    if self.calibration_frames >= self.calibration_threshold:
                        cal = self.color_calibrator.calibrate(
                            warped,
                            detected_squares,
                            flip_board=self.flip_board
                        )
                        if cal:
                            self.state.color_calibrated = True
                            self.state.calibration_state = "complete"
                            self.state.status = "Ready - make your move"
                            print(f"[GUI] Color calibration complete!")
                            # Let occupancy detector initialize from actual detection
                            # (more reliable than assuming 32-square starting position
                            # when detection only sees 20-25 squares)
                else:
                    self.calibration_frames = 0  # Reset if not enough pieces

                # Show calibration progress
                if self.state.frame_count % 30 == 0:
                    print(f"[Calibration] Pieces: {len(detected_squares)}, frames: {self.calibration_frames}/{self.calibration_threshold}")

            # If calibrated, classify pieces by color and detect moves
            if self.state.color_calibrated and board_state:
                # Classify all pieces using calibrated colors
                color_map = self.color_calibrator.classify_all_pieces(
                    warped, detected_squares, flip_board=self.flip_board
                )

                # Count white/black pieces
                self.state.white_pieces = sum(1 for c in color_map.values() if c.value == "white")
                self.state.black_pieces = sum(1 for c in color_map.values() if c.value == "black")

                # Debug: show classification
                if self.state.frame_count % 60 == 0:
                    print(f"[Color] White: {self.state.white_pieces}, Black: {self.state.black_pieces}")

                # Create smoothed board state for occupancy detection
                # Only include squares that passed temporal smoothing
                from src.inference.board_detector import BoardState as BS
                smoothed_board_state = BS(
                    squares={sq: board_state.squares.get(sq, "piece")
                             for sq in detected_squares if sq in board_state.squares},
                    timestamp=board_state.timestamp,
                    frame_number=board_state.frame_number,
                )

                # Detect moves using occupancy-based approach with smoothed state
                detected_move = self.occupancy_detector.update(smoothed_board_state)
                if detected_move:
                    # Move was confirmed (passed stability threshold)
                    move_san = detected_move.piece.split('-')[0][0].upper()
                    # Get proper SAN from the board history
                    if self.occupancy_detector.board.move_stack:
                        last_move = self.occupancy_detector.board.peek()
                        self.occupancy_detector.board.pop()
                        move_san = self.occupancy_detector.board.san(last_move)
                        self.occupancy_detector.board.push(last_move)
                    self.state.moves.append(move_san)
                    print(f"Move confirmed: {move_san}")

            # Draw detections on warped frame (color-coded if calibrated)
            viz = self._draw_detections_with_color(warped, detections, board_state)

            # Draw grid
            viz = self.perspective.draw_grid(viz, warped=True)

            # Draw calibration status
            if not self.state.color_calibrated:
                cv2.putText(viz, f"Calibrating... ({self.calibration_frames}/{self.calibration_threshold})",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            return viz
        else:
            # Not ArUco calibrated - show original with marker overlay
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
                self.state = GUIState()
                self.state.status = "Reset"
                self.state_tracker.reset()
                self.occupancy_detector.reset()
                self.detection_smoother.reset()  # Reset smoother too
                self.color_calibrator = type(self.color_calibrator)(board_size=640)
                self.calibration_frames = 0
                print("Reset!")
            elif key == ord('l'):
                # Lock/unlock baseline
                if self.occupancy_detector.baseline_locked:
                    self.occupancy_detector.unlock_baseline()
                else:
                    self.occupancy_detector.lock_baseline()
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
        "--confidence", "-c", type=float, default=0.25,
        help="Detection confidence threshold (default: 0.25 for better recall)"
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
