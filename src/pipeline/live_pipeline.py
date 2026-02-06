"""Live video streaming pipeline for real-time chess transcription."""

import chess
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Thread, Event
from typing import Callable, Optional

import cv2
import numpy as np
from rich.console import Console
from rich.live import Live
from rich.table import Table

from src.inference.detector import ChessPieceDetector
from src.inference.board_detector import is_starting_position
from src.chess_logic.move_detector import DetectedMove
from src.chess_logic.move_validator import MoveValidator
from src.chess_logic.pgn_generator import LivePGNWriter
from .frame_processor import FrameProcessor, FrameResult

console = Console()


@dataclass
class LiveConfig:
    """Configuration for live processing."""

    # Camera
    camera_index: int = 0
    camera_width: int = 1280
    camera_height: int = 720

    # Processing
    target_fps: int = 30
    stability_threshold: int = 5

    # Output
    output_pgn_path: Optional[str] = None
    show_visualization: bool = True
    window_name: str = "Chess Vision"

    # Game
    white_player: str = "White"
    black_player: str = "Black"
    event_name: str = "Live Game"


class LivePipeline:
    """Real-time chess game transcription from camera feed."""

    def __init__(
        self,
        detector: ChessPieceDetector,
        config: Optional[LiveConfig] = None,
        on_move: Optional[Callable[[chess.Move, str], None]] = None,
    ):
        """Initialize live pipeline.

        Args:
            detector: Trained chess piece detector
            config: Live processing configuration
            on_move: Callback for each validated move (move, san_notation)
        """
        self.detector = detector
        self.config = config or LiveConfig()
        self.on_move = on_move

        # Initialize components
        self.frame_processor = FrameProcessor(
            detector=detector,
            stability_threshold=self.config.stability_threshold,
        )
        self.validator = MoveValidator()

        # PGN writer
        self.pgn_writer = None
        if self.config.output_pgn_path:
            self.pgn_writer = LivePGNWriter(
                output_path=self.config.output_pgn_path,
                event=self.config.event_name,
                white_player=self.config.white_player,
                black_player=self.config.black_player,
            )

        # State
        self.running = False
        self.game_started = False
        self.move_count = 0
        self.last_move_san = ""
        self.fps = 0.0

        # Threading
        self._stop_event = Event()

    def start(self):
        """Start live processing."""
        # Open camera
        cap = cv2.VideoCapture(self.config.camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {self.config.camera_index}")

        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_height)

        console.print(f"[green]Camera opened: {self.config.camera_width}x{self.config.camera_height}")
        console.print("[yellow]Looking for ArUco markers to calibrate...")
        console.print("[yellow]Press 'q' to quit, 'r' to reset, 's' to save PGN")

        self.running = True
        self._stop_event.clear()

        frame_times = []
        last_frame_time = time.perf_counter()

        try:
            while self.running and not self._stop_event.is_set():
                # Capture frame
                ret, frame = cap.read()
                if not ret:
                    console.print("[red]Failed to capture frame")
                    break

                # Calculate FPS
                current_time = time.perf_counter()
                frame_times.append(current_time - last_frame_time)
                last_frame_time = current_time
                if len(frame_times) > 30:
                    frame_times.pop(0)
                self.fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0

                # Process frame
                timestamp = time.time()
                result = self.frame_processor.process(frame, timestamp)

                # Handle game state
                self._process_result(result, timestamp)

                # Show visualization
                if self.config.show_visualization:
                    viz = self._create_visualization(result)
                    cv2.imshow(self.config.window_name, viz)

                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    elif key == ord("r"):
                        self._reset()
                        console.print("[yellow]Game reset")
                    elif key == ord("s"):
                        self._save_pgn()

                # Rate limiting
                target_frame_time = 1.0 / self.config.target_fps
                elapsed = time.perf_counter() - current_time
                if elapsed < target_frame_time:
                    time.sleep(target_frame_time - elapsed)

        finally:
            cap.release()
            if self.config.show_visualization:
                cv2.destroyAllWindows()

            # Final save
            self._save_pgn()
            self.running = False

    def stop(self):
        """Stop live processing."""
        self._stop_event.set()
        self.running = False

    def _process_result(self, result: FrameResult, timestamp: float):
        """Process a frame result."""
        if not self.frame_processor.calibrated:
            return

        # Wait for starting position
        if not self.game_started and result.board_state:
            if is_starting_position(result.board_state):
                self.game_started = True
                console.print("[green]Game started! Starting position detected.")
            return

        # Handle detected move
        if result.detected_move:
            self._handle_move(result.detected_move, timestamp)

    def _handle_move(self, detected_move: DetectedMove, timestamp: float):
        """Handle a detected move."""
        # Validate move
        validation = self.validator.validate(detected_move)

        if validation.is_valid and validation.validated_move:
            move = validation.validated_move
            san = self.validator.board.san(move)

            # Apply move
            self.validator.board.push(move)
            self.move_count += 1
            self.last_move_san = san

            # Save to PGN
            if self.pgn_writer:
                self.pgn_writer.add_move(move, timestamp=timestamp)

            # Callback
            if self.on_move:
                self.on_move(move, san)

            console.print(f"[green]Move {self.move_count}: {san}")

            # Check for game end
            if self.validator.is_game_over():
                result = self.validator.get_game_result()
                console.print(f"[bold green]Game Over: {result}")
                if self.pgn_writer:
                    self.pgn_writer.finalize(result)

    def _create_visualization(self, result: FrameResult) -> np.ndarray:
        """Create visualization frame."""
        if result.warped_frame is None:
            # Not calibrated - show original with message
            viz = result.original_frame.copy()
            self._add_overlay(viz, calibrated=False)
            if result.corners:
                from src.inference.aruco import ArUcoDetector
                temp_detector = ArUcoDetector()
                viz = temp_detector.draw_markers(viz, result.corners)
            return viz

        # Show warped board with detections
        viz = self.frame_processor.visualize(result)
        self._add_overlay(viz, calibrated=True)

        return viz

    def _add_overlay(self, frame: np.ndarray, calibrated: bool):
        """Add status overlay to frame."""
        h, w = frame.shape[:2]

        # Status bar background
        cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)

        # Status text
        if not calibrated:
            status = "Looking for ArUco markers..."
            color = (0, 0, 255)
        elif not self.game_started:
            status = "Waiting for starting position..."
            color = (0, 255, 255)
        else:
            status = f"Move {self.move_count}: {self.last_move_san}" if self.last_move_san else "Game in progress"
            color = (0, 255, 0)

        cv2.putText(frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # FPS and controls
        info = f"FPS: {self.fps:.1f} | q=quit r=reset s=save"
        cv2.putText(frame, info, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Turn indicator
        if self.game_started:
            turn = "White" if self.validator.board.turn else "Black"
            cv2.putText(
                frame, f"Turn: {turn}",
                (w - 120, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2
            )

    def _reset(self):
        """Reset game state."""
        self.frame_processor.reset()
        self.validator = MoveValidator()
        self.game_started = False
        self.move_count = 0
        self.last_move_san = ""

        if self.config.output_pgn_path:
            self.pgn_writer = LivePGNWriter(
                output_path=self.config.output_pgn_path,
                event=self.config.event_name,
                white_player=self.config.white_player,
                black_player=self.config.black_player,
            )

    def _save_pgn(self):
        """Save current PGN."""
        if self.pgn_writer:
            self.pgn_writer.save()
            console.print(f"[green]PGN saved to: {self.config.output_pgn_path}")

    def get_current_pgn(self) -> str:
        """Get current PGN string."""
        if self.pgn_writer:
            return self.pgn_writer.generator.to_pgn()
        return ""

    def get_current_fen(self) -> str:
        """Get current board FEN."""
        return self.validator.board.fen()
