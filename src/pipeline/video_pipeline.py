"""Pre-recorded video processing pipeline."""

import chess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from tqdm import tqdm

from src.inference.detector import ChessPieceDetector
from src.inference.board_detector import is_starting_position
from src.chess_logic.move_detector import DetectedMove
from src.chess_logic.move_validator import MoveValidator
from src.chess_logic.pgn_generator import PGNGenerator
from src.chess_logic.board_state import GameState
from .frame_processor import FrameProcessor, FrameResult

console = Console()


@dataclass
class VideoResult:
    """Result of processing a video."""

    video_path: str
    total_frames: int
    processed_frames: int
    detected_moves: list[DetectedMove]
    validated_moves: list[chess.Move]
    invalid_moves: list[DetectedMove]
    pgn: str
    game_state: GameState
    output_path: Optional[str] = None
    processing_time_seconds: float = 0.0


@dataclass
class ProcessingConfig:
    """Configuration for video processing."""

    # Frame processing
    skip_frames: int = 1  # Process every Nth frame
    start_frame: int = 0
    end_frame: Optional[int] = None

    # Detection
    stability_threshold: int = 5
    require_starting_position: bool = True
    max_calibration_frames: int = 100

    # Output
    save_visualization: bool = False
    visualization_path: Optional[str] = None
    visualization_fps: int = 30


class VideoPipeline:
    """Process pre-recorded chess game videos."""

    def __init__(
        self,
        detector: ChessPieceDetector,
        config: Optional[ProcessingConfig] = None,
    ):
        """Initialize video pipeline.

        Args:
            detector: Trained chess piece detector
            config: Processing configuration
        """
        self.detector = detector
        self.config = config or ProcessingConfig()

        # Initialize components
        self.frame_processor = FrameProcessor(
            detector=detector,
            stability_threshold=self.config.stability_threshold,
        )
        self.validator = MoveValidator()
        self.pgn_generator = PGNGenerator()
        self.game_state = GameState()

        # Tracking
        self.detected_moves: list[DetectedMove] = []
        self.validated_moves: list[chess.Move] = []
        self.invalid_moves: list[DetectedMove] = []

    def process(
        self,
        video_path: str | Path,
        output_pgn: Optional[str | Path] = None,
    ) -> VideoResult:
        """Process a video file.

        Args:
            video_path: Path to input video
            output_pgn: Optional path for PGN output

        Returns:
            VideoResult with processing results
        """
        import time
        start_time = time.perf_counter()

        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        console.print(f"[blue]Processing: {video_path.name}")
        console.print(f"[blue]Frames: {total_frames}, FPS: {fps:.2f}")

        # Reset state
        self._reset()

        # Video writer for visualization
        viz_writer = None
        if self.config.save_visualization and self.config.visualization_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            viz_writer = cv2.VideoWriter(
                self.config.visualization_path,
                fourcc,
                self.config.visualization_fps,
                (640, 640),
            )

        # Process frames
        processed = 0
        calibration_frames = 0
        game_started = False

        end_frame = self.config.end_frame or total_frames

        with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[green]Processing...", total=end_frame)

            frame_idx = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.config.start_frame)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                if frame_idx >= end_frame:
                    break

                progress.update(task, completed=frame_idx)

                # Skip frames if configured
                if frame_idx % self.config.skip_frames != 0:
                    continue

                timestamp = frame_idx / fps if fps > 0 else 0.0

                # Process frame
                result = self.frame_processor.process(frame, timestamp)
                processed += 1

                # Handle calibration phase
                if not self.frame_processor.calibrated:
                    calibration_frames += 1
                    if calibration_frames >= self.config.max_calibration_frames:
                        console.print("[red]Could not calibrate - no ArUco markers found")
                        break
                    continue

                # Wait for starting position
                if (
                    not game_started
                    and self.config.require_starting_position
                    and result.board_state
                ):
                    if is_starting_position(result.board_state):
                        game_started = True
                        console.print(f"[green]Starting position detected at frame {frame_idx}")
                    continue

                # Process detected move
                if result.detected_move:
                    self._handle_move(result.detected_move, timestamp)

                # Write visualization
                if viz_writer:
                    viz = self.frame_processor.visualize(result)
                    viz_writer.write(viz)

        cap.release()
        if viz_writer:
            viz_writer.release()

        # Generate PGN
        pgn = self._generate_pgn()

        # Save PGN if path provided
        if output_pgn:
            self.pgn_generator.save(output_pgn)
            console.print(f"[green]PGN saved to: {output_pgn}")

        processing_time = time.perf_counter() - start_time

        console.print(f"\n[green]Processing complete!")
        console.print(f"  Processed: {processed} frames")
        console.print(f"  Moves detected: {len(self.detected_moves)}")
        console.print(f"  Valid moves: {len(self.validated_moves)}")
        console.print(f"  Invalid moves: {len(self.invalid_moves)}")
        console.print(f"  Time: {processing_time:.2f}s")

        return VideoResult(
            video_path=str(video_path),
            total_frames=total_frames,
            processed_frames=processed,
            detected_moves=self.detected_moves.copy(),
            validated_moves=self.validated_moves.copy(),
            invalid_moves=self.invalid_moves.copy(),
            pgn=pgn,
            game_state=self.game_state,
            output_path=str(output_pgn) if output_pgn else None,
            processing_time_seconds=processing_time,
        )

    def _handle_move(self, detected_move: DetectedMove, timestamp: float):
        """Handle a detected move."""
        self.detected_moves.append(detected_move)

        # Validate move
        result = self.validator.validate(detected_move)

        if result.is_valid and result.validated_move:
            self.validated_moves.append(result.validated_move)
            self.validator.board.push(result.validated_move)
            self.game_state.add_move(result.validated_move, timestamp=timestamp)
            self.pgn_generator.add_move(result.validated_move, timestamp=timestamp)

            console.print(
                f"  [green]Move {len(self.validated_moves)}: "
                f"{result.validated_move.uci()}"
            )
        else:
            self.invalid_moves.append(detected_move)
            console.print(
                f"  [yellow]Invalid move: {detected_move.to_uci()} "
                f"({result.error})"
            )

    def _generate_pgn(self) -> str:
        """Generate final PGN."""
        # Set result if game is over
        if self.validator.is_game_over():
            self.pgn_generator.set_result(self.validator.get_game_result())

        return self.pgn_generator.to_pgn()

    def _reset(self):
        """Reset pipeline state."""
        self.frame_processor.reset()
        self.validator = MoveValidator()
        self.pgn_generator = PGNGenerator()
        self.game_state = GameState()
        self.detected_moves.clear()
        self.validated_moves.clear()
        self.invalid_moves.clear()
