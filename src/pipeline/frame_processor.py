"""Single frame processing for chess detection."""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from src.inference.aruco import ArUcoDetector, BoardCorners
from src.inference.perspective import PerspectiveCorrector
from src.inference.detector import ChessPieceDetector, Detection
from src.inference.board_detector import BoardState, SquareMapper
from src.inference.tracker import StateTracker
from src.inference.occlusion import OcclusionDetector
from src.chess_logic.move_detector import DetectedMove, MoveDetector


@dataclass
class FrameResult:
    """Result of processing a single frame."""

    original_frame: np.ndarray
    warped_frame: Optional[np.ndarray]
    detections: list[Detection]
    board_state: Optional[BoardState]
    corners: Optional[BoardCorners]
    detected_move: Optional[DetectedMove]
    state_changed: bool = False
    has_occlusion: bool = False
    processing_time_ms: float = 0.0


class FrameProcessor:
    """Process individual frames for chess detection."""

    def __init__(
        self,
        detector: ChessPieceDetector,
        output_size: tuple[int, int] = (640, 640),
        enable_occlusion_detection: bool = True,
        stability_threshold: int = 5,
    ):
        """Initialize frame processor.

        Args:
            detector: Trained chess piece detector
            output_size: Size for warped board image
            enable_occlusion_detection: Whether to detect hands/occlusions
            stability_threshold: Frames required for stable state change
        """
        self.detector = detector
        self.output_size = output_size

        # Initialize components
        self.aruco_detector = ArUcoDetector()
        self.perspective = PerspectiveCorrector(output_size)
        self.square_mapper = SquareMapper(output_size[0])
        self.state_tracker = StateTracker(stability_threshold=stability_threshold)
        self.move_detector = MoveDetector()

        if enable_occlusion_detection:
            self.occlusion_detector = OcclusionDetector()
        else:
            self.occlusion_detector = None

        # State
        self.calibrated = False
        self.frame_count = 0

    def calibrate(self, frame: np.ndarray) -> bool:
        """Calibrate perspective from frame.

        Args:
            frame: Frame with visible ArUco markers

        Returns:
            True if calibration successful
        """
        corners = self.aruco_detector.detect(frame)

        if corners is not None and corners.all_found:
            self.perspective.calibrate(corners)
            self.calibrated = True
            return True

        return False

    def process(
        self,
        frame: np.ndarray,
        timestamp: float = 0.0,
    ) -> FrameResult:
        """Process a single frame.

        Args:
            frame: BGR input frame
            timestamp: Frame timestamp

        Returns:
            FrameResult with detection information
        """
        import time
        start_time = time.perf_counter()

        self.frame_count += 1

        # Try to detect corners
        corners = self.aruco_detector.detect(frame)

        # Update calibration if we got good corners
        if corners is not None and corners.all_found:
            self.perspective.calibrate(corners)
            self.calibrated = True

        # If not calibrated, return empty result
        if not self.calibrated:
            return FrameResult(
                original_frame=frame,
                warped_frame=None,
                detections=[],
                board_state=None,
                corners=corners,
                detected_move=None,
                processing_time_ms=(time.perf_counter() - start_time) * 1000,
            )

        # Warp frame
        warped = self.perspective.warp_frame(frame)

        # Detect occlusions
        has_occlusion = False
        if self.occlusion_detector:
            occlusions = self.occlusion_detector.detect_hands(warped)
            has_occlusion = len(occlusions) > 0

        # Detect pieces
        detections = self.detector.detect(warped)

        # Filter occluded detections
        if self.occlusion_detector and has_occlusion:
            detections, _ = self.occlusion_detector.filter_occluded_detections(
                detections, occlusions
            )

        # Update tracker
        board_state, state_changed = self.state_tracker.update(
            detections, timestamp
        )

        # Detect move if state changed
        detected_move = None
        if state_changed:
            detected_move = self.move_detector.update(board_state)

        processing_time = (time.perf_counter() - start_time) * 1000

        return FrameResult(
            original_frame=frame,
            warped_frame=warped,
            detections=detections,
            board_state=board_state,
            corners=corners,
            detected_move=detected_move,
            state_changed=state_changed,
            has_occlusion=has_occlusion,
            processing_time_ms=processing_time,
        )

    def visualize(
        self,
        result: FrameResult,
        show_grid: bool = True,
        show_detections: bool = True,
        show_squares: bool = True,
    ) -> np.ndarray:
        """Create visualization of frame result.

        Args:
            result: Frame processing result
            show_grid: Draw 8x8 grid
            show_detections: Draw detection bounding boxes
            show_squares: Label squares

        Returns:
            Visualization image
        """
        if result.warped_frame is None:
            # Not calibrated - show original with message
            viz = result.original_frame.copy()
            cv2.putText(
                viz,
                "Looking for ArUco markers...",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            if result.corners:
                viz = self.aruco_detector.draw_markers(viz, result.corners)
            return viz

        viz = result.warped_frame.copy()

        # Draw grid
        if show_grid:
            viz = self.perspective.draw_grid(viz, warped=True)

        # Draw detections
        if show_detections:
            viz = self.detector.draw_detections(viz, result.detections)

        # Draw square labels
        if show_squares:
            viz = self.perspective.add_square_labels(viz)

        # Add status info
        status = f"Frame: {self.frame_count} | Pieces: {len(result.detections)}"
        if result.has_occlusion:
            status += " | OCCLUSION"
        if result.detected_move:
            status += f" | Move: {result.detected_move.to_uci()}"

        cv2.putText(
            viz,
            status,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            viz,
            f"{result.processing_time_ms:.1f}ms",
            (10, viz.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        return viz

    def reset(self):
        """Reset processor state."""
        self.state_tracker.reset()
        self.move_detector.reset()
        self.frame_count = 0
        # Keep calibration
