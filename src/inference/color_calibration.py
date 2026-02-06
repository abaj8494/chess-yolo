"""Color calibration for distinguishing white vs black pieces.

At game start, we know piece positions definitively:
- Ranks 1-2: WHITE pieces
- Ranks 7-8: BLACK pieces

We sample actual pixel colors during calibration to build a classifier
that works for the specific chess set and lighting conditions.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple
from enum import Enum


class PieceColor(Enum):
    WHITE = "white"
    BLACK = "black"
    UNKNOWN = "unknown"


@dataclass
class ColorCalibration:
    """Calibration data for piece color classification."""
    white_mean_brightness: float
    black_mean_brightness: float
    threshold: float  # Brightness threshold separating white/black
    white_samples: int
    black_samples: int
    calibrated: bool = False

    def classify(self, brightness: float) -> PieceColor:
        """Classify a piece based on its brightness."""
        if not self.calibrated:
            return PieceColor.UNKNOWN
        return PieceColor.WHITE if brightness > self.threshold else PieceColor.BLACK


class ColorCalibrator:
    """Calibrate piece colors from starting position."""

    # Starting position squares
    WHITE_SQUARES = [
        'a1', 'b1', 'c1', 'd1', 'e1', 'f1', 'g1', 'h1',  # Rank 1
        'a2', 'b2', 'c2', 'd2', 'e2', 'f2', 'g2', 'h2',  # Rank 2
    ]
    BLACK_SQUARES = [
        'a8', 'b8', 'c8', 'd8', 'e8', 'f8', 'g8', 'h8',  # Rank 8
        'a7', 'b7', 'c7', 'd7', 'e7', 'f7', 'g7', 'h7',  # Rank 7
    ]

    def __init__(self, board_size: int = 640):
        self.board_size = board_size
        self.cell_size = board_size / 8
        self.calibration: Optional[ColorCalibration] = None

    def _square_to_roi(self, square: str, flip_board: bool = False) -> Tuple[int, int, int, int]:
        """Convert chess square to ROI coordinates (x1, y1, x2, y2).

        Args:
            square: Chess square notation (e.g., 'e4')
            flip_board: If True, board is flipped (black at bottom)
        """
        file_idx = ord(square[0]) - ord('a')  # 0-7 for a-h
        rank_idx = int(square[1]) - 1  # 0-7 for 1-8

        if flip_board:
            file_idx = 7 - file_idx
            rank_idx = 7 - rank_idx

        # In image coordinates, rank 8 is at top (y=0), rank 1 at bottom
        # So we invert the rank for y coordinate
        x1 = int(file_idx * self.cell_size)
        y1 = int((7 - rank_idx) * self.cell_size)
        x2 = int((file_idx + 1) * self.cell_size)
        y2 = int((8 - rank_idx) * self.cell_size)

        return x1, y1, x2, y2

    def _extract_piece_brightness(
        self,
        frame: np.ndarray,
        square: str,
        flip_board: bool = False,
        margin: float = 0.2
    ) -> Optional[float]:
        """Extract average brightness of piece on a square.

        Args:
            frame: Warped board image (bird's eye view)
            square: Chess square notation
            flip_board: If True, board is flipped
            margin: Fraction of cell to exclude from edges (avoid square colors)
        """
        x1, y1, x2, y2 = self._square_to_roi(square, flip_board)

        # Add margin to focus on center where piece is
        margin_px = int(self.cell_size * margin)
        x1 += margin_px
        y1 += margin_px
        x2 -= margin_px
        y2 -= margin_px

        if x2 <= x1 or y2 <= y1:
            return None

        # Extract ROI
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        # Convert to grayscale and get mean brightness
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi

        return float(np.mean(gray))

    def calibrate(
        self,
        frame: np.ndarray,
        detected_squares: set,
        flip_board: bool = False
    ) -> Optional[ColorCalibration]:
        """Calibrate color thresholds from starting position.

        Args:
            frame: Warped board image
            detected_squares: Set of squares where pieces were detected
            flip_board: If True, board orientation is flipped

        Returns:
            ColorCalibration if successful, None otherwise
        """
        white_brightnesses = []
        black_brightnesses = []

        # Sample white pieces (ranks 1-2)
        for square in self.WHITE_SQUARES:
            if square in detected_squares:
                brightness = self._extract_piece_brightness(frame, square, flip_board)
                if brightness is not None:
                    white_brightnesses.append(brightness)

        # Sample black pieces (ranks 7-8)
        for square in self.BLACK_SQUARES:
            if square in detected_squares:
                brightness = self._extract_piece_brightness(frame, square, flip_board)
                if brightness is not None:
                    black_brightnesses.append(brightness)

        # Need sufficient samples
        min_samples = 4
        if len(white_brightnesses) < min_samples or len(black_brightnesses) < min_samples:
            print(f"[ColorCal] Insufficient samples: white={len(white_brightnesses)}, black={len(black_brightnesses)}")
            return None

        white_mean = np.mean(white_brightnesses)
        black_mean = np.mean(black_brightnesses)

        # Sanity check: white should be brighter than black
        if white_mean <= black_mean:
            print(f"[ColorCal] Warning: white ({white_mean:.1f}) not brighter than black ({black_mean:.1f})")
            # Swap if inverted (could be lighting issue)
            white_mean, black_mean = black_mean, white_mean

        # Threshold is midpoint between means
        threshold = (white_mean + black_mean) / 2

        self.calibration = ColorCalibration(
            white_mean_brightness=white_mean,
            black_mean_brightness=black_mean,
            threshold=threshold,
            white_samples=len(white_brightnesses),
            black_samples=len(black_brightnesses),
            calibrated=True,
        )

        print(f"[ColorCal] Calibrated: white={white_mean:.1f}, black={black_mean:.1f}, threshold={threshold:.1f}")
        print(f"[ColorCal] Samples: white={len(white_brightnesses)}, black={len(black_brightnesses)}")

        return self.calibration

    def classify_piece(
        self,
        frame: np.ndarray,
        square: str,
        flip_board: bool = False
    ) -> PieceColor:
        """Classify a detected piece as white or black.

        Args:
            frame: Warped board image
            square: Square where piece was detected
            flip_board: If True, board is flipped

        Returns:
            PieceColor classification
        """
        if self.calibration is None or not self.calibration.calibrated:
            return PieceColor.UNKNOWN

        brightness = self._extract_piece_brightness(frame, square, flip_board)
        if brightness is None:
            return PieceColor.UNKNOWN

        return self.calibration.classify(brightness)

    def classify_all_pieces(
        self,
        frame: np.ndarray,
        detected_squares: set,
        flip_board: bool = False
    ) -> dict:
        """Classify all detected pieces.

        Args:
            frame: Warped board image
            detected_squares: Set of squares with detected pieces
            flip_board: If True, board is flipped

        Returns:
            Dict mapping square -> PieceColor
        """
        result = {}
        for square in detected_squares:
            result[square] = self.classify_piece(frame, square, flip_board)
        return result

    def is_calibrated(self) -> bool:
        """Check if calibrator has valid calibration."""
        return self.calibration is not None and self.calibration.calibrated

    def get_stats(self) -> dict:
        """Get calibration statistics."""
        if not self.is_calibrated():
            return {"calibrated": False}

        return {
            "calibrated": True,
            "white_brightness": self.calibration.white_mean_brightness,
            "black_brightness": self.calibration.black_mean_brightness,
            "threshold": self.calibration.threshold,
            "separation": self.calibration.white_mean_brightness - self.calibration.black_mean_brightness,
        }
