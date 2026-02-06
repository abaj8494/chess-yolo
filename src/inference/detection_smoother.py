"""Temporal smoothing for chess piece detection.

Filters out noisy detections by requiring consistent detection
across multiple frames before reporting a square as occupied.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Set, Dict, Optional


@dataclass
class SmoothedDetection:
    """Smoothed detection result."""
    occupied: Set[str]  # Squares that are stably occupied
    raw_occupied: Set[str]  # Current frame's raw detections


class DetectionSmoother:
    """Temporal filter for piece detections.

    A square is only reported as occupied if detected in at least
    `min_detections` of the last `window_size` frames.

    Similarly, a square is only removed from occupied set after
    being empty for `min_empty` consecutive frames.
    """

    def __init__(
        self,
        window_size: int = 5,
        min_detections: int = 3,
        min_empty: int = 3,
    ):
        self.window_size = window_size
        self.min_detections = min_detections
        self.min_empty = min_empty

        # Track detection history per square
        # Each entry is a list of booleans (detected=True)
        self.history: Dict[str, list] = defaultdict(list)

        # Currently reported as occupied (smoothed)
        self.smoothed_occupied: Set[str] = set()

        # Track empty frames for hysteresis
        self.empty_frames: Dict[str, int] = defaultdict(int)

    def update(self, detected_squares: Set[str]) -> SmoothedDetection:
        """Update with new frame's detections.

        Args:
            detected_squares: Set of square names detected this frame

        Returns:
            SmoothedDetection with stable occupied squares
        """
        all_squares = self._get_all_squares()

        # Update history for all squares
        for sq in all_squares:
            is_detected = sq in detected_squares

            # Add to history
            self.history[sq].append(is_detected)

            # Keep only last window_size frames
            if len(self.history[sq]) > self.window_size:
                self.history[sq] = self.history[sq][-self.window_size:]

            # Update empty frame counter
            if is_detected:
                self.empty_frames[sq] = 0
            else:
                self.empty_frames[sq] += 1

        # Compute smoothed occupancy
        new_occupied = set()

        for sq in all_squares:
            detections = sum(self.history[sq])
            was_occupied = sq in self.smoothed_occupied

            if was_occupied:
                # Square was occupied - only remove if consistently empty
                if self.empty_frames[sq] >= self.min_empty:
                    # Remove from occupied
                    pass
                else:
                    # Keep as occupied (hysteresis)
                    new_occupied.add(sq)
            else:
                # Square was empty - only add if consistently detected
                if detections >= self.min_detections:
                    new_occupied.add(sq)

        self.smoothed_occupied = new_occupied

        return SmoothedDetection(
            occupied=new_occupied.copy(),
            raw_occupied=detected_squares.copy(),
        )

    def reset(self):
        """Reset smoother state."""
        self.history.clear()
        self.smoothed_occupied.clear()
        self.empty_frames.clear()

    def _get_all_squares(self) -> Set[str]:
        """Get all 64 chess squares."""
        files = 'abcdefgh'
        ranks = '12345678'
        return {f + r for f in files for r in ranks}

    def get_stats(self) -> Dict[str, int]:
        """Get detection statistics."""
        total_detected = sum(
            1 for sq in self._get_all_squares()
            if self.history.get(sq) and self.history[sq][-1]
        )
        return {
            "raw_detected": total_detected,
            "smoothed_detected": len(self.smoothed_occupied),
        }
