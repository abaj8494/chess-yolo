"""Piece tracking across video frames."""

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .board_detector import BoardState, SquareMapper
from .detector import Detection


@dataclass
class TrackedPiece:
    """A piece being tracked across frames."""

    track_id: int
    piece_type: str  # e.g., "white-knight"
    current_square: str
    confidence: float
    frames_seen: int = 1
    last_seen_frame: int = 0
    history: list[str] = field(default_factory=list)  # Recent squares


class StateTracker:
    """Track board state across video frames with temporal smoothing."""

    def __init__(
        self,
        stability_threshold: int = 5,
        memory_frames: int = 10,
        board_size: int = 640,
    ):
        """Initialize tracker.

        Args:
            stability_threshold: Frames required for state change
            memory_frames: Frames to remember piece positions
            board_size: Size of warped board image
        """
        self.stability_threshold = stability_threshold
        self.memory_frames = memory_frames
        self.square_mapper = SquareMapper(board_size)

        # State tracking
        self.stable_state: Optional[BoardState] = None
        self.pending_state: Optional[BoardState] = None
        self.pending_count: int = 0
        self.frame_count: int = 0

        # Recent states for voting
        self.state_history: deque[BoardState] = deque(maxlen=memory_frames)

        # Piece memory for occlusion handling
        self.piece_memory: dict[str, tuple[str, int]] = {}  # square -> (piece, last_seen)

    def update(
        self,
        detections: list[Detection],
        timestamp: float = 0.0,
    ) -> tuple[BoardState, bool]:
        """Update tracker with new detections.

        Args:
            detections: Current frame detections
            timestamp: Frame timestamp

        Returns:
            (current_stable_state, state_changed) tuple
        """
        self.frame_count += 1

        # Build current state
        current_state = self.square_mapper.build_board_state(
            detections,
            frame_number=self.frame_count,
            timestamp=timestamp,
        )

        # Apply piece memory (for occlusion handling)
        current_state = self._apply_memory(current_state)

        # Update memory
        self._update_memory(current_state)

        # Add to history
        self.state_history.append(current_state)

        # Use voting for stability
        voted_state = self._vote_state()

        # Check for state change
        state_changed = False

        if self.stable_state is None:
            # Initialize
            self.stable_state = voted_state
            state_changed = True
        elif voted_state != self.stable_state:
            # Potential state change
            if voted_state == self.pending_state:
                self.pending_count += 1
                if self.pending_count >= self.stability_threshold:
                    self.stable_state = voted_state
                    self.pending_state = None
                    self.pending_count = 0
                    state_changed = True
            else:
                self.pending_state = voted_state
                self.pending_count = 1
        else:
            # State matches, reset pending
            self.pending_state = None
            self.pending_count = 0

        return self.stable_state, state_changed

    def _apply_memory(self, state: BoardState) -> BoardState:
        """Apply piece memory to handle temporary occlusions.

        If a piece was recently seen on a square and nothing else is
        detected there, keep the piece.
        """
        enhanced_squares = state.squares.copy()
        enhanced_confidences = state.confidences.copy()

        for square, (piece, last_seen) in self.piece_memory.items():
            frames_ago = self.frame_count - last_seen

            # If square is empty and piece was recently seen
            if square not in enhanced_squares and frames_ago < self.memory_frames:
                # Decay confidence based on time since last seen
                decay = 1.0 - (frames_ago / self.memory_frames)
                if decay > 0.3:  # Minimum confidence to persist
                    enhanced_squares[square] = piece
                    enhanced_confidences[square] = decay * 0.5  # Lower confidence

        return BoardState(
            squares=enhanced_squares,
            confidences=enhanced_confidences,
            timestamp=state.timestamp,
            frame_number=state.frame_number,
        )

    def _update_memory(self, state: BoardState) -> None:
        """Update piece memory with current detections."""
        for square, piece in state.squares.items():
            # Only update if detected with reasonable confidence
            if state.confidences.get(square, 0) > 0.3:
                self.piece_memory[square] = (piece, self.frame_count)

    def _vote_state(self) -> BoardState:
        """Use voting across recent frames to determine stable state."""
        if len(self.state_history) == 0:
            return BoardState()

        # Count votes for each square
        square_votes: dict[str, dict[str, int]] = {}

        for state in self.state_history:
            for square, piece in state.squares.items():
                if square not in square_votes:
                    square_votes[square] = {}
                square_votes[square][piece] = square_votes[square].get(piece, 0) + 1

        # Build voted state
        voted_squares = {}
        voted_confidences = {}
        threshold = len(self.state_history) / 2  # Majority vote

        for square, votes in square_votes.items():
            # Get piece with most votes
            best_piece = max(votes.keys(), key=lambda p: votes[p])
            vote_count = votes[best_piece]

            if vote_count >= threshold:
                voted_squares[square] = best_piece
                voted_confidences[square] = vote_count / len(self.state_history)

        return BoardState(
            squares=voted_squares,
            confidences=voted_confidences,
            timestamp=self.state_history[-1].timestamp if self.state_history else 0.0,
            frame_number=self.state_history[-1].frame_number if self.state_history else 0,
        )

    def reset(self) -> None:
        """Reset tracker state."""
        self.stable_state = None
        self.pending_state = None
        self.pending_count = 0
        self.frame_count = 0
        self.state_history.clear()
        self.piece_memory.clear()

    def get_stable_state(self) -> Optional[BoardState]:
        """Get current stable board state."""
        return self.stable_state

    def get_stats(self) -> dict:
        """Get tracker statistics."""
        return {
            "frame_count": self.frame_count,
            "history_size": len(self.state_history),
            "memory_size": len(self.piece_memory),
            "pending_count": self.pending_count,
            "has_stable_state": self.stable_state is not None,
        }
