"""Frame differencing for chess move detection.

This approach detects moves by comparing pixel changes between frames,
rather than relying on object detection. Much more robust and reliable.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Set, Tuple, Dict
from enum import Enum


class SquareState(Enum):
    """State of a square based on frame differencing."""
    UNCHANGED = "unchanged"
    EMPTIED = "emptied"      # Piece was removed
    FILLED = "filled"        # Piece was added
    CHANGED = "changed"      # Something changed (unclear direction)


@dataclass
class SquareSignature:
    """Signature of a square's visual content."""
    mean_intensity: float
    std_intensity: float
    histogram: np.ndarray  # Simplified histogram

    def similarity(self, other: "SquareSignature") -> float:
        """Compute similarity score (0-1, higher = more similar)."""
        # Combine intensity and histogram similarity
        intensity_diff = abs(self.mean_intensity - other.mean_intensity) / 255.0
        std_diff = abs(self.std_intensity - other.std_intensity) / 128.0

        # Histogram correlation
        hist_corr = cv2.compareHist(
            self.histogram.astype(np.float32),
            other.histogram.astype(np.float32),
            cv2.HISTCMP_CORREL
        )
        hist_sim = (hist_corr + 1) / 2  # Normalize to 0-1

        # Weighted combination
        similarity = 0.4 * (1 - intensity_diff) + 0.2 * (1 - std_diff) + 0.4 * hist_sim
        return max(0, min(1, similarity))


@dataclass
class FrameDiffResult:
    """Result of frame differencing analysis."""
    emptied_squares: Set[str]
    filled_squares: Set[str]
    changed_squares: Set[str]
    stability_score: float  # 0-1, how stable the overall board is


class FrameDiffDetector:
    """Detect chess moves using frame differencing.

    Works by:
    1. Storing a reference frame when board is stable
    2. Comparing each square's pixels to the reference
    3. Detecting which squares changed significantly
    """

    # Chess square names
    FILES = "abcdefgh"
    RANKS = "87654321"  # Top to bottom in image

    def __init__(
        self,
        board_size: int = 640,
        change_threshold: float = 0.97,  # Similarity below this = changed (higher = more sensitive)
        stability_frames: int = 3,       # Frames to wait before updating reference
        board_margin: int = 0,           # Margin from edges to actual board
    ):
        self.board_size = board_size
        self.board_margin = board_margin
        effective_size = board_size - 2 * board_margin
        self.cell_size = effective_size / 8
        self.change_threshold = change_threshold
        self.stability_frames = stability_frames

        # Reference state
        self.reference_frame: Optional[np.ndarray] = None
        self.reference_signatures: Dict[str, SquareSignature] = {}

        # Stability tracking
        self.stable_count = 0
        self.last_frame: Optional[np.ndarray] = None

        # For detecting piece presence (light vs dark squares)
        self.empty_square_signatures: Dict[str, SquareSignature] = {}
        self.occupied_threshold = 0.6  # Similarity to empty below this = occupied

    def _get_square_roi(self, frame: np.ndarray, square: str) -> np.ndarray:
        """Extract the region of interest for a square."""
        file_idx = self.FILES.index(square[0])
        rank_idx = self.RANKS.index(square[1])

        m = self.board_margin
        x1 = int(m + file_idx * self.cell_size)
        y1 = int(m + rank_idx * self.cell_size)
        x2 = int(x1 + self.cell_size)
        y2 = int(y1 + self.cell_size)

        # Add small margin to avoid edge artifacts
        inner_margin = int(self.cell_size / 8)
        x1 = max(0, x1 + inner_margin)
        y1 = max(0, y1 + inner_margin)
        x2 = min(self.board_size, x2 - inner_margin)
        y2 = min(self.board_size, y2 - inner_margin)

        return frame[y1:y2, x1:x2]

    def _compute_signature(self, roi: np.ndarray) -> SquareSignature:
        """Compute visual signature for a square region."""
        # Convert to grayscale if needed
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi

        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)

        # Compute simplified histogram (16 bins)
        histogram = cv2.calcHist([gray], [0], None, [16], [0, 256])
        histogram = histogram.flatten() / histogram.sum()  # Normalize

        return SquareSignature(
            mean_intensity=mean_intensity,
            std_intensity=std_intensity,
            histogram=histogram,
        )

    def _get_all_squares(self) -> list:
        """Get all 64 square names."""
        return [f"{f}{r}" for r in self.RANKS for f in self.FILES]

    def set_reference(self, frame: np.ndarray):
        """Set the reference frame for comparison.

        Call this when the board is in a known stable state.
        """
        self.reference_frame = frame.copy()
        self.reference_signatures = {}

        for square in self._get_all_squares():
            roi = self._get_square_roi(frame, square)
            self.reference_signatures[square] = self._compute_signature(roi)

        print(f"[FrameDiff] Reference set with {len(self.reference_signatures)} squares", flush=True)

    def calibrate_empty_squares(self, frame: np.ndarray, empty_squares: Set[str]):
        """Calibrate what empty squares look like.

        Args:
            frame: Current frame
            empty_squares: Set of squares known to be empty (e.g., e4, d4 at start)
        """
        for square in empty_squares:
            roi = self._get_square_roi(frame, square)
            self.empty_square_signatures[square] = self._compute_signature(roi)

        print(f"[FrameDiff] Calibrated {len(empty_squares)} empty squares", flush=True)

    def update(self, frame: np.ndarray) -> Optional[FrameDiffResult]:
        """Update with new frame and detect changes.

        Args:
            frame: Warped board image (bird's eye view)

        Returns:
            FrameDiffResult if reference is set, None otherwise
        """
        if self.reference_frame is None:
            # Auto-set reference on first frame
            self.set_reference(frame)
            self.last_frame = frame.copy()
            return None

        # Compute current signatures
        current_signatures = {}
        for square in self._get_all_squares():
            roi = self._get_square_roi(frame, square)
            current_signatures[square] = self._compute_signature(roi)

        # Compare to reference
        emptied = set()
        filled = set()
        changed = set()

        similarities = []
        low_sims = []  # Track squares with low similarity for debugging

        for square in self._get_all_squares():
            ref_sig = self.reference_signatures[square]
            cur_sig = current_signatures[square]

            similarity = ref_sig.similarity(cur_sig)
            similarities.append(similarity)

            # Track low similarities for debugging
            if similarity < 0.95:
                low_sims.append((square, similarity))

            if similarity < self.change_threshold:
                # Square changed significantly - add to changed set
                # We DON'T try to guess emptied vs filled from intensity
                # because it depends on piece color vs square color
                changed.add(square)

        # Debug: print lowest similarities periodically
        if low_sims and self.stable_count % 5 == 0:
            low_sims.sort(key=lambda x: x[1])
            top5 = low_sims[:5]
            print(f"[FrameDiff] Lowest sims: {[(s, f'{v:.2f}') for s, v in top5]}", flush=True)

        # Compute stability score
        stability_score = np.mean(similarities)

        # Track frame-to-frame stability
        if self.last_frame is not None:
            frame_diff = cv2.absdiff(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame,
                cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2GRAY) if len(self.last_frame.shape) == 3 else self.last_frame
            )
            frame_stability = 1.0 - (np.mean(frame_diff) / 255.0)

            if frame_stability > 0.95:  # Very stable
                self.stable_count += 1
            else:
                self.stable_count = 0

        self.last_frame = frame.copy()

        return FrameDiffResult(
            emptied_squares=emptied,
            filled_squares=filled,
            changed_squares=changed,
            stability_score=stability_score,
        )

    def update_reference_for_move(self, from_sq: str, to_sq: str):
        """Update reference signatures after a confirmed move.

        This updates only the affected squares rather than re-capturing
        the entire reference frame.
        """
        if self.last_frame is not None:
            # Update signatures for moved squares
            for square in [from_sq, to_sq]:
                roi = self._get_square_roi(self.last_frame, square)
                self.reference_signatures[square] = self._compute_signature(roi)

            print(f"[FrameDiff] Updated reference for {from_sq}, {to_sq}", flush=True)

    def get_changed_squares_simple(self, frame: np.ndarray) -> Tuple[Set[str], Set[str]]:
        """Simple interface: get emptied and filled squares.

        Returns:
            (emptied_squares, filled_squares)
        """
        result = self.update(frame)
        if result is None:
            return set(), set()
        return result.emptied_squares, result.filled_squares

    def is_stable(self) -> bool:
        """Check if the board has been stable for enough frames."""
        return self.stable_count >= self.stability_frames

    def reset(self):
        """Reset the detector state."""
        self.reference_frame = None
        self.reference_signatures = {}
        self.stable_count = 0
        self.last_frame = None
        print("[FrameDiff] Reset", flush=True)


class FrameDiffMoveDetector:
    """High-level move detection using frame differencing.

    Combines FrameDiffDetector with chess logic for move validation.
    """

    def __init__(self, board_size: int = 640, board_margin: int = 0):
        import chess
        self.chess = chess
        self.board = chess.Board()
        self.diff_detector = FrameDiffDetector(board_size=board_size, board_margin=board_margin)

        # Move confirmation
        self.pending_move: Optional[chess.Move] = None
        self.pending_count: int = 0
        self.required_stable_frames: int = 5  # Much lower than YOLO approach
        self.last_move_frame: int = 0
        self.move_cooldown: int = 10  # Frames after a move before detecting next
        self.frame_count: int = 0

    def reset(self):
        """Reset to starting position."""
        self.board = self.chess.Board()
        self.diff_detector.reset()
        self.pending_move = None
        self.pending_count = 0
        self.frame_count = 0
        print("[FrameDiffMove] Reset to starting position", flush=True)

    def set_reference(self, frame: np.ndarray):
        """Set reference frame (call when board is stable at start)."""
        self.diff_detector.set_reference(frame)

    def _find_matching_move(self, changed: Set[str]) -> Optional["chess.Move"]:
        """Find a legal move matching the observed changes.

        For a normal move: from_square and to_square both change.
        For castling: 4 squares change (king from/to + rook from/to).
        For captures: from_square and to_square both change.
        """
        if len(changed) < 2:
            return None

        for move in self.board.legal_moves:
            from_sq = self.chess.square_name(move.from_square)
            to_sq = self.chess.square_name(move.to_square)

            # Handle castling first (4 squares change)
            if self.board.is_castling(move):
                if to_sq in ('g1', 'g8'):  # Kingside
                    rook_from = 'h1' if to_sq == 'g1' else 'h8'
                    rook_to = 'f1' if to_sq == 'g1' else 'f8'
                    expected = {from_sq, to_sq, rook_from, rook_to}
                    if expected.issubset(changed):
                        return move
                elif to_sq in ('c1', 'c8'):  # Queenside
                    rook_from = 'a1' if to_sq == 'c1' else 'a8'
                    rook_to = 'd1' if to_sq == 'c1' else 'd8'
                    expected = {from_sq, to_sq, rook_from, rook_to}
                    if expected.issubset(changed):
                        return move
            else:
                # Standard move or capture: from and to squares both change
                if from_sq in changed and to_sq in changed:
                    return move

        return None

    def update(self, frame: np.ndarray) -> Optional[str]:
        """Process frame and detect moves.

        Args:
            frame: Warped board image

        Returns:
            Move in SAN notation if confirmed, None otherwise
        """
        self.frame_count += 1

        # Cooldown after last move
        if self.frame_count - self.last_move_frame < self.move_cooldown:
            return None

        # Get frame diff result
        result = self.diff_detector.update(frame)
        if result is None:
            return None

        changed = result.changed_squares

        # Debug output periodically
        if self.frame_count % 10 == 0:
            print(f"[FrameDiffMove] Stability: {result.stability_score:.2f}, "
                  f"Changed: {len(changed)}", flush=True)

        # Filter out hand occlusion (too many changes = probably blocking camera)
        if len(changed) > 10:
            print(f"[FrameDiffMove] Ignoring frame - too many changes ({len(changed)}), likely hand occlusion", flush=True)
            return None

        # Only process if we have clear changes (need at least 2 for a move)
        if len(changed) < 2:
            # Not enough changes - reset pending
            if self.pending_move:
                self.pending_count = max(0, self.pending_count - 1)
                if self.pending_count == 0:
                    self.pending_move = None
            return None

        # Log changes
        print(f"[FrameDiffMove] Changed squares: {changed}", flush=True)

        # Try to match a legal move
        matching_move = self._find_matching_move(changed)

        if matching_move:
            if matching_move == self.pending_move:
                self.pending_count += 1
                print(f"[FrameDiffMove] Candidate: {matching_move.uci()} "
                      f"({self.pending_count}/{self.required_stable_frames})", flush=True)

                if self.pending_count >= self.required_stable_frames:
                    return self._confirm_move(matching_move)
            else:
                self.pending_move = matching_move
                self.pending_count = 1
                print(f"[FrameDiffMove] New candidate: {matching_move.uci()}", flush=True)
        else:
            # No match
            if changed:
                sample_legal = [m.uci() for m in list(self.board.legal_moves)[:5]]
                print(f"[FrameDiffMove] No legal move matches. Sample: {sample_legal}", flush=True)

        return None

    def _confirm_move(self, move: "chess.Move") -> str:
        """Confirm and apply a move."""
        from_sq = self.chess.square_name(move.from_square)
        to_sq = self.chess.square_name(move.to_square)

        # Get SAN before pushing
        san = self.board.san(move)

        # Apply move
        self.board.push(move)

        # Update frame diff reference for affected squares
        self.diff_detector.update_reference_for_move(from_sq, to_sq)

        # Handle castling - also update rook squares
        if self.board.is_castling(move):
            if to_sq == 'g1':
                self.diff_detector.update_reference_for_move('h1', 'f1')
            elif to_sq == 'c1':
                self.diff_detector.update_reference_for_move('a1', 'd1')
            elif to_sq == 'g8':
                self.diff_detector.update_reference_for_move('h8', 'f8')
            elif to_sq == 'c8':
                self.diff_detector.update_reference_for_move('a8', 'd8')

        # Reset pending
        self.pending_move = None
        self.pending_count = 0
        self.last_move_frame = self.frame_count

        print(f"[FrameDiffMove] CONFIRMED: {san} ({move.uci()})", flush=True)
        return san

    def get_board_fen(self) -> str:
        """Get current board position in FEN."""
        return self.board.fen()

    def get_legal_moves(self) -> list:
        """Get legal moves in UCI format."""
        return [m.uci() for m in self.board.legal_moves]
