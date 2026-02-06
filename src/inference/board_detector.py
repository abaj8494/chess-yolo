"""Board state detection from piece detections."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .detector import Detection

# Square names in order (row-major, a8 to h1)
FILES = "abcdefgh"
RANKS = "87654321"  # Top to bottom in image coordinates

# All square names
SQUARES = [f"{f}{r}" for r in RANKS for f in FILES]


@dataclass
class SquareDetection:
    """A detection mapped to a specific square."""

    square: str  # e.g., "e4"
    piece: str  # e.g., "white-knight"
    confidence: float
    detection: Detection

    @property
    def file(self) -> str:
        return self.square[0]

    @property
    def rank(self) -> str:
        return self.square[1]

    @property
    def file_idx(self) -> int:
        return FILES.index(self.file)

    @property
    def rank_idx(self) -> int:
        return RANKS.index(self.rank)


@dataclass
class BoardState:
    """Complete board state from detections."""

    squares: dict[str, str] = field(default_factory=dict)  # square -> piece
    confidences: dict[str, float] = field(default_factory=dict)  # square -> confidence
    timestamp: float = 0.0
    frame_number: int = 0

    def get_piece(self, square: str) -> Optional[str]:
        """Get piece on a square, or None if empty."""
        return self.squares.get(square)

    def is_empty(self, square: str) -> bool:
        """Check if a square is empty."""
        return square not in self.squares

    def get_all_pieces(self, color: Optional[str] = None) -> dict[str, str]:
        """Get all pieces, optionally filtered by color."""
        if color is None:
            return self.squares.copy()
        return {
            sq: piece for sq, piece in self.squares.items()
            if piece.startswith(color)
        }

    def count_pieces(self) -> dict[str, int]:
        """Count pieces by type."""
        counts = {}
        for piece in self.squares.values():
            counts[piece] = counts.get(piece, 0) + 1
        return counts

    def to_fen_board(self) -> str:
        """Convert to FEN board notation (without castling/move info)."""
        rows = []

        for rank in "87654321":
            row = ""
            empty_count = 0

            for file in "abcdefgh":
                square = f"{file}{rank}"
                piece = self.squares.get(square)

                if piece is None:
                    empty_count += 1
                else:
                    if empty_count > 0:
                        row += str(empty_count)
                        empty_count = 0
                    row += self._piece_to_fen(piece)

            if empty_count > 0:
                row += str(empty_count)

            rows.append(row)

        return "/".join(rows)

    def _piece_to_fen(self, piece: str) -> str:
        """Convert piece name to FEN character."""
        piece_map = {
            "king": "k", "queen": "q", "rook": "r",
            "bishop": "b", "knight": "n", "pawn": "p",
        }
        color, piece_type = piece.split("-")
        char = piece_map[piece_type]
        return char.upper() if color == "white" else char

    def __eq__(self, other: "BoardState") -> bool:
        """Compare board states (ignoring confidence and timing)."""
        if not isinstance(other, BoardState):
            return False
        return self.squares == other.squares


class SquareMapper:
    """Map pixel coordinates to chess squares."""

    def __init__(self, board_size: int = 640):
        """Initialize mapper.

        Args:
            board_size: Size of the warped board image (assumed square)
        """
        self.board_size = board_size
        self.cell_size = board_size / 8

    def get_square(self, x: float, y: float) -> str:
        """Map pixel coordinates to square name.

        Args:
            x: X coordinate in warped image
            y: Y coordinate in warped image

        Returns:
            Square name (e.g., "e4")
        """
        # Clamp to valid range
        x = max(0, min(x, self.board_size - 1))
        y = max(0, min(y, self.board_size - 1))

        file_idx = int(x / self.cell_size)
        rank_idx = int(y / self.cell_size)

        file_idx = min(7, file_idx)
        rank_idx = min(7, rank_idx)

        return f"{FILES[file_idx]}{RANKS[rank_idx]}"

    def get_square_center(self, square: str) -> tuple[float, float]:
        """Get pixel coordinates of square center.

        Args:
            square: Square name (e.g., "e4")

        Returns:
            (x, y) center coordinates
        """
        file_idx = FILES.index(square[0])
        rank_idx = RANKS.index(square[1])

        x = (file_idx + 0.5) * self.cell_size
        y = (rank_idx + 0.5) * self.cell_size

        return (x, y)

    def get_square_bounds(self, square: str) -> tuple[float, float, float, float]:
        """Get bounding box of a square.

        Args:
            square: Square name

        Returns:
            (x1, y1, x2, y2) bounding box
        """
        file_idx = FILES.index(square[0])
        rank_idx = RANKS.index(square[1])

        x1 = file_idx * self.cell_size
        y1 = rank_idx * self.cell_size
        x2 = x1 + self.cell_size
        y2 = y1 + self.cell_size

        return (x1, y1, x2, y2)

    def map_detections(
        self,
        detections: list[Detection],
    ) -> list[SquareDetection]:
        """Map detections to squares.

        Args:
            detections: List of piece detections

        Returns:
            List of SquareDetection with square assignments
        """
        square_detections = []

        for det in detections:
            cx, cy = det.center
            square = self.get_square(cx, cy)

            square_detections.append(SquareDetection(
                square=square,
                piece=det.class_name,
                confidence=det.confidence,
                detection=det,
            ))

        return square_detections

    def build_board_state(
        self,
        detections: list[Detection],
        frame_number: int = 0,
        timestamp: float = 0.0,
    ) -> BoardState:
        """Build board state from detections.

        When multiple detections map to the same square, keep the highest
        confidence detection.

        Args:
            detections: List of piece detections
            frame_number: Current frame number
            timestamp: Current timestamp

        Returns:
            BoardState with pieces mapped to squares
        """
        square_dets = self.map_detections(detections)

        # Group by square, keeping highest confidence
        best_per_square: dict[str, SquareDetection] = {}

        for sd in square_dets:
            if sd.square not in best_per_square:
                best_per_square[sd.square] = sd
            elif sd.confidence > best_per_square[sd.square].confidence:
                best_per_square[sd.square] = sd

        # Build state
        squares = {sq: sd.piece for sq, sd in best_per_square.items()}
        confidences = {sq: sd.confidence for sq, sd in best_per_square.items()}

        return BoardState(
            squares=squares,
            confidences=confidences,
            timestamp=timestamp,
            frame_number=frame_number,
        )


def is_starting_position(state: BoardState, tolerance: int = 2) -> bool:
    """Check if board state matches starting position.

    Args:
        state: Board state to check
        tolerance: Number of missing/extra pieces allowed

    Returns:
        True if state is close to starting position
    """
    expected = {
        # White pieces
        "a1": "white-rook", "b1": "white-knight", "c1": "white-bishop", "d1": "white-queen",
        "e1": "white-king", "f1": "white-bishop", "g1": "white-knight", "h1": "white-rook",
        "a2": "white-pawn", "b2": "white-pawn", "c2": "white-pawn", "d2": "white-pawn",
        "e2": "white-pawn", "f2": "white-pawn", "g2": "white-pawn", "h2": "white-pawn",
        # Black pieces
        "a8": "black-rook", "b8": "black-knight", "c8": "black-bishop", "d8": "black-queen",
        "e8": "black-king", "f8": "black-bishop", "g8": "black-knight", "h8": "black-rook",
        "a7": "black-pawn", "b7": "black-pawn", "c7": "black-pawn", "d7": "black-pawn",
        "e7": "black-pawn", "f7": "black-pawn", "g7": "black-pawn", "h7": "black-pawn",
    }

    # Count differences
    differences = 0

    for sq, expected_piece in expected.items():
        actual_piece = state.get_piece(sq)
        if actual_piece != expected_piece:
            differences += 1

    # Check for extra pieces on wrong squares
    for sq in state.squares:
        if sq not in expected:
            differences += 1

    return differences <= tolerance
