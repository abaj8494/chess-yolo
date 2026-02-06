"""Detect chess moves from board state changes."""

import chess
from dataclasses import dataclass
from typing import Optional

from src.inference.board_detector import BoardState


@dataclass
class DetectedMove:
    """A move detected from board state changes."""

    from_square: str
    to_square: str
    piece: str
    is_capture: bool = False
    captured_piece: Optional[str] = None
    promotion: Optional[str] = None
    is_castling: bool = False
    is_en_passant: bool = False
    confidence: float = 1.0

    def to_uci(self) -> str:
        """Convert to UCI notation."""
        uci = f"{self.from_square}{self.to_square}"
        if self.promotion:
            # Get first letter of piece type (e.g., "queen" -> "q")
            promo_char = self.promotion.split("-")[-1][0]
            if promo_char == "k":  # knight
                promo_char = "n"
            uci += promo_char
        return uci

    def to_chess_move(self) -> chess.Move:
        """Convert to python-chess Move object."""
        from_sq = chess.parse_square(self.from_square)
        to_sq = chess.parse_square(self.to_square)

        promotion = None
        if self.promotion:
            piece_type = self.promotion.split("-")[-1]
            promo_map = {
                "queen": chess.QUEEN,
                "rook": chess.ROOK,
                "bishop": chess.BISHOP,
                "knight": chess.KNIGHT,
            }
            promotion = promo_map.get(piece_type)

        return chess.Move(from_sq, to_sq, promotion=promotion)


class MoveDetector:
    """Detect moves by comparing consecutive board states."""

    def __init__(self, board: Optional[chess.Board] = None):
        """Initialize move detector.

        Args:
            board: Optional chess.Board to validate moves against
        """
        self.board = board or chess.Board()
        self.previous_state: Optional[BoardState] = None

    def set_board(self, board: chess.Board):
        """Set the chess board for move validation."""
        self.board = board

    def detect_move(
        self,
        previous_state: BoardState,
        current_state: BoardState,
    ) -> Optional[DetectedMove]:
        """Detect a move from state change.

        Args:
            previous_state: Board state before move
            current_state: Board state after move

        Returns:
            DetectedMove if a valid move detected, None otherwise
        """
        # Find squares that changed
        disappeared = {}  # square -> piece (pieces no longer there)
        appeared = {}     # square -> piece (new pieces or moved pieces)

        # Check previous squares
        for sq, piece in previous_state.squares.items():
            if sq not in current_state.squares:
                disappeared[sq] = piece
            elif current_state.squares[sq] != piece:
                disappeared[sq] = piece
                appeared[sq] = current_state.squares[sq]

        # Check current squares for new pieces
        for sq, piece in current_state.squares.items():
            if sq not in previous_state.squares:
                appeared[sq] = piece

        # Analyze changes
        if len(disappeared) == 0 and len(appeared) == 0:
            # No change
            return None

        # Simple move: one piece disappears, appears elsewhere
        if len(disappeared) == 1 and len(appeared) == 1:
            from_sq = list(disappeared.keys())[0]
            to_sq = list(appeared.keys())[0]
            piece = disappeared[from_sq]

            # Check if same piece type (accounting for promotion)
            appeared_piece = appeared[to_sq]

            is_capture = to_sq in previous_state.squares
            captured = previous_state.squares.get(to_sq) if is_capture else None

            # Check for promotion
            promotion = None
            if piece.endswith("-pawn") and appeared_piece != piece:
                # Pawn changed to another piece = promotion
                promotion = appeared_piece

            return DetectedMove(
                from_square=from_sq,
                to_square=to_sq,
                piece=piece,
                is_capture=is_capture,
                captured_piece=captured,
                promotion=promotion,
            )

        # Capture: one piece disappears, another moves to its square
        if len(disappeared) == 2 and len(appeared) == 1:
            to_sq = list(appeared.keys())[0]
            moving_piece = appeared[to_sq]

            # Find which disappeared piece is the mover
            for from_sq, piece in disappeared.items():
                if from_sq != to_sq:
                    # Check if piece types match
                    if self._pieces_match(piece, moving_piece):
                        captured = disappeared.get(to_sq)
                        return DetectedMove(
                            from_square=from_sq,
                            to_square=to_sq,
                            piece=piece,
                            is_capture=True,
                            captured_piece=captured,
                        )

        # Castling: king and rook both move
        if len(disappeared) == 2 and len(appeared) == 2:
            castling_move = self._detect_castling(disappeared, appeared)
            if castling_move:
                return castling_move

        # En passant: pawn captures diagonally to empty square
        if len(disappeared) == 2 and len(appeared) == 1:
            ep_move = self._detect_en_passant(disappeared, appeared, previous_state)
            if ep_move:
                return ep_move

        return None

    def _pieces_match(self, piece1: str, piece2: str) -> bool:
        """Check if two piece names represent the same piece (or promotion)."""
        # Same piece
        if piece1 == piece2:
            return True

        # Check for promotion: pawn becomes another piece
        if piece1.endswith("-pawn"):
            color = piece1.split("-")[0]
            return piece2.startswith(color)

        return False

    def _detect_castling(
        self,
        disappeared: dict[str, str],
        appeared: dict[str, str],
    ) -> Optional[DetectedMove]:
        """Detect castling move."""
        # Castling patterns
        castling_patterns = [
            # White kingside: e1-g1, h1-f1
            (("e1", "white-king"), ("h1", "white-rook"), "g1", "f1"),
            # White queenside: e1-c1, a1-d1
            (("e1", "white-king"), ("a1", "white-rook"), "c1", "d1"),
            # Black kingside: e8-g8, h8-f8
            (("e8", "black-king"), ("h8", "black-rook"), "g8", "f8"),
            # Black queenside: e8-c8, a8-d8
            (("e8", "black-king"), ("a8", "black-rook"), "c8", "d8"),
        ]

        for (king_from, king_piece), (rook_from, rook_piece), king_to, rook_to in castling_patterns:
            if (
                disappeared.get(king_from) == king_piece and
                disappeared.get(rook_from) == rook_piece and
                appeared.get(king_to) == king_piece and
                appeared.get(rook_to) == rook_piece
            ):
                return DetectedMove(
                    from_square=king_from,
                    to_square=king_to,
                    piece=king_piece,
                    is_castling=True,
                )

        return None

    def _detect_en_passant(
        self,
        disappeared: dict[str, str],
        appeared: dict[str, str],
        previous_state: BoardState,
    ) -> Optional[DetectedMove]:
        """Detect en passant capture."""
        # En passant: pawn moves diagonally, enemy pawn disappears from adjacent square

        # Find the moving pawn
        pawn_squares = [sq for sq, p in disappeared.items() if p.endswith("-pawn")]
        if len(pawn_squares) != 1:
            return None

        moving_pawn_sq = None
        captured_pawn_sq = None

        for sq, piece in disappeared.items():
            if piece.endswith("-pawn"):
                if sq in appeared.values() or any(
                    p.endswith("-pawn") for p in appeared.values()
                ):
                    # This is the moving pawn
                    for to_sq, appeared_piece in appeared.items():
                        if appeared_piece == piece:
                            moving_pawn_sq = sq
                            break
                else:
                    captured_pawn_sq = sq

        if not moving_pawn_sq:
            # Try another approach: find the pawn that moved
            for from_sq, piece in disappeared.items():
                if piece.endswith("-pawn"):
                    for to_sq in appeared:
                        if appeared[to_sq] == piece:
                            moving_pawn_sq = from_sq
                            # The other disappeared pawn is captured
                            for cap_sq, cap_piece in disappeared.items():
                                if cap_sq != from_sq and cap_piece.endswith("-pawn"):
                                    captured_pawn_sq = cap_sq
                            break

        if moving_pawn_sq and captured_pawn_sq:
            to_sq = list(appeared.keys())[0]
            piece = disappeared[moving_pawn_sq]

            return DetectedMove(
                from_square=moving_pawn_sq,
                to_square=to_sq,
                piece=piece,
                is_capture=True,
                captured_piece=disappeared[captured_pawn_sq],
                is_en_passant=True,
            )

        return None

    def update(self, current_state: BoardState) -> Optional[DetectedMove]:
        """Update with new state and detect any move.

        Args:
            current_state: Current board state

        Returns:
            DetectedMove if a move was detected
        """
        if self.previous_state is None:
            self.previous_state = current_state
            return None

        move = self.detect_move(self.previous_state, current_state)

        if move:
            self.previous_state = current_state

        return move

    def reset(self, state: Optional[BoardState] = None):
        """Reset detector state."""
        self.previous_state = state
