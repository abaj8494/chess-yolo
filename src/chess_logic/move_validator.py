"""Validate detected moves against chess rules."""

import chess
from dataclasses import dataclass
from typing import Optional

from .move_detector import DetectedMove


@dataclass
class ValidationResult:
    """Result of move validation."""

    is_valid: bool
    validated_move: Optional[chess.Move] = None
    error: Optional[str] = None
    suggested_moves: list[chess.Move] = None

    def __post_init__(self):
        if self.suggested_moves is None:
            self.suggested_moves = []


class MoveValidator:
    """Validate detected moves using python-chess."""

    def __init__(self, board: Optional[chess.Board] = None):
        """Initialize validator.

        Args:
            board: Chess board to validate against (defaults to starting position)
        """
        self.board = board or chess.Board()

    def set_board(self, board: chess.Board):
        """Set the chess board."""
        self.board = board

    def validate(self, detected_move: DetectedMove) -> ValidationResult:
        """Validate a detected move.

        Args:
            detected_move: Move detected from vision

        Returns:
            ValidationResult with validation status
        """
        try:
            # Convert to chess.Move
            chess_move = detected_move.to_chess_move()

            # Check if legal
            if chess_move in self.board.legal_moves:
                return ValidationResult(
                    is_valid=True,
                    validated_move=chess_move,
                )

            # Try with different promotion pieces
            if detected_move.piece.endswith("-pawn"):
                from_sq = chess.parse_square(detected_move.from_square)
                to_sq = chess.parse_square(detected_move.to_square)

                for promo in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                    promo_move = chess.Move(from_sq, to_sq, promotion=promo)
                    if promo_move in self.board.legal_moves:
                        return ValidationResult(
                            is_valid=True,
                            validated_move=promo_move,
                        )

            # Find similar legal moves
            similar = self._find_similar_moves(detected_move)

            return ValidationResult(
                is_valid=False,
                error=f"Illegal move: {detected_move.to_uci()}",
                suggested_moves=similar,
            )

        except Exception as e:
            return ValidationResult(
                is_valid=False,
                error=f"Validation error: {str(e)}",
            )

    def validate_and_apply(self, detected_move: DetectedMove) -> ValidationResult:
        """Validate and apply move to the board.

        Args:
            detected_move: Move to validate and apply

        Returns:
            ValidationResult
        """
        result = self.validate(detected_move)

        if result.is_valid and result.validated_move:
            self.board.push(result.validated_move)

        return result

    def _find_similar_moves(self, detected_move: DetectedMove) -> list[chess.Move]:
        """Find legal moves similar to the detected move.

        This helps with error recovery when detection is slightly off.
        """
        similar = []

        try:
            from_sq = chess.parse_square(detected_move.from_square)
            to_sq = chess.parse_square(detected_move.to_square)
        except ValueError:
            return similar

        for move in self.board.legal_moves:
            # Same from square
            if move.from_square == from_sq:
                similar.append(move)
            # Same to square
            elif move.to_square == to_sq:
                similar.append(move)
            # Adjacent squares
            elif self._squares_adjacent(move.from_square, from_sq):
                similar.append(move)
            elif self._squares_adjacent(move.to_square, to_sq):
                similar.append(move)

        return similar[:5]  # Limit to 5 suggestions

    def _squares_adjacent(self, sq1: int, sq2: int) -> bool:
        """Check if two squares are adjacent (including diagonal)."""
        file1, rank1 = chess.square_file(sq1), chess.square_rank(sq1)
        file2, rank2 = chess.square_file(sq2), chess.square_rank(sq2)

        return abs(file1 - file2) <= 1 and abs(rank1 - rank2) <= 1

    def get_legal_moves(self) -> list[chess.Move]:
        """Get all legal moves in current position."""
        return list(self.board.legal_moves)

    def is_check(self) -> bool:
        """Check if current side is in check."""
        return self.board.is_check()

    def is_checkmate(self) -> bool:
        """Check if current side is in checkmate."""
        return self.board.is_checkmate()

    def is_stalemate(self) -> bool:
        """Check if position is stalemate."""
        return self.board.is_stalemate()

    def is_game_over(self) -> bool:
        """Check if game is over."""
        return self.board.is_game_over()

    def get_game_result(self) -> Optional[str]:
        """Get game result if game is over."""
        if not self.is_game_over():
            return None
        return self.board.result()
