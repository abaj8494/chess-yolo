"""Chess board state representation and utilities."""

import chess
from dataclasses import dataclass, field
from typing import Optional

from src.inference.board_detector import BoardState as DetectedBoardState


# Map our piece names to python-chess pieces
PIECE_MAP = {
    "white-king": chess.KING,
    "white-queen": chess.QUEEN,
    "white-rook": chess.ROOK,
    "white-bishop": chess.BISHOP,
    "white-knight": chess.KNIGHT,
    "white-pawn": chess.PAWN,
    "black-king": chess.KING,
    "black-queen": chess.QUEEN,
    "black-rook": chess.ROOK,
    "black-bishop": chess.BISHOP,
    "black-knight": chess.KNIGHT,
    "black-pawn": chess.PAWN,
}

# Reverse mapping
CHESS_TO_PIECE_NAME = {
    (chess.KING, chess.WHITE): "white-king",
    (chess.QUEEN, chess.WHITE): "white-queen",
    (chess.ROOK, chess.WHITE): "white-rook",
    (chess.BISHOP, chess.WHITE): "white-bishop",
    (chess.KNIGHT, chess.WHITE): "white-knight",
    (chess.PAWN, chess.WHITE): "white-pawn",
    (chess.KING, chess.BLACK): "black-king",
    (chess.QUEEN, chess.BLACK): "black-queen",
    (chess.ROOK, chess.BLACK): "black-rook",
    (chess.BISHOP, chess.BLACK): "black-bishop",
    (chess.KNIGHT, chess.BLACK): "black-knight",
    (chess.PAWN, chess.BLACK): "black-pawn",
}


def detected_to_chess_board(state: DetectedBoardState) -> chess.Board:
    """Convert detected board state to python-chess Board.

    Note: This creates a board with the current position but without
    information about castling rights, en passant, or move history.

    Args:
        state: Detected board state

    Returns:
        chess.Board with the position
    """
    board = chess.Board(fen=None)  # Empty board
    board.clear()

    for square_name, piece_name in state.squares.items():
        square = chess.parse_square(square_name)
        piece_type = PIECE_MAP.get(piece_name)
        color = chess.WHITE if piece_name.startswith("white") else chess.BLACK

        if piece_type:
            board.set_piece_at(square, chess.Piece(piece_type, color))

    # Try to infer turn from position (heuristic)
    # Count pieces to guess whose turn it is
    white_pieces = sum(1 for sq, p in state.squares.items() if p.startswith("white"))
    black_pieces = sum(1 for sq, p in state.squares.items() if p.startswith("black"))

    # If equal pieces, assume white to move; otherwise the side with fewer pieces just moved
    board.turn = chess.WHITE if white_pieces >= black_pieces else chess.BLACK

    return board


def chess_board_to_detected(board: chess.Board) -> DetectedBoardState:
    """Convert python-chess Board to detected board state.

    Args:
        board: python-chess Board

    Returns:
        DetectedBoardState representation
    """
    squares = {}
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            piece_name = CHESS_TO_PIECE_NAME.get((piece.piece_type, piece.color))
            if piece_name:
                squares[chess.square_name(square)] = piece_name

    return DetectedBoardState(squares=squares)


def get_starting_board_state() -> DetectedBoardState:
    """Get the standard chess starting position as DetectedBoardState."""
    board = chess.Board()
    return chess_board_to_detected(board)


@dataclass
class GameState:
    """Complete game state including history."""

    board: chess.Board = field(default_factory=chess.Board)
    move_history: list[chess.Move] = field(default_factory=list)
    detected_states: list[DetectedBoardState] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)

    @property
    def current_fen(self) -> str:
        return self.board.fen()

    @property
    def move_count(self) -> int:
        return len(self.move_history)

    @property
    def fullmove_number(self) -> int:
        return self.board.fullmove_number

    @property
    def is_game_over(self) -> bool:
        return self.board.is_game_over()

    @property
    def result(self) -> Optional[str]:
        if not self.is_game_over:
            return None
        return self.board.result()

    def reset(self):
        """Reset to starting position."""
        self.board = chess.Board()
        self.move_history.clear()
        self.detected_states.clear()
        self.timestamps.clear()

    def set_position(self, fen: str):
        """Set position from FEN string."""
        self.board = chess.Board(fen)
        self.move_history.clear()
        self.detected_states.clear()
        self.timestamps.clear()

    def add_move(
        self,
        move: chess.Move,
        detected_state: Optional[DetectedBoardState] = None,
        timestamp: float = 0.0,
    ) -> bool:
        """Add a move to the game.

        Args:
            move: The move to make
            detected_state: Optional detected state for reference
            timestamp: Optional timestamp

        Returns:
            True if move was legal and applied
        """
        if move in self.board.legal_moves:
            self.board.push(move)
            self.move_history.append(move)
            if detected_state:
                self.detected_states.append(detected_state)
            self.timestamps.append(timestamp)
            return True
        return False

    def undo_move(self) -> Optional[chess.Move]:
        """Undo the last move.

        Returns:
            The undone move, or None if no moves to undo
        """
        if self.move_history:
            move = self.board.pop()
            self.move_history.pop()
            if self.detected_states:
                self.detected_states.pop()
            if self.timestamps:
                self.timestamps.pop()
            return move
        return None
