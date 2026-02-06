"""PGN (Portable Game Notation) generation from detected moves."""

import chess
import chess.pgn
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Optional

from .board_state import GameState
from .move_detector import DetectedMove


class PGNGenerator:
    """Generate PGN from chess game data."""

    def __init__(
        self,
        event: str = "Recorded Game",
        site: str = "Chess Vision System",
        white_player: str = "White",
        black_player: str = "Black",
        date: Optional[str] = None,
    ):
        """Initialize PGN generator.

        Args:
            event: Event name for PGN header
            site: Site name for PGN header
            white_player: White player name
            black_player: Black player name
            date: Date string (YYYY.MM.DD format), defaults to today
        """
        self.event = event
        self.site = site
        self.white_player = white_player
        self.black_player = black_player
        self.date = date or datetime.now().strftime("%Y.%m.%d")

        # Internal game tracking
        self.game = chess.pgn.Game()
        self.node = self.game
        self.board = chess.Board()
        self.move_count = 0

        self._setup_headers()

    def _setup_headers(self):
        """Set up PGN headers."""
        self.game.headers["Event"] = self.event
        self.game.headers["Site"] = self.site
        self.game.headers["Date"] = self.date
        self.game.headers["Round"] = "1"
        self.game.headers["White"] = self.white_player
        self.game.headers["Black"] = self.black_player
        self.game.headers["Result"] = "*"

    def set_header(self, key: str, value: str):
        """Set a PGN header."""
        self.game.headers[key] = value

    def add_move(
        self,
        move: chess.Move,
        comment: Optional[str] = None,
        timestamp: Optional[float] = None,
    ) -> bool:
        """Add a move to the game.

        Args:
            move: Chess move to add
            comment: Optional comment for this move
            timestamp: Optional timestamp (seconds)

        Returns:
            True if move was added successfully
        """
        if move not in self.board.legal_moves:
            return False

        self.board.push(move)
        self.node = self.node.add_variation(move)
        self.move_count += 1

        # Add comment if provided
        if comment:
            self.node.comment = comment
        elif timestamp is not None:
            # Add timestamp as comment
            self.node.comment = f"[%clk {self._format_time(timestamp)}]"

        # Update result if game is over
        if self.board.is_game_over():
            self.game.headers["Result"] = self.board.result()

        return True

    def add_detected_move(
        self,
        detected_move: DetectedMove,
        timestamp: Optional[float] = None,
    ) -> bool:
        """Add a detected move to the game.

        Args:
            detected_move: Move detected from vision
            timestamp: Optional timestamp

        Returns:
            True if move was added successfully
        """
        chess_move = detected_move.to_chess_move()

        # Build comment with detection info
        comment_parts = []
        if detected_move.confidence < 1.0:
            comment_parts.append(f"conf={detected_move.confidence:.2f}")
        if timestamp is not None:
            comment_parts.append(f"t={timestamp:.1f}s")

        comment = ", ".join(comment_parts) if comment_parts else None

        return self.add_move(chess_move, comment, timestamp)

    def set_result(self, result: str):
        """Set the game result.

        Args:
            result: Result string ("1-0", "0-1", "1/2-1/2", or "*")
        """
        if result in ["1-0", "0-1", "1/2-1/2", "*"]:
            self.game.headers["Result"] = result

    def to_pgn(self) -> str:
        """Generate PGN string.

        Returns:
            Complete PGN string
        """
        return str(self.game)

    def save(self, path: str | Path):
        """Save PGN to file.

        Args:
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            f.write(self.to_pgn())

    def reset(self):
        """Reset generator for a new game."""
        self.game = chess.pgn.Game()
        self.node = self.game
        self.board = chess.Board()
        self.move_count = 0
        self._setup_headers()

    def from_game_state(self, state: GameState) -> str:
        """Generate PGN from GameState object.

        Args:
            state: Complete game state

        Returns:
            PGN string
        """
        self.reset()

        for i, move in enumerate(state.move_history):
            timestamp = state.timestamps[i] if i < len(state.timestamps) else None
            self.add_move(move, timestamp=timestamp)

        if state.is_game_over:
            self.set_result(state.result or "*")

        return self.to_pgn()

    def _format_time(self, seconds: float) -> str:
        """Format time in H:MM:SS format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}:{minutes:02d}:{secs:02d}"

    @staticmethod
    def load_pgn(path: str | Path) -> chess.pgn.Game:
        """Load a PGN file.

        Args:
            path: Path to PGN file

        Returns:
            Parsed game
        """
        with open(path) as f:
            return chess.pgn.read_game(f)

    @staticmethod
    def parse_pgn(pgn_string: str) -> chess.pgn.Game:
        """Parse PGN from string.

        Args:
            pgn_string: PGN content

        Returns:
            Parsed game
        """
        return chess.pgn.read_game(StringIO(pgn_string))


class LivePGNWriter:
    """Write PGN incrementally for live games."""

    def __init__(
        self,
        output_path: str | Path,
        **pgn_kwargs,
    ):
        """Initialize live PGN writer.

        Args:
            output_path: Path to output PGN file
            **pgn_kwargs: Arguments passed to PGNGenerator
        """
        self.output_path = Path(output_path)
        self.generator = PGNGenerator(**pgn_kwargs)
        self._last_save_move = 0

    def add_move(
        self,
        move: chess.Move,
        comment: Optional[str] = None,
        timestamp: Optional[float] = None,
        auto_save: bool = True,
    ) -> bool:
        """Add move and optionally save.

        Args:
            move: Move to add
            comment: Optional comment
            timestamp: Optional timestamp
            auto_save: Whether to save after adding

        Returns:
            True if move added successfully
        """
        result = self.generator.add_move(move, comment, timestamp)

        if result and auto_save:
            self.save()

        return result

    def save(self):
        """Save current PGN to file."""
        self.generator.save(self.output_path)
        self._last_save_move = self.generator.move_count

    def finalize(self, result: Optional[str] = None):
        """Finalize the game and save.

        Args:
            result: Optional result to set
        """
        if result:
            self.generator.set_result(result)
        elif self.generator.board.is_game_over():
            self.generator.set_result(self.generator.board.result())

        self.save()
