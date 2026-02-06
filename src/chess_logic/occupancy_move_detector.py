"""Occupancy-based move detection using chess engine constraints."""

import chess
from dataclasses import dataclass
from typing import Optional, Set

from src.inference.board_detector import BoardState
from .move_detector import DetectedMove


@dataclass
class OccupancyChange:
    """Represents observed changes in board occupancy."""
    emptied: Set[str]  # Squares that became empty
    filled: Set[str]   # Squares that became occupied

    def __repr__(self):
        return f"OccupancyChange(emptied={self.emptied}, filled={self.filled})"


class OccupancyMoveDetector:
    """Detect moves using only occupancy changes + legal move constraints.

    This approach doesn't rely on piece classification accuracy.
    Instead, it:
    1. Detects which squares changed (emptied/filled)
    2. Finds the legal move that matches the observed change
    3. Uses chess rules to filter impossible moves
    """

    def __init__(self):
        self.board = chess.Board()
        self.previous_occupancy: Optional[Set[str]] = None
        self.stable_occupancy: Optional[Set[str]] = None  # Last confirmed stable state
        self.pending_move: Optional[chess.Move] = None
        self.pending_count: int = 0
        self.required_stable_frames: int = 30  # Very high for noisy detection
        self.baseline_locked: bool = False  # Lock baseline to prevent drift
        self.last_move_frame: int = 0  # Frame when last move was confirmed
        self.move_cooldown: int = 60  # Frames to wait after a move before accepting another

    def reset(self):
        """Reset to starting position."""
        self.board = chess.Board()
        self.previous_occupancy = None
        self.stable_occupancy = None
        self.pending_move = None
        self.pending_count = 0
        self.baseline_locked = False
        print("[Occupancy] Reset - will re-initialize on next detection", flush=True)

    def initialize_from_starting_position(self):
        """Initialize baseline from standard starting position.

        This is more reliable than using noisy detection.
        Call this after color calibration completes.
        """
        # Standard starting position squares
        starting_squares = set()
        # White pieces: rows 1 and 2
        for file in 'abcdefgh':
            starting_squares.add(f'{file}1')
            starting_squares.add(f'{file}2')
        # Black pieces: rows 7 and 8
        for file in 'abcdefgh':
            starting_squares.add(f'{file}7')
            starting_squares.add(f'{file}8')

        self.previous_occupancy = starting_squares.copy()
        self.stable_occupancy = starting_squares.copy()
        self.baseline_locked = True  # Lock immediately
        self._no_change_frames = 0
        self._debug_frame_count = 0

        print(f"[Occupancy] Initialized from STARTING POSITION: 32 pieces", flush=True)
        print(f"[Occupancy] Baseline LOCKED - ready for moves!", flush=True)

    def lock_baseline(self):
        """Lock current baseline to prevent drift from noisy detection."""
        self.baseline_locked = True
        print(f"[Occupancy] Baseline LOCKED with {len(self.stable_occupancy) if self.stable_occupancy else 0} pieces", flush=True)

    def unlock_baseline(self):
        """Unlock baseline to allow updates."""
        self.baseline_locked = False
        print("[Occupancy] Baseline UNLOCKED", flush=True)

    def _get_expected_occupancy(self) -> Set[str]:
        """Get expected occupied squares from chess engine."""
        occupied = set()
        for square in chess.SQUARES:
            if self.board.piece_at(square):
                occupied.add(chess.square_name(square))
        return occupied

    def _state_to_occupancy(self, state: BoardState) -> Set[str]:
        """Convert BoardState to set of occupied squares."""
        return set(state.squares.keys())

    def _find_matching_legal_move(self, change: OccupancyChange) -> Optional[chess.Move]:
        """Find a legal move that matches the observed occupancy change.

        Handles cases where:
        1. Both source and destination are visible (standard detection)
        2. Only destination is visible (source was never reliably detected)
        """

        # Get squares that were never reliably detected but should exist per chess engine
        expected = self._get_expected_occupancy()
        detection_baseline = getattr(self, '_detection_baseline', set()) or set()
        undetected_squares = expected - detection_baseline

        for move in self.board.legal_moves:
            from_sq = chess.square_name(move.from_square)
            to_sq = chess.square_name(move.to_square)

            # Case 1: Standard move - piece leaves 'from', arrives at 'to'
            if from_sq in change.emptied and to_sq in change.filled:
                return move

            # Case 2: Source was never detected, but destination appeared
            # This handles pieces that YOLO misses but are really there
            if from_sq in undetected_squares and to_sq in change.filled:
                # Only match if nothing else is in emptied (clean appearance)
                if len(change.emptied) == 0:
                    return move

            # Case 3: Capture - piece leaves 'from', replaces piece at 'to'
            if from_sq in change.emptied and to_sq not in change.filled and to_sq not in change.emptied:
                return move

            # Castling: check for king+rook movement pattern
            if self.board.is_castling(move):
                if self._matches_castling(move, change):
                    return move

            # En passant: pawn captures to empty square, enemy pawn disappears
            if self.board.is_en_passant(move):
                if self._matches_en_passant(move, change):
                    return move

        return None

    def _matches_castling(self, move: chess.Move, change: OccupancyChange) -> bool:
        """Check if occupancy change matches a castling move."""
        from_sq = chess.square_name(move.from_square)
        to_sq = chess.square_name(move.to_square)

        # Kingside castling
        if to_sq in ('g1', 'g8'):
            rook_from = 'h1' if to_sq == 'g1' else 'h8'
            rook_to = 'f1' if to_sq == 'g1' else 'f8'
            expected_emptied = {from_sq, rook_from}
            expected_filled = {to_sq, rook_to}
        # Queenside castling
        elif to_sq in ('c1', 'c8'):
            rook_from = 'a1' if to_sq == 'c1' else 'a8'
            rook_to = 'd1' if to_sq == 'c1' else 'd8'
            expected_emptied = {from_sq, rook_from}
            expected_filled = {to_sq, rook_to}
        else:
            return False

        return expected_emptied <= change.emptied and expected_filled <= change.filled

    def _matches_en_passant(self, move: chess.Move, change: OccupancyChange) -> bool:
        """Check if occupancy change matches an en passant capture."""
        from_sq = chess.square_name(move.from_square)
        to_sq = chess.square_name(move.to_square)

        # The captured pawn is on the same rank as the moving pawn, same file as destination
        captured_sq = chess.square_name(
            chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square))
        )

        expected_emptied = {from_sq, captured_sq}
        expected_filled = {to_sq}

        return expected_emptied <= change.emptied and expected_filled <= change.filled

    def update(self, current_state: BoardState) -> Optional[DetectedMove]:
        """Update with new board state and detect moves.

        Uses a hybrid approach:
        1. Track "visible" squares (those detection can see reliably)
        2. Detect moves by watching for changes in visible squares
        3. Match changes against legal moves

        Args:
            current_state: Current detected board state (occupancy)

        Returns:
            DetectedMove if a move was confirmed
        """
        current_occupancy = self._state_to_occupancy(current_state)

        # Initialize tracking on first call (if not already initialized from starting position)
        if self.previous_occupancy is None:
            self.previous_occupancy = current_occupancy
            if self.stable_occupancy is None:
                self.stable_occupancy = current_occupancy
            self._no_change_frames = 0
            self._detection_baseline = current_occupancy.copy()  # What detection "normally" sees
            print(f"[Occupancy] Initialized with {len(current_occupancy)} detected pieces", flush=True)
            print(f"[Occupancy] Squares: {sorted(current_occupancy)}", flush=True)
            if self.baseline_locked:
                print(f"[Occupancy] Using starting position baseline (32 squares)", flush=True)
            return None

        # Initialize detection baseline if not set
        if not hasattr(self, '_detection_baseline') or self._detection_baseline is None:
            self._detection_baseline = current_occupancy.copy()

        # Compare current vs previous frame (frame-to-frame diff)
        frame_emptied = self.previous_occupancy - current_occupancy
        frame_filled = current_occupancy - self.previous_occupancy

        # Update previous for next frame
        self.previous_occupancy = current_occupancy

        # Debug every 60th frame
        self._debug_frame_count = getattr(self, '_debug_frame_count', 0) + 1
        if self._debug_frame_count % 60 == 0:
            print(f"[Occupancy] Baseline: {len(self.stable_occupancy)}, Detection: {len(current_occupancy)}", flush=True)

        # If no frame-to-frame change, detection is stable
        if not frame_emptied and not frame_filled:
            self._no_change_frames = getattr(self, '_no_change_frames', 0) + 1

            # After many stable frames with no pending move, update detection baseline
            if self._no_change_frames >= 15 and self.pending_move is None:
                if self._detection_baseline != current_occupancy:
                    self._detection_baseline = current_occupancy.copy()
            return None
        else:
            self._no_change_frames = 0

        # === MOVE DETECTION ===
        # Strategy: Look for changes in DETECTION that match legal moves
        # Don't compare against chess baseline (which may have undetected squares)

        # Cooldown check - prevent rapid false moves
        if self._debug_frame_count - self.last_move_frame < self.move_cooldown:
            if self._debug_frame_count % 30 == 0:
                remaining = self.move_cooldown - (self._debug_frame_count - self.last_move_frame)
                print(f"[Occupancy] Cooldown: {remaining} frames remaining", flush=True)
            return None

        # Compare against detection baseline (what detection "normally" sees)
        detection_emptied = self._detection_baseline - current_occupancy
        detection_filled = current_occupancy - self._detection_baseline

        # Also check against chess engine's expected state
        expected_occupancy = self._get_expected_occupancy()
        expected_emptied = expected_occupancy - current_occupancy
        expected_filled = current_occupancy - expected_occupancy

        # Debug: show what changed in detection
        if len(detection_emptied) <= 2 and len(detection_filled) <= 2:
            if detection_emptied or detection_filled:
                print(f"[Occupancy] Detection change: -{detection_emptied}, +{detection_filled}", flush=True)

        # Try to find a matching legal move from detection changes
        change = OccupancyChange(emptied=detection_emptied, filled=detection_filled)
        matching_move = self._find_matching_legal_move(change)

        # If no match from detection, try expected vs current
        if not matching_move and (len(expected_emptied) <= 2 and len(expected_filled) <= 2):
            change = OccupancyChange(emptied=expected_emptied, filled=expected_filled)
            matching_move = self._find_matching_legal_move(change)
            if matching_move:
                print(f"[Occupancy] Matched from expected state change", flush=True)

        if matching_move:
            if matching_move == self.pending_move:
                self.pending_count += 1
                print(f"[Occupancy] Candidate: {matching_move.uci()} ({self.pending_count}/{self.required_stable_frames})", flush=True)

                if self.pending_count >= self.required_stable_frames:
                    return self._confirm_move(matching_move)
            else:
                self.pending_move = matching_move
                self.pending_count = 1
                print(f"[Occupancy] New candidate: {matching_move.uci()} ({self.pending_count}/{self.required_stable_frames})", flush=True)
        else:
            # No match found
            if (len(detection_emptied) <= 2 and len(detection_filled) <= 2 and
                (detection_emptied or detection_filled)):
                legal = [m.uci() for m in list(self.board.legal_moves)[:5]]
                print(f"[Occupancy] No legal move matches. Legal: {legal}", flush=True)

            if self.pending_move and self.pending_count > 0:
                self.pending_count = max(0, self.pending_count - 1)
                if self.pending_count == 0:
                    print(f"[Occupancy] Pending move cancelled", flush=True)
                    self.pending_move = None

        return None

    def _confirm_move(self, move: chess.Move) -> Optional[DetectedMove]:
        """Confirm and apply a move."""
        from_sq = chess.square_name(move.from_square)
        to_sq = chess.square_name(move.to_square)
        piece = self.board.piece_at(move.from_square)

        # Validate: source should be in detection baseline OR in undetected (expected but not seen)
        expected = self._get_expected_occupancy()
        detection_baseline = getattr(self, '_detection_baseline', set()) or set()
        undetected_expected = expected - detection_baseline

        if hasattr(self, '_detection_baseline') and self._detection_baseline:
            if from_sq not in self._detection_baseline and from_sq not in undetected_expected:
                print(f"[Occupancy] REJECTED {move.uci()}: {from_sq} not expected!", flush=True)
                self.pending_move = None
                self.pending_count = 0
                return None

        # Log if move was from undetected square
        if from_sq in undetected_expected:
            print(f"[Occupancy] Note: {from_sq} was undetected but expected (chess engine)", flush=True)

        # Record frame for cooldown
        self.last_move_frame = getattr(self, '_debug_frame_count', 0)

        # Get SAN before pushing
        san = self.board.san(move)

        # Determine move properties
        is_capture = self.board.is_capture(move)
        is_castling = self.board.is_castling(move)
        is_en_passant = self.board.is_en_passant(move)

        # Apply move to board
        self.board.push(move)

        # Update stable occupancy to reflect the move
        # Remove piece from 'from' square, add to 'to' square
        new_stable = self.stable_occupancy.copy()
        new_stable.discard(from_sq)
        new_stable.add(to_sq)

        # Handle castling - rook also moves
        if is_castling:
            if to_sq == 'g1':  # White kingside
                new_stable.discard('h1')
                new_stable.add('f1')
            elif to_sq == 'c1':  # White queenside
                new_stable.discard('a1')
                new_stable.add('d1')
            elif to_sq == 'g8':  # Black kingside
                new_stable.discard('h8')
                new_stable.add('f8')
            elif to_sq == 'c8':  # Black queenside
                new_stable.discard('a8')
                new_stable.add('d8')

        # Handle en passant - captured pawn disappears
        if is_en_passant:
            # Captured pawn is on same rank as moving pawn was, same file as destination
            ep_captured_sq = chess.square_name(
                chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square))
            )
            new_stable.discard(ep_captured_sq)

        # Handle regular capture - just replace, already handled by add to_sq

        self.stable_occupancy = new_stable
        self.previous_occupancy = new_stable

        # Update detection baseline to reflect the move
        # This helps track future moves from this new state
        if hasattr(self, '_detection_baseline') and self._detection_baseline is not None:
            new_det_baseline = self._detection_baseline.copy()
            new_det_baseline.discard(from_sq)
            new_det_baseline.add(to_sq)
            if is_castling:
                if to_sq == 'g1':
                    new_det_baseline.discard('h1')
                    new_det_baseline.add('f1')
                elif to_sq == 'c1':
                    new_det_baseline.discard('a1')
                    new_det_baseline.add('d1')
                elif to_sq == 'g8':
                    new_det_baseline.discard('h8')
                    new_det_baseline.add('f8')
                elif to_sq == 'c8':
                    new_det_baseline.discard('a8')
                    new_det_baseline.add('d8')
            if is_en_passant:
                captured_sq = chess.square_name(
                    chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square))
                )
                new_det_baseline.discard(captured_sq)
            self._detection_baseline = new_det_baseline

        # Reset pending
        self.pending_move = None
        self.pending_count = 0

        print(f"[Occupancy] Confirmed: {san} ({move.uci()})")

        # Build piece name for DetectedMove
        piece_color = "white" if piece.color == chess.WHITE else "black"
        piece_names = {
            chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop",
            chess.ROOK: "rook", chess.QUEEN: "queen", chess.KING: "king"
        }
        piece_name = f"{piece_color}-{piece_names[piece.piece_type]}"

        return DetectedMove(
            from_square=from_sq,
            to_square=to_sq,
            piece=piece_name,
            is_capture=is_capture,
            is_castling=is_castling,
            is_en_passant=is_en_passant,
        )

    def get_legal_moves_uci(self) -> list[str]:
        """Get all legal moves in UCI format."""
        return [m.uci() for m in self.board.legal_moves]

    def get_board_fen(self) -> str:
        """Get current board position in FEN."""
        return self.board.fen()
