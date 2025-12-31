#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Patrick Marshall
# Simplified Stratego Environment with Behavioral Reward Decomposition
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Import useful libraries
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
from __future__ import annotations  # ensure modern typing support across Python versions
from dataclasses import dataclass  # structure small bundles of transition metadata
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple  # expose common typing aliases
import numpy as np  # numerical backbone for board manipulation and randomness

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Constants describing board geometry and piece encoding
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
DIRECTIONS: Tuple[Tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))  # up, down, left, right moves
PIECE_ORDER: Tuple[str, ...] = ("F", "1", "2", "3")  # flag plus three piece ranks per side
PIECE_TO_ID: Dict[str, int] = {"F": 1, "1": 2, "2": 3, "3": 4}  # signed encoding used in observations


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Transition metrics capture the behavior bits described in the paper
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@dataclass(frozen=True)
class TransitionMetrics:
    """Bookkeeping structure used to derive the behavior indicators for each environment step."""

    flag_captured: bool  # whether the learning agent captured the opponent flag
    captured_enemy: bool  # whether any opponent piece was captured
    terminal: bool  # whether this transition concluded the episode


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Helper functions mapping between tuple actions and discrete indices
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
def encode_action(row: int, col: int, direction: int, board_size: int) -> int:
    """Encode (row, col, direction) triples into flat action ids used by the policies."""

    return ((row * board_size) + col) * len(DIRECTIONS) + direction  # pack board index and move direction into single id


def decode_action(action: int, board_size: int) -> Tuple[int, int, int]:
    """Invert encode_action: recover row, column, and direction for a given discrete action."""

    cell, direction = divmod(action, len(DIRECTIONS))  # recover linear cell index and direction
    row, col = divmod(cell, board_size)  # convert linear index back to 2D coordinates
    return row, col, direction


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Behavior indicator functions mirroring the reward decomposition in the paper
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# def b_flag(_: np.ndarray, __: int, metrics: Optional[TransitionMetrics] = None) -> float:
#     """Indicator for flag capture events by the learning agent."""

#     return float(metrics.flag_captured if metrics is not None else 0.0)

def b_capture(_: np.ndarray, __: int, metrics: Optional[TransitionMetrics] = None) -> float:
    """Indicator for capturing any opponent piece."""

    return float(metrics.captured_enemy if metrics is not None else 0.0)


def b_efficiency(_: np.ndarray, __: int, metrics: Optional[TransitionMetrics] = None) -> float:
    """Indicator penalizing the agent for each step until termination."""

    if metrics is None:
        return 1.0
    return 0.0 if metrics.terminal else 1.0


# fixed order aligns behavior functions with reward vector returned by the environment
BEHAVIOR_FUNCTIONS: List[Callable[[np.ndarray, int, Optional[TransitionMetrics]], float]] = [
    # b_flag,
    b_capture,
    b_efficiency,
]


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Behavior bit computation utility
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
def compute_behavior_bits(
    state: np.ndarray,
    action: int,
    next_state: np.ndarray,
    metrics: TransitionMetrics,
) -> Tuple[float, float, float]:
    """Evaluate every behavior function for the transition (state, action, next_state)."""

    return (
        # b_flag(state, action, metrics),  # flag capture indicator
        b_capture(state, action, metrics),  # piece capture indicator
        b_efficiency(state, action, metrics),  # step-count penalty signal
    )


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Stratego-lite environment mirroring the experimental section of the paper
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class StrategoEnv:
    """Gymnasium-style Stratego environment emitting vector rewards for the behavioral actor-critic agent."""

    def __init__(
        self,
        board_size: int = 4,
        max_steps: int = 200,
        seed: Optional[int] = None,
        piece_order: Optional[Sequence[str]] = None,
    ) -> None:
        self.board_size = board_size  # dimension of the square board
        self.max_steps = max_steps  # cap on episode length
        self.num_actions = board_size * board_size * len(DIRECTIONS)  # each square times four directions
        self._rng = np.random.default_rng(seed)  # control stochasticity for reproducible experiments
        self._board: np.ndarray = np.empty((board_size, board_size), dtype=object)  # board entries store (player, piece)
        if piece_order is None:
            order = tuple(PIECE_ORDER)
        else:
            order = tuple(piece_order)
        for token in order:
            if token not in PIECE_TO_ID:
                raise ValueError(f"Unsupported piece token '{token}'. Extend PIECE_TO_ID before using it.")
        self._piece_order: Tuple[str, ...] = order  # desired piece composition per side
        self._valid_pieces = set(order)  # accelerate layout validation against allowed tokens
        per_side_piece_count = len(self._piece_order)
        total_piece_slots = board_size * board_size
        if per_side_piece_count * 2 > total_piece_slots:
            raise ValueError(
                f"Cannot place {per_side_piece_count * 2} pieces on a {board_size}x{board_size} board."
            )
        self._step_count = 0  # track elapsed moves by the learning agent
        self._terminated = False  # whether the flag was captured by either player
        self._truncated = False  # whether the episode ended via time limit
        self._opponent_policy: Optional[Callable[[np.ndarray, np.ndarray], Optional[int]]] = None  # optional self-play opponent
        self._last_opponent_action: Optional[int] = None  # discrete action executed by opponent on previous step
        self._manual_opponent = False  # allow external control of opponent moves
        self._awaiting_opponent = False  # True when waiting for manual opponent action
        self._agent_top = False  # orientation flag: True when learning agent occupies the top row
        self.reset(seed)  # initialize board state immediately

    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # RNG control to replicate training conditions
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    def seed(self, seed: Optional[int] = None) -> None:
        """Reseed the environment RNG for deterministic rollouts."""

        self._rng = np.random.default_rng(seed)

    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # Episode reset consistent with paper’s random piece placement
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    def reset(
        self,
        seed: Optional[int] = None,
        *,
        layout: Optional[List[List[Optional[Tuple[int, str]]]]] = None,
        starting_player: int = 0,
        agent_top: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the board layout, returning the agent observation and initial info dictionary."""

        if seed is not None:
            self.seed(seed)  # allow per-reset reseeding
        if starting_player not in (0, 1):
            raise ValueError("starting_player must be 0 or 1.")
        agent_top_flag = bool(agent_top)
        self._step_count = 0  # reset move counter
        self._terminated = False  # clear terminal flag
        self._truncated = False  # clear truncation flag
        self._last_opponent_action = None  # forget previous opponent move
        self._awaiting_opponent = False  # clear pending manual opponent move
        self._agent_top = agent_top_flag  # remember orientation for subsequent steps
        self._board = np.empty((self.board_size, self.board_size), dtype=object)  # rebuild board container
        if layout is None:
            self._populate_board_default(agent_top=agent_top_flag)  # lay out pieces respecting agent orientation
        else:
            self._apply_layout(layout)  # load explicit board configuration
        obs = self._get_observation(player=0)  # observation from learning agent perspective
        legal_mask = self.legal_action_mask()
        info: Dict[str, Any] = {
            "last_capture": False,  # track whether previous action captured a piece
            "flag_captured": False,  # whether flag capture occurred on transition
            "legal_mask": legal_mask,  # legal action mask for agent
            "opponent_action": None,
            "opponent_capture": False,
            "opponent_flag": False,
            "starting_player": 0,
            "episode_done": False,
            "agent_top": agent_top_flag,
        }

        if starting_player == 1:
            opponent_mask = self.legal_action_mask(player=1)
            if opponent_mask.any():
                opponent_metrics = self._opponent_move()  # opponent takes the opening move
                obs = self._get_observation(player=0)
                legal_mask = self.legal_action_mask()
                terminated = opponent_metrics.flag_captured
                truncated = False
                outcome_reason: Optional[str] = "opponent_flag" if terminated else None
                draw_reason: Optional[str] = None

                if not terminated:
                    if self._only_flags_remaining():
                        truncated = True
                        draw_reason = "only_flags"
                        legal_mask = np.zeros(self.num_actions, dtype=bool)
                    elif not legal_mask.any():
                        opponent_has_moves = self._player_has_available_move(player=1)
                        if not opponent_has_moves:
                            truncated = True
                            draw_reason = "both_players_no_moves"
                            legal_mask = np.zeros(self.num_actions, dtype=bool)
                        else:
                            terminated = True
                            outcome_reason = "agent_no_moves"
                            legal_mask = np.zeros(self.num_actions, dtype=bool)

                self._terminated = terminated
                self._truncated = truncated and not terminated
                info["legal_mask"] = legal_mask
                info["opponent_action"] = self._last_opponent_action
                info["opponent_capture"] = opponent_metrics.captured_enemy
                info["opponent_flag"] = opponent_metrics.flag_captured
                info["starting_player"] = 1
                info["episode_done"] = terminated or truncated
                info["agent_top"] = agent_top_flag
                if terminated:
                    info["outcome"] = "agent_loss"
                    if outcome_reason is not None:
                        info["outcome_reason"] = outcome_reason
                elif truncated:
                    info["outcome"] = "draw"
                    if draw_reason is not None:
                        info["draw_reason"] = draw_reason
            else:
                info["starting_player"] = 0

        # self.render()  # visualize board after reset (enable for debugging)
        return obs, info

    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # Allow injection of a self-play opponent
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    def set_opponent_policy(
        self, policy: Optional[Callable[[np.ndarray, np.ndarray], Optional[int]]]
    ) -> None:
        """Register an opponent policy callable used for self-play evaluation."""

        self._opponent_policy = policy

    def set_manual_opponent(self, enabled: bool) -> None:
        """Enable manual opponent control (skips automatic opponent moves)."""

        self._manual_opponent = bool(enabled)

    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # Main transition function instrumented with behavior metrics
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    def step(
        self, action: int, opponent_action: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, bool, bool, Dict[str, Any]]:
        """Advance the environment one timestep and emit the vector reward described in the paper."""

        if self._terminated or self._truncated:
            raise RuntimeError("Episode already ended. Call reset before stepping again.")
        if self._awaiting_opponent:
            raise RuntimeError("Awaiting opponent move. Call step_opponent before stepping again.")

        prev_obs = self._get_observation(player=0)  # snapshot current observation for behavior computation
        row, col, direction = decode_action(action, self.board_size)  # interpret discrete action
        if not self._is_move_legal((row, col), direction, player=0):
            raise ValueError(f"Illegal action selected: {(row, col, direction)}")

        agent_metrics = self._move_piece((row, col), direction, player=0)  # execute agent move and record metrics
        board_after_agent = self.export_board_state()  # snapshot board before any opponent response
        terminated = agent_metrics.flag_captured  # immediate termination if agent captures flag
        truncated = False  # assume horizon not reached
        self._step_count += 1  # increment step counter
        if self._step_count >= self.max_steps and not terminated:
            truncated = True  # enforce time limit when max steps reached

        agent_win = bool(agent_metrics.flag_captured)  # track whether agent secured a win
        outcome_reason: Optional[str] = "agent_flag" if agent_win else None
        draw_reason: Optional[str] = None

        opponent_metrics = TransitionMetrics(False, False, False)  # default metrics when opponent does nothing
        self._last_opponent_action = None  # clear previous record before optional opponent move
        cached_next_mask: Optional[np.ndarray] = None  # reuse when opponent skips move

        if not terminated and not truncated:
            cached_next_mask = self.legal_action_mask(player=0)  # candidate mask for agent's next turn
            agent_has_future_moves = cached_next_mask.any()
            opponent_has_moves = self._player_has_available_move(player=1)
            only_flags_left = self._only_flags_remaining()

            if only_flags_left:
                truncated = True
                draw_reason = "only_flags"
            elif not opponent_has_moves:
                if agent_has_future_moves:
                    terminated = True
                    agent_win = True
                    outcome_reason = "opponent_no_moves"
                else:
                    truncated = True
                    draw_reason = "both_players_no_moves"
            else:
                cached_next_mask = None  # board will change after opponent acts
                if self._manual_opponent and opponent_action is None:
                    self._awaiting_opponent = True
                    opponent_metrics = TransitionMetrics(False, False, False)
                else:
                    if opponent_action is not None:
                        row_o, col_o, dir_o = decode_action(opponent_action, self.board_size)
                        if not self._is_move_legal((row_o, col_o), dir_o, player=1):
                            raise ValueError(
                                f"Illegal opponent action selected: {(row_o, col_o, dir_o)}"
                            )
                        opponent_metrics = self._move_piece((row_o, col_o), dir_o, player=1)
                        self._last_opponent_action = opponent_action
                    else:
                        opponent_metrics = self._opponent_move()  # allow opponent to react
                    if opponent_metrics.flag_captured:
                        terminated = True  # opponent capturing flag also ends episode
                        agent_win = False
                        outcome_reason = "opponent_flag"

        next_mask: Optional[np.ndarray] = None
        if not terminated and not truncated and not self._awaiting_opponent:
            if cached_next_mask is None:
                cached_next_mask = self.legal_action_mask(player=0)
            only_flags_left = self._only_flags_remaining()
            agent_has_moves = cached_next_mask.any()
            if only_flags_left:
                truncated = True
                draw_reason = "only_flags"
                next_mask = np.zeros(self.num_actions, dtype=bool)
            elif not agent_has_moves:
                opponent_has_moves = self._player_has_available_move(player=1)
                if not opponent_has_moves:
                    truncated = True
                    draw_reason = "both_players_no_moves"
                    next_mask = np.zeros(self.num_actions, dtype=bool)
                else:
                    terminated = True
                    agent_win = False
                    outcome_reason = "agent_no_moves"
                    next_mask = np.zeros(self.num_actions, dtype=bool)
            else:
                next_mask = cached_next_mask

        if terminated or truncated:
            next_mask = np.zeros(self.num_actions, dtype=bool) if next_mask is None else next_mask

        terminal = terminated or truncated  # aggregate terminal condition (may include no-move stalemate)
        self._terminated = terminated  # store final termination flag
        self._truncated = truncated and not terminated  # record truncation only when no terminal win/loss

        metrics = TransitionMetrics(
            flag_captured=agent_metrics.flag_captured,
            captured_enemy=agent_metrics.captured_enemy,
            terminal=terminal,
        )
        next_obs = self._get_observation(player=0)  # build next observation
        b_capture_val, b_eff_val = compute_behavior_bits(prev_obs, action, next_obs, metrics)  # derive behavior indicators
        r_primary = 1.0 if agent_win else 0.0  # primary reward reflects any agent win condition
        reward_vec = np.array([r_primary, b_capture_val, b_eff_val], dtype=np.float32)  # vector reward structure
        info = {
            "last_capture": metrics.captured_enemy,  # expose whether capture occurred
            "flag_captured": metrics.flag_captured,  # propagate flag capture to agent
            "legal_mask": next_mask if not terminal else np.zeros(self.num_actions, dtype=bool),  # expose upcoming mask or empty when finished
            "opponent_action": self._last_opponent_action,  # expose discrete opponent action for visualization tooling
            "opponent_capture": opponent_metrics.captured_enemy,
            "opponent_flag": opponent_metrics.flag_captured,
            "board_after_agent": board_after_agent,  # provide board snapshot after learner move for visualization tools
            "agent_top": self._agent_top,
        }
        if self._awaiting_opponent:
            info["awaiting_opponent"] = True
            info["opponent_mask"] = self.legal_action_mask(player=1)
        if terminated:
            info["outcome"] = "agent_win" if agent_win else "agent_loss"
            if outcome_reason is not None:
                info["outcome_reason"] = outcome_reason
        elif truncated:
            info["outcome"] = "draw"
            if draw_reason is not None:
                info["draw_reason"] = draw_reason
        return next_obs, reward_vec, self._terminated, self._truncated, info

    def step_opponent(self, opponent_action: int) -> Tuple[np.ndarray, bool, bool, Dict[str, Any]]:
        """Apply a manual opponent move after the agent has acted."""

        if self._terminated or self._truncated:
            raise RuntimeError("Episode already ended. Call reset before stepping again.")
        if not self._manual_opponent or not self._awaiting_opponent:
            raise RuntimeError("Opponent move not expected right now.")

        row_o, col_o, dir_o = decode_action(opponent_action, self.board_size)
        if not self._is_move_legal((row_o, col_o), dir_o, player=1):
            raise ValueError(f"Illegal opponent action selected: {(row_o, col_o, dir_o)}")

        opponent_metrics = self._move_piece((row_o, col_o), dir_o, player=1)
        self._last_opponent_action = opponent_action
        self._awaiting_opponent = False

        terminated = opponent_metrics.flag_captured
        truncated = False
        outcome_reason: Optional[str] = "opponent_flag" if terminated else None
        draw_reason: Optional[str] = None

        next_mask: Optional[np.ndarray] = None
        if not terminated:
            if self._only_flags_remaining():
                truncated = True
                draw_reason = "only_flags"
            else:
                next_mask = self.legal_action_mask(player=0)
                if not next_mask.any():
                    opponent_has_moves = self._player_has_available_move(player=1)
                    if not opponent_has_moves:
                        truncated = True
                        draw_reason = "both_players_no_moves"
                    else:
                        terminated = True
                        outcome_reason = "agent_no_moves"

        if terminated or truncated:
            next_mask = np.zeros(self.num_actions, dtype=bool)

        self._terminated = terminated
        self._truncated = truncated and not terminated

        info: Dict[str, Any] = {
            "legal_mask": next_mask,
            "opponent_action": self._last_opponent_action,
            "opponent_capture": opponent_metrics.captured_enemy,
            "opponent_flag": opponent_metrics.flag_captured,
        }
        if terminated:
            info["outcome"] = "agent_loss"
            if outcome_reason is not None:
                info["outcome_reason"] = outcome_reason
        elif truncated:
            info["outcome"] = "draw"
            if draw_reason is not None:
                info["draw_reason"] = draw_reason

        next_obs = self._get_observation(player=0)
        return next_obs, terminated, truncated, info

    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # Legal action mask used by masked policies in the agent
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    def legal_action_mask(self, player: int = 0) -> np.ndarray:
        """Return a boolean mask indicating which moves the given player can execute."""

        mask = np.zeros(self.num_actions, dtype=bool)  # initialize mask as all illegal
        for row in range(self.board_size):
            for col in range(self.board_size):
                piece = self._board[row, col]  # lookup piece occupying square
                if piece is None or piece[0] != player:
                    continue
                for direction_index in range(len(DIRECTIONS)):
                    if self._is_move_legal((row, col), direction_index, player):
                        idx = encode_action(row, col, direction_index, self.board_size)  # map move to action id
                        mask[idx] = True  # mark action as legal
        return mask

    def _player_has_available_move(self, player: int) -> bool:
        """Return True if the specified player has at least one legal move."""

        return self.legal_action_mask(player=player).any()

    def _only_flags_remaining(self) -> bool:
        """Return True when exactly the two flags are the only pieces left on the board."""

        non_flag_found = False
        flag_count = 0
        for row in range(self.board_size):
            for col in range(self.board_size):
                piece = self._board[row, col]
                if piece is None:
                    continue
                _, kind = piece
                if kind == "F":
                    flag_count += 1
                else:
                    non_flag_found = True
                    break
            if non_flag_found:
                break
        return not non_flag_found and flag_count == 2

    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # Random initial board layout respecting sides of the board
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    def _populate_board_default(self, agent_top: bool = False) -> None:
        """Place each player’s pieces on their home row with a fresh random shuffle every reset."""

        self._board.fill(None)  # clear board before placing pieces
        rows_needed = (len(self._piece_order) + self.board_size - 1) // self.board_size
        if rows_needed == 0:
            return
        if agent_top:
            home_rows = {
                0: list(range(rows_needed)),  # learning agent occupies top rows
                1: list(range(self.board_size - 1, self.board_size - rows_needed - 1, -1)),  # opponent on bottom rows
            }
        else:
            home_rows = {
                0: list(range(self.board_size - 1, self.board_size - rows_needed - 1, -1)),  # player 0 occupies bottom rows
                1: list(range(rows_needed)),  # opponent occupies top rows
            }
        for player, rows in home_rows.items():
            pieces = list(self._piece_order)
            self._rng.shuffle(pieces)  # fresh shuffle per player per episode
            # ensure the flag starts on the row closest to this player's baseline
            if rows:
                first_row_slots = min(self.board_size, len(pieces))
                if first_row_slots > 0 and "F" in pieces:
                    flag_index = pieces.index("F")
                    if flag_index >= first_row_slots:
                        swap_target = int(self._rng.integers(0, first_row_slots))
                        pieces[flag_index], pieces[swap_target] = pieces[swap_target], pieces[flag_index]
            piece_idx = 0
            total = len(pieces)
            for row in rows:
                remaining = total - piece_idx
                if remaining <= 0:
                    break
                slots = min(self.board_size, remaining)
                start_col = (self.board_size - slots) // 2  # center pieces when row not fully filled
                for offset in range(slots):
                    col = start_col + offset
                    self._board[row, col] = (player, pieces[piece_idx])
                    piece_idx += 1

    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # Move legality checks enforce game rules and mask correctness
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    def _is_move_legal(self, position: Tuple[int, int], direction: int, player: int) -> bool:
        """Return True if the specified move is allowed under Stratego-lite rules."""

        row, col = position  # unpack current location
        if not (0 <= row < self.board_size and 0 <= col < self.board_size):
            return False  # outside board bounds
        piece = self._board[row, col]
        if piece is None or piece[0] != player:
            return False  # empty square or opponent piece cannot move
        if piece[1] == "F":
            return False  # flags are immobile
        delta_row, delta_col = DIRECTIONS[direction]  # movement offset
        target_row = row + delta_row
        target_col = col + delta_col
        if not (0 <= target_row < self.board_size and 0 <= target_col < self.board_size):
            return False  # target must remain on board
        occupant = self._board[target_row, target_col]
        if occupant is not None and occupant[0] == player:
            return False  # cannot capture own piece
        return True

    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # Move execution logic plus behavior bookkeeping
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    def _move_piece(self, position: Tuple[int, int], direction: int, player: int) -> TransitionMetrics:
        """Execute the move and report whether a capture or flag capture occurred."""

        row, col = position  # origin square
        delta_row, delta_col = DIRECTIONS[direction]  # displacement for chosen direction
        target_row = row + delta_row
        target_col = col + delta_col
        moving_piece = self._board[row, col]  # piece being moved
        target_piece = self._board[target_row, target_col]  # occupant of destination square

        flag_captured = False
        captured_enemy = False

        if target_piece is None:
            self._board[target_row, target_col] = moving_piece  # simple move into open square
            self._board[row, col] = None
        else:
            target_owner, target_kind = target_piece
            if target_kind == "F":
                flag_captured = True
                self._board[target_row, target_col] = moving_piece
                self._board[row, col] = None
            else:
                attacker_kind = moving_piece[1]
                rank_strength = {"1": 3, "2": 2, "3": 1}
                attacker_strength = rank_strength.get(attacker_kind, 0)
                defender_strength = rank_strength.get(target_kind, 0)

                if attacker_kind == target_kind:
                    captured_enemy = True
                    self._board[target_row, target_col] = None
                    self._board[row, col] = None
                elif attacker_strength > defender_strength:
                    captured_enemy = True
                    self._board[target_row, target_col] = moving_piece
                    self._board[row, col] = None
                else:
                    # attacker loses; defender holds position
                    self._board[row, col] = None

        return TransitionMetrics(flag_captured=flag_captured, captured_enemy=captured_enemy, terminal=flag_captured)

    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # Observation encoding matches the neural network input described in the paper
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    def _get_observation(self, player: int) -> np.ndarray:
        """Encode the board from the specified player’s viewpoint as a flattened signed integer array."""

        obs = np.zeros((self.board_size, self.board_size), dtype=np.int8)  # start with empty board encoding
        for row in range(self.board_size):
            for col in range(self.board_size):
                piece = self._board[row, col]  # inspect current square
                if piece is None:
                    continue
                owner, kind = piece  # piece metadata
                value = PIECE_TO_ID[kind]  # integer identifier for rank
                obs[row, col] = value if owner == player else -value  # sign encodes ownership
        return obs.flatten()  # flatten for neural network consumption

    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # Opponent move logic used for random opponent or self-play snapshots
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    def _opponent_move(self) -> TransitionMetrics:
        """Select and execute an opponent move, falling back to random play when no policy is provided."""

        mask = self.legal_action_mask(player=1)  # compute legal moves for opponent side
        legal_indices = np.nonzero(mask)[0]  # list of legal action indices
        if legal_indices.size == 0:
            self._last_opponent_action = None
            return TransitionMetrics(False, False, False)
        action: Optional[int] = None
        if self._opponent_policy is not None:
            obs_opponent = self._get_observation(player=1)  # opponent observes board from their perspective
            try:
                action = self._opponent_policy(obs_opponent, mask.copy())  # query policy for chosen move
            except Exception:
                action = None
        if action is None or not (0 <= action < self.num_actions) or not mask[action]:
            action = int(self._rng.choice(legal_indices))  # fallback to random legal move
        self._last_opponent_action = action
        row, col, direction = decode_action(action, self.board_size)  # decode discrete move
        return self._move_piece((row, col), direction, player=1)

    def _apply_layout(self, layout: List[List[Optional[Tuple[int, str]]]]) -> None:
        """Populate board using an explicit layout description."""

        if len(layout) != self.board_size:
            raise ValueError(f"Layout height {len(layout)} does not match board_size {self.board_size}.")
        for row_idx, row in enumerate(layout):
            if len(row) != self.board_size:
                raise ValueError(f"Layout row {row_idx} has length {len(row)} but expected {self.board_size}.")
            for col_idx, entry in enumerate(row):
                if entry is None:
                    self._board[row_idx, col_idx] = None
                    continue
                player, piece = entry
                if player not in (0, 1):
                    raise ValueError(f"Invalid player id {player} at ({row_idx}, {col_idx}).")
                if piece not in self._valid_pieces:
                    raise ValueError(f"Invalid piece '{piece}' at ({row_idx}, {col_idx}); allowed: {sorted(self._valid_pieces)}.")
                self._board[row_idx, col_idx] = (player, piece)

    def export_board_state(self) -> np.ndarray:
        """Return a shallow copy of the current board configuration for visualization."""

        return self._board.copy()

    def render(self, board: Optional[np.ndarray] = None) -> None:
        """Print a textual rendering of the current board (or an explicit board snapshot)."""

        board_state = self.export_board_state() if board is None else board
        size = board_state.shape[0]
        header = "    " + " ".join(f"{c+1:>2} " for c in range(size))
        print(header)
        for r in range(size):
            row_cells = " ".join(self._format_cell(board_state[r, c]) for c in range(size))
            print(f"{r+1:>3} {row_cells}")

    def _format_cell(self, entry: Optional[Tuple[int, str]]) -> str:
        if entry is None:
            return "..."
        player, piece = entry
        piece_str = f"{player}-{piece}"
        return f"{piece_str:>3}"
    


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Lightweight smoke tests ensure the environment behaves as expected
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
def _run_env_smoke_tests() -> None:
    """Execute basic assertions so changes do not silently violate the paper’s assumptions."""

    env = StrategoEnv()  # instantiate environment for smoke tests
    _, info = env.reset(seed=123)  # deterministic reset for reproducibility
    assert info["legal_mask"].sum() > 0, "At least one legal move must exist after reset."
    _, info_top = env.reset(seed=456, agent_top=True)
    assert info_top["agent_top"] is True, "Agent orientation flag should reflect top-row placement."

    flag_seen = False  # ensure random play can eventually capture flag
    for _ in range(25):
        _, info = env.reset(seed=None)  # randomize layout
        for _ in range(128):
            legal = np.nonzero(info["legal_mask"])[0]  # fetch indices of legal moves
            if legal.size == 0:
                break
            choice = int(np.random.choice(legal))  # random action for smoke test
            _, reward_vec, terminated, truncated, info = env.step(choice)  # advance environment
            if reward_vec[0] >= 1.0:
                flag_seen = True  # record flag capture behavior signal
            if terminated or truncated:
                break
        if flag_seen:
            break
    assert flag_seen, "Expected at least one flag capture over random rollouts."

    env.reset(seed=42)
    mask = env.legal_action_mask()
    for action_idx, allowed in enumerate(mask):
        if not allowed:
            continue
        row, col, direction = decode_action(action_idx, env.board_size)
        assert env._is_move_legal((row, col), direction, player=0), "Mask incorrectly labels an illegal move as legal."


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Allow quick verification by running this module directly
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
if __name__ == "__main__":
    _run_env_smoke_tests()
