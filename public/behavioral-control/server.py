#!/usr/bin/env python3
"""Local inference server for the Stratego simulation UI."""

from __future__ import annotations

import json
import os
import random
import sys
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent import (
    get_aggregated_action_and_probs,
    make_ensemble,
    make_opponent_policy,
    prepare_weights,
)
from env import DIRECTIONS, StrategoEnv, decode_action

AGENT_FILES = {
    3: Path(__file__).resolve().parent / "agents" / "agent3x3.pth",
    5: Path(__file__).resolve().parent / "agents" / "agent5x5.pth",
    7: Path(__file__).resolve().parent / "agents" / "agent7x7.pth",
}

PIECE_SETS = {
    3: ["F", "1", "2"],
    5: ["F", "1", "1", "2", "3"],
    7: [
        "F",
        "1",
        "1",
        "1",
        "2",
        "2",
        "2",
        "2",
        "2",
        "3",
        "3",
        "3",
        "3",
        "3",
    ],
}

DEVICE = torch.device("cpu")
MODEL_CACHE: Dict[int, Dict[str, Any]] = {}
SESSIONS: Dict[str, "Session"] = {}
MAX_ROUNDS = 1000

REASON_LABELS = {
    "agent_flag": "Flag captured",
    "opponent_flag": "Flag captured",
    "opponent_no_moves": "Opponent out of moves",
    "agent_no_moves": "Agent out of moves",
    "only_flags": "Only flags remaining",
    "both_players_no_moves": "No moves remaining",
    "max_steps": "Turn limit reached",
}


def _infer_dims_from_snapshot(snapshot: List[dict]) -> Tuple[int, int]:
    if not snapshot:
        raise ValueError("Snapshot is empty.")
    state = snapshot[0]
    encoder_weight = state.get("encoder.0.weight")
    policy_weight = state.get("policy_head.weight")
    if encoder_weight is None or policy_weight is None:
        raise ValueError("Snapshot missing required model weights for shape inference.")
    obs_dim = int(encoder_weight.shape[1])
    action_dim = int(policy_weight.shape[0])
    return obs_dim, action_dim


def _load_snapshot(path: Path) -> List[dict]:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "snapshot" in payload:
        snapshot = payload["snapshot"]
    elif isinstance(payload, list):
        snapshot = payload
    else:
        raise ValueError(f"Unsupported checkpoint format at {path}.")
    if not isinstance(snapshot, list):
        raise ValueError("Snapshot payload must be a list of state dicts.")
    return snapshot


def _load_agent_bundle(grid_size: int) -> Dict[str, Any]:
    cached = MODEL_CACHE.get(grid_size)
    if cached is not None:
        return cached
    path = AGENT_FILES.get(grid_size)
    if path is None or not path.exists():
        raise FileNotFoundError(f"Missing agent file for {grid_size}x{grid_size} at {path}.")
    snapshot = _load_snapshot(path)
    obs_dim, action_dim = _infer_dims_from_snapshot(snapshot)
    ensemble = make_ensemble(len(snapshot), obs_dim, action_dim)
    for model, state in zip(ensemble, snapshot):
        model.load_state_dict(state)
        model.to(device=DEVICE)
        model.eval()
    bundle = {
        "ensemble": ensemble,
        "snapshot": snapshot,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
    }
    MODEL_CACHE[grid_size] = bundle
    return bundle


def _coerce_weights(
    payload: Optional[Dict[str, Any]],
    fallback: np.ndarray,
) -> np.ndarray:
    if payload is None:
        return fallback
    return np.array(
        [
            float(payload.get("flag", fallback[0])),
            float(payload.get("capture", fallback[1])),
            float(payload.get("efficiency", fallback[2])),
        ],
        dtype=np.float32,
    )


def _serialize_board(board: np.ndarray) -> List[List[Optional[str]]]:
    rows: List[List[Optional[str]]] = []
    for row in board:
        output_row: List[Optional[str]] = []
        for entry in row:
            if entry is None:
                output_row.append(None)
                continue
            player, piece = entry
            actor = "A1" if player == 0 else "A2"
            output_row.append(f"{actor}-{piece}")
        rows.append(output_row)
    return rows


def _decode_move(action: int, board_size: int) -> Tuple[int, int, int, int]:
    row, col, direction = decode_action(action, board_size)
    delta_row, delta_col = DIRECTIONS[direction]
    return row, col, row + delta_row, col + delta_col


def _collect_user_moves(session: "Session") -> List[Dict[str, Any]]:
    mask = session.env.legal_action_mask(player=1)
    actions = np.flatnonzero(mask)
    moves: List[Dict[str, Any]] = []
    for action in actions:
        row, col, to_row, to_col = _decode_move(int(action), session.grid_size)
        moves.append(
            {
                "action": int(action),
                "from": {"row": row, "col": col},
                "to": {"row": to_row, "col": to_col},
            }
        )
    return moves


def _describe_move(
    action: Optional[int],
    board: np.ndarray,
    board_size: int,
    actor: str,
) -> Optional[Dict[str, Any]]:
    if action is None:
        return None
    row, col, to_row, to_col = _decode_move(action, board_size)
    piece_entry = board[row, col]
    piece = piece_entry[1] if piece_entry is not None else "?"
    capture_entry = board[to_row, to_col]
    capture = capture_entry[1] if capture_entry is not None else None
    return {
        "actor": actor,
        "piece": piece,
        "from": {"row": row, "col": col},
        "to": {"row": to_row, "col": to_col},
        "capture": capture,
    }


def _format_move(move: Dict[str, Any]) -> str:
    start = f"({move['from']['row'] + 1},{move['from']['col'] + 1})"
    end = f"({move['to']['row'] + 1},{move['to']['col'] + 1})"
    if move.get("capture"):
        return f"{move['actor']} {move['piece']} x {move['capture']} @ {end}"
    return f"{move['actor']} {move['piece']} -> {end}"


def _format_outcome(winner: Optional[str], reason: Optional[str]) -> str:
    if winner is None:
        return "Round ended"
    label = "Draw" if winner == "draw" else f"{winner} wins"
    if reason:
        return f"{label} ({reason})"
    return label


def _resolve_outcome(info: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    outcome = info.get("outcome")
    reason = info.get("outcome_reason") or info.get("draw_reason")
    if outcome == "agent_win":
        return "A1", REASON_LABELS.get(reason, reason)
    if outcome == "agent_loss":
        return "A2", REASON_LABELS.get(reason, reason)
    if outcome == "draw":
        return "draw", REASON_LABELS.get(reason, reason)
    return None, None


@dataclass
class Session:
    grid_size: int
    rounds: int
    opponent_type: str
    agent1_weights: np.ndarray
    agent2_weights: np.ndarray
    seed: int
    env: StrategoEnv
    ensemble: List[torch.nn.Module]
    snapshot: List[dict]
    obs_dim: int
    action_dim: int
    round_index: int = 1
    results: Dict[str, int] = field(
        default_factory=lambda: {"A1": 0, "A2": 0, "draw": 0}
    )
    obs: Optional[np.ndarray] = None
    legal_mask: Optional[np.ndarray] = None
    done: bool = False
    move_count: int = 0
    last_winner: Optional[str] = None
    last_reason: Optional[str] = None
    starting_player: int = 0
    agent_top: bool = False
    behavior_history: Dict[str, Dict[str, List[float]]] = field(
        default_factory=lambda: {
            "wins": {"A1": [], "A2": []},
            "capture": {"A1": [], "A2": []},
            "efficiency": {"A1": [], "A2": []},
        }
    )
    episode_captures_a1: int = 0
    episode_captures_a2: int = 0
    episode_moves_a1: int = 0
    episode_moves_a2: int = 0
    pending_opponent_move: Optional[Dict[str, Any]] = None
    pending_board: Optional[List[List[Optional[str]]]] = None
    pending_outcome: Optional[Tuple[Optional[str], Optional[str]]] = None
    awaiting_user_move: bool = False

    def reset_round(self) -> Dict[str, Any]:
        self.last_winner = None
        self.last_reason = None
        self.move_count = 0
        self.episode_captures_a1 = 0
        self.episode_captures_a2 = 0
        self.episode_moves_a1 = 0
        self.episode_moves_a2 = 0
        self.pending_opponent_move = None
        self.pending_board = None
        self.pending_outcome = None
        self.awaiting_user_move = False
        rng = random.Random(self.seed + self.round_index * 991)
        self.agent_top = bool(rng.getrandbits(1))
        if self.opponent_type == "user":
            self.starting_player = 0
        else:
            self.starting_player = rng.choice([0, 1])
        reset_seed = rng.randrange(1_000_000)
        info: Dict[str, Any] = {}
        obs, info = self.env.reset(
            seed=reset_seed,
            agent_top=self.agent_top,
            starting_player=self.starting_player,
        )
        self.obs = obs
        self.legal_mask = info.get("legal_mask")
        self.done = bool(info.get("episode_done"))
        if info.get("opponent_action") is not None:
            self.episode_moves_a2 += 1
            if info.get("opponent_capture"):
                self.episode_captures_a2 += 1
        if self.done:
            self.last_winner, self.last_reason = _resolve_outcome(info)
            if self.last_winner is not None:
                self.results[self.last_winner] += 1
            self._record_behavior(self.last_winner, self.last_reason)
        return info

    def ensure_opponent_policy(self) -> None:
        if self.opponent_type == "trained":
            policy = make_opponent_policy(
                self.snapshot,
                self.obs_dim,
                self.action_dim,
                self.agent2_weights,
                DEVICE,
                boltz_agg=True,
            )
            self.env.set_opponent_policy(policy)
        else:
            self.env.set_opponent_policy(None)

    def _record_behavior(self, winner: Optional[str], reason: Optional[str]) -> None:
        a1_win = 1.0 if winner == "A1" else 0.0
        a2_win = 1.0 if winner == "A2" else 0.0
        self.behavior_history["wins"]["A1"].append(a1_win)
        self.behavior_history["wins"]["A2"].append(a2_win)
        self.behavior_history["capture"]["A1"].append(float(self.episode_captures_a1))
        self.behavior_history["capture"]["A2"].append(float(self.episode_captures_a2))
        self.behavior_history["efficiency"]["A1"].append(float(self.episode_moves_a1))
        self.behavior_history["efficiency"]["A2"].append(float(self.episode_moves_a2))


class StrategoHandler(BaseHTTPRequestHandler):
    server_version = "StrategoInference/1.0"
    def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            self._send_json(400, {"error": "Invalid JSON payload."})
            return

        if self.path == "/api/session/start":
            self._handle_start(data)
            return
        if self.path == "/api/session/step":
            self._handle_step(data)
            return
        if self.path == "/api/session/weights":
            self._handle_weights(data)
            return

        self._send_json(404, {"error": "Unknown endpoint."})

    def _handle_start(self, data: Dict[str, Any]) -> None:
        try:
            grid_size = int(data.get("gridSize"))
            rounds = int(data.get("rounds", 1))
            opponent_type = str(data.get("opponentType", "naive"))
        except (TypeError, ValueError):
            self._send_json(400, {"error": "Invalid configuration."})
            return

        if grid_size not in PIECE_SETS:
            self._send_json(400, {"error": "Unsupported grid size."})
            return

        bundle = _load_agent_bundle(grid_size)
        agent1 = data.get("agent1", {})
        agent2 = data.get("agent2", {})
        weights1 = np.array(
            [
                float(agent1.get("flag", 0.0)),
                float(agent1.get("capture", 0.0)),
                float(agent1.get("efficiency", 0.0)),
            ],
            dtype=np.float32,
        )
        weights2 = np.array(
            [
                float(agent2.get("flag", 0.0)),
                float(agent2.get("capture", 0.0)),
                float(agent2.get("efficiency", 0.0)),
            ],
            dtype=np.float32,
        )
        weights1 = prepare_weights(weights1, boltz_agg=True)
        weights2 = prepare_weights(weights2, boltz_agg=True)

        env = StrategoEnv(
            board_size=grid_size,
            max_steps=grid_size * grid_size * 6,
            seed=None,
            piece_order=PIECE_SETS[grid_size],
        )
        rounds = max(1, min(MAX_ROUNDS, rounds))
        session = Session(
            grid_size=grid_size,
            rounds=rounds,
            opponent_type=opponent_type,
            agent1_weights=weights1,
            agent2_weights=weights2,
            seed=random.randrange(1_000_000),
            env=env,
            ensemble=bundle["ensemble"],
            snapshot=bundle["snapshot"],
            obs_dim=bundle["obs_dim"],
            action_dim=bundle["action_dim"],
        )
        session.env.set_manual_opponent(opponent_type == "user")
        session.ensure_opponent_policy()
        session.reset_round()
        session_id = uuid.uuid4().hex
        SESSIONS[session_id] = session

        payload = {"sessionId": session_id, "state": _build_state(session, events=[])}
        self._send_json(200, payload)

    def _handle_step(self, data: Dict[str, Any]) -> None:
        session_id = str(data.get("sessionId", ""))
        session = SESSIONS.get(session_id)
        if session is None:
            self._send_json(404, {"error": "Session not found."})
            return
        user_action = data.get("userAction")
        user_action_int = None
        if user_action is not None:
            try:
                user_action_int = int(user_action)
            except (TypeError, ValueError):
                self._send_json(400, {"error": "Invalid user action."})
                return

        try:
            state = _advance_session(session, user_action=user_action_int)
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        payload = {"sessionId": session_id, "state": state}
        self._send_json(200, payload)

    def _handle_weights(self, data: Dict[str, Any]) -> None:
        session_id = str(data.get("sessionId", ""))
        session = SESSIONS.get(session_id)
        if session is None:
            self._send_json(404, {"error": "Session not found."})
            return
        agent1_payload = data.get("agent1")
        agent2_payload = data.get("agent2")
        try:
            weights1 = _coerce_weights(agent1_payload, session.agent1_weights)
            weights2 = _coerce_weights(agent2_payload, session.agent2_weights)
        except (TypeError, ValueError):
            self._send_json(400, {"error": "Invalid weights payload."})
            return
        session.agent1_weights = prepare_weights(weights1, boltz_agg=True)
        session.agent2_weights = prepare_weights(weights2, boltz_agg=True)
        session.ensure_opponent_policy()
        self._send_json(200, {"status": "ok"})


def _advance_session(
    session: Session, *, user_action: Optional[int] = None
) -> Dict[str, Any]:
    if session.opponent_type == "user":
        return _advance_session_user(session, user_action=user_action)

    events: List[str] = []
    moves: List[Dict[str, Any]] = []

    if session.pending_opponent_move is not None:
        opponent_move = session.pending_opponent_move
        moves.append(opponent_move)
        events.append(_format_move(opponent_move))
        if session.pending_outcome is not None:
            winner, reason = session.pending_outcome
            session.last_winner = winner
            session.last_reason = reason
            if winner:
                session.results[winner] += 1
            session._record_behavior(session.last_winner, session.last_reason)
            session.done = True
            events.append(
                f"Round {session.round_index}: {_format_outcome(session.last_winner, session.last_reason)}"
            )
        session.pending_opponent_move = None
        session.pending_board = None
        session.pending_outcome = None
        complete = session.done and session.round_index >= session.rounds
        return _build_state(session, events=events, moves=moves, complete=complete)

    if session.done:
        if session.round_index >= session.rounds:
            return _build_state(session, events=events, moves=moves, complete=True)
        session.round_index += 1
        session.reset_round()
        events.append(f"Round {session.round_index} ready")
        return _build_state(session, events=events, moves=moves)

    if session.legal_mask is None or not session.legal_mask.any():
        session.done = True
        session.last_winner = "A2"
        session.last_reason = "No legal moves"
        session.results["A2"] += 1
        session._record_behavior(session.last_winner, session.last_reason)
        events.append(
            f"Round {session.round_index}: {_format_outcome(session.last_winner, session.last_reason)}"
        )
        return _build_state(session, events=events, moves=moves)

    board_before = session.env.export_board_state()
    action, _, _ = get_aggregated_action_and_probs(
        session.obs,
        session.legal_mask,
        session.ensemble,
        session.agent1_weights,
        DEVICE,
        boltz_agg=True,
    )
    agent_move = _describe_move(action, board_before, session.grid_size, "A1")
    if agent_move:
        moves.append(agent_move)
        events.append(_format_move(agent_move))

    next_obs, _, terminated, truncated, info = session.env.step(action)
    opponent_action = info.get("opponent_action")
    board_after_agent = info.get("board_after_agent")
    opponent_move = None
    if opponent_action is not None and board_after_agent is not None:
        opponent_move = _describe_move(
            opponent_action,
            board_after_agent,
            session.grid_size,
            "A2",
        )

    session.obs = next_obs
    session.legal_mask = info.get("legal_mask")
    session.move_count += 1
    session.episode_moves_a1 += 1
    if info.get("last_capture"):
        session.episode_captures_a1 += 1
    if info.get("opponent_action") is not None:
        session.episode_moves_a2 += 1
        if info.get("opponent_capture"):
            session.episode_captures_a2 += 1

    if opponent_move is not None and board_after_agent is not None:
        session.pending_opponent_move = opponent_move
        session.pending_board = _serialize_board(board_after_agent)
        if terminated or truncated:
            winner, reason = _resolve_outcome(info)
            if truncated and winner is None:
                winner = "draw"
                reason = REASON_LABELS.get("max_steps", "Turn limit reached")
            session.pending_outcome = (winner, reason)
        return _build_state(
            session,
            events=events,
            moves=moves,
            board_override=session.pending_board,
            winner_override=None,
            reason_override=None,
            complete=False,
        )

    if terminated or truncated:
        session.done = True
        winner, reason = _resolve_outcome(info)
        if truncated and winner is None:
            winner = "draw"
            reason = REASON_LABELS.get("max_steps", "Turn limit reached")
        session.last_winner = winner
        session.last_reason = reason
        if winner:
            session.results[winner] += 1
        session._record_behavior(session.last_winner, session.last_reason)
        events.append(
            f"Round {session.round_index}: {_format_outcome(session.last_winner, session.last_reason)}"
        )

    return _build_state(session, events=events, moves=moves)


def _build_state(
    session: Session,
    *,
    events: List[str],
    moves: Optional[List[Dict[str, Any]]] = None,
    complete: bool = False,
    board_override: Optional[List[List[Optional[str]]]] = None,
    winner_override: Optional[Optional[str]] = None,
    reason_override: Optional[Optional[str]] = None,
    user_moves: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    board = session.env.export_board_state()
    history = session.behavior_history
    episodes = len(history["wins"]["A1"])
    per_side_pieces = len(PIECE_SETS[session.grid_size])
    if episodes == 0:
        averages = {
            "wins": {"A1": 0.0, "A2": 0.0},
            "capture": {"A1": 0.0, "A2": 0.0},
            "efficiency": {"A1": 0.0, "A2": 0.0},
        }
    else:
        averages = {
            "wins": {
                "A1": float(sum(history["wins"]["A1"])) / float(episodes),
                "A2": float(sum(history["wins"]["A2"])) / float(episodes),
            },
            "capture": {
                "A1": float(sum(history["capture"]["A1"]))
                / float(episodes * per_side_pieces),
                "A2": float(sum(history["capture"]["A2"]))
                / float(episodes * per_side_pieces),
            },
            "efficiency": {
                "A1": float(sum(history["efficiency"]["A1"])) / float(episodes),
                "A2": float(sum(history["efficiency"]["A2"])) / float(episodes),
            },
        }
    return {
        "gridSize": session.grid_size,
        "round": session.round_index,
        "totalRounds": session.rounds,
        "results": session.results,
        "board": board_override if board_override is not None else _serialize_board(board),
        "moveCount": session.move_count,
        "winner": session.last_winner if winner_override is None else winner_override,
        "reason": session.last_reason if reason_override is None else reason_override,
        "events": events,
        "lastMoves": moves or [],
        "metricHistory": history,
        "metricAverages": averages,
        "startingPlayer": "A1" if session.starting_player == 0 else "A2",
        "agentTop": session.agent_top,
        "complete": complete,
        "awaitingUserMove": session.awaiting_user_move,
        "userMoves": user_moves or [],
    }


def _advance_session_user(
    session: Session, *, user_action: Optional[int]
) -> Dict[str, Any]:
    events: List[str] = []
    moves: List[Dict[str, Any]] = []

    if session.awaiting_user_move:
        if user_action is None:
            return _build_state(
                session,
                events=events,
                moves=moves,
                user_moves=_collect_user_moves(session),
            )
        mask = session.env.legal_action_mask(player=1)
        if user_action < 0 or user_action >= mask.size or not mask[user_action]:
            raise ValueError("Illegal user action selected.")
        board_before = session.env.export_board_state()
        next_obs, terminated, truncated, info = session.env.step_opponent(user_action)
        user_move = _describe_move(user_action, board_before, session.grid_size, "A2")
        if user_move:
            moves.append(user_move)
            events.append(_format_move(user_move))
        session.obs = next_obs
        session.legal_mask = info.get("legal_mask")
        session.episode_moves_a2 += 1
        if info.get("opponent_capture"):
            session.episode_captures_a2 += 1
        session.awaiting_user_move = False
        if terminated or truncated:
            winner, reason = _resolve_outcome(info)
            if truncated and winner is None:
                winner = "draw"
                reason = REASON_LABELS.get("max_steps", "Turn limit reached")
            session.last_winner = winner
            session.last_reason = reason
            if winner:
                session.results[winner] += 1
            session._record_behavior(session.last_winner, session.last_reason)
            session.done = True
            events.append(
                f"Round {session.round_index}: {_format_outcome(session.last_winner, session.last_reason)}"
            )
            complete = session.done and session.round_index >= session.rounds
            return _build_state(
                session,
                events=events,
                moves=moves,
                complete=complete,
            )
        return _build_state(session, events=events, moves=moves)

    if session.done:
        if session.round_index >= session.rounds:
            return _build_state(session, events=events, moves=moves, complete=True)
        session.round_index += 1
        session.reset_round()
        events.append(f"Round {session.round_index} ready")
        return _build_state(session, events=events, moves=moves)

    if session.legal_mask is None or not session.legal_mask.any():
        session.done = True
        session.last_winner = "A2"
        session.last_reason = "No legal moves"
        session.results["A2"] += 1
        session._record_behavior(session.last_winner, session.last_reason)
        events.append(
            f"Round {session.round_index}: {_format_outcome(session.last_winner, session.last_reason)}"
        )
        return _build_state(session, events=events, moves=moves)

    board_before = session.env.export_board_state()
    action, _, _ = get_aggregated_action_and_probs(
        session.obs,
        session.legal_mask,
        session.ensemble,
        session.agent1_weights,
        DEVICE,
        boltz_agg=True,
    )
    agent_move = _describe_move(action, board_before, session.grid_size, "A1")
    if agent_move:
        moves.append(agent_move)
        events.append(_format_move(agent_move))

    next_obs, _, terminated, truncated, info = session.env.step(action)
    session.obs = next_obs
    session.legal_mask = info.get("legal_mask")
    session.move_count += 1
    session.episode_moves_a1 += 1
    if info.get("last_capture"):
        session.episode_captures_a1 += 1

    if terminated or truncated:
        session.done = True
        winner, reason = _resolve_outcome(info)
        if truncated and winner is None:
            winner = "draw"
            reason = REASON_LABELS.get("max_steps", "Turn limit reached")
        session.last_winner = winner
        session.last_reason = reason
        if winner:
            session.results[winner] += 1
        session._record_behavior(session.last_winner, session.last_reason)
        events.append(
            f"Round {session.round_index}: {_format_outcome(session.last_winner, session.last_reason)}"
        )
        return _build_state(session, events=events, moves=moves)

    if info.get("awaiting_opponent"):
        session.awaiting_user_move = True
        return _build_state(
            session,
            events=events,
            moves=moves,
            user_moves=_collect_user_moves(session),
        )

    return _build_state(session, events=events, moves=moves)


def run_server(host: Optional[str] = None, port: Optional[int] = None) -> None:
    host = host or os.environ.get("HOST", "0.0.0.0")
    port = port or int(os.environ.get("PORT", "8000"))
    server = ThreadingHTTPServer((host, port), StrategoHandler)
    print(f"Stratego inference server running on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    run_server()
