from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple

from env import HordeEnv
from cards.definitions import CARD_POOL
from main import load_agent


@dataclass
class Session:
    env: HordeEnv
    agent1_spec: str
    agent2_spec: str
    agent1_fn: Optional[Any]
    agent2_fn: Optional[Any]
    user_side: Optional[str]
    last_info: Optional[Dict[str, Any]]
    complete: bool
    winner: Optional[str]
    behavior_history: Dict[str, Dict[str, list[float]]]
    pick_counts: Dict[str, Dict[str, Dict[str, int]]]
    action_selection: Dict[str, Any]
    total_rounds: int


SESSIONS: Dict[str, Session] = {}

CARD_NAMES = [card.name for card in CARD_POOL]


def _blank_pick_counts() -> Dict[str, Dict[str, Dict[str, int]]]:
    zeroed = {name: 0 for name in CARD_NAMES}
    return {
        "p0": {"suggested": dict(zeroed), "picked": dict(zeroed)},
        "p1": {"suggested": dict(zeroed), "picked": dict(zeroed)},
    }


def _blank_action_selection() -> Dict[str, Any]:
    def build_cards() -> list[Dict[str, Any]]:
        return [
            {
                "name": name,
                "suggested": 0,
                "picked": 0,
                "pickRate": 0.0,
                "pickGivenSuggested": 0.0,
            }
            for name in CARD_NAMES
        ]

    return {
        "totalRounds": 0,
        "agents": {
            "p0": {"cards": build_cards()},
            "p1": {"cards": build_cards()},
        },
    }


def _update_pick_counts(
    session: Session,
    offers_p0: list[Any],
    offers_p1: list[Any],
    action_p0: int,
    action_p1: int,
) -> None:
    session.total_rounds += 1
    for card in offers_p0:
        name = getattr(card, "name", None)
        if name in session.pick_counts["p0"]["suggested"]:
            session.pick_counts["p0"]["suggested"][name] += 1
    for card in offers_p1:
        name = getattr(card, "name", None)
        if name in session.pick_counts["p1"]["suggested"]:
            session.pick_counts["p1"]["suggested"][name] += 1

    if 0 <= action_p0 < len(offers_p0):
        name = getattr(offers_p0[action_p0], "name", None)
        if name in session.pick_counts["p0"]["picked"]:
            session.pick_counts["p0"]["picked"][name] += 1
    if 0 <= action_p1 < len(offers_p1):
        name = getattr(offers_p1[action_p1], "name", None)
        if name in session.pick_counts["p1"]["picked"]:
            session.pick_counts["p1"]["picked"][name] += 1


def _finalize_action_selection(session: Session) -> None:
    rounds = max(1, session.total_rounds)
    agents: Dict[str, Any] = {}
    for agent in ("p0", "p1"):
        counts = session.pick_counts[agent]
        cards: list[Dict[str, Any]] = []
        for name in CARD_NAMES:
            suggested = counts["suggested"].get(name, 0)
            picked = counts["picked"].get(name, 0)
            cards.append(
                {
                    "name": name,
                    "suggested": suggested,
                    "picked": picked,
                    "pickRate": picked / rounds if rounds else 0.0,
                    "pickGivenSuggested": picked / suggested if suggested else 0.0,
                }
            )
        agents[agent] = {"cards": cards}
    session.action_selection = {"totalRounds": session.total_rounds, "agents": agents}

def _blank_behavior_history() -> Dict[str, Dict[str, list[float]]]:
    return {
        "wins": {"p0": [], "p1": []},
        "rounds": {"p0": [], "p1": []},
    }


def _record_behavior(session: Session, p0_points: int, p1_points: int, rounds_played: int) -> None:
    session.behavior_history["wins"]["p0"].append(float(p0_points))
    session.behavior_history["wins"]["p1"].append(float(p1_points))
    session.behavior_history["rounds"]["p0"].append(float(rounds_played))
    session.behavior_history["rounds"]["p1"].append(float(rounds_played))


def _metric_averages(
    history: Dict[str, Dict[str, list[float]]]
) -> Dict[str, Dict[str, float]]:
    averages: Dict[str, Dict[str, float]] = {}
    for metric, series in history.items():
        episodes = len(series["p0"])
        if episodes == 0:
            averages[metric] = {"p0": 0.0, "p1": 0.0}
        else:
            averages[metric] = {
                "p0": sum(series["p0"]) / episodes,
                "p1": sum(series["p1"]) / episodes,
            }
    return averages


def _parse_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


AGENT_FILES = {
    "easy": "easy_difficulty_agent.pth",
    "medium": "medium_difficulty_agent.pth",
    "hard": "strong_difficulty_agent.pth",
}

AGENT_DIR = os.environ.get("HORDE_AGENT_DIR")


def _agent_path(name: str) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    agent_dir = AGENT_DIR or os.path.join(here, "agents")
    return os.path.join(agent_dir, AGENT_FILES[name])


def _resolve_agent(spec: str, seed: int, env: HordeEnv) -> Optional[Any]:
    if spec == "user":
        return None
    if spec in AGENT_FILES:
        return load_agent(_agent_path(spec), seed, env)
    return load_agent(spec, seed, env)


def _resolve_user_side(agent1: str, agent2: str) -> Optional[str]:
    if agent1 == "user" and agent2 == "user":
        return "both"
    if agent1 == "user":
        return "p0"
    if agent2 == "user":
        return "p1"
    return None


def _build_state(
    session: Session,
    obs: Dict[str, Any],
    awaiting_user_move: bool = False,
    expected_player: Optional[str] = None,
) -> Dict[str, Any]:
    info = session.last_info or {}
    metric_history = session.behavior_history
    return {
        "round": obs.get("round", 0),
        "targetWins": session.env.target_wins,
        "offer": obs.get("offer", []),
        "offerP0": obs.get("offerP0", obs.get("offer", [])),
        "offerP1": obs.get("offerP1", obs.get("offer", [])),
        "score": obs.get("score", {"p0": 0, "p1": 0}),
        "remaining": obs.get("remaining", {}),
        "deckRefreshes": obs.get("deckRefreshes", {"p0": 0, "p1": 0, "total": 0}),
        "log": info.get("log", []),
        "lastOutcome": info.get("outcome"),
        "combatSummary": info.get("combat_summary"),
        "lastPicks": info.get("picks"),
        "complete": session.complete,
        "winner": session.winner,
        "awaitingUserMove": awaiting_user_move,
        "expectedPlayer": expected_player,
        "agents": {"p0": session.agent1_spec, "p1": session.agent2_spec},
        "metricHistory": metric_history,
        "metricAverages": _metric_averages(metric_history),
        "actionSelection": session.action_selection,
    }


def _format_offer(offer: Any) -> str:
    if not offer:
        return "Offer: -"
    parts = []
    for idx, card in enumerate(offer):
        name = getattr(card, "name", None) or card.get("name") if isinstance(card, dict) else None
        parts.append(f"[{idx}] {name or 'Unknown'}")
    return "Offer: " + " | ".join(parts)


def _format_log(raw_log: Any, offer: Any, picks: Dict[str, int]) -> list[str]:
    if not raw_log:
        lines: list[str] = []
        lines.append(_format_offer(offer))
        if picks:
            p0 = picks.get("p0")
            p1 = picks.get("p1")
            lines.append(f"P0 picks: {p0} | P1 picks: {p1}")
        return lines
    return list(raw_log)


def _validate_user_action(action: Any, label: str = "userAction") -> Tuple[Optional[int], Optional[str]]:
    if action is None:
        return None, f"Missing {label} for the human player."
    try:
        value = int(action)
    except (TypeError, ValueError):
        return None, f"{label} must be an integer (0, 1, or 2)."
    if value not in (0, 1, 2):
        return None, f"{label} must be 0, 1, or 2."
    return value, None


def _validate_user_actions(data: Dict[str, Any]) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    a0, err0 = _validate_user_action(data.get("userActionP0"), "userActionP0")
    if err0:
        return None, None, err0
    a1, err1 = _validate_user_action(data.get("userActionP1"), "userActionP1")
    if err1:
        return None, None, err1
    return a0, a1, None


class HordeHandler(BaseHTTPRequestHandler):
    server_version = "HordeSimHTTP/1.0"

    def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        length = _parse_int(self.headers.get("Content-Length"), 0)
        if length <= 0:
            return {}, None
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8")), None
        except json.JSONDecodeError:
            return None, "Invalid JSON body."

    def do_OPTIONS(self) -> None:
        self._send_json(200, {"ok": True})

    def do_POST(self) -> None:
        if self.path == "/api/session/start":
            self._handle_start()
            return
        if self.path == "/api/session/reset":
            self._handle_reset()
            return
        if self.path == "/api/session/step":
            self._handle_step()
            return
        if self.path == "/api/session/weights":
            self._handle_weights()
            return
        self._send_json(404, {"error": "Unknown endpoint."})

    def _handle_start(self) -> None:
        data, error = self._read_json()
        if error:
            self._send_json(400, {"error": error})
            return
        data = data or {}

        rounds = _parse_int(data.get("rounds"), 3)
        seed = _parse_int(data.get("seed"), int(time.time()) % 100000)
        agent1 = str(data.get("agent1", "random")).lower()
        agent2 = str(data.get("agent2", "random")).lower()

        user_side = _resolve_user_side(agent1, agent2)

        env = HordeEnv(seed=None, target_wins=max(1, rounds))
        try:
            agent1_fn = _resolve_agent(agent1, seed + 1, env)
            agent2_fn = _resolve_agent(agent2, seed + 2, env)
        except Exception as exc:  # pragma: no cover - dependency errors reported to client
            self._send_json(400, {"error": f"Failed to load agent: {exc}"})
            return

        obs = env.reset()
        session = Session(
            env=env,
            agent1_spec=agent1,
            agent2_spec=agent2,
            agent1_fn=agent1_fn,
            agent2_fn=agent2_fn,
            user_side=user_side,
            last_info=None,
            complete=False,
            winner=None,
            behavior_history=_blank_behavior_history(),
            pick_counts=_blank_pick_counts(),
            action_selection=_blank_action_selection(),
            total_rounds=0,
        )
        session_id = uuid.uuid4().hex
        SESSIONS[session_id] = session
        state = _build_state(session, obs)
        self._send_json(200, {"sessionId": session_id, "state": state})

    def _handle_step(self) -> None:
        data, error = self._read_json()
        if error:
            self._send_json(400, {"error": error})
            return
        data = data or {}
        session_id = data.get("sessionId")
        if not session_id or session_id not in SESSIONS:
            self._send_json(404, {"error": "Unknown sessionId."})
            return
        session = SESSIONS[session_id]
        obs_p0 = session.env._obs_for("p0")
        obs_p1 = session.env._obs_for("p1")
        obs = obs_p0
        if session.complete:
            self._send_json(200, {"state": _build_state(session, obs)})
            return

        if session.user_side == "both":
            a0, a1, action_error = _validate_user_actions(data)
            if action_error:
                state = _build_state(
                    session,
                    obs,
                    awaiting_user_move=True,
                    expected_player="both",
                )
                self._send_json(200, {"state": state, "error": action_error})
                return
        elif session.user_side:
            action_value, action_error = _validate_user_action(data.get("userAction"))
            if action_error:
                state = _build_state(
                    session,
                    obs,
                    awaiting_user_move=True,
                    expected_player=session.user_side,
                )
                self._send_json(200, {"state": state, "error": action_error})
                return
            if session.user_side == "p0":
                a0 = action_value
                a1 = session.agent2_fn(obs_p1)
            else:
                a0 = session.agent1_fn(obs_p0)
                a1 = action_value
        else:
            a0 = session.agent1_fn(obs_p0)
            a1 = session.agent2_fn(obs_p1)

        offers_p0 = list(session.env.offer_p0)
        offers_p1 = list(session.env.offer_p1)
        _update_pick_counts(session, offers_p0, offers_p1, a0, a1)
        current_offer = offers_p0
        obs, _reward, done, info = session.env.step(a0, a1)
        info = dict(info)
        info["picks"] = {"p0": a0, "p1": a1}
        info["log"] = _format_log(info.get("log"), current_offer, info["picks"])
        session.last_info = info
        if done:
            session.complete = True
            if info.get("p0_points", 0) > info.get("p1_points", 0):
                session.winner = "P0"
            elif info.get("p1_points", 0) > info.get("p0_points", 0):
                session.winner = "P1"
            else:
                session.winner = "DRAW"
            _record_behavior(
                session,
                info.get("p0_points", 0),
                info.get("p1_points", 0),
                obs.get("round", session.env.round_index),
            )
            _finalize_action_selection(session)

        state = _build_state(session, obs)
        self._send_json(200, {"state": state})

    def _handle_reset(self) -> None:
        data, error = self._read_json()
        if error:
            self._send_json(400, {"error": error})
            return
        data = data or {}
        session_id = data.get("sessionId")
        if not session_id or session_id not in SESSIONS:
            self._send_json(404, {"error": "Unknown sessionId."})
            return
        session = SESSIONS[session_id]
        rounds = data.get("rounds")
        if rounds is not None:
            session.env.target_wins = max(1, _parse_int(rounds, session.env.target_wins))
        obs = session.env.reset()
        session.last_info = None
        session.complete = False
        session.winner = None
        state = _build_state(session, obs)
        self._send_json(200, {"state": state})

    def _handle_weights(self) -> None:
        data, error = self._read_json()
        if error:
            self._send_json(400, {"error": error})
            return
        data = data or {}
        session_id = data.get("sessionId")
        if not session_id or session_id not in SESSIONS:
            self._send_json(404, {"error": "Unknown sessionId."})
            return
        session = SESSIONS[session_id]
        agent1 = str(data.get("agent1", session.agent1_spec)).lower()
        agent2 = str(data.get("agent2", session.agent2_spec)).lower()

        user_side = _resolve_user_side(agent1, agent2)
        try:
            agent1_fn = _resolve_agent(agent1, int(time.time()) % 100000, session.env)
            agent2_fn = _resolve_agent(agent2, int(time.time()) % 100000 + 1, session.env)
        except Exception as exc:  # pragma: no cover - dependency errors reported to client
            self._send_json(400, {"error": f"Failed to load agent: {exc}"})
            return

        session.agent1_spec = agent1
        session.agent2_spec = agent2
        session.agent1_fn = agent1_fn
        session.agent2_fn = agent2_fn
        session.user_side = user_side

        obs = session.env._obs()
        state = _build_state(session, obs)
        self._send_json(200, {"state": state})


def run(host: str = "0.0.0.0", port: Optional[int] = None) -> None:
    if port is None:
        port = _parse_int(os.environ.get("PORT"), 8000)
    server = ThreadingHTTPServer((host, port), HordeHandler)
    print(f"Horde server listening on http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run()
