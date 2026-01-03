from __future__ import annotations
import argparse
import os
import random
import time
from collections import Counter
from typing import Callable, Dict, Any, List

from env import HordeEnv


def pick_human(obs: Dict[str, Any]) -> int:
    offer = obs["offer"]
    remaining = obs.get("remaining", {})
    remaining_str = ""
    if remaining:
        remaining_str = f" ({remaining.get('p0', 0)}/{remaining.get('total', 0)} cards remaining)"
    print(f"\nChoose a card:{remaining_str}")
    for i, c in enumerate(offer):
        ab = "-" if c["ability"] is None else f'{c["ability"]["key"]}{c["ability"]["params"]}'
        print(f"  [{i}] {c['name']} ({c['rarity']}) ATK={c['attack']} HP={c['health']} AB={ab}")
    while True:
        s = input("Enter 0, 1, or 2: ").strip()
        if s in ("0", "1", "2"):
            return int(s)
        print("Invalid input.")


def _load_policy_from_checkpoint(path: str, seed: int, env: HordeEnv) -> Callable[[Dict[str, Any]], int]:
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise SystemExit("PyTorch is required to load a trained agent.") from exc

    if not os.path.isfile(path):
        raise SystemExit(f"Agent file not found: {path}")

    checkpoint = torch.load(path, map_location="cpu")
    input_size = int(checkpoint.get("input_size", env.observation_size()))
    hidden_size = int(checkpoint.get("hidden_size", 64))
    if input_size != env.observation_size():
        raise SystemExit(
            f"Agent expects input size {input_size}, but env provides {env.observation_size()}."
        )

    class PolicyNet(nn.Module):
        def __init__(self, input_size: int, hidden_size: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 3),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    policy = PolicyNet(input_size, hidden_size)
    policy.load_state_dict(checkpoint["model_state"])
    policy.eval()
    generator = torch.Generator().manual_seed(seed)

    def act(obs: Dict[str, Any]) -> int:
        obs_vec = env.encode_obs(obs)
        obs_tensor = torch.tensor(obs_vec, dtype=torch.float32)
        with torch.no_grad():
            logits = policy(obs_tensor)
            probs = torch.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1, generator=generator)
            return int(action.item())

    return act


def load_agent(spec: str, seed: int, env: HordeEnv) -> Callable[[Dict[str, Any]], int]:
    if spec.lower() == "random":
        rng = random.SystemRandom()
        return lambda _obs: rng.randrange(3)
    return _load_policy_from_checkpoint(spec, seed, env)


def render_log(log_lines: List[str], delay: float) -> None:
    for line in log_lines:
        print(line)
        time.sleep(delay)


def run_user_vs_agent(seed: int, agent_spec: str, delay: float) -> None:
    env = HordeEnv(seed=seed)
    agent = load_agent(agent_spec, seed + 1, env)
    obs_p0 = env.reset()
    obs_p1 = env._obs_for("p1")

    print("=== HORDE: User vs Agent ===")
    while True:
        print(f"\nScore: You={obs_p0['score']['p0']}  Agent={obs_p0['score']['p1']}")
        a0 = pick_human(obs_p0)
        a1 = agent(obs_p1)

        obs_p0, _reward, done, info = env.step(a0, a1)
        obs_p1 = env._obs_for("p1")
        render_log(info["log"], delay)

        if done:
            if info["p0_points"] > info["p1_points"]:
                print("\nMATCH RESULT: You win!")
            else:
                print("\nMATCH RESULT: Agent wins!")
            break


def _summary_header(agent1: str, agent2: str, episodes: int) -> None:
    print("=== HORDE: Agent vs Agent Summary ===")
    print(f"Agent 1:  {agent1}")
    print(f"Agent 2:  {agent2}")
    print(f"Episodes: {episodes}")


def _print_top_picks(counter: Counter, label: str) -> None:
    print(f"{label} top picks:")
    for name, count in counter.most_common(5):
        print(f"  {name}: {count}")


def run_agent_vs_agent(
    seed: int,
    episodes: int,
    agent1_spec: str,
    agent2_spec: str,
    instant: bool,
    delay: float,
) -> None:
    env = HordeEnv(seed=seed)
    agent1 = load_agent(agent1_spec, seed + 1, env)
    agent2 = load_agent(agent2_spec, seed + 2, env)

    p0_wins = 0
    p1_wins = 0
    draws = 0
    picks0: Counter = Counter()
    picks1: Counter = Counter()

    for ep in range(1, episodes + 1):
        obs_p0 = env.reset()
        obs_p1 = env._obs_for("p1")
        if not instant:
            print(f"\n=== MATCH {ep} ===")
        while True:
            if not instant:
                print(f"\nScore: P0={obs_p0['score']['p0']}  P1={obs_p0['score']['p1']}")
            a0 = agent1(obs_p0)
            a1 = agent2(obs_p1)
            picks0[obs_p0["offer"][a0]["name"]] += 1
            picks1[obs_p1["offer"][a1]["name"]] += 1

            obs_p0, _reward, done, info = env.step(a0, a1)
            obs_p1 = env._obs_for("p1")
            if not instant:
                render_log(info["log"], delay)
            if done:
                if info["p0_points"] > info["p1_points"]:
                    p0_wins += 1
                elif info["p1_points"] > info["p0_points"]:
                    p1_wins += 1
                else:
                    draws += 1
                if not instant:
                    if info["p0_points"] > info["p1_points"]:
                        print("\nMATCH RESULT: P0 wins!")
                    elif info["p1_points"] > info["p0_points"]:
                        print("\nMATCH RESULT: P1 wins!")
                    else:
                        print("\nMATCH RESULT: Draw!")
                break

    if instant:
        _summary_header(agent1_spec, agent2_spec, episodes)
        total = max(1, p0_wins + p1_wins + draws)
        print(f"P0 win rate: {p0_wins / total:.3f}")
        print(f"P1 win rate: {p1_wins / total:.3f}")
        print(f"Draw rate:   {draws / total:.3f}")
        _print_top_picks(picks0, "Agent 1")
        _print_top_picks(picks1, "Agent 2")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["user_vs_agent", "agent_vs_agent"], default="user_vs_agent")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--agent", type=str, default="random")
    parser.add_argument("--agent1", type=str, default="random")
    parser.add_argument("--agent2", type=str, default="random")
    parser.add_argument("--instant", action="store_true")
    parser.add_argument("--delay", type=float, default=0.25)
    args = parser.parse_args()

    if args.mode == "user_vs_agent":
        run_user_vs_agent(args.seed, args.agent, args.delay)
    else:
        if args.episodes is None:
            episodes = 50 if args.instant else 1
        else:
            episodes = max(1, args.episodes)
        run_agent_vs_agent(
            args.seed,
            episodes,
            args.agent1,
            args.agent2,
            args.instant,
            args.delay,
        )
