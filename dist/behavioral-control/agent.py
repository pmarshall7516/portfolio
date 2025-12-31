#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Patrick Marshall
# simple-stratego Behavioral Actor-Critic Training Script
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Import useful libraries
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
from __future__ import annotations  # enable postponed evaluation of annotations for cleaner type hints
import argparse  # parse command line arguments so we can run experiments from the CLI
import copy  # deepcopy tensors and model weights when building opponent snapshots
import csv  # log training and evaluation results to disk for later analysis
import math  # numerical helpers (e.g., exponentiation for importance sampling)
import os  # handle filesystem interactions such as ensuring output directories exist
import random  # seed python-level randomness and perform opponent sampling
from dataclasses import dataclass  # structure per-step rollout data cleanly
from pathlib import Path  # create directories and manage file paths in a portable way
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple  # shared type aliases
import numpy as np  # numerical operations, RNG for evaluation metrics
import torch  # main deep learning framework used for actor and critic networks
from torch import Tensor, optim  # tensor aliases and optimizers for weight updates
from env import StrategoEnv  # simplified Stratego environment defined in env.py
from models import MaskedCategorical, make_ensemble  # neural network modules and masked distribution helper

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Define constants that reflect the reward decomposition described in the paper
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
NUM_BEHAVIORS = 2  # we track two behavior indicators: piece capture and movement efficiency
NUM_LEARNERS = NUM_BEHAVIORS + 1  # the ensemble contains learners for the primary reward plus each behavior
DEFAULT_M_TRAIN = np.array([0.6, 0.3, 0.1], dtype=np.float32)  # default scalarization weights used during training


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Data structure for storing rollout information
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@dataclass
class StepRecord:
    """Per-time-step record used to compute n-step updates for the behavioral actor-critic algorithm."""

    obs: np.ndarray  # flattened observation at time t
    action: int  # discrete action index chosen at time t
    reward_vec: np.ndarray  # vector reward aligned with [primary, b_capture, b_efficiency]
    done: bool  # termination flag for the environment transition
    legal_mask: np.ndarray  # boolean mask of legal actions at time t
    log_pi: List[float]  # log-probabilities of the selected action under each learner policy
    log_C: float  # log-probability of the selected action under the behavior (aggregated) policy
    eligibility: float  # eligibility trace scalar carried through the rollout


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Utility helpers for reproducibility and file management
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
def set_global_seed(seed: int) -> None:
    """Seed every RNG we rely on so experiments mirror the methodology in the paper."""

    random.seed(seed)  # seed python RNG
    np.random.seed(seed)  # seed numpy RNG
    torch.manual_seed(seed)  # seed CPU torch RNG
    if torch.cuda.is_available():  # seed CUDA RNGs when running on GPU to keep behavior deterministic
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    """Create a directory (including parents) if needed for logging artifacts."""

    Path(path).mkdir(parents=True, exist_ok=True)


def project_to_simplex(weights: np.ndarray) -> np.ndarray:
    """Project a weight vector onto the probability simplex."""

    v = np.asarray(weights, dtype=np.float64)
    if v.ndim != 1:
        raise ValueError("Scalarization weights must be a flat vector.")
    if np.allclose(v.sum(), 1.0) and np.all(v >= 0.0):
        projected = v
    else:
        n = v.shape[0]
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho_candidates = u + (1.0 - cssv) / np.arange(1, n + 1)
        rho = int(np.where(rho_candidates > 0)[0][-1])
        theta = (cssv[rho] - 1.0) / (rho + 1)
        projected = np.clip(v - theta, a_min=0.0, a_max=None)
    total = projected.sum()
    if total <= 0.0 or not np.isfinite(total):
        raise ValueError("Unable to normalize scalarization weights onto the simplex.")
    projected = projected / total
    return projected.astype(np.float32)


def ensure_simplex(weights: np.ndarray) -> np.ndarray:
    """Return a float32 simplex vector, raising on invalid inputs."""

    projected = project_to_simplex(weights)
    if np.any(projected < -1e-6) or np.any(projected > 1.0 + 1e-6):
        raise ValueError("Scalarization weights must lie in [0, 1] after projection.")
    return projected


def prepare_weights(weights: np.ndarray, boltz_agg: bool, *, force_simplex: bool = False) -> np.ndarray:
    """Normalize or pass through scalarization weights based on aggregation mode."""

    weights_arr = np.asarray(weights, dtype=np.float32)
    if boltz_agg and not force_simplex:
        return weights_arr
    return ensure_simplex(weights_arr)


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Aggregation and action selection routines
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
def _aggregate_policies_linear(
    per_policy_probs: Sequence[Tensor],
    legal_mask: Tensor,
    weights: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """Blend per-learner distributions linearly, enforcing coverage on legal actions."""

    stacked = torch.stack(per_policy_probs, dim=0)  # shape: [num_learners, num_actions]
    raw = torch.mv(stacked.transpose(0, 1), weights)  # aggregate weighted probabilities over learners
    raw = torch.clamp(raw, min=0.0)  # maintain non-negative support
    raw = raw * legal_mask  # zero out probabilities on illegal actions
    legal_indices = torch.nonzero(legal_mask > 0, as_tuple=False).squeeze(-1)  # identify all valid moves
    if legal_indices.numel() == 0:
        raise ValueError("No legal actions available for aggregation.")
    raw = raw + eps * legal_mask
    total = raw.sum()  # compute normalization constant over legal actions
    if total <= 0 or torch.isnan(total):
        probs = torch.zeros_like(raw)
        probs[legal_indices] = 1.0 / legal_indices.numel()
        return probs
    return raw / total  # normalize to obtain the final behavior distribution C(a|s)


def _aggregate_policies_boltzmann(
    per_policy_probs: Sequence[Tensor],
    legal_mask: Tensor,
    weights: Tensor,
) -> Tensor:
    """Boltzmann aggregation over learner policies supporting unconstrained weights."""

    stacked = torch.stack(per_policy_probs, dim=0)  # shape: [num_learners, num_actions]
    logits = torch.mv(stacked.transpose(0, 1), weights)
    legal_indices = torch.nonzero(legal_mask > 0, as_tuple=False).squeeze(-1)
    if legal_indices.numel() == 0:
        raise ValueError("No legal actions available for aggregation.")
    valid_logits = logits[legal_indices]
    max_logit = torch.max(valid_logits)
    shifted = valid_logits - max_logit
    exp_vals = torch.exp(shifted)
    denom = exp_vals.sum()
    probs = torch.zeros_like(logits)
    if denom <= 0 or torch.isnan(denom):
        probs[legal_indices] = 1.0 / legal_indices.numel()
        return probs
    probs[legal_indices] = exp_vals / denom
    return probs


def aggregate_policies(
    per_policy_probs: Sequence[Tensor],
    legal_mask: Tensor,
    weights: Tensor,
    *,
    boltz_agg: bool,
    eps: float = 1e-8,
) -> Tensor:
    """Dispatch aggregation to either the linear or Boltzmann combiner."""

    if boltz_agg:
        return _aggregate_policies_boltzmann(per_policy_probs, legal_mask, weights)
    return _aggregate_policies_linear(per_policy_probs, legal_mask, weights, eps=eps)


def _classify_episode_outcome(
    info: Dict[str, Any],
    *,
    truncated: bool,
    last_reward_vec: Optional[np.ndarray],
) -> str:
    """Map environment termination metadata to a descriptive outcome label."""

    outcome = info.get("outcome")
    reason = info.get("outcome_reason")
    draw_reason = info.get("draw_reason")

    if outcome == "agent_win":
        if reason == "agent_flag":
            return "win_flag_capture"
        if reason == "opponent_no_moves":
            return "win_piece_capture"
        return "win_other"

    if outcome == "agent_loss":
        if reason == "opponent_flag":
            return "loss_flag_taken"
        if reason == "agent_no_moves":
            return "loss_no_pieces"
        return "loss_other"

    if outcome == "draw" or draw_reason is not None or truncated:
        if draw_reason in ("only_flags", "both_players_no_moves"):
            return "tie_no_pieces"
        if truncated and draw_reason is None:
            return "tie_max_steps"
        return "tie_other"

    if last_reward_vec is not None and last_reward_vec[0] >= 1.0:
        return "win_other"
    return "other"


def get_aggregated_action_and_probs(
    obs: np.ndarray,
    legal_mask: np.ndarray,
    ensemble: Sequence[torch.nn.Module],
    weights: np.ndarray,
    device: torch.device,
    *,
    boltz_agg: bool,
) -> Tuple[int, Tensor, List[Tensor]]:
    """Sample an action from the aggregated distribution and return per-learner probabilities for logging."""

    weights_arr = prepare_weights(weights, boltz_agg)
    obs_tensor = torch.from_numpy(obs).to(device=device, dtype=torch.float32)  # lift observation onto device
    mask_tensor = torch.from_numpy(legal_mask.astype(np.float32)).to(device=device)  # convert mask to tensor form
    weights_tensor = torch.from_numpy(weights_arr.astype(np.float32)).to(device=device)  # align weights with device
    per_probs: List[Tensor] = []  # collect each learner’s masked probability vector

    prev_training_flags = [model.training for model in ensemble]  # remember original train/eval modes
    for model in ensemble:
        model.eval()  # ensure deterministic forward passes during data collection

    with torch.no_grad():  # avoid tracking gradients while acting in the environment
        for model in ensemble:
            outputs = model(obs_tensor)  # forward pass through actor-critic network
            logits = outputs["logits"].to(device=device, dtype=torch.float32)
            dist = MaskedCategorical(logits, mask_tensor)  # create masked distribution per learner
            per_probs.append(dist.probs)  # collect normalized probabilities for each learner
        aggregate = aggregate_policies(per_probs, mask_tensor, weights_tensor, boltz_agg=boltz_agg)  # build behavior policy
        action = int(torch.multinomial(aggregate, num_samples=1).item())  # sample an action index

    for model, was_training in zip(ensemble, prev_training_flags):
        model.train(was_training)  # restore original training modes so subsequent updates are correct

    return action, aggregate, per_probs  # expose sampled action and distribution diagnostics to caller


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Snapshotting helpers to support self-play opponents
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
def snapshot_ensemble(ensemble: Sequence[torch.nn.Module]) -> List[Dict[str, Tensor]]:
    """Capture and return a deep copy of every learner’s state dict to freeze their policy."""

    return [copy.deepcopy(model.state_dict()) for model in ensemble]  # deep copies preserve weights for frozen opponents


def make_opponent_policy(
    snapshot: Sequence[Dict[str, Tensor]],
    obs_dim: int,
    action_dim: int,
    weights: np.ndarray,
    device: torch.device,
    *,
    boltz_agg: bool,
) -> Callable[[np.ndarray, np.ndarray], Optional[int]]:
    """Instantiate a frozen opponent policy constructed from a saved ensemble snapshot."""

    opponent_models = make_ensemble(NUM_LEARNERS, obs_dim, action_dim)  # new models with identical architecture
    for model, state in zip(opponent_models, snapshot):
        model.load_state_dict(state)  # load saved weights
        model.to(device)
        model.eval()
    prepared_weights = prepare_weights(weights, boltz_agg)
    weights_tensor = torch.from_numpy(prepared_weights).to(device)  # freeze scalarization weights on device

    def policy(obs: np.ndarray, legal_mask: np.ndarray) -> Optional[int]:
        """Callable used by the environment to select actions for the opponent."""

        obs_tensor = torch.from_numpy(obs).to(device=device, dtype=torch.float32)
        mask_tensor = torch.from_numpy(legal_mask.astype(np.float32)).to(device=device)
        with torch.no_grad():
            per_probs = []
            for model in opponent_models:
                outputs = model(obs_tensor)
                logits = outputs["logits"]
                dist = MaskedCategorical(logits, mask_tensor)
                per_probs.append(dist.probs)
            aggregate = aggregate_policies(per_probs, mask_tensor, weights_tensor, boltz_agg=boltz_agg)  # rebuild ensemble policy distribution
            legal_indices = torch.nonzero(mask_tensor > 0, as_tuple=False).squeeze(-1)  # legal action indices for sampling guard
            if legal_indices.numel() == 0:
                return None  # malformed mask; fall back handled outside
            action_tensor = torch.multinomial(aggregate, num_samples=1)  # sample opponent move from aggregate
            return int(action_tensor.item())

    return policy


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# n-step target computation with importance sampling corrections
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
def compute_n_step_targets(
    trajectory: Sequence[StepRecord],
    tau: int,
    n_step: int,
    gamma: float,
    learner_idx: int,
    ensemble: Sequence[torch.nn.Module],
    device: torch.device,
) -> Tuple[float, float]:
    """Compute discounted returns and importance ratios as described in Algorithm 1 of the paper."""

    T = len(trajectory)  # total number of recorded steps
    horizon = min(n_step, T - tau)  # actual rollout window for this update
    j_end = tau + horizon - 1  # last index included in the product

    discounted = 0.0  # running total for discounted returns over the n-step window
    for offset, step in enumerate(trajectory[tau : tau + horizon]):
        discounted += (gamma ** offset) * float(step.reward_vec[learner_idx])  # weight learner-specific reward by gamma power

    if tau + n_step < T:
        next_obs = torch.from_numpy(trajectory[tau + n_step].obs).to(device=device, dtype=torch.float32)  # bootstrap from subsequent state
        with torch.no_grad():
            next_val = ensemble[learner_idx](next_obs)["values"]  # bootstrap from critic
            if isinstance(next_val, Tensor):
                next_val_scalar = float(next_val.squeeze().item())
            else:
                next_val_scalar = float(next_val)
        discounted += (gamma ** n_step) * next_val_scalar  # append bootstrapped critic value at horizon boundary

    log_rho = 0.0  # accumulation buffer for importance sampling correction in log-space
    for j in range(tau, j_end + 1):
        log_rho += trajectory[j].log_pi[learner_idx] - trajectory[j].log_C  # accumulate log-importance ratios
    log_rho = float(np.clip(log_rho, -10.0, 10.0))  # mild clipping to stabilize exponentiation
    rho = math.exp(log_rho)  # convert back to multiplicative ratio

    return discounted, rho


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Parameter update routine for all learners
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
def perform_update_step(
    trajectory: Sequence[StepRecord],
    tau: int,
    ensemble: Sequence[torch.nn.Module],
    actor_opts: Sequence[optim.Optimizer],
    critic_opts: Sequence[optim.Optimizer],
    n_step: int,
    gamma: float,
    device: torch.device,
) -> List[float]:
    """Update actors and critics using the off-policy n-step objectives defined in the paper."""

    rho_values: List[float] = []  # collect importance ratios for logging
    obs_tensor = torch.from_numpy(trajectory[tau].obs).to(device=device, dtype=torch.float32)  # state at update index tau
    legal_mask_tensor = torch.from_numpy(trajectory[tau].legal_mask.astype(np.float32)).to(device)  # mask frozen from rollout
    action_tensor = torch.tensor(trajectory[tau].action, device=device, dtype=torch.long)  # discrete action executed at tau
    eligibility = trajectory[tau].eligibility  # eligibility trace weight carried through trajectory

    for learner_idx in range(NUM_LEARNERS):  # update each reward component learner independently
        target, rho = compute_n_step_targets(
            trajectory,
            tau,
            n_step,
            gamma,
            learner_idx,
            ensemble,
            device,
        )
        rho_values.append(rho)  # log ratio for diagnostics
        target_tensor = torch.tensor(target, device=device, dtype=torch.float32)  # scalar TD target

        critic_opts[learner_idx].zero_grad(set_to_none=True)  # reset gradients for critic update
        outputs = ensemble[learner_idx](obs_tensor)  # forward pass to obtain current value estimate
        value_pred = outputs["values"]  # predicted scalar value from critic head
        if isinstance(value_pred, Tensor):
            value_scalar = value_pred.squeeze()
        else:
            value_scalar = torch.tensor(value_pred, device=device, dtype=torch.float32)
        advantage = target_tensor - value_scalar  # compute TD error for this learner
        critic_loss = rho * 0.5 * advantage.pow(2)  # squared-error TD loss scaled by importance ratio
        critic_loss.backward()  # backpropagate through critic parameters
        critic_opts[learner_idx].step()  # apply critic update

        actor_opts[learner_idx].zero_grad(set_to_none=True)  # clear actor gradients
        outputs_actor = ensemble[learner_idx](obs_tensor)  # reuse forward pass for actor head
        dist = MaskedCategorical(outputs_actor["logits"], legal_mask_tensor)  # masked action distribution under learner
        log_pi_action = dist.log_prob(action_tensor)  # log-probability of the rollout action
        actor_loss = -eligibility * rho * advantage.detach() * log_pi_action  # policy gradient with importance correction
        actor_loss.backward()  # accumulate gradients on actor parameters
        actor_opts[learner_idx].step()  # perform actor update

    return rho_values


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Rollout collection and online updating
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
def train_episode(
    env: StrategoEnv,
    ensemble: Sequence[torch.nn.Module],
    actor_opts: Sequence[optim.Optimizer],
    critic_opts: Sequence[optim.Optimizer],
    weights: np.ndarray,
    n_step: int,
    gamma: float,
    device: torch.device,
    starting_player: Optional[int] = None,
    agent_top: Optional[bool] = None,
    *,
    boltz_agg: bool,
) -> Tuple[List[StepRecord], float, np.ndarray, float, List[float], str]:
    """Run a single training episode, collect data, and perform streaming updates."""

    weights = prepare_weights(weights, boltz_agg)
    if starting_player is None:
        opening_player = random.choice((0, 1))
    else:
        if starting_player not in (0, 1):
            raise ValueError("starting_player must be 0 or 1.")
        opening_player = starting_player
    if agent_top is None:
        agent_top_flag = bool(random.getrandbits(1))
    else:
        agent_top_flag = bool(agent_top)
    obs, info = env.reset(starting_player=opening_player, agent_top=agent_top_flag)  # start new episode and obtain initial observation
    legal_mask = info["legal_mask"]  # mask identifying legal moves at the initial state
    done = bool(info.get("episode_done", False))  # loop control flag that mirrors Gym termination semantics
    eligibility = 1.0  # reset eligibility trace multiplier
    trajectory: List[StepRecord] = []  # container for n-step update data
    episode_return = 0.0  # running total of primary reward
    behavior_sums = np.zeros(NUM_BEHAVIORS, dtype=np.float32)  # aggregate behavior counters for logging
    rho_diagnostics: List[float] = []  # on-the-fly importance ratio diagnostics
    final_info = info
    final_truncated = False
    last_reward_vec: Optional[np.ndarray] = None

    while not done:
        if not legal_mask.any():
            done = True
            break
        action, C_probs, per_probs = get_aggregated_action_and_probs(
            obs,
            legal_mask,
            ensemble,
            weights,
            device,
            boltz_agg=boltz_agg,
        )  # sample action under behavior policy
        log_pi = [float(torch.log(probs[action].clamp_min(1e-8)).item()) for probs in per_probs]  # log-probs per learner for IS corrections
        log_C = float(torch.log(C_probs[action].clamp_min(1e-8)).item())  # log-prob under aggregated behavior policy

        next_obs, reward_vec, terminated, truncated, info = env.step(action)  # transition environment with chosen action
        done = terminated or truncated  # track whether episode concluded this step
        final_info = info
        final_truncated = truncated
        last_reward_vec = reward_vec

        trajectory.append(
            StepRecord(
                obs=obs,
                action=action,
                reward_vec=reward_vec,
                done=done,
                legal_mask=legal_mask.copy(),
                log_pi=log_pi,
                log_C=log_C,
                eligibility=eligibility,
            )
        )

        episode_return += float(reward_vec[0])  # accumulate primary reward for logging
        behavior_sums += reward_vec[1:]  # track auxiliary behavior indicators

        if len(trajectory) >= n_step:
            tau = len(trajectory) - n_step  # index of earliest transition eligible for update
            rho_values = perform_update_step(
                trajectory,
                tau,
                ensemble,
                actor_opts,
                critic_opts,
                n_step,
                gamma,
                device,
            )
            rho_diagnostics.extend(rho_values)

        eligibility *= gamma  # decay eligibility trace
        obs = next_obs  # advance to next observation
        legal_mask = info["legal_mask"]  # update legal action mask from environment info

    T = len(trajectory)  # total steps collected this episode
    tail_start = max(0, T - n_step + 1)  # starting index for remaining updates after loop exits
    for tau in range(tail_start, T):
        rho_values = perform_update_step(
            trajectory,
            tau,
            ensemble,
            actor_opts,
            critic_opts,
            n_step,
            gamma,
            device,
        )
        rho_diagnostics.extend(rho_values)

    outcome_label = _classify_episode_outcome(final_info, truncated=final_truncated, last_reward_vec=last_reward_vec)
    win_flag = 1.0 if outcome_label.startswith("win_") else 0.0
    return trajectory, episode_return, behavior_sums, win_flag, rho_diagnostics, outcome_label


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Evaluation routine mirroring the experimental setup in the paper
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
def evaluate_agent(
    env: StrategoEnv,
    ensemble: Sequence[torch.nn.Module],
    weights: np.ndarray,
    episodes: int,
    device: torch.device,
    *,
    boltz_agg: bool,
) -> Dict[str, float]:
    """Run several evaluation episodes using inference weights and record aggregate metrics."""

    weights = prepare_weights(weights, boltz_agg)
    returns: List[float] = []  # per-episode primary returns
    behavior_counts: List[np.ndarray] = []  # per-episode behavior indicator sums
    wins: List[float] = []  # binary flag capture outcomes

    previous_modes = [model.training for model in ensemble]  # remember training/eval modes to restore later
    for model in ensemble:
        model.eval()

    for _ in range(episodes):
        agent_top_flag = bool(random.getrandbits(1))
        obs, info = env.reset(agent_top=agent_top_flag)  # begin evaluation episode with fresh state
        env.set_opponent_policy(None)  # evaluation against default random opponent
        legal_mask = info["legal_mask"]  # initial legal move mask
        done = False  # termination tracker
        episode_return = 0.0  # cumulative return placeholder
        behavior_sum = np.zeros(NUM_BEHAVIORS, dtype=np.float32)  # track behavior counts for this run

        while not done:
            action, _, _ = get_aggregated_action_and_probs(
                obs,
                legal_mask,
                ensemble,
                weights,
                device,
                boltz_agg=boltz_agg,
            )  # greedy rollout using inference weights
            obs, reward_vec, terminated, truncated, info = env.step(action)  # transition environment
            done = terminated or truncated  # break loop when episode ends
            episode_return += float(reward_vec[0])  # add primary reward
            behavior_sum += reward_vec[1:]  # accumulate behavior metrics
            legal_mask = info["legal_mask"]  # refresh legal moves

        returns.append(episode_return)  # store performance metric
        behavior_counts.append(behavior_sum)  # cache behavior stats
        wins.append(1.0 if reward_vec[1] == 1.0 else 0.0)  # record win indicator from final reward

    for model, mode in zip(ensemble, previous_modes):
        model.train(mode)

    behavior_array = np.stack(behavior_counts, axis=0)  # convert list of arrays into matrix for averaging
    return {
        "episode": 0.0,  # placeholder updated by caller
        "return_primary": float(np.mean(returns)),
        # "b_flag_sum": float(np.mean(behavior_array[:, 0])),
        "b_capture_sum": float(np.mean(behavior_array[:, 0])),
        "b_eff_sum": float(np.mean(behavior_array[:, 1])),
        "win_rate": float(np.mean(wins)),
        "rho_mean": 0.0,
        "rho_std": 0.0,
        "rho_p95": 0.0,
    }


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Argument parsing and device selection
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
def parse_args() -> argparse.Namespace:
    """Expose the hyperparameters described in the paper via CLI for reproducibility."""

    parser = argparse.ArgumentParser(description="Behavioral Actor-Critic training script for Stratego-lite.")
    parser.add_argument("--episodes", type=int, default=100, help="Number of training episodes.")
    parser.add_argument("--eval_every", type=int, default=20, help="Frequency (in episodes) for evaluations.")
    parser.add_argument("--eval_episodes", type=int, default=50, help="Number of evaluation episodes per sweep.")
    parser.add_argument("--n_step", type=int, default=3, help="n-step horizon used for returns.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--lr_actor", type=float, default=3e-4, help="Learning rate for actor updates.")
    parser.add_argument("--lr_critic", type=float, default=3e-4, help="Learning rate for critic updates.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max_steps", type=int, default=200, help="Maximum steps per environment episode.")
    parser.add_argument("--m_train", nargs=NUM_LEARNERS, type=float, default=DEFAULT_M_TRAIN.tolist(), help="Training scalarization weights.")
    parser.add_argument("--m_infer", nargs=NUM_LEARNERS, type=float, default=DEFAULT_M_TRAIN.tolist(), help="Inference scalarization weights.")
    parser.add_argument(
        "--boltz_agg",
        dest="boltz_agg",
        action="store_true",
        default=True,
        help="Use Boltzmann aggregation (allows unconstrained weights).",
    )
    parser.add_argument(
        "--no-boltz-agg",
        dest="boltz_agg",
        action="store_false",
        help="Disable Boltzmann aggregation and revert to simplex-constrained mixing.",
    )
    parser.add_argument("--output_dir", type=str, default="runs", help="Directory for log CSV files.")
    parser.add_argument("--selfplay_prob", type=float, default=0.0, help="Probability of sampling a frozen self-play opponent.")
    parser.add_argument("--checkpoint_interval", type=int, default=50, help="Episodes between snapshotting the ensemble.")
    parser.add_argument("--device", type=str, default="auto", help="Device specifier: cpu, cuda, or auto.")
    return parser.parse_args()


def select_device(flag: str) -> torch.device:
    """Resolve device string to a torch.device, defaulting to CUDA when available."""

    if flag == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(flag)


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Main training loop tying all components together
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
def main() -> None:
    """Orchestrate training, evaluation, logging, and opponent snapshotting."""

    args = parse_args()  # pull hyperparameters from CLI
    device = select_device(args.device)  # resolve torch device selection
    set_global_seed(args.seed)  # synchronize RNGs for reproducibility

    m_train = np.asarray(args.m_train, dtype=np.float32)  # training scalarization weights
    m_infer = np.asarray(args.m_infer, dtype=np.float32)  # inference scalarization weights
    if m_train.shape[0] != NUM_LEARNERS or m_infer.shape[0] != NUM_LEARNERS:
        raise ValueError(f"Expected {NUM_LEARNERS} scalarization weights for train/infer.")
    m_train = prepare_weights(m_train, args.boltz_agg)
    m_infer = prepare_weights(m_infer, args.boltz_agg)

    train_env = StrategoEnv(max_steps=args.max_steps, seed=args.seed)  # environment used for online updates
    eval_env = StrategoEnv(max_steps=args.max_steps, seed=args.seed + 1)  # separate evaluator with different seed

    obs_dim = train_env.reset()[0].shape[0]  # flatten observation dimensionality
    action_dim = train_env.num_actions  # discrete action space size shared by all learners

    ensemble = make_ensemble(NUM_LEARNERS, obs_dim, action_dim)  # instantiate actor-critic learners
    for model in ensemble:
        model.to(device)
        model.train()

    actor_opts = [optim.Adam(model.parameters(), lr=args.lr_actor) for model in ensemble]  # optimizer per actor head
    critic_opts = [optim.Adam(model.parameters(), lr=args.lr_critic) for model in ensemble]  # optimizer per critic head

    ensure_dir(args.output_dir)  # guarantee logging directory exists
    train_log_path = os.path.join(args.output_dir, "train_log.csv")  # training metrics log file
    eval_log_path = os.path.join(args.output_dir, "eval_log.csv")  # evaluation metrics log file

    train_log_exists = Path(train_log_path).exists()  # detect whether to write CSV header
    eval_log_exists = Path(eval_log_path).exists()

    fields = [
                "episode",
                "return_primary",
                "b_capture_sum",
                "b_eff_sum",
                "win_rate",
                "rho_mean",
                "rho_std",
                "rho_p95",
            ]

    with open(train_log_path, "a", newline="") as train_file, open(eval_log_path, "a", newline="") as eval_file:  # append to logs so training can resume seamlessly
        train_writer = csv.DictWriter(
            train_file,
            fieldnames=fields,
        )
        eval_writer = csv.DictWriter(
            eval_file,
            fieldnames=fields,
        )
        if not train_log_exists:
            train_writer.writeheader()
        if not eval_log_exists:
            eval_writer.writeheader()

        opponent_pool: List[List[Dict[str, Tensor]]] = []  # rotating buffer of frozen opponents

        for episode in range(1, args.episodes + 1):
            if opponent_pool and random.random() < args.selfplay_prob:
                snapshot = random.choice(opponent_pool)  # sample opponent snapshot for self-play
                opponent_policy = make_opponent_policy(
                    snapshot,
                    obs_dim,
                    action_dim,
                    m_train,
                    device,
                    boltz_agg=args.boltz_agg,
                )  # reconstruct frozen policy
                train_env.set_opponent_policy(opponent_policy)  # inject into environment
            else:
                train_env.set_opponent_policy(None)

            _, episode_return, behavior_sums, win_flag, rho_values, _ = train_episode(
                train_env,
                ensemble,
                actor_opts,
                critic_opts,
                m_train,
                args.n_step,
                args.gamma,
                device,
                boltz_agg=args.boltz_agg,
            )

            rho_array = np.asarray(rho_values, dtype=np.float32) if rho_values else np.zeros(1, dtype=np.float32)  # avoid empty stats
            train_writer.writerow(
                {
                    "episode": episode,
                    "return_primary": episode_return,
                    "b_capture_sum": behavior_sums[0],
                    "b_eff_sum": behavior_sums[1],
                    "win_rate": win_flag,
                    "rho_mean": float(rho_array.mean()),
                    "rho_std": float(rho_array.std()),
                    "rho_p95": float(np.percentile(rho_array, 95)),
                }
            )
            train_file.flush()  # ensure metrics hit disk before next iteration

            if episode % args.checkpoint_interval == 0:
                opponent_pool.append(snapshot_ensemble(ensemble))  # stash current policies for future self-play

            if args.eval_every > 0 and episode % args.eval_every == 0:
                metrics = evaluate_agent(
                    eval_env,
                    ensemble,
                    m_infer,
                    args.eval_episodes,
                    device,
                    boltz_agg=args.boltz_agg,
                )  # periodic evaluation sweep
                metrics["episode"] = episode  # stamp episode number before saving
                eval_writer.writerow(metrics)  # append evaluation metrics row
                eval_file.flush()  # force evaluation log to persist immediately


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Standard script entry point
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
if __name__ == "__main__":
    main()
