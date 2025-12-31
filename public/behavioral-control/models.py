#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Patrick Marshall
# Neural network modules for the behavioral actor-critic ensemble
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Import useful libraries
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
from __future__ import annotations  # future annotations keep type hints concise
from typing import Dict, List, Sequence, Type  # typing helpers used across the module
import torch  # deep learning framework
from torch import Tensor, nn  # tensor alias and neural network building blocks

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Masked categorical distribution to enforce legal action support
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class MaskedCategorical:
    """Categorical distribution that zeroes probability on illegal actions while maintaining coverage."""

    def __init__(self, logits: Tensor, legal_mask: Tensor, eps: float = 1e-8) -> None:
        """Construct the distribution given raw logits and a boolean mask of legal actions."""

        if logits.ndim != 1:
            raise ValueError("MaskedCategorical expects 1-D logits.")
        if legal_mask.shape != logits.shape:
            raise ValueError("Logits and legal mask must share shape.")

        device = logits.device  # retain device for subsequent ops
        dtype = logits.dtype  # ensure mask uses same precision as logits
        mask = legal_mask.to(device=device, dtype=dtype)  # promote mask tensor to logits dtype/device
        masked_logits = logits.clone()  # avoid in-place modification of caller-owned tensor
        masked_logits = masked_logits.masked_fill(mask < 0.5, torch.finfo(dtype).min)  # force illegal logits to -inf
        probs = torch.softmax(masked_logits, dim=-1)  # convert logits to probabilities
        probs = probs * mask  # explicitly remove probability mass on illegal actions
        total = probs.sum()  # normalization constant over legal actions
        if total <= 0 or torch.isnan(total):
            legal_count = mask.sum()
            if legal_count <= 0:
                raise ValueError("No legal actions available to construct distribution.")
            probs = mask / legal_count  # distribute probability uniformly across available moves
        else:
            probs = probs / (total + eps)  # normalize and keep coverage with epsilon
        self.probs = probs

    def sample(self) -> Tensor:
        """Sample a legal action index according to the masked distribution."""

        return torch.multinomial(self.probs, num_samples=1).squeeze(-1)  # draw according to masked probabilities

    def log_prob(self, action_idx: Tensor) -> Tensor:
        """Return the log-probability of the provided action index tensor."""

        if action_idx.ndim == 0:
            action_idx = action_idx.unsqueeze(0)
        gathered = self.probs.gather(0, action_idx)  # extract probability of requested action(s)
        return torch.log(gathered + 1e-8)  # add epsilon for numerical safety

    def entropy(self) -> Tensor:
        """Compute entropy for diagnostics (supports exploration analysis)."""

        safe_probs = torch.clamp(self.probs, min=1e-8)  # avoid log(0)
        return -(safe_probs * torch.log(safe_probs)).sum()  # classic categorical entropy


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Actor-critic network with shared encoder and separate heads
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class ActorCritic(nn.Module):
    """Minimal actor-critic used for each reward component learner in the ensemble."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (128, 128),
        activation_cls: Type[nn.Module] = nn.ReLU,
    ) -> None:
        """Initialize the network architecture from observation dimension and action space size."""

        super().__init__()
        layers: List[nn.Module] = []  # container for hidden MLP layers
        input_dim = obs_dim  # track current layer width during construction
        for size in hidden_sizes:
            layers.append(nn.Linear(input_dim, size))  # dense layer
            layers.append(activation_cls())  # non-linearity
            input_dim = size  # update fan-out for next layer
        self.encoder = nn.Sequential(*layers) if layers else nn.Identity()  # shared torso across actor/critic
        self.policy_head = nn.Linear(input_dim, action_dim)  # generates logits for MaskedCategorical
        self.value_head = nn.Linear(input_dim, 1)  # outputs scalar value used in critic updates

    def forward(self, obs: Tensor) -> Dict[str, Tensor]:
        """Return masked policy logits and value estimates for the provided observation."""

        if obs.ndim == 1:
            obs = obs.unsqueeze(0)  # promote to batch dimension when receiving flat vector
        features = self.encoder(obs)  # shared representation
        logits = self.policy_head(features)  # actor logits
        values = self.value_head(features).squeeze(-1)  # critic scalar(s)
        if logits.shape[0] == 1:
            logits = logits.squeeze(0)  # drop batch dimension for single sample usage
            values = values.squeeze(0)
        return {"logits": logits, "values": values}  # unified dict interface for caller


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Factory for building the ensemble specified in the paper
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
def make_ensemble(
    num_learners: int,
    obs_dim: int,
    action_dim: int,
    hidden_sizes: Sequence[int] = (128, 128),
    activation_cls: Type[nn.Module] = nn.ReLU,
) -> List[ActorCritic]:
    """Instantiate K+1 actor-critics, one for the primary reward and each behavior component."""

    return [
        ActorCritic(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=hidden_sizes, activation_cls=activation_cls)
        for _ in range(num_learners)
    ]  # create ensemble with identical architecture per learner
