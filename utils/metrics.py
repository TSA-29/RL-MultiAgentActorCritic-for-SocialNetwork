from __future__ import annotations

import numpy as np
import torch


def consensus_degree(opinions: torch.Tensor | np.ndarray) -> torch.Tensor:
    tensor = _as_tensor(opinions)
    if tensor.ndim != 1:
        raise ValueError("opinions must be a 1D tensor.")
    if tensor.numel() < 2:
        return torch.tensor(1.0, dtype=torch.float32, device=tensor.device)

    pairwise_gap = torch.abs(tensor.unsqueeze(1) - tensor.unsqueeze(0))
    upper_indices = torch.triu_indices(tensor.numel(), tensor.numel(), offset=1, device=tensor.device)
    mean_gap = pairwise_gap[upper_indices[0], upper_indices[1]].mean()
    return (1.0 - (mean_gap / 2.0)).clamp(0.0, 1.0)


def cross_community_weight(
    weight_matrix: torch.Tensor | np.ndarray,
    community_labels: torch.Tensor | np.ndarray,
) -> torch.Tensor:
    weights = _as_tensor(weight_matrix)
    labels = _as_tensor(community_labels, dtype=torch.long, device=weights.device)
    community_mismatch = labels.unsqueeze(1) != labels.unsqueeze(0)
    if not torch.any(community_mismatch):
        return torch.tensor(0.0, dtype=torch.float32, device=weights.device)
    return weights[community_mismatch].mean()


def harmful_bridge_mass(
    *,
    previous_weights: torch.Tensor | np.ndarray,
    updated_weights: torch.Tensor | np.ndarray,
    community_labels: torch.Tensor | np.ndarray,
    opinions: torch.Tensor | np.ndarray,
) -> torch.Tensor:
    previous = _as_tensor(previous_weights)
    updated = _as_tensor(updated_weights, device=previous.device)
    labels = _as_tensor(community_labels, dtype=torch.long, device=previous.device)
    views = _as_tensor(opinions, device=previous.device)

    community_mismatch = labels.unsqueeze(1) != labels.unsqueeze(0)
    if not torch.any(community_mismatch):
        return torch.tensor(0.0, dtype=torch.float32, device=previous.device)

    positive_shift = (updated - previous).clamp_min(0.0)
    opinion_gap = torch.abs(views.unsqueeze(1) - views.unsqueeze(0))
    harmful = positive_shift[community_mismatch] * opinion_gap[community_mismatch]
    return harmful.mean() if harmful.numel() > 0 else torch.tensor(0.0, dtype=torch.float32, device=previous.device)


def _as_tensor(
    value: torch.Tensor | np.ndarray,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.to(dtype=dtype)
    else:
        tensor = torch.as_tensor(value, dtype=dtype)
    if device is not None:
        tensor = tensor.to(device=device)
    return tensor
