from __future__ import annotations

import torch


def build_scale_free_weight_matrix(
    *,
    num_nodes: int,
    attachment_edges: int,
    generator: torch.Generator | None = None,
    small_value: float = 1e-8,
) -> torch.Tensor:
    """Create a directed scale-free graph and normalize rows into weights."""

    if num_nodes < 2:
        raise ValueError("num_nodes must be at least 2.")
    if attachment_edges < 1:
        raise ValueError("attachment_edges must be at least 1.")

    seed_size = min(num_nodes, max(3, attachment_edges + 1))
    adjacency = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)

    for row in range(seed_size):
        for col in range(seed_size):
            if row != col:
                adjacency[row, col] = 1.0

    for new_node in range(seed_size, num_nodes):
        current_graph = adjacency[:new_node, :new_node]
        target_scores = current_graph.sum(dim=0) + 1.0
        source_scores = current_graph.sum(dim=1) + 1.0
        edge_budget = min(attachment_edges, new_node)

        out_targets = _sample_without_replacement(target_scores, edge_budget, generator)
        in_sources = _sample_without_replacement(source_scores, edge_budget, generator)

        adjacency[new_node, out_targets] = 1.0
        adjacency[in_sources, new_node] = 1.0

    raw_weights = torch.rand((num_nodes, num_nodes), generator=generator, dtype=torch.float32) + small_value
    weight_matrix = adjacency * raw_weights
    return normalize_rows(weight_matrix, small_value=small_value)


def sample_polarized_opinions(
    *,
    num_nodes: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample polarized-but-bridgeable initial opinions."""

    negative_count = int(round(0.4 * num_nodes))
    neutral_count = int(round(0.2 * num_nodes))
    positive_count = num_nodes - negative_count - neutral_count

    negative = _sample_normal(negative_count, mean=-0.7, std=0.15, generator=generator)
    neutral = _sample_normal(neutral_count, mean=0.0, std=0.15, generator=generator)
    positive = _sample_normal(positive_count, mean=0.7, std=0.15, generator=generator)

    opinions = torch.cat((negative, neutral, positive), dim=0).clamp(-1.0, 1.0)
    permutation = torch.randperm(num_nodes, generator=generator)
    return opinions[permutation]


def hegselmann_krause_update(
    *,
    opinions: torch.Tensor,
    weight_matrix: torch.Tensor,
    confidence_bound: float,
    self_belief: float,
    small_value: float = 1e-8,
) -> torch.Tensor:
    """Apply a weighted HK opinion update with a fixed self-belief prior."""

    opinion_gap = torch.abs(opinions.unsqueeze(1) - opinions.unsqueeze(0))
    confidence_mask = (opinion_gap <= confidence_bound).to(dtype=torch.float32)

    identity = torch.eye(opinions.size(0), dtype=torch.float32, device=opinions.device)
    effective_weights = (weight_matrix * confidence_mask) + (self_belief * identity)
    normalized_weights = effective_weights / effective_weights.sum(dim=1, keepdim=True).clamp_min(small_value)
    return (normalized_weights @ opinions).clamp(-1.0, 1.0)


def normalize_rows(matrix: torch.Tensor, *, small_value: float = 1e-8) -> torch.Tensor:
    row_sums = matrix.sum(dim=1, keepdim=True)
    normalized = matrix.clone()
    non_zero_rows = row_sums.squeeze(-1) > small_value

    if torch.any(non_zero_rows):
        normalized[non_zero_rows] = normalized[non_zero_rows] / row_sums[non_zero_rows]

    zero_rows = torch.nonzero(~non_zero_rows, as_tuple=False).flatten()
    if zero_rows.numel() > 0:
        normalized[zero_rows] = 0.0
        normalized[zero_rows, zero_rows] = 1.0
    return normalized


def _sample_without_replacement(
    weights: torch.Tensor,
    sample_count: int,
    generator: torch.Generator | None,
) -> torch.Tensor:
    probabilities = weights / weights.sum().clamp_min(1e-8)
    return torch.multinomial(
        probabilities,
        num_samples=sample_count,
        replacement=False,
        generator=generator,
    )


def _sample_normal(
    sample_count: int,
    *,
    mean: float,
    std: float,
    generator: torch.Generator | None,
) -> torch.Tensor:
    if sample_count <= 0:
        return torch.empty(0, dtype=torch.float32)
    noise = torch.randn(sample_count, generator=generator, dtype=torch.float32)
    return (noise * std) + mean
