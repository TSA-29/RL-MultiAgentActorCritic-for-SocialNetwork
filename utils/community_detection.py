from __future__ import annotations

from collections.abc import Iterable

import networkx as nx
import numpy as np
import torch


AVAILABLE_DETECTORS = ("label_propagation", "louvain")


def available_detectors() -> tuple[str, ...]:
    return AVAILABLE_DETECTORS


def detect_communities(
    *,
    weight_matrix: torch.Tensor | np.ndarray,
    opinions: torch.Tensor | np.ndarray,
    detector: str,
    seed: int | None,
    temperature: float = 0.35,
) -> np.ndarray:
    """Detect communities on an opinion-aware affinity graph."""

    detector_name = str(detector).lower()
    if detector_name not in AVAILABLE_DETECTORS:
        raise ValueError(
            f"Unsupported detector '{detector}'. Available detectors: {', '.join(AVAILABLE_DETECTORS)}."
        )

    affinity = build_affinity_matrix(weight_matrix=weight_matrix, opinions=opinions, temperature=temperature)
    graph = nx.Graph()
    node_count = affinity.shape[0]
    graph.add_nodes_from(range(node_count))

    rows, cols = np.where(np.triu(affinity, k=1) > 0.0)
    for row, col in zip(rows.tolist(), cols.tolist()):
        graph.add_edge(row, col, weight=float(affinity[row, col]))

    if detector_name == "louvain":
        communities = nx.algorithms.community.louvain_communities(graph, weight="weight", seed=seed)
    else:
        communities = list(nx.algorithms.community.asyn_lpa_communities(graph, weight="weight", seed=seed))

    return canonicalize_communities(communities=communities, node_count=node_count)


def build_affinity_matrix(
    *,
    weight_matrix: torch.Tensor | np.ndarray,
    opinions: torch.Tensor | np.ndarray,
    temperature: float = 0.35,
) -> np.ndarray:
    """Build the undirected opinion-aware affinity graph used by detectors."""

    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")

    weights = _to_numpy(weight_matrix)
    views = _to_numpy(opinions).reshape(-1)
    if weights.shape != (views.size, views.size):
        raise ValueError("weight_matrix must have shape (num_nodes, num_nodes).")

    symmetric_weights = 0.5 * (weights + weights.T)
    opinion_gap = np.abs(views[:, None] - views[None, :])
    affinity = symmetric_weights * np.exp(-opinion_gap / temperature)
    np.fill_diagonal(affinity, 0.0)
    return affinity.astype(np.float32, copy=False)


def canonicalize_communities(
    *,
    communities: Iterable[Iterable[int]],
    node_count: int,
) -> np.ndarray:
    """Convert an arbitrary community iterable into stable integer labels."""

    seen_nodes: set[int] = set()
    normalized: list[list[int]] = []
    for community in communities:
        nodes = sorted({int(node) for node in community})
        if not nodes:
            continue
        normalized.append(nodes)
        seen_nodes.update(nodes)

    for node in range(node_count):
        if node not in seen_nodes:
            normalized.append([node])

    normalized.sort(key=lambda nodes: (-len(nodes), nodes[0], tuple(nodes)))
    labels = np.empty(node_count, dtype=np.int32)
    for label, nodes in enumerate(normalized):
        labels[np.asarray(nodes, dtype=np.int32)] = label
    return labels


def _to_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(value, dtype=np.float32)
