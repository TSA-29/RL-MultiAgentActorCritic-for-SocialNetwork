from __future__ import annotations

from typing import Mapping

import numpy as np


def zero_policy(observation: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    active_count = _active_agent_count(observation)
    membership = np.asarray(observation["community_membership"], dtype=np.float32)
    node_count = int(np.asarray(observation["opinions"]).size)

    actions: dict[str, np.ndarray] = {}
    for agent_index in range(active_count):
        source_nodes = np.flatnonzero(membership[agent_index] > 0.5)
        actions[f"community_{agent_index}"] = np.zeros((source_nodes.size, node_count), dtype=np.float32)
    return actions


def heuristic_bridge_policy(
    observation: Mapping[str, np.ndarray],
    *,
    confidence_bound: float,
) -> dict[str, np.ndarray]:
    opinions = np.asarray(observation["opinions"], dtype=np.float32).reshape(-1)
    labels = np.asarray(observation["community_labels"], dtype=np.int32).reshape(-1)
    membership = np.asarray(observation["community_membership"], dtype=np.float32)
    active_count = _active_agent_count(observation)
    global_mean = float(opinions.mean())
    denominator = max(confidence_bound, 1e-6)

    actions: dict[str, np.ndarray] = {}
    for agent_index in range(active_count):
        source_nodes = np.flatnonzero(membership[agent_index] > 0.5)
        local_action = np.zeros((source_nodes.size, opinions.size), dtype=np.float32)

        for local_row, source_node in enumerate(source_nodes.tolist()):
            opinion_gap = np.abs(opinions - opinions[source_node])
            within_bound = opinion_gap <= confidence_bound

            center_bonus = 1.5 * (1.0 - np.abs(opinions - global_mean))
            moderate_bridge_bonus = (
                (labels != labels[source_node]).astype(np.float32)
                * within_bound.astype(np.float32)
                * (1.0 - (opinion_gap / denominator))
            )
            local_bonus = (
                0.25
                * (labels == labels[source_node]).astype(np.float32)
                * within_bound.astype(np.float32)
                * (1.0 - (opinion_gap / denominator))
            )

            score = center_bonus + moderate_bridge_bonus + local_bonus
            score[source_node] = score[source_node] + 0.25
            score = score - score.mean()
            local_action[local_row] = np.clip(score, -1.0, 1.0)

        actions[f"community_{agent_index}"] = local_action
    return actions


def _active_agent_count(observation: Mapping[str, np.ndarray]) -> int:
    active_mask = np.asarray(observation["active_agent_mask"], dtype=np.float32).reshape(-1)
    return int(np.count_nonzero(active_mask > 0.5))
