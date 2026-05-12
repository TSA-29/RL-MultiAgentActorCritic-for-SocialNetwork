from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor


def _clone_to_cpu(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _clone_to_cpu(inner_value) for key, inner_value in value.items()}
    if isinstance(value, np.ndarray):
        return torch.from_numpy(np.array(value, copy=True))
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    return torch.as_tensor(value)


@dataclass(frozen=True)
class ReplayTransition:
    global_state: dict[str, Any]
    joint_action: Tensor
    reward: float
    next_global_state: dict[str, Any]
    done: bool


class ReplayBuffer:
    """Simple replay buffer storing full joint actions for centralized training."""

    def __init__(
        self,
        capacity: int,
        *,
        seed: int | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be a positive integer.")

        self.capacity = int(capacity)
        self._rng = random.Random(seed)
        self._storage: list[ReplayTransition] = []
        self._position = 0

    def __len__(self) -> int:
        return len(self._storage)

    def clear(self) -> None:
        self._storage.clear()
        self._position = 0

    def add(
        self,
        *,
        global_state: Mapping[str, Any],
        joint_action: Tensor | np.ndarray,
        reward: float,
        next_global_state: Mapping[str, Any],
        done: bool,
    ) -> None:
        transition = ReplayTransition(
            global_state=_clone_to_cpu(global_state),
            joint_action=_clone_to_cpu(joint_action).to(dtype=torch.float32),
            reward=float(reward),
            next_global_state=_clone_to_cpu(next_global_state),
            done=bool(done),
        )

        if len(self._storage) < self.capacity:
            self._storage.append(transition)
        else:
            self._storage[self._position] = transition
        self._position = (self._position + 1) % self.capacity

    def sample(self, batch_size: int) -> list[ReplayTransition]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if batch_size > len(self._storage):
            raise ValueError(
                f"Cannot sample {batch_size} transitions from a buffer containing {len(self._storage)} items."
            )
        return self._rng.sample(self._storage, batch_size)
