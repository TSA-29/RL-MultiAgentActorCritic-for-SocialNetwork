from __future__ import annotations

from typing import Any, Mapping

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from .network_factory import (
    build_scale_free_weight_matrix,
    hegselmann_krause_update,
    sample_polarized_opinions,
)
from utils.community_detection import detect_communities
from utils.metrics import consensus_degree, cross_community_weight, harmful_bridge_mass


class SocialNetworkEnv(gym.Env):
    """Directed social-network environment for consensus-oriented MAAC training."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        num_nodes: int = 50,
        max_steps: int = 100,
        confidence_bound: float = 0.7,
        consensus_threshold: float = 0.95,
        community_update_freq: int = 10,
        attachment_edges: int = 2,
        community_detector: str = "louvain",
        community_temperature: float = 0.35,
        action_scale: float = 1.0,
        intervention_interval: int = 1,
        hold_last_action: bool = True,
        terminal_bonus: float = 1.0,
        step_penalty: float = 0.01,
        action_penalty_coef: float = 0.01,
        bridge_penalty_coef: float = 0.25,
        self_belief: float = 1.0,
        small_value: float = 1e-8,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()

        if num_nodes < 2:
            raise ValueError("num_nodes must be at least 2.")
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1.")
        if not 0.0 < confidence_bound <= 2.0:
            raise ValueError("confidence_bound must be in (0, 2].")
        if not 0.0 < consensus_threshold <= 1.0:
            raise ValueError("consensus_threshold must be in (0, 1].")
        if community_update_freq < 1:
            raise ValueError("community_update_freq must be at least 1.")
        if attachment_edges < 1:
            raise ValueError("attachment_edges must be at least 1.")
        if action_scale <= 0.0:
            raise ValueError("action_scale must be positive.")
        if intervention_interval < 1:
            raise ValueError("intervention_interval must be at least 1.")
        if self_belief <= 0.0:
            raise ValueError("self_belief must be positive.")
        if small_value <= 0.0:
            raise ValueError("small_value must be positive.")

        self.num_nodes = int(num_nodes)
        self.max_steps = int(max_steps)
        self.confidence_bound = float(confidence_bound)
        self.consensus_threshold = float(consensus_threshold)
        self.community_update_freq = int(community_update_freq)
        self.attachment_edges = min(int(attachment_edges), self.num_nodes - 1)
        self.community_detector = str(community_detector)
        self.community_temperature = float(community_temperature)
        self.action_scale = float(action_scale)
        self.intervention_interval = int(intervention_interval)
        self.hold_last_action = bool(hold_last_action)
        self.terminal_bonus = float(terminal_bonus)
        self.step_penalty = float(step_penalty)
        self.action_penalty_coef = float(action_penalty_coef)
        self.bridge_penalty_coef = float(bridge_penalty_coef)
        self.self_belief = float(self_belief)
        self.small_value = float(small_value)
        self.device = torch.device(device)

        self.max_communities = self.num_nodes
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_nodes, self.num_nodes),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(
            {
                "opinions": spaces.Box(-1.0, 1.0, shape=(self.num_nodes,), dtype=np.float32),
                "weight_matrix": spaces.Box(0.0, 1.0, shape=(self.num_nodes, self.num_nodes), dtype=np.float32),
                "community_labels": spaces.Box(
                    low=0,
                    high=self.num_nodes - 1,
                    shape=(self.num_nodes,),
                    dtype=np.int32,
                ),
                "community_membership": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.max_communities, self.num_nodes),
                    dtype=np.float32,
                ),
                "action_mask": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.max_communities, self.num_nodes, self.num_nodes),
                    dtype=np.float32,
                ),
                "active_agent_mask": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.max_communities,),
                    dtype=np.float32,
                ),
                "global_consensus": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
                "step_count": spaces.Box(low=0, high=self.max_steps, shape=(1,), dtype=np.int32),
            }
        )

        self.current_step = 0
        self.weight_matrix: torch.Tensor | None = None
        self.opinions: torch.Tensor | None = None
        self.community_labels: torch.Tensor | None = None
        self.community_membership: torch.Tensor | None = None
        self.action_mask: torch.Tensor | None = None
        self.active_agent_mask: torch.Tensor | None = None
        self.community_nodes: list[torch.Tensor] = []
        self.num_active_agents = 0
        self.last_applied_action: torch.Tensor | None = None

        self._episode_seed: int | None = None
        self._torch_generator: torch.Generator | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = 0
        self._episode_seed = seed
        self.last_applied_action = None

        torch_seed = int(self.np_random.integers(0, 2**31 - 1))
        self._torch_generator = torch.Generator(device="cpu").manual_seed(torch_seed)

        self.weight_matrix = build_scale_free_weight_matrix(
            num_nodes=self.num_nodes,
            attachment_edges=self.attachment_edges,
            generator=self._torch_generator,
            small_value=self.small_value,
        ).to(self.device)
        self.opinions = sample_polarized_opinions(
            num_nodes=self.num_nodes,
            generator=self._torch_generator,
        ).to(self.device)
        self._refresh_communities()

        consensus = consensus_degree(self.opinions)
        observation = self._get_observation(consensus)
        zero = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        info = self._build_info(
            consensus=consensus,
            consensus_gain=zero,
            terminal_bonus=zero,
            action_penalty=zero,
            bridge_penalty=zero,
            step_penalty=zero,
            reward=zero,
            harmful_bridge=zero,
            action_magnitude=zero,
            community_refreshed=True,
            applied_action_source="none",
            used_fresh_intervention=False,
            used_held_action=False,
        )
        return observation, info

    def step(
        self,
        action: Mapping[int | str, np.ndarray | torch.Tensor] | np.ndarray | torch.Tensor,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if self.weight_matrix is None or self.opinions is None or self.community_labels is None:
            raise RuntimeError("Call reset() before step().")

        previous_weights = self.weight_matrix.clone()
        previous_labels = self.community_labels.clone()
        previous_consensus = consensus_degree(self.opinions)

        formatted_action, applied_action_source, used_fresh_intervention, used_held_action = (
            self._resolve_step_action(action)
        )
        scaled_action = formatted_action * self.action_scale
        self.weight_matrix = self._apply_action_deltas(self.weight_matrix, scaled_action)

        action_magnitude = scaled_action.abs().mean()
        harmful_bridge = harmful_bridge_mass(
            previous_weights=previous_weights,
            updated_weights=self.weight_matrix,
            community_labels=previous_labels,
            opinions=self.opinions,
        )

        self.opinions = hegselmann_krause_update(
            opinions=self.opinions,
            weight_matrix=self.weight_matrix,
            confidence_bound=self.confidence_bound,
            self_belief=self.self_belief,
            small_value=self.small_value,
        )
        self.current_step += 1

        community_refreshed = False
        if self.current_step % self.community_update_freq == 0:
            self._refresh_communities()
            community_refreshed = True

        current_consensus = consensus_degree(self.opinions)
        consensus_gain = current_consensus - previous_consensus
        action_penalty = self.action_penalty_coef * action_magnitude
        bridge_penalty = self.bridge_penalty_coef * harmful_bridge
        step_penalty = torch.tensor(self.step_penalty, dtype=torch.float32, device=self.device)

        terminated = bool(current_consensus >= self.consensus_threshold)
        terminal_bonus = torch.tensor(
            self.terminal_bonus if terminated else 0.0,
            dtype=torch.float32,
            device=self.device,
        )
        reward = consensus_gain + terminal_bonus - step_penalty - action_penalty - bridge_penalty

        truncated = self.current_step >= self.max_steps and not terminated
        observation = self._get_observation(current_consensus)
        info = self._build_info(
            consensus=current_consensus,
            consensus_gain=consensus_gain,
            terminal_bonus=terminal_bonus,
            action_penalty=action_penalty,
            bridge_penalty=bridge_penalty,
            step_penalty=step_penalty,
            reward=reward,
            harmful_bridge=harmful_bridge,
            action_magnitude=action_magnitude,
            community_refreshed=community_refreshed,
            applied_action_source=applied_action_source,
            used_fresh_intervention=used_fresh_intervention,
            used_held_action=used_held_action,
        )
        return observation, float(reward.item()), terminated, truncated, info

    def get_agent_ids(self) -> list[str]:
        return [f"community_{agent_index}" for agent_index in range(self.num_active_agents)]

    def should_request_new_action(self) -> bool:
        return self.last_applied_action is None or (self.current_step % self.intervention_interval == 0)

    def schedule_action(
        self,
        action: Mapping[int | str, np.ndarray | torch.Tensor] | np.ndarray | torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.should_request_new_action():
            if action is None:
                raise ValueError("A fresh action is required on intervention steps.")
            return self._format_action(action)

        if self.hold_last_action:
            if self.last_applied_action is None:
                raise RuntimeError("No previous action is available to hold for this step.")
            return self.last_applied_action

        return torch.zeros((self.num_nodes, self.num_nodes), dtype=torch.float32, device=self.device)

    def get_runtime_communities(self) -> dict[str, dict[str, np.ndarray | list[int]]]:
        if self.community_membership is None:
            raise RuntimeError("Call reset() before requesting runtime communities.")

        runtime: dict[str, dict[str, np.ndarray | list[int]]] = {}
        for agent_index, nodes in enumerate(self.community_nodes):
            community_mask = torch.zeros(self.num_nodes, dtype=torch.float32, device=self.device)
            community_mask[nodes] = 1.0
            runtime[f"community_{agent_index}"] = {
                "source_nodes": nodes.detach().cpu().tolist(),
                "community_mask": self._to_numpy(community_mask),
                "local_action_mask": np.ones((nodes.numel(), self.num_nodes), dtype=np.float32),
            }
        return runtime

    def _refresh_communities(self) -> None:
        if self.weight_matrix is None or self.opinions is None:
            raise RuntimeError("Communities cannot be refreshed before reset().")

        labels = detect_communities(
            weight_matrix=self.weight_matrix,
            opinions=self.opinions,
            detector=self.community_detector,
            seed=self._community_seed(),
            temperature=self.community_temperature,
        )
        self.community_labels = torch.as_tensor(labels, dtype=torch.long, device=self.device)

        self.community_nodes = []
        community_membership = torch.zeros(
            (self.max_communities, self.num_nodes),
            dtype=torch.float32,
            device=self.device,
        )
        action_mask = torch.zeros(
            (self.max_communities, self.num_nodes, self.num_nodes),
            dtype=torch.float32,
            device=self.device,
        )
        active_agent_mask = torch.zeros(self.max_communities, dtype=torch.float32, device=self.device)

        unique_labels = torch.unique(self.community_labels, sorted=True)
        for agent_index, label in enumerate(unique_labels.tolist()):
            nodes = torch.nonzero(self.community_labels == label, as_tuple=False).flatten()
            self.community_nodes.append(nodes)
            community_membership[agent_index, nodes] = 1.0
            action_mask[agent_index, nodes, :] = 1.0
            active_agent_mask[agent_index] = 1.0

        self.num_active_agents = len(self.community_nodes)
        self.community_membership = community_membership
        self.action_mask = action_mask
        self.active_agent_mask = active_agent_mask

    def _community_seed(self) -> int | None:
        if self._episode_seed is None:
            return None
        return int(self._episode_seed + (9973 * self.current_step))

    def _resolve_step_action(
        self,
        action: Mapping[int | str, np.ndarray | torch.Tensor] | np.ndarray | torch.Tensor,
    ) -> tuple[torch.Tensor, str, bool, bool]:
        if self.should_request_new_action():
            formatted_action = self._format_action(action)
            self.last_applied_action = formatted_action.detach().clone()
            return formatted_action, "fresh", True, False

        if self.hold_last_action:
            if self.last_applied_action is None:
                raise RuntimeError("No previous action is available to hold for this step.")
            return self.last_applied_action, "held", False, True

        return (
            torch.zeros((self.num_nodes, self.num_nodes), dtype=torch.float32, device=self.device),
            "none",
            False,
            False,
        )

    def _format_action(
        self,
        action: Mapping[int | str, np.ndarray | torch.Tensor] | np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        if self.community_membership is None:
            raise RuntimeError("Call reset() before formatting actions.")

        joint_action = torch.zeros((self.num_nodes, self.num_nodes), dtype=torch.float32, device=self.device)
        if isinstance(action, Mapping):
            for key, value in action.items():
                agent_index = self._parse_agent_id(key)
                if agent_index >= self.num_active_agents:
                    continue

                nodes = self.community_nodes[agent_index]
                local_action = torch.as_tensor(value, dtype=torch.float32, device=self.device)
                if local_action.shape == (self.num_nodes, self.num_nodes):
                    joint_action[nodes] = local_action[nodes]
                    continue
                if local_action.shape != (nodes.numel(), self.num_nodes):
                    raise ValueError(
                        "Local community actions must have shape (community_size, num_nodes) "
                        "or a full (num_nodes, num_nodes) matrix."
                    )
                joint_action[nodes] = local_action
            return joint_action.clamp(-1.0, 1.0)

        action_tensor = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        if action_tensor.shape != (self.num_nodes, self.num_nodes):
            raise ValueError(
                f"Action tensor must have shape {(self.num_nodes, self.num_nodes)}, "
                f"got {tuple(action_tensor.shape)}."
            )
        return action_tensor.clamp(-1.0, 1.0)

    def _apply_action_deltas(self, weights: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        base_logits = torch.log(weights.clamp_min(self.small_value))
        return torch.softmax(base_logits + delta, dim=-1)

    def _get_observation(self, consensus: torch.Tensor) -> dict[str, np.ndarray]:
        assert self.weight_matrix is not None
        assert self.opinions is not None
        assert self.community_labels is not None
        assert self.community_membership is not None
        assert self.action_mask is not None
        assert self.active_agent_mask is not None

        return {
            "opinions": self._to_numpy(self.opinions),
            "weight_matrix": self._to_numpy(self.weight_matrix),
            "community_labels": self.community_labels.detach().cpu().numpy().astype(np.int32, copy=False),
            "community_membership": self._to_numpy(self.community_membership),
            "action_mask": self._to_numpy(self.action_mask),
            "active_agent_mask": self._to_numpy(self.active_agent_mask),
            "global_consensus": np.array([float(consensus.item())], dtype=np.float32),
            "step_count": np.array([self.current_step], dtype=np.int32),
        }

    def _build_info(
        self,
        *,
        consensus: torch.Tensor,
        consensus_gain: torch.Tensor,
        terminal_bonus: torch.Tensor,
        action_penalty: torch.Tensor,
        bridge_penalty: torch.Tensor,
        step_penalty: torch.Tensor,
        reward: torch.Tensor,
        harmful_bridge: torch.Tensor,
        action_magnitude: torch.Tensor,
        community_refreshed: bool,
        applied_action_source: str,
        used_fresh_intervention: bool,
        used_held_action: bool,
    ) -> dict[str, Any]:
        assert self.weight_matrix is not None
        assert self.community_labels is not None

        cross_weight = cross_community_weight(self.weight_matrix, self.community_labels)
        return {
            "intervention_interval": self.intervention_interval,
            "hold_last_action": self.hold_last_action,
            "intervention_mode": (
                "stepwise"
                if self.intervention_interval == 1
                else ("periodic_hold" if self.hold_last_action else "periodic_no_hold")
            ),
            "applied_action_source": applied_action_source,
            "used_fresh_intervention": used_fresh_intervention,
            "used_held_action": used_held_action,
            "global_consensus": float(consensus.item()),
            "consensus_gain": float(consensus_gain.item()),
            "terminal_bonus": float(terminal_bonus.item()),
            "step_penalty": float(step_penalty.item()),
            "action_penalty": float(action_penalty.item()),
            "bridge_penalty": float(bridge_penalty.item()),
            "harmful_bridge_mass": float(harmful_bridge.item()),
            "action_magnitude": float(action_magnitude.item()),
            "reward": float(reward.item()),
            "cross_community_weight": float(cross_weight.item()),
            "community_refreshed": community_refreshed,
            "num_active_agents": self.num_active_agents,
            "agent_ids": self.get_agent_ids(),
            "community_nodes": {
                f"community_{agent_index}": nodes.detach().cpu().tolist()
                for agent_index, nodes in enumerate(self.community_nodes)
            },
        }

    @staticmethod
    def _parse_agent_id(key: int | str) -> int:
        if isinstance(key, int):
            return key
        digits = "".join(character for character in str(key) if character.isdigit())
        if digits == "":
            raise ValueError(f"Could not parse community id from action key: {key}")
        return int(digits)

    @staticmethod
    def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy().astype(np.float32, copy=False)
