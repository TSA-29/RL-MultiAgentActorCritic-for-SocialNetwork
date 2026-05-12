from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Normal

from .common import (
    CentralizedCritic,
    CriticConfig,
    GraphEncoderConfig,
    ReplayBuffer,
    RuntimeCommunityActor,
    SharedActorConfig,
)


def _parse_agent_index(key: int | str) -> int:
    if isinstance(key, int):
        return key
    digits = "".join(character for character in str(key) if character.isdigit())
    if not digits:
        raise ValueError(f"Could not parse an agent id from key '{key}'.")
    return int(digits)


@dataclass(frozen=True)
class MAACConfig:
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    entropy_coef: float = 0.2
    init_log_std: float = -0.5
    min_log_std: float = -5.0
    max_log_std: float = 1.0
    gradient_clip_norm: float | None = 1.0
    replay_capacity: int = 100_000
    policy_epsilon: float = 1e-6


class MAACAgent(nn.Module):
    """Shared-actor MAAC trainer with a centralized critic."""

    def __init__(
        self,
        num_nodes: int,
        *,
        actor_config: SharedActorConfig | None = None,
        critic_config: CriticConfig | None = None,
        maac_config: MAACConfig | None = None,
        replay_buffer: ReplayBuffer | None = None,
        seed: int | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()

        if num_nodes <= 0:
            raise ValueError("num_nodes must be positive.")

        requested_device = torch.device(device)
        self.num_nodes = int(num_nodes)
        self.config = maac_config or MAACConfig()
        self.actor = RuntimeCommunityActor(actor_config).to(requested_device)
        self.critic = CentralizedCritic(critic_config).to(requested_device)
        self._initialize_modules(device=requested_device)
        # Lazy layers materialize during _initialize_modules; move the full modules again
        # so the live agent stays on the requested device before checkpointing or rollout.
        self.actor = self.actor.to(requested_device)
        self.critic = self.critic.to(requested_device)
        self.target_critic = deepcopy(self.critic).to(requested_device)
        self.target_critic.requires_grad_(False)

        self.log_std = nn.Parameter(
            torch.tensor(float(self.config.init_log_std), dtype=torch.float32, device=requested_device)
        )
        self.actor_optimizer = torch.optim.Adam([*self.actor.parameters(), self.log_std], lr=self.config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.critic_lr)
        self.replay_buffer = replay_buffer or ReplayBuffer(capacity=self.config.replay_capacity, seed=seed)

    @property
    def device(self) -> torch.device:
        return next(self.actor.parameters()).device

    def select_actions(
        self,
        global_state: Mapping[str, Any],
        *,
        deterministic: bool = False,
        as_numpy: bool = True,
    ) -> dict[str, np.ndarray | Tensor]:
        state = self._state_to_device(global_state)
        actions, _, _ = self._sample_joint_action(state, deterministic=deterministic)

        formatted: dict[str, np.ndarray | Tensor] = {}
        for agent_id, local_action in actions.items():
            if as_numpy:
                formatted[agent_id] = local_action.detach().cpu().numpy().astype(np.float32, copy=False)
            else:
                formatted[agent_id] = local_action.detach()
        return formatted

    def assemble_joint_action(
        self,
        global_state: Mapping[str, Any],
        actions: Mapping[int | str, np.ndarray | Tensor],
    ) -> Tensor:
        state = self._state_to_device(global_state)
        joint_action = torch.zeros((self.num_nodes, self.num_nodes), dtype=torch.float32, device=self.device)

        for key, value in actions.items():
            agent_index = _parse_agent_index(key)
            source_nodes = self._source_nodes(state, agent_index)
            local_action = torch.as_tensor(value, dtype=torch.float32, device=self.device)

            if local_action.shape == (self.num_nodes, self.num_nodes):
                joint_action[source_nodes] = local_action[source_nodes]
            elif local_action.shape == (source_nodes.numel(), self.num_nodes):
                joint_action[source_nodes] = local_action
            else:
                raise ValueError(
                    f"Action for agent {key} must have shape ({source_nodes.numel()}, {self.num_nodes}) "
                    f"or ({self.num_nodes}, {self.num_nodes})."
                )
        return joint_action.clamp(-1.0, 1.0)

    def store_transition(
        self,
        *,
        global_state: Mapping[str, Any],
        actions: Mapping[int | str, np.ndarray | Tensor] | np.ndarray | Tensor,
        reward: float,
        next_global_state: Mapping[str, Any],
        done: bool,
    ) -> None:
        if isinstance(actions, Mapping):
            joint_action = self.assemble_joint_action(global_state, actions)
        else:
            joint_action = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
            if joint_action.shape != (self.num_nodes, self.num_nodes):
                raise ValueError(
                    f"Joint action must have shape {(self.num_nodes, self.num_nodes)}, "
                    f"got {tuple(joint_action.shape)}."
                )
            joint_action = joint_action.clamp(-1.0, 1.0)
        self.replay_buffer.add(
            global_state=global_state,
            joint_action=joint_action.detach().cpu(),
            reward=float(reward),
            next_global_state=next_global_state,
            done=done,
        )

    def update(self, batch_size: int | None = None) -> dict[str, float]:
        batch_size = batch_size or self.config.batch_size
        if len(self.replay_buffer) < batch_size:
            return {}

        transitions = self.replay_buffer.sample(batch_size)
        critic_targets: list[Tensor] = []
        current_qs: list[Tensor] = []

        with torch.no_grad():
            for transition in transitions:
                next_state = self._state_to_device(transition.next_global_state)
                _, next_log_prob, next_joint_action = self._sample_joint_action(next_state, deterministic=False)
                target_q = self.target_critic(next_state, next_joint_action)
                reward = torch.tensor(float(transition.reward), dtype=torch.float32, device=self.device)
                done = torch.tensor(float(transition.done), dtype=torch.float32, device=self.device)
                bellman_target = reward + ((1.0 - done) * self.config.gamma * (target_q - (self.config.entropy_coef * next_log_prob)))
                critic_targets.append(bellman_target)

        for transition, target in zip(transitions, critic_targets):
            state = self._state_to_device(transition.global_state)
            joint_action = transition.joint_action.to(device=self.device, dtype=torch.float32)
            q_value = self.critic(state, joint_action)
            current_qs.append(q_value)

        critic_loss = torch.stack([(q_value - target).pow(2) for q_value, target in zip(current_qs, critic_targets)]).mean()
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self._clip_gradients(self.critic.parameters())
        self.critic_optimizer.step()

        self.critic.requires_grad_(False)
        try:
            actor_terms: list[Tensor] = []
            sampled_q_values: list[Tensor] = []
            sampled_log_probs: list[Tensor] = []
            for transition in transitions:
                state = self._state_to_device(transition.global_state)
                _, log_prob, joint_action = self._sample_joint_action(state, deterministic=False)
                q_value = self.critic(state, joint_action)
                actor_terms.append((self.config.entropy_coef * log_prob) - q_value)
                sampled_q_values.append(q_value)
                sampled_log_probs.append(log_prob)

            actor_loss = torch.stack(actor_terms).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self._clip_gradients(self.actor.parameters())
            self._clip_gradients([self.log_std])
            self.actor_optimizer.step()
            with torch.no_grad():
                self.log_std.clamp_(self.config.min_log_std, self.config.max_log_std)
        finally:
            self.critic.requires_grad_(True)

        self._soft_update(self.target_critic, self.critic)
        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "mean_q": float(torch.stack(current_qs).mean().item()),
            "mean_target_q": float(torch.stack(critic_targets).mean().item()),
            "mean_policy_q": float(torch.stack(sampled_q_values).mean().item()),
            "mean_log_prob": float(torch.stack(sampled_log_probs).mean().item()),
            "log_std": float(self.log_std.item()),
        }

    def build_checkpoint(self) -> dict[str, Any]:
        return {
            "num_nodes": self.num_nodes,
            "actor_config": asdict(self.actor.config),
            "critic_config": asdict(self.critic.config),
            "maac_config": asdict(self.config),
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "target_critic_state_dict": self.target_critic.state_dict(),
            "log_std": self.log_std.detach().cpu(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
        }

    def _initialize_modules(self, *, device: torch.device) -> None:
        dummy_membership = torch.zeros((self.num_nodes, self.num_nodes), dtype=torch.float32, device=device)
        dummy_membership[0] = 1.0
        dummy_action_mask = torch.zeros(
            (self.num_nodes, self.num_nodes, self.num_nodes),
            dtype=torch.float32,
            device=device,
        )
        dummy_action_mask[0] = 1.0
        dummy_state = {
            "opinions": torch.zeros((self.num_nodes,), dtype=torch.float32, device=device),
            "weight_matrix": torch.eye(self.num_nodes, dtype=torch.float32, device=device),
            "community_membership": dummy_membership,
            "action_mask": dummy_action_mask,
            "active_agent_mask": torch.cat(
                (
                    torch.ones(1, dtype=torch.float32, device=device),
                    torch.zeros(self.num_nodes - 1, dtype=torch.float32, device=device),
                )
            ),
            "global_consensus": torch.zeros((1,), dtype=torch.float32, device=device),
            "step_count": torch.zeros((1,), dtype=torch.float32, device=device),
        }
        community_mask = dummy_membership[0]
        local_action_mask = torch.ones((self.num_nodes, self.num_nodes), dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = self.actor(dummy_state, community_mask=community_mask, local_action_mask=local_action_mask)
            self.critic(dummy_state, logits)

    def load_checkpoint(self, checkpoint: Mapping[str, Any], *, load_optimizers: bool = True) -> None:
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.target_critic.load_state_dict(checkpoint["target_critic_state_dict"])
        self.log_std.data.copy_(torch.as_tensor(checkpoint["log_std"], dtype=torch.float32, device=self.device))

        if load_optimizers:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Mapping[str, Any],
        *,
        device: str | torch.device = "cpu",
        load_optimizers: bool = False,
    ) -> "MAACAgent":
        actor_config = _shared_actor_config_from_dict(checkpoint["actor_config"])
        critic_config = _critic_config_from_dict(checkpoint["critic_config"])
        maac_config = _maac_config_from_dict(checkpoint["maac_config"])
        agent = cls(
            num_nodes=int(checkpoint["num_nodes"]),
            actor_config=actor_config,
            critic_config=critic_config,
            maac_config=maac_config,
            device=device,
        )
        agent.load_checkpoint(checkpoint, load_optimizers=load_optimizers)
        return agent

    def _sample_joint_action(
        self,
        state: Mapping[str, Tensor],
        *,
        deterministic: bool,
    ) -> tuple[dict[str, Tensor], Tensor, Tensor]:
        total_log_prob = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        joint_action = torch.zeros((self.num_nodes, self.num_nodes), dtype=torch.float32, device=self.device)
        local_actions: dict[str, Tensor] = {}

        for agent_index in self._active_agent_indices(state):
            community_mask = self._community_mask(state, agent_index)
            source_nodes = self._source_nodes(state, agent_index)
            local_action_mask = self._local_action_mask(state, agent_index, source_nodes)
            logits = self.actor(
                state,
                community_mask=community_mask,
                local_action_mask=local_action_mask,
            )
            sampled_action, log_prob = self._sample_policy(logits, local_action_mask, deterministic=deterministic)
            joint_action[source_nodes] = sampled_action
            local_actions[f"community_{agent_index}"] = sampled_action
            total_log_prob = total_log_prob + log_prob

        return local_actions, total_log_prob, joint_action

    def _sample_policy(
        self,
        logits: Tensor,
        action_mask: Tensor,
        *,
        deterministic: bool,
    ) -> tuple[Tensor, Tensor]:
        if logits.numel() == 0:
            zero = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            return logits, zero

        clipped_log_std = self.log_std.clamp(self.config.min_log_std, self.config.max_log_std)
        std = clipped_log_std.exp()
        distribution = Normal(logits, std)
        raw_action = logits if deterministic else distribution.rsample()
        squashed_action = torch.tanh(raw_action)
        action = squashed_action * action_mask

        if deterministic:
            return action, torch.tensor(0.0, dtype=torch.float32, device=self.device)

        correction = torch.log((1.0 - squashed_action.pow(2)).clamp_min(self.config.policy_epsilon))
        log_prob = (distribution.log_prob(raw_action) - correction) * action_mask
        return action, log_prob.sum()

    def _active_agent_indices(self, state: Mapping[str, Tensor]) -> list[int]:
        active_mask = state.get("active_agent_mask")
        if active_mask is None:
            membership = state["community_membership"]
            return list(range(membership.size(0)))

        flat_active = active_mask.reshape(-1)
        return [index for index, value in enumerate(flat_active.tolist()) if value > 0.5]

    def _community_mask(self, state: Mapping[str, Tensor], agent_index: int) -> Tensor:
        membership = state["community_membership"]
        return membership[agent_index]

    def _source_nodes(self, state: Mapping[str, Tensor], agent_index: int) -> Tensor:
        community_mask = self._community_mask(state, agent_index)
        return torch.nonzero(community_mask > 0.5, as_tuple=False).flatten()

    def _local_action_mask(
        self,
        state: Mapping[str, Tensor],
        agent_index: int,
        source_nodes: Tensor,
    ) -> Tensor:
        action_mask = state.get("action_mask")
        if action_mask is None:
            return torch.ones((source_nodes.numel(), self.num_nodes), dtype=torch.float32, device=self.device)
        return action_mask[agent_index, source_nodes].to(dtype=torch.float32)

    def _state_to_device(self, state: Mapping[str, Any]) -> dict[str, Tensor]:
        return {
            key: torch.as_tensor(value, device=self.device).to(
                dtype=torch.float32 if key not in {"community_labels", "step_count"} else torch.float32
            )
            for key, value in state.items()
        }

    def _clip_gradients(self, parameters: Any) -> None:
        if self.config.gradient_clip_norm is None:
            return
        torch.nn.utils.clip_grad_norm_(list(parameters), max_norm=self.config.gradient_clip_norm)

    def _soft_update(self, target: nn.Module, source: nn.Module) -> None:
        with torch.no_grad():
            for target_parameter, source_parameter in zip(target.parameters(), source.parameters()):
                target_parameter.data.mul_(1.0 - self.config.tau).add_(self.config.tau * source_parameter.data)
            for target_buffer, source_buffer in zip(target.buffers(), source.buffers()):
                target_buffer.copy_(source_buffer)


def _graph_encoder_config_from_dict(payload: Mapping[str, Any]) -> GraphEncoderConfig:
    return GraphEncoderConfig(
        gnn_hidden_dims=tuple(payload.get("gnn_hidden_dims", (128, 128))),
        graph_hidden_dims=tuple(payload.get("graph_hidden_dims", (128,))),
        graph_embedding_dim=int(payload.get("graph_embedding_dim", 128)),
        activation=str(payload.get("activation", "relu")),
        dropout=float(payload.get("dropout", 0.0)),
        layer_norm=bool(payload.get("layer_norm", True)),
        add_self_loops=bool(payload.get("add_self_loops", True)),
        step_count_scale=float(payload.get("step_count_scale", 100.0)),
    )


def _shared_actor_config_from_dict(payload: Mapping[str, Any]) -> SharedActorConfig:
    return SharedActorConfig(
        encoder=_graph_encoder_config_from_dict(payload.get("encoder", {})),
        edge_hidden_dims=tuple(payload.get("edge_hidden_dims", (128, 64))),
        activation=str(payload.get("activation", "relu")),
        dropout=float(payload.get("dropout", 0.0)),
        layer_norm=bool(payload.get("layer_norm", True)),
    )


def _critic_config_from_dict(payload: Mapping[str, Any]) -> CriticConfig:
    return CriticConfig(
        encoder=_graph_encoder_config_from_dict(payload.get("encoder", {})),
        edge_hidden_dims=tuple(payload.get("edge_hidden_dims", (128, 64))),
        value_hidden_dims=tuple(payload.get("value_hidden_dims", (256, 128))),
        activation=str(payload.get("activation", "relu")),
        dropout=float(payload.get("dropout", 0.0)),
        layer_norm=bool(payload.get("layer_norm", True)),
    )


def _maac_config_from_dict(payload: Mapping[str, Any]) -> MAACConfig:
    return MAACConfig(
        batch_size=int(payload.get("batch_size", 256)),
        gamma=float(payload.get("gamma", 0.99)),
        tau=float(payload.get("tau", 0.005)),
        actor_lr=float(payload.get("actor_lr", 3e-4)),
        critic_lr=float(payload.get("critic_lr", 3e-4)),
        entropy_coef=float(payload.get("entropy_coef", 0.2)),
        init_log_std=float(payload.get("init_log_std", -0.5)),
        min_log_std=float(payload.get("min_log_std", -5.0)),
        max_log_std=float(payload.get("max_log_std", 1.0)),
        gradient_clip_norm=(
            None if payload.get("gradient_clip_norm", 1.0) is None else float(payload.get("gradient_clip_norm", 1.0))
        ),
        replay_capacity=int(payload.get("replay_capacity", 100_000)),
        policy_epsilon=float(payload.get("policy_epsilon", 1e-6)),
    )


__all__ = ["MAACAgent", "MAACConfig"]
