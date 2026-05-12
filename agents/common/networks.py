from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn


def _resolve_activation(name: str) -> type[nn.Module]:
    activations = {
        "elu": nn.ELU,
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
    }
    try:
        return activations[name.lower()]
    except KeyError as error:
        supported = ", ".join(sorted(activations))
        raise ValueError(f"Unsupported activation '{name}'. Available options: {supported}.") from error


def _build_mlp(
    input_dim: int | None,
    hidden_dims: Sequence[int],
    output_dim: int,
    *,
    activation: str = "relu",
    dropout: float = 0.0,
    layer_norm: bool = False,
    activate_final: bool = False,
) -> nn.Sequential:
    activation_cls = _resolve_activation(activation)
    layers: list[nn.Module] = []
    previous_dim = input_dim
    all_dims = [*hidden_dims, output_dim]

    for index, hidden_dim in enumerate(all_dims):
        is_last = index == len(all_dims) - 1
        linear = nn.LazyLinear(hidden_dim) if previous_dim is None else nn.Linear(previous_dim, hidden_dim)
        layers.append(linear)

        if not is_last or activate_final:
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(activation_cls())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))

        previous_dim = hidden_dim

    return nn.Sequential(*layers)


def _to_tensor(
    value: Any,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> Tensor:
    if isinstance(value, np.ndarray):
        tensor = torch.from_numpy(np.array(value, copy=True))
    else:
        tensor = torch.as_tensor(value)
    tensor = tensor.to(dtype=dtype)
    if device is not None:
        tensor = tensor.to(device=device)
    return tensor


def _module_device(module: nn.Module) -> torch.device:
    parameter = next(module.parameters(), None)
    if parameter is not None:
        return parameter.device
    return torch.device("cpu")


@dataclass(frozen=True)
class GraphEncoderConfig:
    gnn_hidden_dims: tuple[int, ...] = (128, 128)
    graph_hidden_dims: tuple[int, ...] = (128,)
    graph_embedding_dim: int = 128
    activation: str = "relu"
    dropout: float = 0.0
    layer_norm: bool = True
    add_self_loops: bool = True
    step_count_scale: float = 100.0


@dataclass(frozen=True)
class SharedActorConfig:
    encoder: GraphEncoderConfig = field(default_factory=GraphEncoderConfig)
    edge_hidden_dims: tuple[int, ...] = (128, 64)
    activation: str = "relu"
    dropout: float = 0.0
    layer_norm: bool = True


@dataclass(frozen=True)
class CriticConfig:
    encoder: GraphEncoderConfig = field(default_factory=GraphEncoderConfig)
    edge_hidden_dims: tuple[int, ...] = (128, 64)
    value_hidden_dims: tuple[int, ...] = (256, 128)
    activation: str = "relu"
    dropout: float = 0.0
    layer_norm: bool = True


@dataclass
class EncodedGraphState:
    node_embeddings: Tensor
    graph_embedding: Tensor
    adjacency: Tensor
    opinions: Tensor


class GraphConvolution(nn.Module):
    """Simple GCN-style message passing over a weighted adjacency matrix."""

    def __init__(self, input_dim: int | None, output_dim: int, *, add_self_loops: bool = True) -> None:
        super().__init__()
        self.message_linear = nn.LazyLinear(output_dim) if input_dim is None else nn.Linear(input_dim, output_dim)
        self.root_linear = (
            nn.LazyLinear(output_dim, bias=False) if input_dim is None else nn.Linear(input_dim, output_dim, bias=False)
        )
        self.add_self_loops = add_self_loops

    def forward(self, node_features: Tensor, adjacency: Tensor) -> Tensor:
        propagated_adjacency = adjacency
        if self.add_self_loops:
            identity = torch.eye(adjacency.size(-1), device=adjacency.device, dtype=adjacency.dtype)
            propagated_adjacency = adjacency + identity

        degree = propagated_adjacency.sum(dim=-1).clamp_min(1e-6)
        inv_sqrt_degree = degree.pow(-0.5)
        normalized_adjacency = inv_sqrt_degree.unsqueeze(-1) * propagated_adjacency * inv_sqrt_degree.unsqueeze(-2)
        messages = normalized_adjacency @ node_features
        return self.message_linear(messages) + self.root_linear(node_features)


class GraphObservationEncoder(nn.Module):
    """Encode the global graph state into node and graph embeddings."""

    def __init__(self, config: GraphEncoderConfig) -> None:
        super().__init__()
        if not config.gnn_hidden_dims:
            raise ValueError("GraphEncoderConfig.gnn_hidden_dims must contain at least one hidden dimension.")

        self.config = config
        self.activation = _resolve_activation(config.activation)()
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()

        self.gnn_layers = nn.ModuleList()
        self.gnn_norms = nn.ModuleList()
        previous_dim: int | None = None
        for hidden_dim in config.gnn_hidden_dims:
            self.gnn_layers.append(
                GraphConvolution(previous_dim, hidden_dim, add_self_loops=config.add_self_loops)
            )
            self.gnn_norms.append(nn.LayerNorm(hidden_dim) if config.layer_norm else nn.Identity())
            previous_dim = hidden_dim

        self.graph_projection = _build_mlp(
            input_dim=None,
            hidden_dims=config.graph_hidden_dims,
            output_dim=config.graph_embedding_dim,
            activation=config.activation,
            dropout=config.dropout,
            layer_norm=config.layer_norm,
        )

    def forward(self, state: Mapping[str, Any], *, pool_mask_key: str | None = None) -> EncodedGraphState:
        device = _module_device(self)
        adjacency = _to_tensor(state["weight_matrix"], device=device)
        opinions = _to_tensor(state["opinions"], device=device).reshape(-1)

        if adjacency.ndim != 2:
            raise ValueError("GraphObservationEncoder expects an unbatched (num_nodes, num_nodes) adjacency matrix.")
        if opinions.numel() != adjacency.size(0):
            raise ValueError("Opinion vector length must match the graph node count.")

        out_degree = adjacency.sum(dim=-1, keepdim=True)
        in_degree = adjacency.sum(dim=-2).unsqueeze(-1)
        node_features = [opinions.unsqueeze(-1), out_degree, in_degree]

        pool_mask: Tensor | None = None
        if pool_mask_key is not None and pool_mask_key in state:
            pool_mask = _to_tensor(state[pool_mask_key], device=device).reshape(-1)
            if pool_mask.numel() != adjacency.size(0):
                raise ValueError(f"Pool mask '{pool_mask_key}' must have shape (num_nodes,).")
            node_features.append(pool_mask.unsqueeze(-1))

        membership = state.get("community_membership")
        if membership is not None:
            membership_tensor = _to_tensor(membership, device=device)
            if membership_tensor.ndim != 2:
                raise ValueError("community_membership must have shape (num_communities, num_nodes).")
            community_sizes = membership_tensor.sum(dim=-1)
            node_community_size = torch.einsum("an,a->n", membership_tensor, community_sizes)
            node_features.append((node_community_size / adjacency.size(0)).unsqueeze(-1))

        global_features: list[Tensor] = []
        if "global_consensus" in state:
            global_features.append(_to_tensor(state["global_consensus"], device=device).reshape(1))
        if "step_count" in state:
            step_count = _to_tensor(state["step_count"], device=device).reshape(1) / self.config.step_count_scale
            global_features.append(step_count)
        if "active_agent_mask" in state:
            active_agent_mask = _to_tensor(state["active_agent_mask"], device=device).reshape(-1)
            global_features.append(active_agent_mask.mean().reshape(1))
        if pool_mask is not None:
            global_features.append(pool_mask.mean().reshape(1))

        if global_features:
            repeated = torch.cat(global_features).unsqueeze(0).expand(adjacency.size(0), -1)
            node_features.append(repeated)

        node_embeddings = torch.cat(node_features, dim=-1)
        for gnn_layer, layer_norm in zip(self.gnn_layers, self.gnn_norms):
            node_embeddings = gnn_layer(node_embeddings, adjacency)
            node_embeddings = layer_norm(node_embeddings)
            node_embeddings = self.activation(node_embeddings)
            node_embeddings = self.dropout(node_embeddings)

        if pool_mask is None:
            pooled = node_embeddings.mean(dim=0)
        else:
            masked = node_embeddings * pool_mask.unsqueeze(-1)
            pooled = masked.sum(dim=0) / pool_mask.sum().clamp_min(1.0)

        graph_embedding = self.graph_projection(pooled)
        return EncodedGraphState(
            node_embeddings=node_embeddings,
            graph_embedding=graph_embedding,
            adjacency=adjacency,
            opinions=opinions,
        )


class RuntimeCommunityActor(nn.Module):
    """Shared actor applied once per active community."""

    def __init__(self, config: SharedActorConfig | None = None) -> None:
        super().__init__()
        self.config = config or SharedActorConfig()
        self.encoder = GraphObservationEncoder(self.config.encoder)
        node_dim = self.config.encoder.gnn_hidden_dims[-1]
        edge_input_dim = (2 * node_dim) + 2 + self.config.encoder.graph_embedding_dim
        self.edge_head = _build_mlp(
            input_dim=edge_input_dim,
            hidden_dims=self.config.edge_hidden_dims,
            output_dim=1,
            activation=self.config.activation,
            dropout=self.config.dropout,
            layer_norm=self.config.layer_norm,
        )

    def forward(
        self,
        state: Mapping[str, Any],
        *,
        community_mask: Tensor,
        local_action_mask: Tensor,
    ) -> Tensor:
        encoded = self.encoder({**state, "community_mask": community_mask}, pool_mask_key="community_mask")
        source_nodes = torch.nonzero(community_mask > 0.5, as_tuple=False).flatten()
        if source_nodes.numel() == 0:
            return torch.zeros((0, encoded.adjacency.size(0)), dtype=torch.float32, device=encoded.adjacency.device)

        source_embeddings = encoded.node_embeddings[source_nodes]
        target_embeddings = encoded.node_embeddings
        local_size, num_nodes = source_embeddings.size(0), target_embeddings.size(0)

        source_expand = source_embeddings.unsqueeze(1).expand(local_size, num_nodes, -1)
        target_expand = target_embeddings.unsqueeze(0).expand(local_size, num_nodes, -1)
        current_weights = encoded.adjacency[source_nodes].unsqueeze(-1)
        opinion_gap = torch.abs(
            encoded.opinions[source_nodes].unsqueeze(1) - encoded.opinions.unsqueeze(0)
        ).unsqueeze(-1)
        global_context = encoded.graph_embedding.unsqueeze(0).unsqueeze(0).expand(local_size, num_nodes, -1)

        features = torch.cat((source_expand, target_expand, current_weights, opinion_gap, global_context), dim=-1)
        logits = self.edge_head(features).squeeze(-1)
        return logits * local_action_mask


class CentralizedCritic(nn.Module):
    """Centralized critic over the global graph state and assembled joint action matrix."""

    def __init__(self, config: CriticConfig | None = None) -> None:
        super().__init__()
        self.config = config or CriticConfig()
        self.encoder = GraphObservationEncoder(self.config.encoder)
        node_dim = self.config.encoder.gnn_hidden_dims[-1]
        edge_input_dim = (2 * node_dim) + 3 + self.config.encoder.graph_embedding_dim
        self.edge_head = _build_mlp(
            input_dim=edge_input_dim,
            hidden_dims=self.config.edge_hidden_dims,
            output_dim=self.config.encoder.graph_embedding_dim,
            activation=self.config.activation,
            dropout=self.config.dropout,
            layer_norm=self.config.layer_norm,
        )
        self.value_head = _build_mlp(
            input_dim=2 * self.config.encoder.graph_embedding_dim,
            hidden_dims=self.config.value_hidden_dims,
            output_dim=1,
            activation=self.config.activation,
            dropout=self.config.dropout,
            layer_norm=self.config.layer_norm,
        )

    def forward(self, state: Mapping[str, Any], joint_action: Tensor | np.ndarray) -> Tensor:
        encoded = self.encoder(state)
        action_tensor = _to_tensor(joint_action, device=encoded.adjacency.device)
        if action_tensor.shape != encoded.adjacency.shape:
            raise ValueError(
                f"Critic expected joint_action shape {tuple(encoded.adjacency.shape)}, got {tuple(action_tensor.shape)}."
            )

        num_nodes = encoded.adjacency.size(0)
        source_expand = encoded.node_embeddings.unsqueeze(1).expand(num_nodes, num_nodes, -1)
        target_expand = encoded.node_embeddings.unsqueeze(0).expand(num_nodes, num_nodes, -1)
        adjacency = encoded.adjacency.unsqueeze(-1)
        action_feature = action_tensor.unsqueeze(-1)
        opinion_gap = torch.abs(encoded.opinions.unsqueeze(1) - encoded.opinions.unsqueeze(0)).unsqueeze(-1)
        global_context = encoded.graph_embedding.unsqueeze(0).unsqueeze(0).expand(num_nodes, num_nodes, -1)

        features = torch.cat(
            (source_expand, target_expand, adjacency, action_feature, opinion_gap, global_context),
            dim=-1,
        )
        edge_embeddings = self.edge_head(features)
        pooled_action_embedding = edge_embeddings.mean(dim=(0, 1))
        critic_input = torch.cat((encoded.graph_embedding, pooled_action_embedding), dim=-1)
        return self.value_head(critic_input).reshape(())
