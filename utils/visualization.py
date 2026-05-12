from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, Union

import numpy as np

from envs.social_network_env import SocialNetworkEnv


PolicyAction = Union[Mapping[str, np.ndarray], np.ndarray]
PolicySelector = Callable[[SocialNetworkEnv, Mapping[str, np.ndarray]], PolicyAction]


@dataclass(frozen=True)
class EpisodeTrace:
    policy_name: str
    detector: str
    seed: int
    max_steps: int
    intervention_interval: int
    intervention_mode: str
    consensus_threshold: float
    terminated: bool
    truncated: bool
    total_reward: float
    opinions: list[np.ndarray]
    weight_matrices: list[np.ndarray]
    community_labels: list[np.ndarray]
    consensus: list[float]
    cross_community_weight: list[float]
    rewards: list[float]
    action_magnitudes: list[float]
    harmful_bridge_mass: list[float]
    action_sources: list[str]
    fresh_action_steps: list[int]
    held_action_steps: list[int]
    community_refresh_steps: list[int]

    @property
    def step_count(self) -> int:
        return len(self.rewards)

    @property
    def final_consensus(self) -> float:
        return float(self.consensus[-1])

    @property
    def initial_community_count(self) -> int:
        return int(np.unique(self.community_labels[0]).size)

    @property
    def final_community_count(self) -> int:
        return int(np.unique(self.community_labels[-1]).size)


def rollout_episode_trace(
    *,
    env_kwargs: Mapping[str, Any],
    seed: int,
    policy_name: str,
    action_selector: PolicySelector,
) -> EpisodeTrace:
    env = SocialNetworkEnv(**dict(env_kwargs))
    observation, info = env.reset(seed=int(seed))
    total_reward = 0.0
    done = False
    terminated = False
    truncated = False

    opinions = [np.asarray(observation["opinions"], dtype=np.float32).copy()]
    weight_matrices = [np.asarray(observation["weight_matrix"], dtype=np.float32).copy()]
    community_labels = [np.asarray(observation["community_labels"], dtype=np.int32).copy()]
    consensus = [float(info["global_consensus"])]
    cross_community_weight = [float(info["cross_community_weight"])]
    rewards: list[float] = []
    action_magnitudes: list[float] = []
    harmful_bridge_mass: list[float] = []
    action_sources: list[str] = []
    fresh_action_steps: list[int] = []
    held_action_steps: list[int] = []
    community_refresh_steps: list[int] = [0]

    while not done:
        if env.should_request_new_action():
            action = env.schedule_action(action_selector(env, observation))
        else:
            action = env.schedule_action()

        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        step_index = len(rewards) + 1
        total_reward += float(reward)
        opinions.append(np.asarray(observation["opinions"], dtype=np.float32).copy())
        weight_matrices.append(np.asarray(observation["weight_matrix"], dtype=np.float32).copy())
        community_labels.append(np.asarray(observation["community_labels"], dtype=np.int32).copy())
        consensus.append(float(info["global_consensus"]))
        cross_community_weight.append(float(info["cross_community_weight"]))
        rewards.append(float(reward))
        action_magnitudes.append(float(info["action_magnitude"]))
        harmful_bridge_mass.append(float(info["harmful_bridge_mass"]))
        action_sources.append(str(info["applied_action_source"]))

        if bool(info["used_fresh_intervention"]):
            fresh_action_steps.append(step_index)
        if bool(info["used_held_action"]):
            held_action_steps.append(step_index)
        if bool(info["community_refreshed"]):
            community_refresh_steps.append(step_index)

    return EpisodeTrace(
        policy_name=policy_name,
        detector=env.community_detector,
        seed=int(seed),
        max_steps=env.max_steps,
        intervention_interval=env.intervention_interval,
        intervention_mode=str(info["intervention_mode"]),
        consensus_threshold=env.consensus_threshold,
        terminated=bool(terminated),
        truncated=bool(truncated),
        total_reward=total_reward,
        opinions=opinions,
        weight_matrices=weight_matrices,
        community_labels=community_labels,
        consensus=consensus,
        cross_community_weight=cross_community_weight,
        rewards=rewards,
        action_magnitudes=action_magnitudes,
        harmful_bridge_mass=harmful_bridge_mass,
        action_sources=action_sources,
        fresh_action_steps=fresh_action_steps,
        held_action_steps=held_action_steps,
        community_refresh_steps=community_refresh_steps,
    )


def plot_detector_comparison(
    *,
    rows: Sequence[Mapping[str, object]],
    output_path: str | Path,
) -> None:
    """Render a detector-comparison bar chart if matplotlib is available."""

    pyplot = _require_matplotlib()

    labels = [str(row["label"]) for row in rows]
    values = [float(row["mean_final_consensus"]) for row in rows]

    figure, axis = pyplot.subplots(figsize=(10, 4))
    axis.bar(labels, values, color="#4a7a5c")
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Mean Final Consensus")
    axis.set_title("Detector / Policy Comparison")
    figure.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    pyplot.close(figure)


def plot_training_diagnostics(
    *,
    updates: Sequence[Mapping[str, object]],
    output_path: str | Path,
    title: str | None = None,
    smoothing_window: int = 25,
) -> None:
    """Render MAAC actor/critic loss curves from saved training updates."""

    if smoothing_window < 1:
        raise ValueError("smoothing_window must be at least 1.")
    if not updates:
        raise ValueError("updates must not be empty.")

    pyplot = _require_matplotlib()
    x_values = np.asarray(
        [int(row.get("env_step", row.get("update", index + 1))) for index, row in enumerate(updates)],
        dtype=np.int32,
    )
    actor_loss = np.asarray([float(row["actor_loss"]) for row in updates], dtype=np.float32)
    critic_loss = np.asarray([float(row["critic_loss"]) for row in updates], dtype=np.float32)
    smoothed_actor = _causal_moving_average(actor_loss, window=min(smoothing_window, actor_loss.size))
    smoothed_critic = _causal_moving_average(critic_loss, window=min(smoothing_window, critic_loss.size))

    figure, axes = pyplot.subplots(1, 2, figsize=(14, 5.5), facecolor="#f6f1e8", constrained_layout=True)
    for axis in axes:
        axis.set_facecolor("#fdfbf7")
        axis.grid(alpha=0.22, linewidth=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.set_xlabel("Environment step")

    axes[0].plot(x_values, critic_loss, color="#a5b4fc", alpha=0.45, linewidth=1.2, label="Raw")
    axes[0].plot(x_values, smoothed_critic, color="#1d4ed8", linewidth=2.2, label="Smoothed")
    axes[0].set_title("Critic Loss", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Loss")
    axes[0].legend(frameon=False, loc="upper right")

    axes[1].plot(x_values, actor_loss, color="#c4b5fd", alpha=0.45, linewidth=1.2, label="Raw")
    axes[1].plot(x_values, smoothed_actor, color="#4338ca", linewidth=2.2, label="Smoothed")
    axes[1].set_title("Actor Loss", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("Loss")
    axes[1].legend(frameon=False, loc="upper right")

    final_step = int(x_values[-1])
    for axis in axes:
        axis.axvline(final_step, color="#475569", linestyle=":", linewidth=1.0, alpha=0.8)

    figure.suptitle(
        title or "MAAC Training Diagnostics",
        x=0.05,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color="#111827",
    )
    figure.text(
        0.05,
        0.955,
        (
            f"updates={len(updates)} | smoothing_window={min(smoothing_window, len(updates))} "
            f"| final_step={final_step}"
        ),
        ha="left",
        fontsize=10,
        color="#475569",
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    pyplot.close(figure)


def plot_episode_storyboard(
    *,
    trace: EpisodeTrace,
    output_path: str | Path,
    title: str | None = None,
) -> None:
    pyplot = _require_matplotlib()
    color_map = pyplot.colormaps["coolwarm"]

    figure = pyplot.figure(figsize=(16, 10), facecolor="#f6f1e8", constrained_layout=True)
    axes = figure.subplot_mosaic(
        [
            ["trajectory", "trajectory", "initial"],
            ["trajectory", "trajectory", "final"],
            ["metrics", "metrics", "summary"],
        ],
        gridspec_kw={"width_ratios": [1.6, 1.6, 1.25], "height_ratios": [1.2, 1.2, 0.9]},
    )
    for axis in axes.values():
        axis.set_facecolor("#fdfbf7")

    positions = _build_network_positions(
        initial_weights=trace.weight_matrices[0],
        final_weights=trace.weight_matrices[-1],
        seed=trace.seed,
    )
    snapshot_artist = _plot_network_snapshot(
        axis=axes["initial"],
        positions=positions,
        weight_matrix=trace.weight_matrices[0],
        opinions=trace.opinions[0],
        community_labels=trace.community_labels[0],
        title="Initial Network",
        color_map=color_map,
    )
    _plot_network_snapshot(
        axis=axes["final"],
        positions=positions,
        weight_matrix=trace.weight_matrices[-1],
        opinions=trace.opinions[-1],
        community_labels=trace.community_labels[-1],
        title="Final Network",
        color_map=color_map,
    )
    _plot_opinion_trajectories(axes["trajectory"], trace, color_map=color_map)
    _plot_metric_panel(axes["metrics"], trace)
    _plot_summary_panel(axes["summary"], trace)

    figure.colorbar(
        snapshot_artist,
        ax=[axes["initial"], axes["final"]],
        fraction=0.05,
        pad=0.02,
        label="Opinion value",
    )
    figure.suptitle(
        title or f"Opinion Dynamics Storyboard | {trace.policy_name}",
        x=0.05,
        ha="left",
        fontsize=20,
        fontweight="bold",
        color="#111827",
    )
    figure.text(
        0.05,
        0.955,
        (
            f"detector={trace.detector} | seed={trace.seed} | mode={trace.intervention_mode} "
            f"| fresh interventions={len(trace.fresh_action_steps)}"
        ),
        ha="left",
        fontsize=11,
        color="#475569",
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    pyplot.close(figure)


def _plot_opinion_trajectories(axis: Any, trace: EpisodeTrace, *, color_map: Any) -> None:
    opinion_matrix = np.stack(trace.opinions, axis=0)
    time_points = np.arange(opinion_matrix.shape[0], dtype=np.int32)
    start_colors = color_map((opinion_matrix[0] + 1.0) / 2.0)

    for node_index in range(opinion_matrix.shape[1]):
        trajectory = opinion_matrix[:, node_index]
        line_width = 1.0 + (0.9 * abs(float(trajectory[-1] - trajectory[0])))
        axis.plot(
            time_points,
            trajectory,
            color=start_colors[node_index],
            alpha=0.72,
            linewidth=line_width,
            zorder=2,
        )

    mean_opinion = opinion_matrix.mean(axis=1)
    for step in trace.fresh_action_steps:
        axis.axvline(step, color="#d97706", linewidth=2.0, alpha=0.09, zorder=1)

    axis.plot(time_points, mean_opinion, color="#111827", linewidth=2.8, label="Mean opinion", zorder=3)
    if trace.fresh_action_steps:
        action_steps = np.asarray(trace.fresh_action_steps, dtype=np.int32)
        axis.scatter(
            action_steps,
            mean_opinion[action_steps],
            s=34,
            color="#d97706",
            edgecolors="#ffffff",
            linewidths=0.7,
            zorder=4,
            label="Fresh intervention",
        )
    if trace.community_refresh_steps:
        refresh_steps = np.asarray(trace.community_refresh_steps, dtype=np.int32)
        axis.scatter(
            refresh_steps,
            np.full(refresh_steps.shape, -1.02, dtype=np.float32),
            s=24,
            color="#2563eb",
            edgecolors="none",
            clip_on=False,
            label="Community refresh",
            zorder=4,
        )

    axis.set_title("Opinion Trajectories", loc="left", fontsize=14, fontweight="bold", color="#111827")
    axis.set_xlabel("Environment step")
    axis.set_ylabel("Opinion value")
    axis.set_xlim(0, int(time_points[-1]))
    axis.set_ylim(-1.05, 1.05)
    axis.grid(alpha=0.25, linewidth=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(loc="upper right", frameon=False)


def _plot_metric_panel(axis: Any, trace: EpisodeTrace) -> None:
    time_points = np.arange(len(trace.consensus), dtype=np.int32)
    consensus = np.asarray(trace.consensus, dtype=np.float32)
    cross_weight = np.asarray(trace.cross_community_weight, dtype=np.float32)
    harmful_bridge = np.concatenate(([0.0], np.asarray(trace.harmful_bridge_mass, dtype=np.float32)))

    secondary_axis = axis.twinx()
    axis.plot(time_points, consensus, color="#0f766e", linewidth=2.5, label="Consensus")
    axis.axhline(
        trace.consensus_threshold,
        color="#0f766e",
        linewidth=1.2,
        linestyle="--",
        alpha=0.65,
        label="Consensus target",
    )
    secondary_axis.plot(
        time_points,
        cross_weight,
        color="#b45309",
        linewidth=2.0,
        label="Cross-community weight",
    )
    secondary_axis.fill_between(
        time_points,
        0.0,
        harmful_bridge,
        color="#dc2626",
        alpha=0.14,
        label="Harmful bridge mass",
    )

    for step in trace.fresh_action_steps:
        axis.axvline(step, color="#d97706", linewidth=1.8, alpha=0.08)

    axis.set_title("Consensus and Bridge Dynamics", loc="left", fontsize=14, fontweight="bold", color="#111827")
    axis.set_xlabel("Environment step")
    axis.set_ylabel("Consensus")
    secondary_axis.set_ylabel("Bridge weight")
    axis.set_xlim(0, int(time_points[-1]))
    axis.set_ylim(0.0, 1.02)
    secondary_axis.set_ylim(bottom=0.0)
    axis.grid(alpha=0.22, linewidth=0.8)
    axis.spines["top"].set_visible(False)
    secondary_axis.spines["top"].set_visible(False)

    left_handles, left_labels = axis.get_legend_handles_labels()
    right_handles, right_labels = secondary_axis.get_legend_handles_labels()
    axis.legend(left_handles + right_handles, left_labels + right_labels, loc="upper left", frameon=False, ncol=2)


def _plot_summary_panel(axis: Any, trace: EpisodeTrace) -> None:
    axis.axis("off")
    axis.set_title("Run Summary", loc="left", fontsize=14, fontweight="bold", color="#111827")

    outcome = "success" if trace.terminated else ("max steps" if trace.truncated else "stopped")
    summary_lines = [
        ("Policy", trace.policy_name),
        ("Detector", trace.detector),
        ("Seed", str(trace.seed)),
        ("Outcome", outcome),
        ("Steps", f"{trace.step_count} / {trace.max_steps}"),
        ("Final consensus", f"{trace.final_consensus:.3f}"),
        ("Episode reward", f"{trace.total_reward:.3f}"),
        ("Intervention cadence", f"every {trace.intervention_interval} step(s)"),
        ("Communities", f"{trace.initial_community_count} -> {trace.final_community_count}"),
        ("Action reuse", f"{len(trace.held_action_steps)} held steps"),
    ]

    y = 0.95
    for label, value in summary_lines:
        axis.text(0.0, y, label.upper(), fontsize=8, fontweight="bold", color="#64748b", transform=axis.transAxes)
        axis.text(0.0, y - 0.045, value, fontsize=11, color="#0f172a", transform=axis.transAxes)
        y -= 0.085

    axis.text(
        0.0,
        0.0,
        "Node fill = opinion\nNode outline = detected community\nEdges = strongest mutual attention weights",
        fontsize=9,
        color="#475569",
        transform=axis.transAxes,
        va="bottom",
    )


def _plot_network_snapshot(
    *,
    axis: Any,
    positions: Mapping[int, np.ndarray],
    weight_matrix: np.ndarray,
    opinions: np.ndarray,
    community_labels: np.ndarray,
    title: str,
    color_map: Any,
) -> Any:
    axis.set_title(title, loc="left", fontsize=14, fontweight="bold", color="#111827")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)

    symmetrized_weights = 0.5 * (weight_matrix + weight_matrix.T)
    strong_edges = _select_strong_edges(symmetrized_weights, max_edges=max(12, opinions.size * 2))
    max_weight = max((weight for _, _, weight in strong_edges), default=1.0)

    for source_index, target_index, weight in strong_edges:
        start_x, start_y = positions[source_index]
        end_x, end_y = positions[target_index]
        scaled_weight = 0.0 if max_weight <= 0.0 else (weight / max_weight)
        axis.plot(
            [start_x, end_x],
            [start_y, end_y],
            color="#64748b",
            alpha=0.08 + (0.28 * scaled_weight),
            linewidth=0.8 + (2.0 * scaled_weight),
            zorder=1,
        )

    coordinates = np.asarray([positions[index] for index in range(opinions.size)], dtype=np.float32)
    unique_labels = sorted({int(label) for label in np.asarray(community_labels).tolist()})
    community_palette = _require_matplotlib().colormaps["tab20"](
        np.linspace(0.05, 0.95, max(len(unique_labels), 1))
    )
    label_colors = {
        label: community_palette[index % len(community_palette)]
        for index, label in enumerate(unique_labels)
    }
    border_colors = [label_colors[int(label)] for label in community_labels]

    strengths = symmetrized_weights.sum(axis=1)
    normalized_strength = strengths / max(float(strengths.max()), 1e-6)
    node_sizes = 220.0 + (460.0 * normalized_strength)
    scatter = axis.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=opinions,
        cmap=color_map,
        vmin=-1.0,
        vmax=1.0,
        s=node_sizes,
        edgecolors=border_colors,
        linewidths=2.0,
        zorder=2,
    )

    if opinions.size <= 24:
        for node_index, (x_coord, y_coord) in enumerate(coordinates.tolist()):
            axis.text(
                x_coord,
                y_coord,
                str(node_index),
                ha="center",
                va="center",
                fontsize=7,
                color="#0f172a",
                zorder=3,
            )

    axis.margins(0.18)
    axis.text(
        0.02,
        0.02,
        f"communities={len(unique_labels)} | edges shown={len(strong_edges)}",
        transform=axis.transAxes,
        fontsize=9,
        color="#475569",
    )
    return scatter


def _build_network_positions(
    *,
    initial_weights: np.ndarray,
    final_weights: np.ndarray,
    seed: int,
) -> dict[int, np.ndarray]:
    try:
        import networkx as nx
    except ModuleNotFoundError as error:
        raise RuntimeError("networkx is required for episode storyboard layouts.") from error

    combined = 0.25 * (
        initial_weights
        + initial_weights.T
        + final_weights
        + final_weights.T
    )
    graph = nx.from_numpy_array(combined)
    return nx.spring_layout(graph, seed=int(seed), weight="weight")


def _select_strong_edges(weight_matrix: np.ndarray, *, max_edges: int) -> list[tuple[int, int, float]]:
    upper_indices = np.triu_indices_from(weight_matrix, k=1)
    edge_weights = weight_matrix[upper_indices]
    ranked_indices = np.argsort(edge_weights)[::-1]

    selected: list[tuple[int, int, float]] = []
    for edge_index in ranked_indices:
        weight = float(edge_weights[edge_index])
        if weight <= 0.0:
            continue
        selected.append(
            (
                int(upper_indices[0][edge_index]),
                int(upper_indices[1][edge_index]),
                weight,
            )
        )
        if len(selected) >= max_edges:
            break
    return selected


def _causal_moving_average(values: np.ndarray, *, window: int) -> np.ndarray:
    if values.ndim != 1:
        raise ValueError("values must be a 1D array.")
    if window < 1:
        raise ValueError("window must be at least 1.")
    if values.size == 0:
        return values.astype(np.float32, copy=True)

    smoothed = np.empty(values.size, dtype=np.float32)
    cumulative = np.cumsum(values, dtype=np.float64)
    for index in range(values.size):
        start = max(0, index - window + 1)
        total = cumulative[index] - (cumulative[start - 1] if start > 0 else 0.0)
        smoothed[index] = total / float(index - start + 1)
    return smoothed


def _require_matplotlib() -> Any:
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as pyplot
    except ModuleNotFoundError as error:
        raise RuntimeError("matplotlib is required for visualization helpers.") from error
    return pyplot
