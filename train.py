from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from agents import MAACAgent, MAACConfig
from agents.common import CriticConfig, GraphEncoderConfig, SharedActorConfig
from envs.social_network_env import SocialNetworkEnv
from utils.baselines import heuristic_bridge_policy, zero_policy
from utils.config_io import dump_json, load_config
from utils.evaluation import format_summary_table, rollout_policy


DEFAULT_CONFIG_PATH = Path("config/hyperparams.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the rebuilt MAAC social-network pipeline.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="Path to the YAML/JSON config.")
    parser.add_argument("--episodes", type=int, default=None, help="Override the configured episode count.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override the configured batch size.")
    parser.add_argument(
        "--eval-every",
        type=int,
        default=None,
        help="Benchmark the current policy every N episodes and update the best checkpoint from that ranking.",
    )
    parser.add_argument(
        "--community-detector",
        type=str,
        default=None,
        choices=("label_propagation", "louvain"),
        help="Detector used for this training run.",
    )
    parser.add_argument("--device", type=str, default=None, help="Torch device override, for example cpu or cuda.")
    parser.add_argument("--seed", type=int, default=None, help="Global seed override.")
    parser.add_argument(
        "--deterministic-actions",
        action="store_true",
        help="Use deterministic actor outputs during data collection.",
    )
    parser.add_argument(
        "--diagnostics-only",
        action="store_true",
        help="Run the zero-policy and heuristic diagnostics and exit without training.",
    )
    return parser.parse_args()


def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def resolve_device(requested: str | None) -> torch.device:
    if requested is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"CUDA was requested via --device {requested}, but torch.cuda.is_available() is False.")
    return device


def build_env_kwargs(config: dict[str, Any], *, detector: str, device: torch.device) -> dict[str, Any]:
    environment = config["environment"]
    return {
        "num_nodes": int(environment["num_nodes"]),
        "max_steps": int(environment["max_steps"]),
        "confidence_bound": float(environment["confidence_bound"]),
        "consensus_threshold": float(environment["consensus_threshold"]),
        "community_update_freq": int(environment["community_update_freq"]),
        "attachment_edges": int(environment["attachment_edges"]),
        "community_detector": detector,
        "community_temperature": float(environment["community_temperature"]),
        "action_scale": float(environment["action_scale"]),
        "intervention_interval": int(environment.get("intervention_interval", 1)),
        "hold_last_action": bool(environment.get("hold_last_action", True)),
        "terminal_bonus": float(environment["terminal_bonus"]),
        "step_penalty": float(environment["step_penalty"]),
        "action_penalty_coef": float(environment["action_penalty_coef"]),
        "bridge_penalty_coef": float(environment["bridge_penalty_coef"]),
        "self_belief": float(environment["self_belief"]),
        "device": device,
    }


def build_encoder_config(config: dict[str, Any]) -> GraphEncoderConfig:
    encoder = config["model"]["encoder"]
    return GraphEncoderConfig(
        gnn_hidden_dims=tuple(encoder["gnn_hidden_dims"]),
        graph_hidden_dims=tuple(encoder["graph_hidden_dims"]),
        graph_embedding_dim=int(encoder["graph_embedding_dim"]),
        activation=str(encoder["activation"]),
        dropout=float(encoder["dropout"]),
        layer_norm=bool(encoder["layer_norm"]),
        add_self_loops=bool(encoder["add_self_loops"]),
        step_count_scale=float(encoder["step_count_scale"]),
    )


def build_agent(config: dict[str, Any], *, num_nodes: int, device: torch.device, seed: int | None) -> MAACAgent:
    encoder_config = build_encoder_config(config)
    actor_cfg = config["model"]["actor"]
    critic_cfg = config["model"]["critic"]
    training_cfg = config["training"]

    agent = MAACAgent(
        num_nodes=num_nodes,
        actor_config=SharedActorConfig(
            encoder=encoder_config,
            edge_hidden_dims=tuple(actor_cfg["edge_hidden_dims"]),
            activation=str(actor_cfg["activation"]),
            dropout=float(actor_cfg["dropout"]),
            layer_norm=bool(actor_cfg["layer_norm"]),
        ),
        critic_config=CriticConfig(
            encoder=encoder_config,
            edge_hidden_dims=tuple(critic_cfg["edge_hidden_dims"]),
            value_hidden_dims=tuple(critic_cfg["value_hidden_dims"]),
            activation=str(critic_cfg["activation"]),
            dropout=float(critic_cfg["dropout"]),
            layer_norm=bool(critic_cfg["layer_norm"]),
        ),
        maac_config=MAACConfig(
            batch_size=int(training_cfg["batch_size"]),
            gamma=float(training_cfg["gamma"]),
            tau=float(training_cfg["tau"]),
            actor_lr=float(training_cfg["actor_lr"]),
            critic_lr=float(training_cfg["critic_lr"]),
            entropy_coef=float(training_cfg["entropy_coef"]),
            init_log_std=float(training_cfg["init_log_std"]),
            min_log_std=float(training_cfg["min_log_std"]),
            max_log_std=float(training_cfg["max_log_std"]),
            gradient_clip_norm=(
                None
                if training_cfg["gradient_clip_norm"] is None
                else float(training_cfg["gradient_clip_norm"])
            ),
            replay_capacity=int(training_cfg["replay_capacity"]),
            policy_epsilon=float(training_cfg["policy_epsilon"]),
        ),
        seed=seed,
        device=device,
    )
    return agent


def build_train_seeds(config: dict[str, Any], *, episodes: int, fallback_seed: int | None) -> list[int]:
    configured = list(config["training"].get("train_seeds", []))
    if configured:
        seeds = []
        while len(seeds) < episodes:
            seeds.extend(int(seed) for seed in configured)
        return seeds[:episodes]

    base = 0 if fallback_seed is None else int(fallback_seed)
    return [base + offset for offset in range(episodes)]


def resolve_trained_eval_settings(config: dict[str, Any]) -> dict[str, Any]:
    evaluation_cfg = config["evaluation"]
    policy_seed = evaluation_cfg.get("policy_seed", 0)
    return {
        "deterministic": bool(evaluation_cfg["deterministic_policy"]),
        "policy_seed": None if policy_seed is None else int(policy_seed),
    }


def run_baseline_suite(
    config: dict[str, Any],
    *,
    detector: str,
    device: torch.device,
) -> list[dict[str, Any]]:
    evaluation_cfg = config["evaluation"]
    seeds = [int(seed) for seed in evaluation_cfg["benchmark_seeds"]]
    env_kwargs = build_env_kwargs(config, detector=detector, device=device)

    zero_summary = rollout_policy(
        env_kwargs=env_kwargs,
        seeds=seeds,
        policy_name=f"{detector}:zero",
        action_selector=lambda env, observation: zero_policy(observation),
    )
    heuristic_summary = rollout_policy(
        env_kwargs=env_kwargs,
        seeds=seeds,
        policy_name=f"{detector}:heuristic",
        action_selector=lambda env, observation: heuristic_bridge_policy(
            observation,
            confidence_bound=env.confidence_bound,
        ),
    )
    return [zero_summary, heuristic_summary]


def run_trained_suite(
    config: dict[str, Any],
    *,
    detector: str,
    device: torch.device,
    agent: MAACAgent,
) -> dict[str, Any]:
    evaluation_cfg = config["evaluation"]
    seeds = [int(seed) for seed in evaluation_cfg["benchmark_seeds"]]
    env_kwargs = build_env_kwargs(config, detector=detector, device=device)
    eval_settings = resolve_trained_eval_settings(config)
    eval_settings["device"] = str(device)
    policy_seed = None if eval_settings["deterministic"] else eval_settings["policy_seed"]
    rng_state = capture_rng_state() if policy_seed is not None else None
    if policy_seed is not None:
        set_seed(policy_seed)

    was_training = agent.training
    agent.eval()
    try:
        with torch.no_grad():
            summary = rollout_policy(
                env_kwargs=env_kwargs,
                seeds=seeds,
                policy_name=f"{detector}:trained",
                action_selector=lambda env, observation: agent.select_actions(
                    observation,
                    deterministic=bool(eval_settings["deterministic"]),
                    as_numpy=True,
                ),
            )
    finally:
        agent.train(was_training)
        if rng_state is not None:
            restore_rng_state(rng_state)

    summary["evaluation_settings"] = eval_settings
    return summary


def run_benchmark_suite(
    config: dict[str, Any],
    *,
    detector: str,
    device: torch.device,
    agent: MAACAgent,
    diagnostics_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    trained_summary = run_trained_suite(config, detector=detector, device=device, agent=agent)
    return trained_summary, [*diagnostics_rows, trained_summary]


def benchmark_rank(summary_row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(summary_row["success_rate"]),
        float(summary_row["mean_final_consensus"]),
        float(summary_row["mean_reward"]),
        -float(summary_row["mean_steps"]),
    )


def is_benchmark_improvement(
    candidate_summary: dict[str, Any],
    best_summary: dict[str, Any] | None,
) -> bool:
    if best_summary is None:
        return True
    return benchmark_rank(candidate_summary) > benchmark_rank(best_summary)


def build_checkpoint_info(
    *,
    checkpoint_type: str,
    selected_at_episode: int | None,
    benchmark_summary: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    evaluation_settings = dict(benchmark_summary.get("evaluation_settings") or resolve_trained_eval_settings(config))
    return {
        "checkpoint_type": checkpoint_type,
        "selected_at_episode": None if selected_at_episode is None else int(selected_at_episode),
        "benchmark_metrics": {
            "success_rate": float(benchmark_summary["success_rate"]),
            "mean_final_consensus": float(benchmark_summary["mean_final_consensus"]),
            "mean_reward": float(benchmark_summary["mean_reward"]),
            "mean_steps": float(benchmark_summary["mean_steps"]),
        },
        "evaluation_settings": evaluation_settings,
        "benchmark_seeds": [int(seed) for seed in config["evaluation"]["benchmark_seeds"]],
    }


def save_checkpoint(
    *,
    path: Path,
    agent: MAACAgent,
    config: dict[str, Any],
    detector: str,
    summary_rows: list[dict[str, Any]] | None = None,
    checkpoint_info: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "agent": agent.build_checkpoint(),
            "config": config,
            "community_detector": detector,
            "summary_rows": summary_rows or [],
            "checkpoint_info": checkpoint_info or {},
        },
        path,
    )


def train(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    training_cfg = config["training"]
    detector = str(training_cfg["community_detector"])
    seed = None if training_cfg["seed"] is None else int(training_cfg["seed"])
    device = resolve_device(training_cfg["device"])
    set_seed(seed)

    env_kwargs = build_env_kwargs(config, detector=detector, device=device)
    diagnostics_rows = run_baseline_suite(config, detector=detector, device=device)
    print(format_summary_table(diagnostics_rows))
    if args.diagnostics_only:
        return {"diagnostics": diagnostics_rows}

    env = SocialNetworkEnv(**env_kwargs)
    agent = build_agent(config, num_nodes=env.num_nodes, device=device, seed=seed)
    episodes = int(training_cfg["episodes"])
    learning_starts = int(training_cfg["learning_starts"])
    update_every = int(training_cfg["update_every"])
    log_every = int(training_cfg["log_every"])
    eval_every = int(training_cfg.get("eval_every", episodes))
    if eval_every <= 0:
        eval_every = episodes
    deterministic_actions = bool(training_cfg["deterministic_actions"])
    train_seeds = build_train_seeds(config, episodes=episodes, fallback_seed=seed)

    reward_window: list[float] = []
    consensus_window: list[float] = []
    step_window: list[float] = []
    success_window: list[float] = []
    actor_loss_window: list[float] = []
    critic_loss_window: list[float] = []
    training_updates: list[dict[str, Any]] = []
    update_count = 0
    total_env_steps = 0
    latest_summary: list[dict[str, Any]] = diagnostics_rows[:]
    latest_trained_summary: dict[str, Any] | None = None
    best_trained_summary: dict[str, Any] | None = None
    best_summary_rows: list[dict[str, Any]] = diagnostics_rows[:]
    best_checkpoint_episode: int | None = None

    checkpoint_dir = Path(training_cfg["checkpoint_dir"])
    prefix = str(training_cfg["checkpoint_prefix"])
    latest_checkpoint = checkpoint_dir / f"{prefix}_{detector}_latest.pt"
    best_checkpoint = checkpoint_dir / f"{prefix}_{detector}_best.pt"
    summary_path = checkpoint_dir / f"{prefix}_{detector}_summary.json"

    for episode in range(1, episodes + 1):
        observation, info = env.reset(seed=int(train_seeds[episode - 1]))
        done = False
        episode_reward = 0.0
        episode_steps = 0
        final_consensus = float(info["global_consensus"])
        terminated = False

        while not done:
            if env.should_request_new_action():
                selected_actions = agent.select_actions(
                    observation,
                    deterministic=deterministic_actions,
                    as_numpy=True,
                )
                applied_action = env.schedule_action(selected_actions)
            else:
                applied_action = env.schedule_action()

            next_observation, reward, terminated, truncated, info = env.step(applied_action)
            done = terminated or truncated

            agent.store_transition(
                global_state=observation,
                actions=applied_action,
                reward=reward,
                next_global_state=next_observation,
                done=done,
            )

            total_env_steps += 1
            if len(agent.replay_buffer) >= learning_starts and total_env_steps % update_every == 0:
                update_metrics = agent.update(int(training_cfg["batch_size"]))
                if update_metrics:
                    actor_loss = float(update_metrics["actor_loss"])
                    critic_loss = float(update_metrics["critic_loss"])
                    actor_loss_window.append(actor_loss)
                    critic_loss_window.append(critic_loss)
                    update_count += 1
                    training_updates.append(
                        {
                            "update": update_count,
                            "episode": episode,
                            "env_step": total_env_steps,
                            "buffer_size": len(agent.replay_buffer),
                            "actor_loss": actor_loss,
                            "critic_loss": critic_loss,
                            "mean_q": float(update_metrics["mean_q"]),
                            "mean_target_q": float(update_metrics["mean_target_q"]),
                            "mean_policy_q": float(update_metrics["mean_policy_q"]),
                            "mean_log_prob": float(update_metrics["mean_log_prob"]),
                            "log_std": float(update_metrics["log_std"]),
                        }
                    )

            observation = next_observation
            episode_reward += float(reward)
            episode_steps += 1
            final_consensus = float(info["global_consensus"])

        reward_window.append(episode_reward)
        consensus_window.append(final_consensus)
        step_window.append(float(episode_steps))
        success_window.append(float(terminated))

        if episode % log_every == 0 or episode == 1 or episode == episodes:
            avg_reward = float(np.mean(reward_window)) if reward_window else 0.0
            avg_consensus = float(np.mean(consensus_window)) if consensus_window else 0.0
            avg_steps = float(np.mean(step_window)) if step_window else 0.0
            success_rate = float(np.mean(success_window)) if success_window else 0.0
            avg_actor_loss = float(np.mean(actor_loss_window)) if actor_loss_window else float("nan")
            avg_critic_loss = float(np.mean(critic_loss_window)) if critic_loss_window else float("nan")

            print(
                f"Episode {episode:04d}/{episodes:04d} | "
                f"avg_reward={avg_reward:.4f} | "
                f"consensus={avg_consensus:.4f} | "
                f"avg_steps={avg_steps:.1f} | "
                f"success_rate={success_rate:.2f} | "
                f"actor_loss={avg_actor_loss:.4f} | "
                f"critic_loss={avg_critic_loss:.4f} | "
                f"buffer_size={len(agent.replay_buffer)}"
            )

            reward_window.clear()
            consensus_window.clear()
            step_window.clear()
            success_window.clear()
            actor_loss_window.clear()
            critic_loss_window.clear()

        should_run_benchmark_eval = episode % eval_every == 0 or episode == episodes
        if should_run_benchmark_eval:
            trained_summary, benchmark_rows = run_benchmark_suite(
                config,
                detector=detector,
                device=device,
                agent=agent,
                diagnostics_rows=diagnostics_rows,
            )
            latest_trained_summary = trained_summary
            latest_summary = benchmark_rows

            print(
                f"Benchmark {episode:04d}/{episodes:04d} | "
                f"success_rate={float(trained_summary['success_rate']):.2f} | "
                f"consensus={float(trained_summary['mean_final_consensus']):.4f} | "
                f"reward={float(trained_summary['mean_reward']):.4f} | "
                f"steps={float(trained_summary['mean_steps']):.1f}"
            )

            if is_benchmark_improvement(trained_summary, best_trained_summary):
                best_trained_summary = trained_summary
                best_summary_rows = benchmark_rows
                best_checkpoint_episode = episode
                best_checkpoint_info = build_checkpoint_info(
                    checkpoint_type="best",
                    selected_at_episode=episode,
                    benchmark_summary=trained_summary,
                    config=config,
                )
                save_checkpoint(
                    path=best_checkpoint,
                    agent=agent,
                    config=config,
                    detector=detector,
                    summary_rows=benchmark_rows,
                    checkpoint_info=best_checkpoint_info,
                )
                print(
                    f"Saved new best checkpoint | "
                    f"episode={episode} | "
                    f"success_rate={float(trained_summary['success_rate']):.2f} | "
                    f"consensus={float(trained_summary['mean_final_consensus']):.4f} | "
                    f"reward={float(trained_summary['mean_reward']):.4f} | "
                    f"steps={float(trained_summary['mean_steps']):.1f} | "
                    f"path={best_checkpoint}"
                )

    if latest_trained_summary is None:
        latest_trained_summary, latest_summary = run_benchmark_suite(
            config,
            detector=detector,
            device=device,
            agent=agent,
            diagnostics_rows=diagnostics_rows,
        )
    if best_trained_summary is None:
        best_trained_summary = latest_trained_summary
        best_summary_rows = latest_summary
        best_checkpoint_episode = episodes
        save_checkpoint(
            path=best_checkpoint,
            agent=agent,
            config=config,
            detector=detector,
            summary_rows=best_summary_rows,
            checkpoint_info=build_checkpoint_info(
                checkpoint_type="best",
                selected_at_episode=best_checkpoint_episode,
                benchmark_summary=best_trained_summary,
                config=config,
            ),
        )

    print(format_summary_table(latest_summary))

    save_checkpoint(
        path=latest_checkpoint,
        agent=agent,
        config=config,
        detector=detector,
        summary_rows=latest_summary,
        checkpoint_info=build_checkpoint_info(
            checkpoint_type="latest",
            selected_at_episode=episodes,
            benchmark_summary=latest_trained_summary,
            config=config,
        ),
    )
    dump_json(
        summary_path,
        {
            "community_detector": detector,
            "rows": latest_summary,
            "best_rows": best_summary_rows,
            "latest_checkpoint": str(latest_checkpoint),
            "best_checkpoint": str(best_checkpoint),
            "best_checkpoint_episode": best_checkpoint_episode,
            "training_history": {
                "episodes": episodes,
                "learning_starts": learning_starts,
                "update_every": update_every,
                "total_env_steps": total_env_steps,
                "total_updates": update_count,
                "updates": training_updates,
            },
        },
    )
    return {
        "diagnostics": diagnostics_rows,
        "trained": latest_trained_summary,
        "best_trained": best_trained_summary,
        "latest_checkpoint": str(latest_checkpoint),
        "best_checkpoint": str(best_checkpoint),
        "summary_path": str(summary_path),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    training_cfg = config["training"]
    if args.episodes is not None:
        training_cfg["episodes"] = int(args.episodes)
    if args.batch_size is not None:
        training_cfg["batch_size"] = int(args.batch_size)
    if args.eval_every is not None:
        training_cfg["eval_every"] = int(args.eval_every)
    if args.community_detector is not None:
        training_cfg["community_detector"] = str(args.community_detector)
    if args.device is not None:
        training_cfg["device"] = str(args.device)
    if args.seed is not None:
        training_cfg["seed"] = int(args.seed)
    if args.deterministic_actions:
        training_cfg["deterministic_actions"] = True

    train(config, args)


if __name__ == "__main__":
    main()
