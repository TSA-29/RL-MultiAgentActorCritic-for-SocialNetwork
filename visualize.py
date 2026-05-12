from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from utils.baselines import heuristic_bridge_policy, zero_policy
from utils.community_detection import available_detectors
from utils.config_io import load_config
from utils.visualization import plot_episode_storyboard, plot_training_diagnostics, rollout_episode_trace

import evaluate
import train


DEFAULT_CONFIG_PATH = Path("config/hyperparams.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render either an episode storyboard or MAAC training-loss diagnostics.",
    )
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="Path to the YAML/JSON config.")
    parser.add_argument(
        "--kind",
        type=str,
        default="episode",
        choices=("episode", "training"),
        help="Which visualization to render.",
    )
    parser.add_argument(
        "--community-detector",
        type=str,
        default=None,
        choices=available_detectors(),
        help="Detector to visualize. Defaults to the detector stored in the config.",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="trained",
        choices=("trained", "heuristic", "zero"),
        help="Policy to roll out for the visualization.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory containing detector-specific checkpoints saved by train.py.",
    )
    parser.add_argument(
        "--checkpoint-type",
        type=str,
        default="best",
        choices=("latest", "best"),
        help="Checkpoint variant used when --policy trained is selected.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Episode seed to visualize. Defaults to the first benchmark seed from the config.",
    )
    parser.add_argument("--device", type=str, default=None, help="Torch device override.")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Force deterministic actor outputs for trained-policy rollouts.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="PNG output path. Defaults to artifacts/plots/<detector>_<policy>_seed<seed>.png.",
    )
    parser.add_argument(
        "--summary-path",
        type=str,
        default=None,
        help="Training summary JSON used by --kind training. Defaults to artifacts/checkpoints/<prefix>_<detector>_summary.json.",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=25,
        help="Moving-average window used by --kind training.",
    )
    return parser.parse_args()


def build_default_output_path(
    *,
    detector: str,
    policy: str,
    seed: int,
    checkpoint_type: str,
) -> Path:
    suffix = f"_{checkpoint_type}" if policy == "trained" else ""
    return Path("artifacts/plots") / f"{detector}_{policy}_seed{seed}{suffix}.png"


def build_default_training_output_path(
    *,
    checkpoint_prefix: str,
    detector: str,
) -> Path:
    return Path("artifacts/plots") / f"{checkpoint_prefix}_{detector}_training_diagnostics.png"


def render_trace(
    *,
    config: dict[str, Any],
    detector: str,
    policy: str,
    seed: int,
    device: str | None,
    checkpoint_dir: Path | None,
    checkpoint_type: str,
    deterministic_override: bool,
) -> tuple[str, Any]:
    resolved_device = train.resolve_device(device or config["training"]["device"])
    env_kwargs = train.build_env_kwargs(config, detector=detector, device=resolved_device)

    if policy == "zero":
        label = f"{detector}:zero"
        trace = rollout_episode_trace(
            env_kwargs=env_kwargs,
            seed=seed,
            policy_name=label,
            action_selector=lambda env, observation: zero_policy(observation),
        )
        return label, trace

    if policy == "heuristic":
        label = f"{detector}:heuristic"
        trace = rollout_episode_trace(
            env_kwargs=env_kwargs,
            seed=seed,
            policy_name=label,
            action_selector=lambda env, observation: heuristic_bridge_policy(
                observation,
                confidence_bound=env.confidence_bound,
            ),
        )
        return label, trace

    prefix = str(config["training"]["checkpoint_prefix"])
    agent, checkpoint_path, _ = evaluate.load_agent_for_detector(
        detector=detector,
        checkpoint_dir=checkpoint_dir,
        device=resolved_device,
        prefix=prefix,
        checkpoint_type=checkpoint_type,
    )
    if agent is None:
        expected_path = checkpoint_path or (checkpoint_dir / f"{prefix}_{detector}_{checkpoint_type}.pt")
        raise FileNotFoundError(
            f"Could not find the requested checkpoint for detector '{detector}' at '{expected_path}'."
        )

    eval_settings = train.resolve_trained_eval_settings(config)
    deterministic = bool(deterministic_override or eval_settings["deterministic"])
    policy_seed = None if deterministic else eval_settings["policy_seed"]
    rng_state = train.capture_rng_state() if policy_seed is not None else None
    if policy_seed is not None:
        train.set_seed(int(policy_seed))

    label = f"{detector}:trained"
    was_training = agent.training
    agent.eval()
    try:
        trace = rollout_episode_trace(
            env_kwargs=env_kwargs,
            seed=seed,
            policy_name=label,
            action_selector=lambda env, observation: agent.select_actions(
                observation,
                deterministic=deterministic,
                as_numpy=True,
            ),
        )
    finally:
        agent.train(was_training)
        if rng_state is not None:
            train.restore_rng_state(rng_state)

    return label, trace


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    detector = str(args.community_detector or config["training"]["community_detector"])
    checkpoint_prefix = str(config["training"]["checkpoint_prefix"])
    checkpoint_dir = (
        Path(args.checkpoint_dir)
        if args.checkpoint_dir is not None
        else Path(config["training"]["checkpoint_dir"])
    )
    if args.kind == "training":
        summary_path = (
            Path(args.summary_path)
            if args.summary_path is not None
            else checkpoint_dir / f"{checkpoint_prefix}_{detector}_summary.json"
        )
        output_path = (
            Path(args.output)
            if args.output is not None
            else build_default_training_output_path(
                checkpoint_prefix=checkpoint_prefix,
                detector=detector,
            )
        )
        summary_payload = load_config(summary_path)
        training_history = dict(summary_payload.get("training_history") or {})
        updates = list(training_history.get("updates") or [])
        if not updates:
            raise ValueError(
                f"No training-history updates were found in '{summary_path}'. "
                "Re-run training with the current code to generate actor/critic loss data."
            )
        plot_training_diagnostics(
            updates=updates,
            output_path=output_path,
            title=f"MAAC Training Diagnostics | {detector}",
            smoothing_window=int(args.smoothing_window),
        )
        print(f"Saved training diagnostics to {output_path}")
        return

    seed = int(args.seed if args.seed is not None else config["evaluation"]["benchmark_seeds"][0])
    output_path = (
        Path(args.output)
        if args.output is not None
        else build_default_output_path(
            detector=detector,
            policy=args.policy,
            seed=seed,
            checkpoint_type=args.checkpoint_type,
        )
    )

    label, trace = render_trace(
        config=config,
        detector=detector,
        policy=args.policy,
        seed=seed,
        device=args.device,
        checkpoint_dir=checkpoint_dir,
        checkpoint_type=args.checkpoint_type,
        deterministic_override=bool(args.deterministic),
    )
    plot_episode_storyboard(trace=trace, output_path=output_path, title=f"Opinion Dynamics Storyboard | {label}")
    print(f"Saved visualization to {output_path}")


if __name__ == "__main__":
    main()
