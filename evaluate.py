from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from agents import MAACAgent
from train import build_env_kwargs, resolve_device, run_trained_suite
from utils.baselines import heuristic_bridge_policy, zero_policy
from utils.community_detection import available_detectors
from utils.config_io import dump_json, load_config
from utils.evaluation import format_summary_table, rollout_policy


DEFAULT_CONFIG_PATH = Path("config/hyperparams.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate detectors, baselines, and trained MAAC checkpoints.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="Path to the YAML/JSON config.")
    parser.add_argument(
        "--community-detector",
        type=str,
        default="all",
        choices=(*available_detectors(), "all"),
        help="Detector to evaluate. Use 'all' to compare both detectors.",
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
        default="latest",
        choices=("latest", "best"),
        help="Which trained checkpoint to evaluate for the trained-policy row.",
    )
    parser.add_argument("--device", type=str, default=None, help="Torch device override.")
    return parser.parse_args()


def load_agent_for_detector(
    *,
    detector: str,
    checkpoint_dir: Path | None,
    device: torch.device,
    prefix: str,
    checkpoint_type: str = "latest",
) -> tuple[MAACAgent | None, Path | None, dict[str, Any]]:
    if checkpoint_dir is None:
        return None, None, {}

    checkpoint_path = checkpoint_dir / f"{prefix}_{detector}_{checkpoint_type}.pt"
    if not checkpoint_path.exists():
        return None, checkpoint_path, {}

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_info = dict(checkpoint.get("checkpoint_info") or {})

    print(f"Loaded checkpoint | detector={detector} | type={checkpoint_type} | path={checkpoint_path}")
    if checkpoint_info:
        metrics = checkpoint_info.get("benchmark_metrics") or {}
        eval_settings = checkpoint_info.get("evaluation_settings") or {}
        print(
            f"Saved checkpoint metadata | "
            f"episode={checkpoint_info.get('selected_at_episode')} | "
            f"success_rate={float(metrics.get('success_rate', 0.0)):.2f} | "
            f"consensus={float(metrics.get('mean_final_consensus', 0.0)):.4f} | "
            f"reward={float(metrics.get('mean_reward', 0.0)):.4f} | "
            f"steps={float(metrics.get('mean_steps', 0.0)):.1f} | "
            f"deterministic={bool(eval_settings.get('deterministic', False))} | "
            f"policy_seed={eval_settings.get('policy_seed')} | "
            f"device={eval_settings.get('device')}"
        )
        if (
            eval_settings.get("device") is not None
            and str(eval_settings.get("device")) != str(device)
            and not bool(eval_settings.get("deterministic", False))
        ):
            print(
                f"Warning: checkpoint benchmark used device={eval_settings.get('device')}, "
                f"but evaluation is running on device={device}. "
                f"Stochastic rollouts may not reproduce exactly across devices."
            )

    return (
        MAACAgent.from_checkpoint(checkpoint["agent"], device=device, load_optimizers=False),
        checkpoint_path,
        checkpoint_info,
    )


def evaluate_detector(
    *,
    config: dict[str, Any],
    detector: str,
    device: torch.device,
    checkpoint_dir: Path | None,
    checkpoint_type: str = "latest",
) -> list[dict[str, Any]]:
    env_kwargs = build_env_kwargs(config, detector=detector, device=device)
    seeds = [int(seed) for seed in config["evaluation"]["benchmark_seeds"]]
    rows = [
        rollout_policy(
            env_kwargs=env_kwargs,
            seeds=seeds,
            policy_name=f"{detector}:zero",
            action_selector=lambda env, observation: zero_policy(observation),
        ),
        rollout_policy(
            env_kwargs=env_kwargs,
            seeds=seeds,
            policy_name=f"{detector}:heuristic",
            action_selector=lambda env, observation: heuristic_bridge_policy(
                observation,
                confidence_bound=env.confidence_bound,
            ),
        ),
    ]

    prefix = str(config["training"]["checkpoint_prefix"])
    agent, checkpoint_path, checkpoint_info = load_agent_for_detector(
        detector=detector,
        checkpoint_dir=checkpoint_dir,
        device=device,
        prefix=prefix,
        checkpoint_type=checkpoint_type,
    )
    if agent is not None:
        trained_row = run_trained_suite(config, detector=detector, device=device, agent=agent)
        trained_row["checkpoint_type"] = checkpoint_type
        trained_row["checkpoint_path"] = None if checkpoint_path is None else str(checkpoint_path)
        trained_row["checkpoint_info"] = checkpoint_info
        rows.append(trained_row)
    return rows


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(args.device or config["training"]["device"])
    checkpoint_dir = (
        Path(args.checkpoint_dir)
        if args.checkpoint_dir is not None
        else Path(config["training"]["checkpoint_dir"])
    )

    detectors = list(available_detectors()) if args.community_detector == "all" else [args.community_detector]
    summary_rows: list[dict[str, Any]] = []
    for detector in detectors:
        summary_rows.extend(
            evaluate_detector(
                config=config,
                detector=detector,
                device=device,
                checkpoint_dir=checkpoint_dir,
                checkpoint_type=args.checkpoint_type,
            )
        )

    print(format_summary_table(summary_rows))
    summary_name = "evaluation_summary.json"
    if args.checkpoint_type != "latest":
        summary_name = f"evaluation_summary_{args.checkpoint_type}.json"
    summary_path = checkpoint_dir / summary_name
    dump_json(summary_path, {"checkpoint_type": args.checkpoint_type, "rows": summary_rows})


if __name__ == "__main__":
    main()
