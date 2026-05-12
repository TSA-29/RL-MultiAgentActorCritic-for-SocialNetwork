from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from envs.social_network_env import SocialNetworkEnv


ActionSelector = Callable[[SocialNetworkEnv, dict[str, np.ndarray]], dict[str, np.ndarray]]


def rollout_policy(
    *,
    env_kwargs: dict[str, Any],
    seeds: Sequence[int],
    policy_name: str,
    action_selector: ActionSelector,
) -> dict[str, Any]:
    episode_rows: list[dict[str, Any]] = []

    for seed in seeds:
        env = SocialNetworkEnv(**env_kwargs)
        observation, info = env.reset(seed=int(seed))
        total_reward = 0.0
        done = False
        step_count = 0
        last_info = info

        while not done:
            if env.should_request_new_action():
                action = env.schedule_action(action_selector(env, observation))
            else:
                action = env.schedule_action()

            observation, reward, terminated, truncated, last_info = env.step(action)
            total_reward += float(reward)
            step_count += 1
            done = terminated or truncated

        episode_rows.append(
            {
                "seed": int(seed),
                "total_reward": total_reward,
                "final_consensus": float(last_info["global_consensus"]),
                "steps": step_count,
                "success": bool(last_info["global_consensus"] >= env.consensus_threshold),
                "harmful_bridge_mass": float(last_info["harmful_bridge_mass"]),
                "cross_community_weight": float(last_info["cross_community_weight"]),
            }
        )

    return summarize_rollouts(policy_name=policy_name, episode_rows=episode_rows)


def summarize_rollouts(*, policy_name: str, episode_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not episode_rows:
        raise ValueError("episode_rows must not be empty.")

    return {
        "label": policy_name,
        "episodes": len(episode_rows),
        "mean_reward": float(np.mean([row["total_reward"] for row in episode_rows])),
        "mean_final_consensus": float(np.mean([row["final_consensus"] for row in episode_rows])),
        "mean_steps": float(np.mean([row["steps"] for row in episode_rows])),
        "success_rate": float(np.mean([float(row["success"]) for row in episode_rows])),
        "mean_harmful_bridge_mass": float(np.mean([row["harmful_bridge_mass"] for row in episode_rows])),
        "mean_cross_community_weight": float(np.mean([row["cross_community_weight"] for row in episode_rows])),
        "episodes_detail": list(episode_rows),
    }


def format_summary_table(rows: Sequence[dict[str, Any]]) -> str:
    headers = (
        ("label", "Label"),
        ("mean_final_consensus", "Consensus"),
        ("success_rate", "Success"),
        ("mean_reward", "Reward"),
        ("mean_steps", "Steps"),
        ("mean_harmful_bridge_mass", "HarmfulBridge"),
    )
    table_rows = []
    for row in rows:
        table_rows.append(
            [
                str(row["label"]),
                f"{float(row['mean_final_consensus']):.4f}",
                f"{float(row['success_rate']):.2f}",
                f"{float(row['mean_reward']):.4f}",
                f"{float(row['mean_steps']):.1f}",
                f"{float(row['mean_harmful_bridge_mass']):.4f}",
            ]
        )

    widths = [
        max(len(header), *(len(values[column_index]) for values in table_rows))
        for column_index, (_, header) in enumerate(headers)
    ]
    header_line = " | ".join(header.ljust(widths[index]) for index, (_, header) in enumerate(headers))
    separator = "-+-".join("-" * width for width in widths)
    body = [
        " | ".join(values[index].ljust(widths[index]) for index in range(len(values)))
        for values in table_rows
    ]
    return "\n".join([header_line, separator, *body])
