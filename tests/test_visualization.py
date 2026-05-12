from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from utils.baselines import zero_policy
from utils.config_io import load_config
from utils.visualization import plot_episode_storyboard, plot_training_diagnostics, rollout_episode_trace

import train


class VisualizationTests(unittest.TestCase):
    def _small_env_kwargs(self) -> dict:
        config = load_config("config/hyperparams.yaml")
        config["environment"]["num_nodes"] = 10
        config["environment"]["max_steps"] = 5
        config["environment"]["community_update_freq"] = 2
        return train.build_env_kwargs(config, detector="louvain", device=train.resolve_device("cpu"))

    def test_rollout_episode_trace_records_stepwise_history(self) -> None:
        trace = rollout_episode_trace(
            env_kwargs=self._small_env_kwargs(),
            seed=101,
            policy_name="louvain:zero",
            action_selector=lambda env, observation: zero_policy(observation),
        )

        self.assertEqual(trace.policy_name, "louvain:zero")
        self.assertEqual(trace.seed, 101)
        self.assertEqual(len(trace.opinions), trace.step_count + 1)
        self.assertEqual(len(trace.weight_matrices), trace.step_count + 1)
        self.assertEqual(len(trace.community_labels), trace.step_count + 1)
        self.assertEqual(len(trace.consensus), trace.step_count + 1)
        self.assertEqual(len(trace.cross_community_weight), trace.step_count + 1)
        self.assertEqual(len(trace.rewards), trace.step_count)
        self.assertEqual(len(trace.action_magnitudes), trace.step_count)
        self.assertEqual(len(trace.harmful_bridge_mass), trace.step_count)
        self.assertEqual(len(trace.action_sources), trace.step_count)
        self.assertGreaterEqual(trace.initial_community_count, 1)
        self.assertGreaterEqual(trace.final_community_count, 1)

    def test_plot_episode_storyboard_creates_png_when_matplotlib_is_available(self) -> None:
        try:
            import matplotlib.pyplot  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("matplotlib is not installed in the current environment.")

        trace = rollout_episode_trace(
            env_kwargs=self._small_env_kwargs(),
            seed=102,
            policy_name="louvain:zero",
            action_selector=lambda env, observation: zero_policy(observation),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "storyboard.png"
            plot_episode_storyboard(trace=trace, output_path=output_path)
            self.assertTrue(output_path.exists())

    def test_plot_training_diagnostics_creates_png_when_matplotlib_is_available(self) -> None:
        try:
            import matplotlib.pyplot  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("matplotlib is not installed in the current environment.")

        updates = [
            {"update": 1, "env_step": 20, "actor_loss": -0.8, "critic_loss": 3.5},
            {"update": 2, "env_step": 40, "actor_loss": -1.1, "critic_loss": 2.9},
            {"update": 3, "env_step": 60, "actor_loss": -0.9, "critic_loss": 2.1},
            {"update": 4, "env_step": 80, "actor_loss": -0.6, "critic_loss": 1.8},
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "training_diagnostics.png"
            plot_training_diagnostics(updates=updates, output_path=output_path, smoothing_window=2)
            self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()
