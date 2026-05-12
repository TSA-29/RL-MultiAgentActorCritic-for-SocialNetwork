from __future__ import annotations

import tempfile
import unittest
from argparse import Namespace
from json import loads
from pathlib import Path

import numpy as np
import torch

import evaluate
import train
from agents import MAACAgent
from envs.social_network_env import SocialNetworkEnv
from utils.config_io import load_config
from utils.evaluation import format_summary_table


class PipelineTests(unittest.TestCase):
    def _small_config(self, checkpoint_dir: Path) -> dict:
        config = load_config("config/hyperparams.yaml")
        config["environment"]["num_nodes"] = 10
        config["environment"]["max_steps"] = 5
        config["model"]["encoder"]["gnn_hidden_dims"] = [32]
        config["model"]["encoder"]["graph_hidden_dims"] = [32]
        config["model"]["encoder"]["graph_embedding_dim"] = 32
        config["model"]["actor"]["edge_hidden_dims"] = [32]
        config["model"]["critic"]["edge_hidden_dims"] = [32]
        config["model"]["critic"]["value_hidden_dims"] = [64]
        config["training"]["episodes"] = 1
        config["training"]["batch_size"] = 2
        config["training"]["learning_starts"] = 1
        config["training"]["log_every"] = 1
        config["training"]["eval_every"] = 1
        config["training"]["train_seeds"] = [7]
        config["training"]["checkpoint_dir"] = str(checkpoint_dir)
        config["evaluation"]["benchmark_seeds"] = [101]
        return config

    def test_training_smoke_saves_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self._small_config(Path(temp_dir))
            result = train.train(config, Namespace(diagnostics_only=False))

            self.assertTrue(Path(result["latest_checkpoint"]).exists())
            self.assertTrue(Path(result["best_checkpoint"]).exists())
            self.assertTrue(Path(result["summary_path"]).exists())
            summary_payload = loads(Path(result["summary_path"]).read_text(encoding="utf-8"))
            self.assertIn("training_history", summary_payload)

    def test_evaluation_can_load_best_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir)
            config = self._small_config(checkpoint_dir)
            result = train.train(config, Namespace(diagnostics_only=False))

            rows = evaluate.evaluate_detector(
                config=config,
                detector=config["training"]["community_detector"],
                device=train.resolve_device(config["training"]["device"]),
                checkpoint_dir=checkpoint_dir,
                checkpoint_type="best",
            )

            self.assertIn(f"{config['training']['community_detector']}:trained", {row["label"] for row in rows})
            trained_row = next(row for row in rows if row["label"].endswith(":trained"))
            best_summary = result["best_trained"]
            self.assertAlmostEqual(trained_row["success_rate"], best_summary["success_rate"], places=7)
            self.assertAlmostEqual(
                trained_row["mean_final_consensus"],
                best_summary["mean_final_consensus"],
                places=7,
            )
            self.assertAlmostEqual(trained_row["mean_reward"], best_summary["mean_reward"], places=7)
            self.assertAlmostEqual(trained_row["mean_steps"], best_summary["mean_steps"], places=7)

    def test_checkpoint_reload_preserves_deterministic_actions(self) -> None:
        config = load_config("config/hyperparams.yaml")
        config["environment"]["num_nodes"] = 10
        config["model"]["encoder"]["gnn_hidden_dims"] = [32]
        config["model"]["encoder"]["graph_hidden_dims"] = [32]
        config["model"]["encoder"]["graph_embedding_dim"] = 32
        config["model"]["actor"]["edge_hidden_dims"] = [32]
        config["model"]["critic"]["edge_hidden_dims"] = [32]
        config["model"]["critic"]["value_hidden_dims"] = [64]

        device = torch.device("cpu")
        env = SocialNetworkEnv(
            **train.build_env_kwargs(config, detector=config["training"]["community_detector"], device=device)
        )
        observation, _ = env.reset(seed=9)
        agent = train.build_agent(config, num_nodes=env.num_nodes, device=device, seed=9)

        actions_before = agent.select_actions(observation, deterministic=True, as_numpy=True)
        checkpoint = agent.build_checkpoint()
        reloaded = MAACAgent.from_checkpoint(checkpoint, device=device)
        actions_after = reloaded.select_actions(observation, deterministic=True, as_numpy=True)

        self.assertEqual(set(actions_before), set(actions_after))
        for key in actions_before:
            np.testing.assert_allclose(actions_before[key], actions_after[key], atol=1e-6)

    def test_build_agent_respects_requested_device(self) -> None:
        config = load_config("config/hyperparams.yaml")
        config["environment"]["num_nodes"] = 10
        config["model"]["encoder"]["gnn_hidden_dims"] = [32]
        config["model"]["encoder"]["graph_hidden_dims"] = [32]
        config["model"]["encoder"]["graph_embedding_dim"] = 32
        config["model"]["actor"]["edge_hidden_dims"] = [32]
        config["model"]["critic"]["edge_hidden_dims"] = [32]
        config["model"]["critic"]["value_hidden_dims"] = [64]

        agent = train.build_agent(config, num_nodes=10, device=torch.device("cpu"), seed=7)
        self.assertEqual(agent.device.type, "cpu")

    def test_evaluation_table_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self._small_config(Path(temp_dir))
            device = torch.device("cpu")
            rows = []
            for detector in ("louvain", "label_propagation"):
                rows.extend(
                    evaluate.evaluate_detector(
                        config=config,
                        detector=detector,
                        device=device,
                        checkpoint_dir=Path(temp_dir),
                    )
                )
            table = format_summary_table(rows)
            self.assertIn("louvain:zero", table)
            self.assertIn("label_propagation:heuristic", table)


if __name__ == "__main__":
    unittest.main()
