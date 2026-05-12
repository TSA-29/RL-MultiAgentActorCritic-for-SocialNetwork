from __future__ import annotations

import unittest

import numpy as np
import torch

from envs.network_factory import hegselmann_krause_update
from envs.social_network_env import SocialNetworkEnv


class SocialNetworkEnvTests(unittest.TestCase):
    def test_action_update_keeps_rows_stochastic(self) -> None:
        env = SocialNetworkEnv(num_nodes=12, max_steps=2, device="cpu")
        observation, _ = env.reset(seed=3)
        random_action = np.random.uniform(-1.0, 1.0, size=(env.num_nodes, env.num_nodes)).astype(np.float32)
        next_observation, _, _, _, _ = env.step(random_action)
        row_sums = next_observation["weight_matrix"].sum(axis=1)
        self.assertTrue(np.allclose(row_sums, 1.0, atol=1e-5))

    def test_action_can_create_new_edges(self) -> None:
        env = SocialNetworkEnv(num_nodes=12, max_steps=2, device="cpu")
        observation, _ = env.reset(seed=4)
        weight_matrix = observation["weight_matrix"]

        zero_locations = np.argwhere(weight_matrix <= 0.0)
        self.assertGreater(len(zero_locations), 0)
        row_index, col_index = map(int, zero_locations[0])

        action = np.full((env.num_nodes, env.num_nodes), -1.0, dtype=np.float32)
        action[row_index, col_index] = 1.0
        next_observation, _, _, _, _ = env.step(action)

        self.assertGreater(next_observation["weight_matrix"][row_index, col_index], weight_matrix[row_index, col_index])

    def test_hk_confidence_mask_blocks_far_neighbors(self) -> None:
        opinions = torch.tensor([-1.0, 1.0], dtype=torch.float32)
        weight_matrix = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)

        blocked = hegselmann_krause_update(
            opinions=opinions,
            weight_matrix=weight_matrix,
            confidence_bound=0.5,
            self_belief=1.0,
        )
        allowed = hegselmann_krause_update(
            opinions=opinions,
            weight_matrix=weight_matrix,
            confidence_bound=2.0,
            self_belief=1.0,
        )

        self.assertTrue(torch.allclose(blocked, opinions))
        self.assertLess(torch.abs(allowed[0] - allowed[1]).item(), torch.abs(opinions[0] - opinions[1]).item())

    def test_reward_penalties_are_reported(self) -> None:
        env = SocialNetworkEnv(num_nodes=10, max_steps=2, device="cpu")
        env.reset(seed=5)
        action = np.ones((env.num_nodes, env.num_nodes), dtype=np.float32)
        _, reward, _, _, info = env.step(action)

        self.assertIn("action_penalty", info)
        self.assertIn("bridge_penalty", info)
        self.assertGreaterEqual(info["action_penalty"], 0.0)
        self.assertGreaterEqual(info["bridge_penalty"], 0.0)
        reconstructed = (
            info["consensus_gain"]
            + info["terminal_bonus"]
            - info["step_penalty"]
            - info["action_penalty"]
            - info["bridge_penalty"]
        )
        self.assertAlmostEqual(reward, reconstructed, places=5)


if __name__ == "__main__":
    unittest.main()
