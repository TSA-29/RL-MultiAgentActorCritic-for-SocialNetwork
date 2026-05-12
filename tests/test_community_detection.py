from __future__ import annotations

import unittest

import numpy as np

from utils.community_detection import canonicalize_communities, detect_communities


class CommunityDetectionTests(unittest.TestCase):
    def test_detector_seed_reproducibility(self) -> None:
        weight_matrix = np.array(
            [
                [0.0, 0.7, 0.3, 0.0],
                [0.7, 0.0, 0.3, 0.0],
                [0.0, 0.3, 0.0, 0.7],
                [0.0, 0.3, 0.7, 0.0],
            ],
            dtype=np.float32,
        )
        opinions = np.array([-0.8, -0.6, 0.6, 0.8], dtype=np.float32)

        for detector in ("louvain", "label_propagation"):
            labels_a = detect_communities(
                weight_matrix=weight_matrix,
                opinions=opinions,
                detector=detector,
                seed=11,
            )
            labels_b = detect_communities(
                weight_matrix=weight_matrix,
                opinions=opinions,
                detector=detector,
                seed=11,
            )
            np.testing.assert_array_equal(labels_a, labels_b)

    def test_canonical_community_labels_are_stable(self) -> None:
        labels = canonicalize_communities(communities=[[3, 4], [0, 1, 2]], node_count=5)
        np.testing.assert_array_equal(labels, np.array([0, 0, 0, 1, 1], dtype=np.int32))


if __name__ == "__main__":
    unittest.main()
