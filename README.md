# RL Multi-Agent Actor-Critic for Social Network Consensus

This repository implements a community-aware multi-agent reinforcement learning pipeline for steering a polarized social network toward consensus.

The environment is a weighted directed social graph where each node has an opinion in `[-1, 1]`. Opinions evolve through bounded-confidence dynamics inspired by the Hegselmann-Krause model. The controller does not directly edit opinions; instead, it changes edge weights so future social influence patterns move the network toward consensus.

The main learning method is a graph-based Multi-Agent Actor-Critic (MAAC) system. Runtime agents correspond to detected communities, not individual nodes.

## Key Ideas

- Directed weighted social networks with row-normalized influence weights
- Polarized initial opinions sampled around negative, neutral, and positive clusters
- Bounded-confidence opinion updates
- Community detection using Louvain or label propagation
- Community-level MAAC intervention over edge-weight changes
- Reward shaping for consensus gain, intervention cost, episode length, and harmful cross-community bridges
- Built-in zero-action and heuristic baselines
- Training, evaluation, and visualization entry points

## Repository Structure

```text
.
|-- agents/                    # MAAC implementation and shared neural networks
|-- config/                    # YAML experiment configurations
|-- envs/                      # Social network environment and graph factory
|-- tests/                     # Regression tests
|-- utils/                     # Baselines, metrics, evaluation, visualization helpers
|-- train.py                   # Train MAAC policies
|-- evaluate.py                # Compare detectors, baselines, and checkpoints
|-- visualize.py               # Render rollout and training diagnostic plots
|-- PROJECT_EXPLAINER.md       # Detailed project explanation
|-- RUN_AND_EVALUATE.md        # Extended command guide
|-- requirements.txt           # Python dependencies
```

Generated checkpoints and plots are written under `artifacts/`, which is ignored by Git.

## Setup

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\pip.exe install -r requirements.txt
```

The dependencies are:

- `torch`
- `gymnasium`
- `numpy`
- `networkx`
- `pyyaml`
- `matplotlib`

Run all commands from the project root.

## Configurations

Two experiment configs are included:

- `config/hyperparams.yaml`: step-wise intervention with `intervention_interval: 1`
- `config/hyperparams_periodic.yaml`: periodic intervention with held actions and `intervention_interval: 3`

Both modes use the same environment and training pipeline. With periodic intervention, the agent proposes a new action every `K` steps and the environment reuses the last action between intervention steps.

## Baseline Diagnostics

Run built-in zero-policy and heuristic-policy diagnostics:

```powershell
.\.venv\Scripts\python.exe train.py --diagnostics-only --device cuda
```

For CPU:

```powershell
.\.venv\Scripts\python.exe train.py --diagnostics-only --device cpu
```

Run diagnostics with the periodic config:

```powershell
.\.venv\Scripts\python.exe train.py --config config/hyperparams_periodic.yaml --diagnostics-only --device cuda
```

## Training

Train the default detector from `config/hyperparams.yaml`:

```powershell
.\.venv\Scripts\python.exe train.py --device cuda
```

Train with the periodic config:

```powershell
.\.venv\Scripts\python.exe train.py --config config/hyperparams_periodic.yaml --device cuda
```

Train a specific community detector:

```powershell
.\.venv\Scripts\python.exe train.py --community-detector label_propagation --device cuda
```

Useful overrides:

- `--config`
- `--episodes`
- `--batch-size`
- `--eval-every`
- `--community-detector`
- `--device`
- `--seed`
- `--deterministic-actions`

Training writes checkpoints and summaries to:

```text
artifacts/checkpoints/maac_<detector>_latest.pt
artifacts/checkpoints/maac_<detector>_best.pt
artifacts/checkpoints/maac_<detector>_summary.json
```

## Evaluation

Compare both detectors, baselines, and available trained checkpoints:

```powershell
.\.venv\Scripts\python.exe evaluate.py --community-detector all --device cuda
```

Evaluate periodic-config checkpoints:

```powershell
.\.venv\Scripts\python.exe evaluate.py --config config/hyperparams_periodic.yaml --community-detector all --device cuda
```

Evaluate one detector:

```powershell
.\.venv\Scripts\python.exe evaluate.py --community-detector louvain --device cuda
```

If checkpoints were saved elsewhere:

```powershell
.\.venv\Scripts\python.exe evaluate.py --checkpoint-dir path\to\checkpoints --device cuda
```

## Visualization

Render an episode storyboard for the best trained checkpoint:

```powershell
.\.venv\Scripts\python.exe visualize.py --config config/hyperparams_periodic.yaml --policy trained --checkpoint-type best --device cuda
```

Visualize the heuristic baseline:

```powershell
.\.venv\Scripts\python.exe visualize.py --config config/hyperparams_periodic.yaml --policy heuristic --device cuda
```

Choose a seed and output path:

```powershell
.\.venv\Scripts\python.exe visualize.py --config config/hyperparams_periodic.yaml --policy trained --seed 103 --output artifacts/plots/periodic_seed103.png --device cuda
```

Render training diagnostics from a saved summary:

```powershell
.\.venv\Scripts\python.exe visualize.py --kind training --community-detector louvain --device cuda
```

## Tests

Run the regression suite:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## More Detail

For a full explanation of the modeling assumptions, environment dynamics, reward design, MAAC architecture, and experiment interpretation, see `PROJECT_EXPLAINER.md`.

For a longer command guide, including smoke tests and experiment hygiene notes, see `RUN_AND_EVALUATE.md`.
