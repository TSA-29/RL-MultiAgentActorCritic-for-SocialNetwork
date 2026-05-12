# Run and Evaluate

This project has two runnable entry points:

- `train.py` for detector-specific MAAC training
- `evaluate.py` for detector comparison, baselines, and trained checkpoint evaluation
- `visualize.py` for episode-level rollout storyboards

The environment now supports two intervention schedules inside the same codebase:

- Baseline step-wise intervention: `intervention_interval: 1`
- Periodic intervention with held actions: `intervention_interval: K` and `hold_last_action: true`

In both modes:

- opinions evolve every environment step
- rewards are computed every environment step
- `train.py` and `evaluate.py` use the same pipeline

With `intervention_interval > 1`, the agent only proposes a fresh action every `K` steps. The environment reuses the last applied joint action on the in-between steps.

## Setup

Install the required packages in your environment:

```powershell
.\.venv\Scripts\pip.exe install -r requirements.txt
```

If you are training on GPU, confirm PyTorch sees CUDA:

```powershell
@'
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no-cuda")
'@ | .\.venv\Scripts\python.exe -
```

Run commands from the project root:

```powershell
cd C:\Users\Panda\Desktop\Uni\Tsinghua\Y1S2\RL\Project
```

The current config already uses CUDA for training:

```json
"training": {
  "device": "cuda"
}
```

If you keep that setting, `train.py` does not need a `--device` override. The examples below use `--device cuda` explicitly so the runtime choice is obvious.

## Intervention Config

Predefined configs in this repo:

- `config/hyperparams.yaml`: baseline step-wise intervention with `intervention_interval: 1`
- `config/hyperparams_periodic.yaml`: periodic intervention with held actions and `intervention_interval: 3`

Baseline step-wise mode:

```json
"environment": {
  "intervention_interval": 1,
  "hold_last_action": true
}
```

Periodic intervention from `config/hyperparams_periodic.yaml`:

```json
"environment": {
  "intervention_interval": 3,
  "hold_last_action": true
}
```

Recommended experiment hygiene:

- Keep `config/hyperparams.yaml` as the baseline config
- Run periodic experiments with `--config config/hyperparams_periodic.yaml`
- Use a different `checkpoint_prefix` or `checkpoint_dir` when comparing modes so checkpoints do not overwrite each other

## Baseline Diagnostics

Run the zero-policy and heuristic checks before training:

```powershell
.\.venv\Scripts\python.exe train.py --diagnostics-only --device cuda
```

Run the same diagnostics with the periodic config:

```powershell
.\.venv\Scripts\python.exe train.py --config config/hyperparams_periodic.yaml --diagnostics-only --device cuda
```

This prints a table with:

- `Consensus`: mean final consensus over the benchmark seeds
- `Success`: fraction of episodes reaching the consensus threshold
- `Reward`: mean episode return
- `Steps`: mean number of environment steps
- `HarmfulBridge`: mean added harmful cross-community mass

## Train One Detector

Train the default detector from `config/hyperparams.yaml`:

```powershell
.\.venv\Scripts\python.exe train.py --device cuda
```

Train the periodic config from `config/hyperparams_periodic.yaml`:

```powershell
.\.venv\Scripts\python.exe train.py --config config/hyperparams_periodic.yaml --device cuda
```

Train the label-propagation variant:

```powershell
.\.venv\Scripts\python.exe train.py --community-detector label_propagation --device cuda
```

Train the label-propagation variant with the periodic config:

```powershell
.\.venv\Scripts\python.exe train.py --config config/hyperparams_periodic.yaml --community-detector label_propagation --device cuda
```

Useful overrides:

- `--config`
- `--episodes`
- `--batch-size`
- `--community-detector`
- `--device`
- `--seed`
- `--deterministic-actions`

Training writes:

- `artifacts/checkpoints/maac_<detector>_latest.pt`
- `artifacts/checkpoints/maac_<detector>_best.pt`
- `artifacts/checkpoints/maac_<detector>_summary.json`

If you are comparing baseline vs periodic intervention, keep separate checkpoint names. Example:

```json
"training": {
  "checkpoint_prefix": "maac_periodic"
}
```

## Evaluate Detectors

Compare both detectors using any available checkpoints plus the built-in baselines:

```powershell
.\.venv\Scripts\python.exe evaluate.py --community-detector all --device cuda
```

Evaluate checkpoints produced by the periodic config:

```powershell
.\.venv\Scripts\python.exe evaluate.py --config config/hyperparams_periodic.yaml --community-detector all --device cuda
```

Evaluate just one detector:

```powershell
.\.venv\Scripts\python.exe evaluate.py --community-detector louvain --device cuda
```

If checkpoints were saved elsewhere:

```powershell
.\.venv\Scripts\python.exe evaluate.py --checkpoint-dir path\to\checkpoints --device cuda
```

## Visualize One Episode

Render a richer episode storyboard with:

- node-level opinion trajectories over time
- initial and final network snapshots
- consensus and cross-community bridge dynamics

Visualize the best trained checkpoint from the periodic config:

```powershell
.\.venv\Scripts\python.exe visualize.py --config config/hyperparams_periodic.yaml --policy trained --checkpoint-type best --device cuda
```

Visualize the heuristic baseline instead:

```powershell
.\.venv\Scripts\python.exe visualize.py --config config/hyperparams_periodic.yaml --policy heuristic --device cuda
```

Pick a specific evaluation seed and output file:

```powershell
.\.venv\Scripts\python.exe visualize.py --config config/hyperparams_periodic.yaml --policy trained --seed 103 --output artifacts/plots/periodic_seed103.png --device cuda
```

Useful overrides:

- `--policy trained|heuristic|zero`
- `--checkpoint-type latest|best`
- `--seed`
- `--community-detector`
- `--output`
- `--deterministic`

## Smoke Test

Quick one-episode training smoke from Python:

```powershell
@'
from argparse import Namespace
from utils.config_io import load_config
import train

config = load_config("config/hyperparams.yaml")
config["environment"]["num_nodes"] = 10
config["environment"]["max_steps"] = 5
config["training"]["episodes"] = 1
config["training"]["batch_size"] = 2
config["training"]["learning_starts"] = 1
config["training"]["log_every"] = 1
config["training"]["device"] = "cuda"
config["training"]["train_seeds"] = [7]
config["evaluation"]["benchmark_seeds"] = [101]
train.train(config, Namespace(diagnostics_only=False))
'@ | .\.venv\Scripts\python.exe -
```

If you want to smoke test periodic intervention instead of the baseline, also set:

```python
config["environment"]["intervention_interval"] = 3
config["environment"]["hold_last_action"] = True
```

## Tests

Run the regression suite:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
