# Project Explainer: Community-Aware Multi-Agent Reinforcement Learning for Opinion Dynamics

## 1. What This Project Is

This repository implements a reinforcement learning pipeline for steering a polarized social network toward consensus by modifying the strengths of interpersonal influence edges over time.

At a high level:

- The environment is a weighted directed social graph.
- Each node has a scalar opinion in `[-1, 1]`.
- Opinions evolve according to a bounded-confidence update rule derived from the Hegselmann-Krause model.
- The controller does not directly edit opinions. Instead, it changes the graph's edge weights, which indirectly changes future opinion evolution.
- The main learning method is a graph-based Multi-Agent Actor-Critic (`MAAC`) system in which each detected community is treated as one runtime agent.

The codebase currently represents a MAAC-focused rebuild. A single-agent SAC baseline is not implemented in the current version.

## 2. Research Question and Modeling Idea

The central research question is:

How can an intervention policy reshape interaction weights in a polarized social network so that consensus is reached faster, while avoiding harmful cross-community links and excessively large interventions?

The modeling logic is:

1. A social network is represented as a directed weighted graph.
2. Agents influence one another through row-normalized weights.
3. Opinions only move toward others who are within a confidence bound.
4. Reinforcement learning proposes edge-weight adjustments.
5. The reward encourages increased consensus and penalizes both intervention magnitude and harmful bridges.
6. Community detection is used to decompose the control problem into multiple local decision makers that share parameters.

This is therefore not a generic multi-agent RL benchmark. It is a structured control problem over dynamical social influence networks.

## 3. Repository Structure

The main files are:

- `train.py`: end-to-end MAAC training and checkpointing.
- `evaluate.py`: detector comparison, baseline evaluation, and trained-policy evaluation.
- `visualize.py`: rollout visualization entry point.
- `envs/social_network_env.py`: the main environment.
- `envs/network_factory.py`: graph generation, opinion initialization, and bounded-confidence dynamics.
- `agents/maac_agent.py`: MAAC learner.
- `agents/common/networks.py`: graph encoder, shared actor, and centralized critic.
- `agents/common/replay_buffer.py`: replay memory for off-policy learning.
- `utils/community_detection.py`: Louvain and label propagation community detection.
- `utils/baselines.py`: zero-action and heuristic baseline policies.
- `utils/evaluation.py`: rollout and summary helpers.
- `utils/visualization.py`: episode trace collection and figure generation.
- `config/hyperparams.yaml`: baseline stepwise-intervention configuration.
- `config/hyperparams_periodic.yaml`: periodic intervention configuration with held actions.
- `tests/`: regression tests for the environment, pipeline, detectors, and visualization.

## 4. Formal Problem Setup

### 4.1 State

At each environment step, the state includes:

- `opinions`: one scalar opinion per node.
- `weight_matrix`: the directed influence matrix.
- `community_labels`: integer community assignments.
- `community_membership`: one-hot community membership matrix.
- `action_mask`: valid local action locations for each runtime agent.
- `active_agent_mask`: which community slots are active.
- `global_consensus`: a scalar summary metric.
- `step_count`: current time step.

The environment exposes these as a Gymnasium dictionary observation.

### 4.2 Action

The action is a matrix of edge-weight deltas with shape `(num_nodes, num_nodes)`.

Interpretation:

- Each row corresponds to a source node.
- Each column corresponds to a target node.
- Positive values increase relative emphasis on a target.
- Negative values decrease relative emphasis.
- Actions are clipped to `[-1, 1]`.

When MAAC is used, each community-level actor only fills the rows belonging to nodes in its own community, and the full joint action matrix is assembled from those local actions.

### 4.3 Transition Dynamics

A step proceeds as follows:

1. The chosen action is scaled by `action_scale`.
2. The action is applied in logit space to the current row-stochastic weight matrix.
3. A row-wise softmax produces the updated weight matrix, keeping every row normalized.
4. Opinions are updated using a bounded-confidence Hegselmann-Krause rule.
5. Communities may be recomputed if the refresh interval is reached.

This means the policy controls social influence patterns, not belief values directly.

### 4.4 Objective

The environment reward is:

`reward = consensus_gain + terminal_bonus - step_penalty - action_penalty - bridge_penalty`

where:

- `consensus_gain` is the change in global consensus from the previous step.
- `terminal_bonus` is granted if the consensus threshold is reached.
- `step_penalty` discourages unnecessarily long episodes.
- `action_penalty` discourages large intervention magnitudes.
- `bridge_penalty` discourages harmful cross-community strengthening.

Episodes terminate either when the consensus threshold is reached or when `max_steps` is exhausted.

## 5. Social Network Environment

### 5.1 Initial Graph

The graph is generated by a scale-free construction in `envs/network_factory.py`.

Important details:

- The network is directed.
- New nodes attach using preferential sampling.
- Random positive raw edge weights are assigned.
- Rows are normalized so each node distributes unit influence mass across its outgoing neighbors.

This creates heterogeneous connectivity rather than a fully uniform graph.

### 5.2 Initial Opinions

Initial opinions are sampled from a polarized mixture:

- roughly `40%` near `-0.7`
- roughly `20%` near `0.0`
- roughly `40%` near `+0.7`

Gaussian noise is added and values are clipped to `[-1, 1]`.

This produces an initially polarized but not completely disconnected ideological landscape.

### 5.3 Opinion Dynamics

The opinion update uses a weighted Hegselmann-Krause style rule:

- Nodes only attend to neighbors whose opinion difference is within `confidence_bound`.
- A fixed self-belief term is added on the diagonal.
- Effective weights are renormalized row-wise.
- The next opinion vector is the weighted average under these filtered weights.

This encodes bounded confidence: agents ignore others who are too far away ideologically.

### 5.4 Community Detection

The environment periodically detects communities using an opinion-aware affinity graph:

- The directed weight matrix is symmetrized.
- Each pairwise affinity is downweighted by `exp(-opinion_gap / temperature)`.
- Community detection is then run on the resulting undirected graph.

Supported detectors:

- `louvain`
- `label_propagation`

The resulting communities define the runtime multi-agent decomposition.

### 5.5 Consensus Metric

Consensus is measured from the mean pairwise opinion gap:

- pairwise opinion gaps are computed over all unordered node pairs
- the mean gap is scaled into a score in `[0, 1]`
- higher is better

Perfect agreement gives a consensus score of `1.0`.

### 5.6 Harmful Bridge Mass

The project distinguishes between merely increasing cross-community edges and increasing them in a way that is likely harmful.

Harmful bridge mass is computed by:

- taking only positive increases in edge weight
- restricting to cross-community edges
- weighting those increases by the opinion gap between the connected nodes

This makes the penalty especially strong when the policy reinforces ties across highly disagreeing communities.

## 6. Why This Is Multi-Agent

The code does not assign one RL agent per node. Instead, it assigns one runtime agent per detected community.

That design has several consequences:

- The number of active agents can change over time.
- Agents correspond to mesoscopic social groups rather than individual people.
- The policy structure is parameter-sharing rather than fully independent learning.
- Community detection becomes part of the control pipeline, not just an analysis tool.

This is a strong modeling assumption: communities are treated as the right intervention granularity.

## 7. MAAC Architecture

### 7.1 Shared Graph Encoder

The graph encoder is implemented in `agents/common/networks.py`.

Its purpose is to transform the raw graph state into:

- node embeddings
- a pooled graph embedding

Input information used by the encoder includes:

- node opinions
- weighted out-degree
- weighted in-degree
- optional community mask
- community size information
- global consensus
- normalized step count
- fraction of active agents

The default encoder configuration is:

- GNN hidden dimensions: `[128, 128]`
- graph MLP hidden dimensions: `[128]`
- graph embedding dimension: `128`
- activation: `ReLU`
- layer normalization enabled
- dropout `0.0`

This is a graph-aware representation learner, not a flattened MLP over the whole state.

### 7.2 Shared Community Actor

The actor is shared across all runtime communities.

For one active community, it:

1. encodes the global graph while marking the focal community
2. extracts embeddings for source nodes inside that community
3. pairs each source node with all possible target nodes
4. combines source embedding, target embedding, current edge weight, opinion gap, and global graph context
5. predicts an edge logit for every local source-target pair

The default actor head uses:

- edge MLP hidden dimensions: `[128, 64]`

The actor therefore outputs local rewiring proposals for the nodes that belong to one community.

### 7.3 Centralized Critic

The critic is centralized.

It evaluates:

- the full global graph state
- the assembled joint action matrix across all communities

Its default architecture uses:

- edge-processing hidden dimensions: `[128, 64]`
- value head hidden dimensions: `[256, 128]`

The critic thus sees the whole coordination pattern, which is important because one community's rewiring decisions can affect everyone through future opinion dynamics.

### 7.4 Stochastic Policy Parameterization

The actor outputs mean logits. Exploration is introduced by:

- a learned global `log_std`
- Gaussian sampling
- `tanh` squashing

This is structurally similar to soft actor-critic style continuous control.

In deterministic mode, the actor uses the mean logits directly and returns zero log-probability.

### 7.5 Replay Buffer

Training is off-policy.

The replay buffer stores:

- full global state
- full joint action matrix
- scalar reward
- next global state
- terminal flag

Transitions are cloned to CPU before storage. Sampling is uniform without prioritization.

## 8. Learning Algorithm

The learner in `agents/maac_agent.py` is a shared-actor MAAC variant with a target critic.

For each sampled transition:

1. The target critic evaluates a next-state sampled action.
2. A Bellman target is formed:

`target = reward + gamma * (target_q - entropy_coef * next_log_prob)` for nonterminal transitions.

3. The critic minimizes mean squared error to this target.
4. The actor maximizes value while retaining an entropy term, implemented as minimizing:

`actor_loss = mean(entropy_coef * log_prob - q_value)`

5. The target critic is updated by soft interpolation with parameter `tau`.

Important notes:

- There is one critic, not a double-critic SAC variant.
- There is one shared actor for all communities.
- Gradient clipping is used.
- The policy standard deviation is a single learned scalar parameter, not a state-dependent variance head.

## 9. Training Pipeline

The training script follows this sequence:

1. Load configuration.
2. Resolve the device and seeds.
3. Run baseline diagnostics before any learning.
4. Construct the environment and MAAC agent.
5. Train for the configured number of episodes.
6. Periodically benchmark the trained policy on held benchmark seeds.
7. Save both latest and best checkpoints.
8. Write a JSON summary of metrics.

### 9.1 Diagnostics Before Learning

Before training starts, the script evaluates two non-learning policies:

- `zero`: applies no intervention
- `heuristic`: a hand-designed bridging policy

These diagnostics provide a sanity check and an interpretable baseline.

### 9.2 Episode Loop

Within each episode:

- the environment is reset with a seed from `train_seeds`
- a new action is requested only when intervention timing requires it
- otherwise the previous action may be reused
- transitions are stored after every environment step
- updates begin once the replay buffer reaches `learning_starts`
- gradient updates occur every `update_every` environment steps

### 9.3 Benchmarking and Checkpoint Selection

The policy is periodically benchmarked on `evaluation.benchmark_seeds`.

Checkpoint selection ranks policies by:

1. success rate
2. mean final consensus
3. mean reward
4. negative mean steps

This means faster successful consensus is preferred over slow or unreliable performance.

## 10. Intervention Modes

The project supports two intervention regimes in the same codebase.

### 10.1 Stepwise Intervention

In `config/hyperparams.yaml`:

- `intervention_interval = 1`
- the controller proposes a fresh action every environment step

### 10.2 Periodic Intervention with Held Actions

In `config/hyperparams_periodic.yaml`:

- `intervention_interval = 3`
- `hold_last_action = true`

This means the agent proposes a fresh intervention only every third step, and the previous action is reused on intermediate steps.

Academically, this is important because it changes the control problem from continuous intervention to sparse intervention with persistence.

## 11. Configurations in This Repository

### 11.1 Baseline Config

The default baseline config uses:

- `num_nodes = 20`
- `max_steps = 100`
- `confidence_bound = 0.7`
- `consensus_threshold = 0.95`
- `community_update_freq = 20`
- `attachment_edges = 2`
- `action_scale = 1.0`
- `terminal_bonus = 1.0`

Training defaults:

- `episodes = 150`
- `batch_size = 32`
- `learning_starts = 1800`
- `update_every = 20`
- `gamma = 0.99`
- `tau = 0.002`
- `actor_lr = 5e-5`
- `critic_lr = 1e-5`
- `entropy_coef = 0.02`

### 11.2 Periodic Config

The periodic config changes the task substantially:

- `confidence_bound = 0.3`
- `action_scale = 0.4`
- `intervention_interval = 3`
- `terminal_bonus = 2.0`
- `episodes = 1000`
- `learning_starts = 3000`
- checkpoint prefix changes to `maac_periodic`

This is not just a minor ablation. It creates a harder and more delayed control setting.

## 12. Baseline Policies

### 12.1 Zero Policy

The zero policy emits all-zero action matrices for each active community. It is the "no intervention" baseline.

### 12.2 Heuristic Bridge Policy

The heuristic policy scores candidate targets using:

- closeness to the global mean opinion
- moderate cross-community bridging within the confidence bound
- weaker within-community encouragement

It then centers and clips the scores to form actions.

This heuristic is useful because it encodes a domain intuition: connect communities, but only where the ideological distance is still bridgeable.

## 13. Evaluation Pipeline

`evaluate.py` evaluates, per detector:

- zero policy
- heuristic policy
- trained policy, if a checkpoint exists

Metrics aggregated over benchmark seeds include:

- mean reward
- mean final consensus
- mean steps
- success rate
- mean harmful bridge mass
- mean cross-community weight

Evaluation outputs are written as JSON into `artifacts/checkpoints/`.

## 14. Visualization Pipeline

The visualization code produces a multi-panel storyboard for a single rollout.

It records:

- opinion trajectories over time
- weight matrices over time
- community labels over time
- consensus trajectory
- cross-community weight trajectory
- harmful bridge mass
- intervention timing
- community refresh timing

The final plot includes:

- initial network snapshot
- final network snapshot
- opinion trajectory panel
- consensus and bridge dynamics panel
- textual run summary

This is valuable for qualitative analysis, not only for presentation.

## 15. Output Artifacts

Training writes:

- `artifacts/checkpoints/<prefix>_<detector>_latest.pt`
- `artifacts/checkpoints/<prefix>_<detector>_best.pt`
- `artifacts/checkpoints/<prefix>_<detector>_summary.json`

Evaluation writes:

- `artifacts/checkpoints/evaluation_summary.json`
- or `evaluation_summary_best.json` when best checkpoints are evaluated

Visualization writes:

- PNG figures under `artifacts/plots/`

Checkpoint files store:

- agent weights
- model configuration
- training configuration
- detector name
- summary rows
- checkpoint metadata

## 16. What the Tests Cover

The test suite checks:

- row-stochasticity of the updated weight matrix
- the ability of actions to create or strengthen edges
- bounded-confidence blocking behavior in the HK update
- reward accounting consistency
- end-to-end training smoke with checkpoint creation
- checkpoint reload fidelity for deterministic actions
- evaluation loading of saved checkpoints
- community-detection seed reproducibility
- episode-trace bookkeeping
- visualization file generation when matplotlib is available

This is a regression-oriented test suite. It verifies implementation stability, but it is not a proof of scientific validity.

## 17. Current Empirical State in the Repository

The repository already contains saved summaries in `artifacts/checkpoints/`.

Examples:

- The saved best periodic Louvain checkpoint reports mean final consensus around `0.879`, success rate `0.60`, and mean steps `58.8`.
- The saved best stepwise Louvain checkpoint reports mean final consensus around `0.886`, success rate `0.60`, and mean steps `60.2`.

These are repository-local stored results, not a full experimental study. They should be treated as implementation outputs rather than a complete scientific conclusion.

## 18. Important Implementation Boundaries

### 18.1 SAC Is Not Implemented Here

`agents/sac_agent.py` is only a scaffold that raises `NotImplementedError`.

So academically, the present codebase should be described as:

- a MAAC-based graph RL system
- with diagnostic baselines
- not a side-by-side finished SAC versus MAAC benchmark

### 18.2 The Critic Design Is Simpler Than Modern SAC Variants

The current learner uses:

- one centralized critic
- one target critic
- no twin Q-functions

That is a practical design choice, but it should not be conflated with full double-Q SAC-style stabilization.

### 18.3 Community Detection Is Part of the State Construction

Communities are not fixed.

They are recomputed during the episode, which means:

- the multi-agent decomposition changes over time
- control and representation depend on detector behavior
- performance is partly a function of the chosen community detector

### 18.4 The Environment Is a Stylized Model

This is not an empirical social science simulator calibrated to real data. It is a stylized controlled environment with:

- synthetic scale-free graphs
- synthetic polarized opinions
- bounded-confidence opinion updating
- RL-based adaptive rewiring

It is best interpreted as a computational experiment framework.

## 19. How To Describe This Project Academically

A concise accurate description would be:

"This project studies consensus control in synthetic polarized social networks using a graph-based multi-agent actor-critic architecture. Communities are detected online from an opinion-aware affinity graph and treated as runtime agents with shared parameters. Each agent proposes local rewiring interventions over outgoing influence weights, while a centralized critic evaluates the global state and joint action. The environment combines bounded-confidence opinion dynamics with rewards that encourage consensus gains while penalizing intervention magnitude and harmful cross-community bridging."

## 20. How To Run The Main Workflows

Install dependencies:

```powershell
.\.venv\Scripts\pip.exe install -r requirements.txt
```

Train:

```powershell
.\.venv\Scripts\python.exe train.py --device cuda
```

Train the periodic configuration:

```powershell
.\.venv\Scripts\python.exe train.py --config config/hyperparams_periodic.yaml --device cuda
```

Evaluate:

```powershell
.\.venv\Scripts\python.exe evaluate.py --community-detector all --device cuda
```

Visualize:

```powershell
.\.venv\Scripts\python.exe visualize.py --config config/hyperparams_periodic.yaml --policy trained --checkpoint-type best --device cuda
```

Run tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## 21. Final Takeaway

This repository is best understood as an experimental RL framework for adaptive rewiring of social influence networks under bounded-confidence opinion dynamics.

Its main scientific ingredients are:

- synthetic polarized graph generation
- dynamic opinion-aware community detection
- community-level multi-agent decomposition
- graph neural state encoding
- shared-actor centralized-critic learning
- multi-objective reward shaping for consensus control

Its main practical output is a reproducible pipeline for training, evaluating, and visualizing MAAC policies in this controlled setting.
