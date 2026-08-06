# CS-SLSD — Controlled MNIST Weekly Update

This version creates a controlled continual semi-supervised MNIST experiment with:

- fixed class-prior stream;
- recorded class distribution and total-variation change;
- centroid-refined pseudo-labels with fixed semantic references;
- corrected diagonal empirical Fisher estimation;
- EWC and online EWC support;
- reservoir replay when the buffer becomes full;
- fixed random seeds and repeated runs;
- mean ± standard-deviation summaries;
- pseudo-label precision and coverage;
- feature-centroid drift and parameter-change measurements;
- component ablations.

## Setup

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
# .venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Quick validation

```bash
python run_experiment.py --method centroid_ewc --seed 0
```

## Main repeated comparison

```bash
python run_repeated.py --seeds 0 1 2 3 4
python plot_results.py
```

## Quick parameter sweep

```bash
python run_sweep.py --quick
```

## Ablation study

```bash
python run_ablations.py
```

## Main methods

- `naive`: confidence-only pseudo-labels.
- `centroid`: confidence + fixed centroid agreement.
- `centroid_ewc`: centroid refinement + EWC.
- `centroid_replay`: centroid refinement + reservoir replay.
- `centroid_replay_ewc`: combined model with replay and online EWC refresh.

MNIST labels in stream batches are hidden from training and used only to calculate research metrics.
