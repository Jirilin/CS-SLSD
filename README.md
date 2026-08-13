# CS-SLSD — Aug 13 Weekly Research Update

This version turns the earlier MNIST prototype into a shared controlled experiment for **MNIST, CIFAR-10 and SVHN** and compares:

1. `offline` — train only on initial trusted labels; never update on the stream.
2. `naive` — confidence pseudo-labels; update on current batch only.
3. `replay` — confidence pseudo-labels + reservoir replay.
4. `ewc` — confidence pseudo-labels + Fisher-based EWC.
5. `proposed` — centroid-refined pseudo-labels + reservoir replay + online EWC.

## 1. Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2. Test

```bash
python -m pytest tests -v
```

## 3. Smoke test first

```bash
python run_experiment.py --dataset mnist --method proposed --seed 0 --smoke
python run_experiment.py --dataset cifar10 --method proposed --seed 0 --smoke
python run_experiment.py --dataset svhn --method proposed --seed 0 --smoke
```

Smoke tests are only pipeline checks, not dissertation results.

## 4. One real run

```bash
python run_experiment.py --dataset mnist --method proposed --seed 0
```

## 5. Quick weekly comparison

```bash
python run_extended.py --quick
python aggregate_results.py --results-dir results/extended
python plot_results.py --results-dir results/extended
python run_analysis.py --results-dir results/extended
```

## 6. Full experiment matrix

This is 3 datasets × 5 methods × 5 seeds = **75 runs**.

```bash
python run_extended.py
python aggregate_results.py --results-dir results/extended
python plot_results.py --results-dir results/extended
python run_analysis.py --results-dir results/extended
```

Run it when you have enough time. Do not claim the full 75-run study is complete until the files exist.

## Output locations

- Per-batch metrics: `results/extended/metrics_*.csv`
- Per-class accuracy: `results/extended/class_accuracy_*.csv`
- Task-pair accuracy: `results/extended/task_accuracy_*.csv`
- Run summaries: `results/extended/summary_*.json`
- Mean ± SD table: `results/extended/comparison_mean_std.csv`
- Dissertation table: `results/extended/dissertation_comparison_table.csv`
- Figures: `results/extended/figures/`
- Change analysis: `results/extended/change_correlation_*.csv`
