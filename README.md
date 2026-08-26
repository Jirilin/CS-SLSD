# CS-SLSD — Continual Semi-Supervised Learning from Streaming Data

Research prototype for controlled continual semi-supervised image-classification experiments on **MNIST, CIFAR-10 and SVHN**.

## Research question

Can a classifier continue learning from a changing, mostly unlabelled image stream while maintaining pseudo-label reliability and reducing catastrophic forgetting?

## Compared methods

- `offline`: initial trusted labels only; no stream adaptation.
- `naive`: confidence pseudo-labels; current batch only.
- `replay`: confidence pseudo-labels + reservoir replay.
- `ewc`: confidence pseudo-labels + Fisher-based EWC.
- `proposed`: centroid-refined pseudo-labels + reservoir replay + online EWC.

The current `proposed` implementation is a practical prototype and **is not claimed to be an exact implementation of SDSL's adaptive flat-region/minimax replay**.

## Core pipeline

1. Build a trusted initial labelled set.
2. Convert the remaining benchmark data into controlled sequential batches.
3. Train the base CNN.
4. Generate pseudo-labels from incoming unlabelled samples.
5. Apply replay and/or EWC depending on the selected method.
6. Evaluate accuracy, forgetting, pseudo-label quality, drift, parameter change and runtime after each batch.
7. Repeat with multiple seeds and aggregate mean ± standard deviation.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Test

```bash
python -m pytest tests -v
```

## Smoke-test the full pipeline

```bash
python run_experiment.py --dataset mnist --method proposed --seed 0 --smoke
python run_experiment.py --dataset cifar10 --method proposed --seed 0 --smoke
python run_experiment.py --dataset svhn --method proposed --seed 0 --smoke
```

Smoke results are pipeline checks only and must not be used as dissertation evidence.

## Multi-dataset experiments

Quick validation:

```bash
python run_extended.py --quick
```

Full matrix (3 datasets × 5 methods × 5 seeds = 75 runs):

```bash
python run_extended.py
```

Then aggregate and analyse:

```bash
python aggregate_results.py --results-dir results/extended
python plot_results.py --results-dir results/extended
python run_analysis.py --results-dir results/extended
python create_final_figures.py --results-dir results/extended
python generate_pipeline_diagram.py
```

## Ablations

```bash
python run_ablations.py
```

## Submission-readiness check

```bash
python validate_submission.py --results-dir results/extended
```

This repository check does not replace the official university submission checklist.

## Important outputs

- Per-run metrics: `results/extended/metrics_*.csv`
- Per-class accuracy: `results/extended/class_accuracy_*.csv`
- Task-group accuracy: `results/extended/task_accuracy_*.csv`
- Run summaries: `results/extended/summary_*.json`
- Mean ± SD: `results/extended/comparison_mean_std.csv`
- Final figures: `results/final_figures/`
- Final table: `results/final_tables/final_comparison_table.csv`
- Readiness report: `results/submission_readiness.csv`

## Finalisation documents

- `docs/WEEKLY_UPDATE_AUG26.md`
- `docs/FINAL_PRESENTATION_STRUCTURE.md`
- `docs/PAPERWORK_ETHICS_CHECKLIST.md`
- `docs/FIRST_DRAFT_REVIEW_CHECKLIST.md`
- `report_drafts/REVISED_ANALYSIS_DISCUSSION.md`
- `report_drafts/CONCLUSION_FUTURE_WORK.md`
- `report_drafts/FRONT_MATTER_TEMPLATE.md`
