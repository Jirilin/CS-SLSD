# Continual Semi-Supervised Learning from Streaming Data

This repository contains the final-week implementation package for the MSc dissertation project.

## Core idea
The project tests whether an image classifier can keep learning from a changing stream of mostly unlabelled data without forgetting earlier knowledge.

## Implemented methods
- Offline baseline
- Naive online pseudo-labelling
- Replay baseline
- EWC baseline
- Proposed prototype: centroid-refined pseudo-labelling + reservoir replay + online EWC

## Datasets
- MNIST
- CIFAR-10
- SVHN

The datasets are downloaded automatically by Torchvision into `./data`. Do not submit the `data/` folder.

## Quick validation
```bash
python -m pytest tests -v
python run_experiment.py --dataset mnist --method proposed --seed 0 --smoke
python run_experiment.py --dataset cifar10 --method proposed --seed 0 --smoke
python run_experiment.py --dataset svhn --method proposed --seed 0 --smoke
```

## Full final run
```bash
python run_final_pipeline.py --full
```

If interrupted:
```bash
python run_final_pipeline.py --full --skip-tests
```

`run_extended.py` supports `--resume`, so completed summaries are skipped.

## Result processing
```bash
python aggregate_results.py --results-dir results/extended
python plot_results.py --results-dir results/extended
python run_analysis.py --results-dir results/extended
python create_final_figures.py --results-dir results/extended
python validate_submission.py --results-dir results/extended
```

## Important outputs
- `results/extended/dissertation_comparison_table.csv`
- `results/extended/comparison_mean_std.csv`
- `results/extended/change_correlation_summary.csv`
- `results/extended/figures/`
- `results/final_figures/`
- `results/extended/environment_reproducibility.json`
- `results/extended/submission_manifest.csv`

## Report and presentation materials
- `report_drafts/FINAL_REPORT_MASTER_DRAFT.md`
- `docs/FINAL_PRESENTATION_SCRIPT.md`
- `docs/VIVA_PREPARATION.md`
- `docs/DEMO_RUNBOOK.md`
- `docs/FINAL_SUBMISSION_CHECKLIST.md`
- `docs/SUPERVISOR_FINAL_REVIEW.md`
- `docs/REFERENCE_AUDIT_HARVARD.md`

## Important limitation
The implemented proposed prototype is inspired by SDSL and operationalises robust pseudo-labelling with centroid references. Its anti-forgetting mechanism is reservoir replay + online EWC. It is not a full reproduction of the original SDSL minimax flat-region replay solver. State this clearly in the report unless the minimax component is implemented and verified.
