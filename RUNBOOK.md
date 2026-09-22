# CS-SLSD Dissertation Runbook

## Project
Continual Semi-Supervised Learning from Streaming Data  
MSc Artificial Intelligence — Oxford Brookes University  
Module: COMP7039

## Repository and backup locations
- GitHub: https://github.com/Jirilin/CS-SLSD.git
- Google Drive: https://drive.google.com/drive/folders/1hRGVg1P4oa4-ZlWCTiSVFeM1gj32CDsH?usp=share_link

## 1. Workflow
1. Set up environment.
2. Install dependencies.
3. Compile and test code.
4. Run a smoke test.
5. Run one full SDSL experiment.
6. Run the SDSL matrix.
7. Aggregate SDSL results.
8. Run baseline matrix.
9. Compare SDSL with baselines.
10. Generate figures.
11. Validate result integrity.
12. Commit and archive final artefacts.

Smoke/debug outputs must not be mixed with final dissertation evidence.

## 2. Main files
Final SDSL:
- `sdsl_config.py`
- `sdsl_model.py`
- `sdsl_pseudolabel.py`
- `gradient_subspace.py`
- `flat_minimax.py`
- `run_sdsl_full.py`
- `run_sdsl_matrix.py`
- `aggregate_sdsl.py`

Utilities:
- `dataset_stream.py`
- `seed_utils.py`
- `metrics.py`

Baselines:
- `run_experiment.py`
- `run_baseline_matrix.py`
- `replay_buffer.py`
- `ewc.py`
- `compare_baselines_and_sdsl.py`

Visualisation:
- `plot_dissertation_results.py`

## 3. Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements_sdsl.txt
python -m pip install scipy
```

Check versions:
```bash
python --version
python -c "import torch; print(torch.__version__)"
python -c "import torchvision; print(torchvision.__version__)"
python -c "import scipy; print(scipy.__version__)"
```

## 4. Validation
```bash
python -m py_compile   sdsl_model.py sdsl_pseudolabel.py gradient_subspace.py flat_minimax.py   run_sdsl_full.py run_sdsl_matrix.py aggregate_sdsl.py   run_experiment.py run_baseline_matrix.py compare_baselines_and_sdsl.py
```

```bash
python -m pytest tests -v
```

Unit tests validate software behaviour; they do not establish the scientific hypothesis.

## 5. Smoke test
```bash
python run_sdsl_full.py --dataset mnist --seed 0 --smoke
```

Purpose: catch dependency, dataset, tensor-shape, device, optimisation and output-writing errors before full experiments.

Do not use smoke-test metrics as final dissertation evidence.

## 6. One full SDSL run
```bash
python run_sdsl_full.py --dataset mnist --seed 0
```

Use `cifar10` or `svhn` for the other datasets.

## 7. Full SDSL matrix
```bash
python run_sdsl_matrix.py   --datasets mnist cifar10 svhn   --seeds 0 1 2 3 4
```

Expected: 3 datasets × 5 seeds = 15 full SDSL runs.

## 8. Aggregate SDSL
```bash
python aggregate_sdsl.py --results-dir results/sdsl_full
```

Expected:
- `results/sdsl_full/sdsl_runs.csv`
- `results/sdsl_full/sdsl_mean_std.csv`

## 9. Baseline matrix
```bash
python run_baseline_matrix.py   --datasets mnist cifar10 svhn   --methods offline naive replay ewc   --seeds 0 1 2 3 4   --output-dir results/baselines
```

Expected: 4 methods × 3 datasets × 5 seeds = 60 baseline runs.

Definitions:
- Offline: joint-supervised oracle/reference using full stream ground truth.
- Naive: confidence pseudo-labels with direct updating.
- Replay: confidence pseudo-labels + reservoir replay.
- EWC: confidence pseudo-labels + Elastic Weight Consolidation.

## 10. Final comparison
```bash
python compare_baselines_and_sdsl.py   --baseline-dir results/baselines   --sdsl-dir results/sdsl_full
```

Use only the cleaned final aggregate in the dissertation.

## 11. Figures
```bash
python plot_dissertation_results.py
```

Recommended figures:
- MNIST stream accuracy
- CIFAR-10 stream accuracy
- SVHN stream accuracy
- final accuracy comparison
- forgetting
- pseudo-label precision
- pseudo-label coverage
- projected-minimax diagnostics
- runtime

## 12. Result integrity
Check duplicates:
```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv("results/final_method_runs.csv")
keys = ["dataset","method","seed"]
print("Rows:", len(df))
print("Duplicate keys:", df.duplicated(keys).sum())
print(df.groupby(["dataset","method"]).size())
PY
```

For a complete 75-run set:
- 75 rows
- 0 duplicate `(dataset, method, seed)` keys
- 5 runs per dataset/method combination

Also verify:
- no smoke/debug runs;
- no stale aggregate CSV;
- no missing seeds;
- report hyperparameters match JSON configs;
- baseline and SDSL pseudo-label thresholds are disclosed;
- all dissertation figures trace back to CSV/JSON outputs.

## 13. Evidence chain
```text
source code
→ raw CSV/JSON
→ seed aggregation
→ mean ± SD
→ comparison table
→ figures
→ dissertation claims
```

## 14. Git
```bash
git status
git add .
git commit -m "Final dissertation implementation and reproducibility documentation"
git push
git rev-parse HEAD
```

Record the repository URL and final commit hash in the dissertation.

## 15. Viva demo
Use:
```bash
python -m pytest tests -v
python run_sdsl_full.py --dataset mnist --seed 0 --smoke
```

Then show existing final result artefacts instead of starting the full experiment matrix live.

## 16. Final archive
Retain:
- source code
- requirements files
- tests
- raw run outputs
- aggregated results
- figures
- project log
- ethics documents
- report
- presentation
- this runbook

Do not commit:
- `.venv/`
- `__pycache__/`
- downloaded benchmark data
- secrets/tokens

Suggested `.gitignore`:
```gitignore
.venv/
__pycache__/
*.pyc
.DS_Store
data/
.env
```

## 17. Limitations to disclose
- controlled benchmark stream rather than unrestricted real-world drift;
- class-prior shift is the principal simulated change;
- pseudo-label quality is dataset-dependent;
- compact CNN capacity;
- archived baseline and SDSL thresholds may differ;
- flatness diagnostics are empirical, not formal proofs;
- component ablations are future work unless actually executed.

## 18. Key reference
Ren, W., Wang, P., Li, X., Hughes, C.E. and Fu, Y. (2022) ‘Semi-supervised Drifted Stream Learning with Short Lookback’, *Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*, pp. 1504–1513. DOI: 10.1145/3534678.3539297.
