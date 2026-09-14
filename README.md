# CS-SLSD — Continual Semi-Supervised Learning from Streaming Data

MSc Artificial Intelligence dissertation project at Oxford Brookes University.

## Research question
Can a continual semi-supervised image classifier adapt to a changing, predominantly unlabelled stream while retaining knowledge acquired from earlier data?

## Final method
The final implementation is an SDSL-inspired image-stream pipeline with a compact CNN (`SDSLVisionNet`), three-stage pseudo-label generation (classifier prediction, current-stream centroid refinement, invariant semantic reconstruction), low-rank gradient-subspace memory, and projected minimax replay.

## Experimental protocol
- Initial trusted labels: 100 samples per class
- Stream batches: 20
- Stream batch size: 256
- Dominant-pair fraction: approximately 70%
- Seeds: 0, 1, 2, 3, 4
- Datasets: MNIST, CIFAR-10, SVHN
- Final subspace rank: 8

Ground-truth labels of stream observations are hidden from the SDSL learner and retained only for evaluation metrics.

## Compared methods
- `offline`: joint-supervised oracle using true labels of the full stream; not a deployable continual semi-supervised method.
- `naive`: confidence pseudo-labels and direct updating.
- `replay`: confidence pseudo-labels plus reservoir replay.
- `ewc`: confidence pseudo-labels plus Elastic Weight Consolidation.
- `sdsl_full`: three-stage pseudo-labelling plus gradient-subspace projected minimax replay.

## Install
```bash
git clone https://github.com/Jirilin/CS-SLSD.git
cd CS-SLSD
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements_sdsl.txt
python -m pip install scipy
```

## Tests
```bash
python -m pytest tests -v
```
The unit tests check software behaviour such as tensor compatibility, pseudo-label generation, and parameter updates. Passing tests does not by itself prove the research hypothesis.

## Smoke test
```bash
python run_sdsl_full.py --dataset mnist --seed 0 --smoke
```
A smoke test is a shortened end-to-end validation run. Do not mix smoke outputs with final dissertation evidence.

## One full SDSL run
```bash
python run_sdsl_full.py --dataset mnist --seed 0
```

## Full SDSL matrix
```bash
python run_sdsl_matrix.py --datasets mnist cifar10 svhn --seeds 0 1 2 3 4
```
This launches 15 SDSL runs.

## Aggregate SDSL
```bash
python aggregate_sdsl.py --results-dir results/sdsl_full
```

## Full baseline matrix
```bash
python run_baseline_matrix.py   --datasets mnist cifar10 svhn   --methods offline naive replay ewc   --seeds 0 1 2 3 4   --output-dir results/baselines
```
This launches 60 baseline runs.

## Compare baselines and SDSL
```bash
python compare_baselines_and_sdsl.py   --baseline-dir results/baselines   --sdsl-dir results/sdsl_full
```

## Generate dissertation figures
```bash
python plot_dissertation_results.py
```

## Reproducibility
Seeds `0, 1, 2, 3, 4` are used because training and data ordering are stochastic. Results are aggregated as mean ± standard deviation.

## Main SDSL results
| Dataset | Final accuracy | Mean stream accuracy | Pseudo-label precision | Class-wise forgetting |
|---|---:|---:|---:|---:|
| MNIST | 0.8330 ± 0.0138 | 0.7509 ± 0.0252 | 0.7154 ± 0.0310 | 0.0551 ± 0.0281 |
| CIFAR-10 | 0.3742 ± 0.0105 | 0.3410 ± 0.0105 | 0.3098 ± 0.0126 | 0.0887 ± 0.0195 |
| SVHN | 0.1659 ± 0.0375 | 0.1299 ± 0.0285 | 0.1326 ± 0.0222 | 0.0765 ± 0.0267 |

Interpretation:
- MNIST is the strongest success case.
- CIFAR-10 shows modest adaptation with noisy pseudo-labels.
- SVHN is a weak/failure case in absolute performance.
- Pseudo-label quality is a central constraint.

## Viva demo
```bash
python -m pytest tests -v
python run_sdsl_full.py --dataset mnist --seed 0 --smoke
```
Then show existing final artefacts and explain:
`raw run → CSV/JSON → aggregation → mean ± SD → figures → dissertation evidence`.

## Limitations
- Simulated benchmark streams, mainly controlled class-prior shift.
- Compact CNN.
- Pseudo-label quality falls on CIFAR-10 and SVHN.
- Some archived baselines and SDSL use different pseudo-label thresholds.
- Flatness diagnostics are empirical, not formal proofs.
- Future work: matched thresholds, stronger encoders, realistic drift, component ablations.

## Key reference
Ren, W., Wang, P., Li, X., Hughes, C. E. and Fu, Y. (2022), *Semi-supervised Drifted Stream Learning with Short Lookback*, KDD 2022, pp. 1504–1513, DOI: 10.1145/3534678.3539297.

## Repository
https://github.com/Jirilin/CS-SLSD
