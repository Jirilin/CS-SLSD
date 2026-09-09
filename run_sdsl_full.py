from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from dataset_stream import ControlledVisionStream
from gradient_subspace import ParameterVectorizer, GradientSubspaceMemory, gradient_vector_on_loader
from metrics import (accuracy, class_accuracy, task_accuracy_vector, snapshot,
                     parameter_change, forgetting_from_history,
                     average_incremental_accuracy, backward_transfer_proxy)
from sdsl_config import SDSLConfig
from sdsl_model import SDSLVisionNet
from sdsl_pseudolabel import SDSLRobustPseudoLabeler
from flat_minimax import SDSLFlatMinimaxSolver
from seed_utils import set_global_seed, choose_device


def tensor_loader(x, y, batch_size, shuffle=True):
    if x is None or y is None or x.numel() == 0:
        return None
    return DataLoader(TensorDataset(x.detach().cpu(), y.detach().cpu()),
                      batch_size=batch_size, shuffle=shuffle)


def materialise_loader(loader, limit=None):
    xs, ys = [], []
    n = 0
    for x, y in loader:
        if limit is not None and n >= limit:
            break
        if limit is not None and n + len(x) > limit:
            keep = limit - n
            x, y = x[:keep], y[:keep]
        xs.append(x.cpu()); ys.append(y.cpu()); n += len(x)
    return torch.cat(xs, 0), torch.cat(ys, 0)


def combine(gold_x, gold_y, extra_x, extra_y):
    if extra_x is None or extra_y is None or extra_x.numel() == 0:
        return gold_x, gold_y
    return torch.cat([gold_x, extra_x.detach().cpu()], 0), torch.cat([gold_y, extra_y.detach().cpu()], 0)


def standard_train(model, loader, device, epochs, lr, weight_decay=1e-4):
    if loader is None or epochs <= 0:
        return float("nan")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    losses = []
    for _ in range(epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().item()))
    return float(np.mean(losses)) if losses else float("nan")


def flat_replay_train(model, loader, device, solver, epochs):
    stats = []
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            stats.append(solver.step(x, y))
    if not stats:
        return {k: float("nan") for k in ["replay_loss", "worst_loss", "sharpness_gap", "flat_width", "projected_gradient_fraction", "update_norm"]}
    return {
        "replay_loss": float(np.mean([s.clean_loss for s in stats])),
        "worst_loss": float(np.mean([s.worst_loss for s in stats])),
        "sharpness_gap": float(np.mean([s.sharpness_gap for s in stats])),
        "flat_width": float(np.mean([s.perturb_norm for s in stats])),
        "projected_gradient_fraction": float(np.mean([s.projected_gradient_fraction for s in stats])),
        "update_norm": float(np.mean([s.update_norm for s in stats])),
    }


def parse_args():
    p = argparse.ArgumentParser(description="Full SDSL generation-replay + minimax flat-region solver")
    p.add_argument("--dataset", choices=["mnist", "cifar10", "svhn"], default="mnist")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="results/sdsl_full")
    p.add_argument("--stream-batches", type=int, default=None)
    p.add_argument("--stream-batch-size", type=int, default=None)
    p.add_argument("--initial-epochs", type=int, default=None)
    p.add_argument("--replay-epochs", type=int, default=None)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--semantic-rank", type=int, default=None)
    p.add_argument("--subspace-rank", type=int, default=None)
    p.add_argument("--ascent-steps", type=int, default=None)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    overrides = {"dataset": args.dataset, "seed": args.seed}
    for name in ["stream_batches", "stream_batch_size", "initial_epochs", "replay_epochs",
                 "semantic_rank", "subspace_rank", "ascent_steps"]:
        value = getattr(args, name)
        if value is not None:
            overrides[name] = value
    if args.threshold is not None:
        overrides["pseudo_threshold"] = args.threshold
    if args.smoke:
        overrides.update(stream_batches=2, stream_batch_size=96, initial_epochs=1,
                         generation_warmup_epochs=1, replay_epochs=1, ascent_steps=1,
                         subspace_batches=1)
    cfg = SDSLConfig(**overrides)

    set_global_seed(cfg.seed)
    device = choose_device(cfg.device)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    stream = ControlledVisionStream(cfg.dataset, cfg.data_root, cfg.seed, cfg.initial_per_class,
                                    cfg.stream_batches, cfg.stream_batch_size, cfg.dominant_fraction)
    spec = stream.spec
    distribution = stream.distribution_table()
    distribution.to_csv(out / f"distribution_{spec.name}_seed{cfg.seed}.csv", index=False)

    model = SDSLVisionNet(spec.in_channels, spec.num_classes).to(device)
    initial_train = stream.initial_loader(cfg.train_batch_size)
    initial_eval = stream.initial_eval_loader(cfg.test_batch_size)
    test_loader = stream.test_loader(cfg.test_batch_size)
    task_loaders = stream.class_group_test_loaders(cfg.test_batch_size)
    gold_x, gold_y = materialise_loader(initial_eval)

    start = time.perf_counter()
    initial_loss = standard_train(model, initial_train, device, cfg.initial_epochs, cfg.initial_lr, cfg.weight_decay)

    labeler = SDSLRobustPseudoLabeler(
        model, device, spec.num_classes, cfg.semantic_rank, cfg.semantic_blend,
        cfg.centroid_iterations, cfg.centroid_temperature, cfg.classifier_weight,
        cfg.pseudo_threshold, cfg.min_accept_fraction,
    )
    labeler.fit_invariant_semantics(initial_eval)

    subspace = GradientSubspaceMemory(cfg.subspace_rank)
    vectorizer = ParameterVectorizer(model)
    # Initial trusted-gradient direction seeds M for the first stream replay.
    init_grad = gradient_vector_on_loader(model, initial_eval, device, vectorizer, cfg.subspace_batches)
    subspace.add(init_grad)
    solver = SDSLFlatMinimaxSolver(model, subspace, cfg.replay_lr, cfg.ascent_lr,
                                   cfg.ascent_steps, cfg.max_perturb_norm,
                                   cfg.gradient_clip, cfg.weight_decay)

    initial_acc = accuracy(model, test_loader, device)
    class_hist = [class_accuracy(model, test_loader, device, spec.num_classes)]
    task_hist = [task_accuracy_vector(model, task_loaders, device)]
    previous_snapshot = snapshot(model)
    previous_x = previous_y = None
    rows = [{
        "dataset": spec.name, "method": "sdsl_full", "seed": cfg.seed, "batch": -1,
        "test_accuracy": initial_acc, "generation_loss": initial_loss,
        "pseudo_precision": np.nan, "pseudo_coverage": np.nan,
        "initial_pseudo_precision": np.nan, "cluster_precision": np.nan,
        "semantic_precision": np.nan, "classifier_cluster_agreement": np.nan,
        "semantic_shift": 0.0, "distribution_tv": 0.0,
        "parameter_change": 0.0, "replay_loss": np.nan, "worst_loss": np.nan,
        "sharpness_gap": np.nan, "flat_width": 0.0,
        "projected_gradient_fraction": 0.0, "update_norm": 0.0,
        "subspace_rank": len(subspace), "accepted_count": 0, "batch_seconds": 0.0,
    }]

    for batch in stream.batches():
        tick = time.perf_counter()

        # SDSL Step 1: refresh encoder/classifier with gold D plus D_hat(t-1).
        warm_x, warm_y = combine(gold_x, gold_y, previous_x, previous_y)
        warm_loader = tensor_loader(warm_x, warm_y, cfg.train_batch_size, shuffle=True)
        generation_loss = standard_train(model, warm_loader, device, cfg.generation_warmup_epochs,
                                         cfg.warmup_lr, cfg.weight_decay)

        # SDSL Steps 2 & 3: centroid adjustment + invariant semantic reconstruction.
        pl = labeler.generate(batch.images, batch.hidden_labels)

        # Replay uses always-available gold D + current pseudo-labeled stream D_hat(t).
        replay_x, replay_y = combine(gold_x, gold_y, pl.images, pl.labels)
        replay_loader = tensor_loader(replay_x, replay_y, cfg.train_batch_size, shuffle=True)
        flat_stats = flat_replay_train(model, replay_loader, device, solver, cfg.replay_epochs)

        # Current short lookback becomes the knowledge source for t+1.
        keep = min(cfg.lookback_limit, pl.images.size(0))
        if keep > 0:
            # Highest-confidence samples are retained; no ground-truth labels are consulted.
            idx = torch.topk(pl.confidence, k=keep).indices
            previous_x = pl.images[idx].detach().cpu()
            previous_y = pl.labels[idx].detach().cpu()
            look_loader = tensor_loader(previous_x, previous_y, cfg.train_batch_size, shuffle=False)
            g_old = gradient_vector_on_loader(model, look_loader, device, vectorizer, cfg.subspace_batches)
            subspace.add(g_old)

        current_acc = accuracy(model, test_loader, device)
        class_hist.append(class_accuracy(model, test_loader, device, spec.num_classes))
        task_hist.append(task_accuracy_vector(model, task_loaders, device))
        change = parameter_change(previous_snapshot, model)
        previous_snapshot = snapshot(model)
        dist = float(distribution.iloc[batch.batch_id].total_variation_from_previous)
        rows.append({
            "dataset": spec.name, "method": "sdsl_full", "seed": cfg.seed, "batch": batch.batch_id,
            "test_accuracy": current_acc, "generation_loss": generation_loss,
            "pseudo_precision": pl.precision, "pseudo_coverage": pl.coverage,
            "initial_pseudo_precision": pl.initial_precision,
            "cluster_precision": pl.cluster_precision,
            "semantic_precision": pl.semantic_precision,
            "classifier_cluster_agreement": pl.classifier_cluster_agreement,
            "semantic_shift": pl.semantic_shift, "distribution_tv": dist,
            "parameter_change": change, **flat_stats,
            "subspace_rank": len(subspace), "accepted_count": pl.accepted_count,
            "batch_seconds": time.perf_counter() - tick,
        })
        print(f"[{spec.name} seed={cfg.seed}] batch {batch.batch_id:02d} "
              f"acc={current_acc:.4f} pl_prec={pl.precision:.4f} cov={pl.coverage:.3f} "
              f"width={flat_stats['flat_width']:.4f} sharp={flat_stats['sharpness_gap']:.4f}")

    df = pd.DataFrame(rows)
    stem = f"{spec.name}_sdsl_full_seed{cfg.seed}"
    df.to_csv(out / f"metrics_{stem}.csv", index=False)
    pd.DataFrame(class_hist, columns=[f"class_{c}_accuracy" for c in range(spec.num_classes)]).assign(
        batch=[-1] + list(range(cfg.stream_batches))).to_csv(out / f"class_accuracy_{stem}.csv", index=False)
    pd.DataFrame(task_hist, columns=[f"task_{i}_accuracy" for i in range(5)]).assign(
        batch=[-1] + list(range(cfg.stream_batches))).to_csv(out / f"task_accuracy_{stem}.csv", index=False)

    summary = {
        "dataset": spec.name,
        "method": "sdsl_full",
        "seed": cfg.seed,
        "initial_accuracy": float(df.iloc[0].test_accuracy),
        "final_accuracy": float(df.iloc[-1].test_accuracy),
        "mean_stream_accuracy": float(df[df.batch >= 0].test_accuracy.mean()),
        "classwise_forgetting": forgetting_from_history(class_hist),
        "average_incremental_accuracy": average_incremental_accuracy(task_hist),
        "backward_transfer_proxy": backward_transfer_proxy(task_hist),
        "mean_pseudo_precision": float(df.pseudo_precision.mean(skipna=True)),
        "mean_pseudo_coverage": float(df.pseudo_coverage.mean(skipna=True)),
        "mean_initial_pseudo_precision": float(df.initial_pseudo_precision.mean(skipna=True)),
        "mean_cluster_precision": float(df.cluster_precision.mean(skipna=True)),
        "mean_semantic_precision": float(df.semantic_precision.mean(skipna=True)),
        "mean_sharpness_gap": float(df.sharpness_gap.mean(skipna=True)),
        "mean_flat_width": float(df.flat_width.mean(skipna=True)),
        "mean_projected_gradient_fraction": float(df.projected_gradient_fraction.mean(skipna=True)),
        "mean_parameter_change": float(df.parameter_change.mean(skipna=True)),
        "mean_batch_seconds": float(df[df.batch >= 0].batch_seconds.mean()),
        "total_seconds": float(time.perf_counter() - start),
        "final_subspace_rank": len(subspace),
        "config": cfg.__dict__,
    }
    with open(out / f"summary_{stem}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
