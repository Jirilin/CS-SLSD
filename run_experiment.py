from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from config import ExperimentConfig
from seed_utils import set_global_seed, choose_device
from sdsl_model import SDSLVisionNet
from dataset_stream import ControlledVisionStream
from ewc import OnlineEWC
from replay_buffer import ReservoirReplayBuffer
from metrics import (
    accuracy,
    class_accuracy,
    task_accuracy_vector,
    snapshot,
    parameter_change,
    forgetting_from_history,
    average_incremental_accuracy,
    backward_transfer_proxy,
)


# "offline" is an oracle/reference condition: it is allowed to use the
# ground-truth labels of the complete stream and trains jointly on all data.
# The other three methods never train on hidden stream labels.
METHODS = ["offline", "naive", "replay", "ewc"]


def make_loader(
    x: torch.Tensor | None,
    y: torch.Tensor | None,
    batch_size: int = 64,
    shuffle: bool = True,
) -> DataLoader | None:
    if x is None or y is None or x.numel() == 0:
        return None
    dataset = TensorDataset(x.detach().cpu(), y.detach().cpu())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def collect_loader(loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    xs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    for x, y in loader:
        xs.append(x.detach().cpu())
        ys.append(y.detach().cpu())
    if not xs:
        raise RuntimeError("Cannot collect an empty DataLoader.")
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def train_epochs(
    model: torch.nn.Module,
    loader: DataLoader | None,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    ewc: OnlineEWC | None = None,
) -> float:
    if loader is None or epochs <= 0:
        return float("nan")

    losses: List[float] = []
    for _ in range(epochs):
        model.train()
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            if ewc is not None:
                loss = loss + ewc.penalty()

            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def confidence_pseudo_labels(
    model: torch.nn.Module,
    images: torch.Tensor,
    hidden_labels: torch.Tensor,
    device: torch.device,
    threshold: float,
):
    """
    Generate confidence-filtered pseudo-labels.

    IMPORTANT:
    ``hidden_labels`` are used only after pseudo-label selection to calculate
    evaluation precision.  They are never used to select or train pseudo-labels.
    """
    model.eval()

    x = images.to(device)
    probs = torch.softmax(model(x), dim=1)
    conf, pred = probs.max(dim=1)
    mask = conf >= threshold

    accepted_x = x[mask]
    accepted_y = pred[mask]
    coverage = float(mask.float().mean().item())

    # Evaluation only.  Move every tensor participating in CPU indexing to CPU
    # to avoid MPS/CPU mask-device errors.
    pred_cpu = pred.detach().cpu()
    mask_cpu = mask.detach().cpu()
    hidden_cpu = hidden_labels.detach().cpu()

    if bool(mask_cpu.any().item()):
        precision = float(
            (pred_cpu[mask_cpu] == hidden_cpu[mask_cpu])
            .float()
            .mean()
            .item()
        )
        mean_confidence = float(conf[mask].mean().item())
    else:
        precision = float("nan")
        mean_confidence = float("nan")

    return accepted_x, accepted_y, coverage, precision, mean_confidence


@torch.no_grad()
def fit_reference_centroids(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> torch.Tensor:
    """Fit one normalised trusted centroid per class for diagnostics only."""
    model.eval()
    sums = None
    counts = torch.zeros(num_classes, device=device)

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        features = F.normalize(model.forward_features(x), dim=1)

        if sums is None:
            sums = torch.zeros(
                num_classes,
                features.size(1),
                device=device,
            )

        for c in range(num_classes):
            mask = y == c
            if mask.any():
                sums[c] += features[mask].sum(dim=0)
                counts[c] += mask.sum()

    if sums is None or (counts == 0).any():
        raise RuntimeError("Trusted reference set must contain every class.")

    return F.normalize(sums / counts.unsqueeze(1), dim=1)


@torch.no_grad()
def feature_centroid_drift(
    model: torch.nn.Module,
    images: torch.Tensor,
    hidden_labels: torch.Tensor,
    reference_centroids: torch.Tensor,
    device: torch.device,
    num_classes: int,
) -> float:
    """
    Evaluation-only feature drift.

    Hidden labels are used only for diagnostic grouping and never for training.
    """
    model.eval()
    features = F.normalize(model.forward_features(images.to(device)), dim=1)
    hidden = hidden_labels.to(device)

    drifts: List[float] = []
    for c in range(num_classes):
        mask = hidden == c
        if mask.any():
            current = F.normalize(
                features[mask].mean(dim=0, keepdim=True),
                dim=1,
            ).squeeze(0)
            drifts.append(
                float(torch.norm(current - reference_centroids[c], p=2).cpu())
            )
    return float(np.mean(drifts)) if drifts else float("nan")


def parse_args():
    p = argparse.ArgumentParser(
        description="Dissertation baseline experiments for continual semi-supervised vision."
    )
    p.add_argument("--dataset", choices=["mnist", "cifar10", "svhn"], default="mnist")
    p.add_argument("--method", choices=METHODS, default="naive")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--threshold", type=float, default=0.90)
    p.add_argument("--ewc-lambda", type=float, default=50.0)
    p.add_argument("--replay-capacity", type=int, default=1000)
    p.add_argument("--replay-samples", type=int, default=128)
    p.add_argument("--fisher-samples", type=int, default=None)

    p.add_argument("--initial-epochs", type=int, default=None)
    p.add_argument("--online-epochs", type=int, default=None)
    p.add_argument("--offline-epochs", type=int, default=None)
    p.add_argument("--initial-lr", type=float, default=1e-3)
    p.add_argument("--online-lr", type=float, default=5e-4)
    p.add_argument("--offline-lr", type=float, default=1e-3)

    p.add_argument("--stream-batches", type=int, default=None)
    p.add_argument("--stream-batch-size", type=int, default=None)
    p.add_argument("--output-dir", default="results/baselines")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def save_tables(
    output: Path,
    stem: str,
    rows: List[Dict],
    class_hist: List[np.ndarray],
    task_hist: List[np.ndarray],
    batch_ids: List[int],
    num_classes: int,
):
    df = pd.DataFrame(rows)
    df.to_csv(output / f"metrics_{stem}.csv", index=False)

    class_df = pd.DataFrame(
        class_hist,
        columns=[f"class_{c}_accuracy" for c in range(num_classes)],
    )
    class_df.insert(0, "batch", batch_ids)
    class_df.to_csv(output / f"class_accuracy_{stem}.csv", index=False)

    task_df = pd.DataFrame(
        task_hist,
        columns=[f"task_{i}_accuracy" for i in range(5)],
    )
    task_df.insert(0, "batch", batch_ids)
    task_df.to_csv(output / f"task_accuracy_{stem}.csv", index=False)

    return df


def run_offline_joint(
    args,
    cfg: ExperimentConfig,
    stream: ControlledVisionStream,
    spec,
    stream_batches,
    test_loader,
    task_loaders,
    output: Path,
    distribution: pd.DataFrame,
    initial_accuracy_value: float,
    initial_class: np.ndarray,
    initial_task: np.ndarray,
):
    """
    Oracle joint-training reference.

    It uses the true labels from ALL stream batches at once.  Therefore:
      * it is NOT a deployable semi-supervised continual learner;
      * it is an upper/reference condition;
      * stream-trajectory metrics such as mean stream accuracy and forgetting
        are not directly comparable and are saved as NaN.
    """
    start = time.perf_counter()

    # Re-seed before constructing the offline model so it starts from the same
    # deterministic parameter initialisation family as other methods.
    set_global_seed(cfg.seed)
    device = choose_device(cfg.device)
    model = SDSLVisionNet(spec.in_channels, spec.num_classes).to(device)

    initial_x, initial_y = collect_loader(
        stream.initial_loader(cfg.train_batch_size, shuffle=False)
    )
    stream_x = torch.cat([b.images.detach().cpu() for b in stream_batches], dim=0)
    stream_y = torch.cat([b.hidden_labels.detach().cpu() for b in stream_batches], dim=0)

    joint_x = torch.cat([initial_x, stream_x], dim=0)
    joint_y = torch.cat([initial_y, stream_y], dim=0)

    joint_loader = make_loader(
        joint_x,
        joint_y,
        batch_size=cfg.train_batch_size,
        shuffle=True,
    )
    offline_epochs = args.offline_epochs or cfg.initial_epochs
    optimizer = torch.optim.Adam(model.parameters(), lr=args.offline_lr)

    train_start = time.perf_counter()
    training_loss = train_epochs(
        model,
        joint_loader,
        optimizer,
        device,
        offline_epochs,
    )
    train_seconds = time.perf_counter() - train_start

    final_acc = accuracy(model, test_loader, device)
    final_class = class_accuracy(model, test_loader, device, spec.num_classes)
    final_task = task_accuracy_vector(model, task_loaders, device)

    # Only two legitimate evaluation points exist for a true joint/offline
    # reference: the initial-only model (from the common initial phase) and the
    # fully joint oracle.  We do not fabricate a stream trajectory.
    rows = [
        {
            "dataset": spec.name,
            "batch": -1,
            "method": "offline",
            "seed": cfg.seed,
            "test_accuracy": initial_accuracy_value,
            "pseudo_precision": np.nan,
            "pseudo_coverage": np.nan,
            "accepted_count": 0,
            "rejected_count": 0,
            "distribution_tv": 0.0,
            "feature_centroid_drift": np.nan,
            "parameter_change": np.nan,
            "training_loss": np.nan,
            "buffer_size": 0,
            "batch_seconds": 0.0,
            "evaluation_scope": "common_initial_model",
        },
        {
            "dataset": spec.name,
            "batch": cfg.stream_batches,
            "method": "offline",
            "seed": cfg.seed,
            "test_accuracy": final_acc,
            "pseudo_precision": np.nan,
            "pseudo_coverage": np.nan,
            "accepted_count": int(len(joint_y)),
            "rejected_count": 0,
            "distribution_tv": np.nan,
            "feature_centroid_drift": np.nan,
            "parameter_change": np.nan,
            "training_loss": training_loss,
            "buffer_size": 0,
            "batch_seconds": train_seconds,
            "evaluation_scope": "oracle_joint_all_stream_labels",
        },
    ]

    stem = (
        f"{spec.name}_offline_seed{cfg.seed}"
        f"_epochs{offline_epochs}_lr{args.offline_lr:g}"
    )
    df = save_tables(
        output,
        stem,
        rows,
        [initial_class, final_class],
        [initial_task, final_task],
        [-1, cfg.stream_batches],
        spec.num_classes,
    )

    summary = {
        "dataset": spec.name,
        "method": "offline",
        "seed": cfg.seed,
        "reference_type": "oracle_joint_supervised",
        "trajectory_comparable": False,
        "initial_accuracy": float(initial_accuracy_value),
        "final_accuracy": float(final_acc),
        "mean_stream_accuracy": float("nan"),
        "classwise_forgetting": float("nan"),
        "average_incremental_accuracy": float("nan"),
        "backward_transfer_proxy": float("nan"),
        "mean_pseudo_precision": float("nan"),
        "mean_pseudo_coverage": float("nan"),
        "mean_distribution_tv": float(distribution.total_variation_from_previous.mean()),
        "mean_feature_centroid_drift": float("nan"),
        "mean_parameter_change": float("nan"),
        "mean_batch_seconds": float("nan"),
        "total_seconds": float(time.perf_counter() - start),
        "offline_training_seconds": float(train_seconds),
        "offline_epochs": int(offline_epochs),
        "offline_train_examples": int(len(joint_y)),
        "threshold": float("nan"),
        "ewc_lambda": float("nan"),
    }

    with open(output / f"summary_{stem}.json", "w") as f:
        json.dump(summary, f, indent=2, allow_nan=True)

    print(json.dumps(summary, indent=2, allow_nan=True))


def main():
    args = parse_args()

    overrides = {
        "dataset": args.dataset,
        "seed": args.seed,
        "confidence_threshold": args.threshold,
        "ewc_lambda": args.ewc_lambda,
        "replay_capacity": args.replay_capacity,
        "replay_samples": args.replay_samples,
    }
    for arg_name, cfg_name in [
        ("initial_epochs", "initial_epochs"),
        ("online_epochs", "online_epochs"),
        ("stream_batches", "stream_batches"),
        ("stream_batch_size", "stream_batch_size"),
        ("fisher_samples", "fisher_samples"),
    ]:
        value = getattr(args, arg_name)
        if value is not None:
            overrides[cfg_name] = value

    if args.smoke:
        overrides.update(
            initial_epochs=1,
            online_epochs=1,
            stream_batches=2,
            stream_batch_size=128,
            fisher_samples=100,
        )
        if args.offline_epochs is None:
            args.offline_epochs = 1

    cfg = ExperimentConfig(**overrides)
    set_global_seed(cfg.seed)
    device = choose_device(cfg.device)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    stream = ControlledVisionStream(
        cfg.dataset,
        cfg.data_root,
        cfg.seed,
        cfg.initial_per_class,
        cfg.stream_batches,
        cfg.stream_batch_size,
        cfg.dominant_fraction,
    )
    spec = stream.spec
    distribution = stream.distribution_table()
    distribution.to_csv(
        output / f"distribution_{spec.name}_seed{cfg.seed}.csv",
        index=False,
    )

    # Materialise the deterministic stream once.  Every method for a given
    # dataset/seed therefore sees the same batch sequence.
    stream_batches = list(stream.batches())

    initial_train = stream.initial_loader(cfg.train_batch_size)
    initial_eval = stream.initial_eval_loader(cfg.test_batch_size)
    test_loader = stream.test_loader(cfg.test_batch_size)
    task_loaders = stream.class_group_test_loaders(cfg.test_batch_size)

    # Common backbone with SDSL.
    model = SDSLVisionNet(spec.in_channels, spec.num_classes).to(device)
    initial_optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.initial_lr,
    )

    start_time = time.perf_counter()
    train_epochs(
        model,
        initial_train,
        initial_optimizer,
        device,
        cfg.initial_epochs,
    )

    initial_acc = accuracy(model, test_loader, device)
    initial_class = class_accuracy(model, test_loader, device, spec.num_classes)
    initial_task = task_accuracy_vector(model, task_loaders, device)

    # Offline joint reference is deliberately handled separately because it
    # has access to labels unavailable to the continual semi-supervised methods.
    if args.method == "offline":
        run_offline_joint(
            args=args,
            cfg=cfg,
            stream=stream,
            spec=spec,
            stream_batches=stream_batches,
            test_loader=test_loader,
            task_loaders=task_loaders,
            output=output,
            distribution=distribution,
            initial_accuracy_value=initial_acc,
            initial_class=initial_class,
            initial_task=initial_task,
        )
        return

    reference_centroids = fit_reference_centroids(
        model,
        initial_eval,
        device,
        spec.num_classes,
    )

    replay = (
        ReservoirReplayBuffer(cfg.replay_capacity, cfg.seed)
        if args.method == "replay"
        else None
    )
    ewc = (
        OnlineEWC(model, device, cfg.ewc_lambda, cfg.online_ewc_gamma)
        if args.method == "ewc"
        else None
    )
    if ewc is not None:
        ewc.consolidate(
            initial_eval,
            cfg.fisher_samples,
            use_true_labels=True,
        )

    online_optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.online_lr,
    )

    class_hist = [initial_class]
    task_hist = [initial_task]
    rows: List[Dict] = [
        {
            "dataset": spec.name,
            "batch": -1,
            "method": args.method,
            "seed": cfg.seed,
            "test_accuracy": initial_acc,
            "pseudo_precision": np.nan,
            "pseudo_coverage": np.nan,
            "accepted_count": 0,
            "rejected_count": 0,
            "mean_accepted_confidence": np.nan,
            "distribution_tv": 0.0,
            "feature_centroid_drift": 0.0,
            "parameter_change": 0.0,
            "training_loss": np.nan,
            "buffer_size": 0,
            "buffer_replaced": 0,
            "buffer_total_seen": 0,
            "batch_seconds": 0.0,
        }
    ]
    previous = snapshot(model)

    for batch in stream_batches:
        batch_start = time.perf_counter()

        accepted_x, accepted_y, coverage, precision, mean_conf = (
            confidence_pseudo_labels(
                model,
                batch.images,
                batch.hidden_labels,
                device,
                cfg.confidence_threshold,
            )
        )
        accepted_count = int(accepted_y.numel())
        rejected_count = int(len(batch.images) - accepted_count)

        train_x, train_y = accepted_x, accepted_y
        replaced = 0

        if replay is not None and accepted_count > 0:
            # Add the current pseudo-labelled batch, then sample uniformly
            # from bounded reservoir memory.
            stats = replay.add_batch(accepted_x, accepted_y)
            replaced = stats.replaced
            old = replay.sample(cfg.replay_samples, device)
            if old is not None:
                rx, ry = old
                train_x = torch.cat([accepted_x, rx], dim=0)
                train_y = torch.cat([accepted_y, ry], dim=0)

        loader = make_loader(
            train_x,
            train_y,
            cfg.train_batch_size,
            shuffle=True,
        )

        loss = train_epochs(
            model,
            loader,
            online_optimizer,
            device,
            cfg.online_epochs,
            ewc=ewc,
        )

        drift = feature_centroid_drift(
            model,
            batch.images,
            batch.hidden_labels,
            reference_centroids,
            device,
            spec.num_classes,
        )

        change = parameter_change(previous, model)
        previous = snapshot(model)

        current_acc = accuracy(model, test_loader, device)
        class_hist.append(
            class_accuracy(model, test_loader, device, spec.num_classes)
        )
        task_hist.append(task_accuracy_vector(model, task_loaders, device))

        dist_row = distribution.iloc[batch.batch_id]

        rows.append(
            {
                "dataset": spec.name,
                "batch": batch.batch_id,
                "method": args.method,
                "seed": cfg.seed,
                "test_accuracy": current_acc,
                "pseudo_precision": precision,
                "pseudo_coverage": coverage,
                "accepted_count": accepted_count,
                "rejected_count": rejected_count,
                "mean_accepted_confidence": mean_conf,
                "distribution_tv": float(
                    dist_row.total_variation_from_previous
                ),
                "feature_centroid_drift": drift,
                "parameter_change": change,
                "training_loss": loss,
                "buffer_size": len(replay) if replay is not None else 0,
                "buffer_replaced": replaced,
                "buffer_total_seen": replay.seen if replay is not None else 0,
                "batch_seconds": time.perf_counter() - batch_start,
            }
        )

    total_seconds = time.perf_counter() - start_time

    stem = (
        f"{spec.name}_{args.method}_seed{cfg.seed}"
        f"_lam{cfg.ewc_lambda:g}_thr{cfg.confidence_threshold:g}"
    )
    batch_ids = [-1] + list(range(cfg.stream_batches))
    df = save_tables(
        output,
        stem,
        rows,
        class_hist,
        task_hist,
        batch_ids,
        spec.num_classes,
    )

    summary = {
        "dataset": spec.name,
        "method": args.method,
        "seed": cfg.seed,
        "reference_type": "continual_semi_supervised",
        "trajectory_comparable": True,
        "initial_accuracy": float(df.iloc[0].test_accuracy),
        "final_accuracy": float(df.iloc[-1].test_accuracy),
        "mean_stream_accuracy": float(
            df[df.batch >= 0].test_accuracy.mean()
        ),
        "classwise_forgetting": forgetting_from_history(class_hist),
        "average_incremental_accuracy": average_incremental_accuracy(task_hist),
        "backward_transfer_proxy": backward_transfer_proxy(task_hist),
        "mean_pseudo_precision": (
            float(df.pseudo_precision.mean(skipna=True))
            if df.pseudo_precision.notna().any()
            else float("nan")
        ),
        "mean_pseudo_coverage": (
            float(df.pseudo_coverage.mean(skipna=True))
            if df.pseudo_coverage.notna().any()
            else float("nan")
        ),
        "mean_distribution_tv": float(df.distribution_tv.mean()),
        "mean_feature_centroid_drift": float(
            df.feature_centroid_drift.mean(skipna=True)
        ),
        "mean_parameter_change": float(df.parameter_change.mean()),
        "mean_batch_seconds": float(
            df[df.batch >= 0].batch_seconds.mean()
        ),
        "total_seconds": float(total_seconds),
        "final_buffer_size": int(df.iloc[-1].buffer_size),
        "ewc_lambda": float(cfg.ewc_lambda),
        "threshold": float(cfg.confidence_threshold),
        "initial_lr": float(args.initial_lr),
        "online_lr": float(args.online_lr),
    }

    if ewc is not None:
        summary.update(ewc.fisher_summary())

    with open(output / f"summary_{stem}.json", "w") as f:
        json.dump(summary, f, indent=2, allow_nan=True)

    print(json.dumps(summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
