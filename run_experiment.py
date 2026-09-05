from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from config import ExperimentConfig
from seed_utils import set_global_seed, choose_device
from models import VisionCNN
from dataset_stream import ControlledVisionStream
from centroid_pseudolabel import CentroidRefinedPseudoLabeler
from ewc import OnlineEWC
from replay_buffer import ReservoirReplayBuffer
from metrics import accuracy, class_accuracy, task_accuracy_vector, snapshot, parameter_change, forgetting_from_history, average_incremental_accuracy, backward_transfer_proxy


METHODS = ["offline", "naive", "replay", "ewc", "proposed"]


def make_loader(x, y, batch_size=64, shuffle=True):
    if x is None or y is None or x.numel() == 0:
        return None
    return DataLoader(TensorDataset(x.detach().cpu(), y.detach().cpu()), batch_size=batch_size, shuffle=shuffle)


def train_epochs(model, loader, optimizer, device, epochs: int, ewc=None):
    if loader is None or epochs <= 0:
        return float("nan")
    losses = []
    for _ in range(epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(x), y)
            if ewc is not None:
                loss = loss + ewc.penalty()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def confidence_pseudo_labels(model, images, hidden_labels, device, threshold):
    model.eval()
    with torch.no_grad():
        x = images.to(device)
        probs = torch.softmax(model(x), dim=1)
        conf, pred = probs.max(1)
        mask = conf >= threshold
        accepted_x, accepted_y = x[mask], pred[mask]
        coverage = float(mask.float().mean().item())
        precision = float("nan")
        if mask.any():
            precision = float((pred[mask].cpu() == hidden_labels[mask]).float().mean().item())
    return accepted_x, accepted_y, coverage, precision


def parse_args():
    p = argparse.ArgumentParser(description="Continual semi-supervised vision experiment")
    p.add_argument("--dataset", choices=["mnist", "cifar10", "svhn"], default="mnist")
    p.add_argument("--method", choices=METHODS, default="proposed")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ewc-lambda", type=float, default=50.0)
    p.add_argument("--threshold", type=float, default=0.90)
    p.add_argument("--replay-capacity", type=int, default=1000)
    p.add_argument("--replay-samples", type=int, default=128)
    p.add_argument("--initial-epochs", type=int, default=None)
    p.add_argument("--online-epochs", type=int, default=None)
    p.add_argument("--stream-batches", type=int, default=None)
    p.add_argument("--stream-batch-size", type=int, default=None)
    p.add_argument("--fisher-samples", type=int, default=None)
    p.add_argument("--online-consolidate-every", type=int, default=5)
    p.add_argument("--output-dir", default="results/single")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    overrides = dict(dataset=args.dataset, seed=args.seed, ewc_lambda=args.ewc_lambda,
                     confidence_threshold=args.threshold, replay_capacity=args.replay_capacity,
                     replay_samples=args.replay_samples, online_consolidate_every=args.online_consolidate_every)
    for arg_name, cfg_name in [
        ("initial_epochs", "initial_epochs"), ("online_epochs", "online_epochs"),
        ("stream_batches", "stream_batches"), ("stream_batch_size", "stream_batch_size"),
        ("fisher_samples", "fisher_samples")
    ]:
        value = getattr(args, arg_name)
        if value is not None:
            overrides[cfg_name] = value
    if args.smoke:
        overrides.update(initial_epochs=1, online_epochs=1, stream_batches=2,
                         stream_batch_size=128, fisher_samples=100)
    cfg = ExperimentConfig(**overrides)
    set_global_seed(cfg.seed)
    device = choose_device(cfg.device)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    stream = ControlledVisionStream(cfg.dataset, cfg.data_root, cfg.seed, cfg.initial_per_class,
                                    cfg.stream_batches, cfg.stream_batch_size, cfg.dominant_fraction)
    spec = stream.spec
    distribution = stream.distribution_table()
    distribution.to_csv(output / ("distribution_%s_seed%d.csv" % (spec.name, cfg.seed)), index=False)

    model = VisionCNN(spec.in_channels, spec.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    initial_train = stream.initial_loader(cfg.train_batch_size)
    initial_eval = stream.initial_eval_loader(cfg.test_batch_size)
    test_loader = stream.test_loader(cfg.test_batch_size)
    task_loaders = stream.class_group_test_loaders(cfg.test_batch_size)

    start_time = time.perf_counter()
    train_epochs(model, initial_train, optimizer, device, cfg.initial_epochs)

    labeler = CentroidRefinedPseudoLabeler(model, device, spec.num_classes,
                                           cfg.confidence_threshold, cfg.centroid_weight,
                                           cfg.centroid_temperature)
    labeler.fit_reference_centroids(initial_eval)
    reference_centroids = labeler.centroids.detach().clone()

    uses_replay = args.method in ["replay", "proposed"]
    uses_ewc = args.method in ["ewc", "proposed"]
    uses_centroid = args.method == "proposed"
    replay = ReservoirReplayBuffer(cfg.replay_capacity, cfg.seed) if uses_replay else None
    ewc = OnlineEWC(model, device, cfg.ewc_lambda, cfg.online_ewc_gamma) if uses_ewc else None
    if ewc is not None:
        ewc.consolidate(initial_eval, cfg.fisher_samples, use_true_labels=True)

    initial_acc = accuracy(model, test_loader, device)
    class_hist = [class_accuracy(model, test_loader, device, spec.num_classes)]
    task_hist = [task_accuracy_vector(model, task_loaders, device)]
    rows = [{
        "dataset": spec.name, "batch": -1, "method": args.method, "seed": cfg.seed,
        "test_accuracy": initial_acc, "pseudo_precision": np.nan, "pseudo_coverage": np.nan,
        "agreement": np.nan, "accepted_count": 0, "rejected_count": 0,
        "mean_accepted_confidence": np.nan, "distribution_tv": 0.0,
        "feature_centroid_drift": 0.0, "parameter_change": 0.0,
        "training_loss": np.nan, "buffer_size": 0, "buffer_replaced": 0,
        "buffer_total_seen": 0, "batch_seconds": 0.0,
    }]
    previous = snapshot(model)

    for batch in stream.batches():
        batch_start = time.perf_counter()
        accepted_x = accepted_y = None
        precision = coverage = agreement = float("nan")
        accepted_count = rejected_count = 0
        mean_accepted_confidence = float("nan")

        if args.method != "offline":
            if uses_centroid:
                pl = labeler.generate(batch.images, batch.hidden_labels)
                accepted_x, accepted_y = pl.accepted_images, pl.pseudo_labels
                coverage, precision, agreement = pl.coverage, pl.precision, pl.classifier_centroid_agreement
                accepted_count, rejected_count = pl.accepted_count, pl.rejected_count
                mean_accepted_confidence = pl.mean_accepted_confidence
            else:
                accepted_x, accepted_y, coverage, precision = confidence_pseudo_labels(
                    model, batch.images, batch.hidden_labels, device, cfg.confidence_threshold
                )
                accepted_count = int(accepted_y.numel()) if accepted_y is not None else 0
                rejected_count = int(len(batch.images) - accepted_count)

        replaced = 0
        train_x, train_y = accepted_x, accepted_y
        if replay is not None and accepted_x is not None and accepted_x.numel() > 0:
            stats = replay.add_batch(accepted_x, accepted_y)
            replaced = stats.replaced
            old = replay.sample(cfg.replay_samples, device)
            if old is not None:
                rx, ry = old
                train_x = torch.cat([accepted_x, rx], 0)
                train_y = torch.cat([accepted_y, ry], 0)

        loader = make_loader(train_x, train_y, cfg.train_batch_size)
        loss = float("nan")
        if args.method != "offline" and loader is not None:
            loss = train_epochs(model, loader, optimizer, device, cfg.online_epochs, ewc)

        with torch.no_grad():
            features = torch.nn.functional.normalize(model.forward_features(batch.images.to(device)), dim=1)
            drifts = []
            hidden = batch.hidden_labels.to(device)
            for c in range(spec.num_classes):
                m = hidden == c
                if m.any():
                    current = torch.nn.functional.normalize(features[m].mean(0, keepdim=True), dim=1).squeeze(0)
                    drifts.append(float(torch.norm(current - reference_centroids[c], p=2).cpu()))
            centroid_drift = float(np.mean(drifts)) if drifts else float("nan")

        # Proposed method refreshes Fisher online from trusted replay memory.
        if args.method == "proposed" and ewc is not None and replay is not None and \
                cfg.online_consolidate_every > 0 and (batch.batch_id + 1) % cfg.online_consolidate_every == 0:
            sample = replay.sample(min(cfg.fisher_samples, len(replay)), device)
            if sample is not None:
                fx, fy = sample
                fisher_loader = make_loader(fx, fy, cfg.train_batch_size, shuffle=False)
                ewc.consolidate(fisher_loader, min(cfg.fisher_samples, len(replay)), use_true_labels=True)

        change = parameter_change(previous, model)
        previous = snapshot(model)
        current_acc = accuracy(model, test_loader, device)
        class_hist.append(class_accuracy(model, test_loader, device, spec.num_classes))
        task_hist.append(task_accuracy_vector(model, task_loaders, device))
        dist_row = distribution.iloc[batch.batch_id]
        rows.append({
            "dataset": spec.name, "batch": batch.batch_id, "method": args.method, "seed": cfg.seed,
            "test_accuracy": current_acc, "pseudo_precision": precision,
            "pseudo_coverage": coverage, "agreement": agreement,
            "accepted_count": accepted_count, "rejected_count": rejected_count,
            "mean_accepted_confidence": mean_accepted_confidence,
            "distribution_tv": float(dist_row.total_variation_from_previous),
            "feature_centroid_drift": centroid_drift, "parameter_change": change,
            "training_loss": loss, "buffer_size": len(replay) if replay else 0,
            "buffer_replaced": replaced,
            "buffer_total_seen": replay.seen if replay else 0,
            "batch_seconds": time.perf_counter() - batch_start,
        })

    total_seconds = time.perf_counter() - start_time
    df = pd.DataFrame(rows)
    stem = "%s_%s_seed%d_lam%g_thr%g" % (spec.name, args.method, cfg.seed, cfg.ewc_lambda, cfg.confidence_threshold)
    df.to_csv(output / ("metrics_%s.csv" % stem), index=False)

    class_df = pd.DataFrame(class_hist, columns=["class_%d_accuracy" % c for c in range(spec.num_classes)])
    class_df.insert(0, "batch", [-1] + list(range(cfg.stream_batches)))
    class_df.to_csv(output / ("class_accuracy_%s.csv" % stem), index=False)

    task_df = pd.DataFrame(task_hist, columns=["task_%d_accuracy" % i for i in range(5)])
    task_df.insert(0, "batch", [-1] + list(range(cfg.stream_batches)))
    task_df.to_csv(output / ("task_accuracy_%s.csv" % stem), index=False)

    summary = {
        "dataset": spec.name,
        "method": args.method,
        "seed": cfg.seed,
        "initial_accuracy": float(df.iloc[0].test_accuracy),
        "final_accuracy": float(df.iloc[-1].test_accuracy),
        "mean_stream_accuracy": float(df[df.batch >= 0].test_accuracy.mean()),
        "classwise_forgetting": forgetting_from_history(class_hist),
        "average_incremental_accuracy": average_incremental_accuracy(task_hist),
        "backward_transfer_proxy": backward_transfer_proxy(task_hist),
        "mean_pseudo_precision": float(df.pseudo_precision.mean(skipna=True)) if df.pseudo_precision.notna().any() else float("nan"),
        "mean_pseudo_coverage": float(df.pseudo_coverage.mean(skipna=True)) if df.pseudo_coverage.notna().any() else float("nan"),
        "mean_distribution_tv": float(df.distribution_tv.mean()),
        "mean_feature_centroid_drift": float(df.feature_centroid_drift.mean(skipna=True)),
        "mean_parameter_change": float(df.parameter_change.mean()),
        "mean_batch_seconds": float(df[df.batch >= 0].batch_seconds.mean()),
        "total_seconds": float(total_seconds),
        "final_buffer_size": int(df.iloc[-1].buffer_size),
        "ewc_lambda": cfg.ewc_lambda,
        "threshold": cfg.confidence_threshold,
    }
    if ewc is not None:
        summary.update(ewc.fisher_summary())
    with open(output / ("summary_%s.json" % stem), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
