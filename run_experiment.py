from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from config import ExperimentConfig
from seed_utils import set_global_seed, choose_device
from models import SimpleCNN
from mnist_stream import FrozenMNISTStream
from centroid_pseudolabel import CentroidRefinedPseudoLabeler
from ewc import OnlineEWC
from replay_buffer import ReservoirReplayBuffer
from metrics import accuracy, class_accuracy, snapshot, parameter_change, forgetting_from_history


def train_epochs(model, loader, optimizer, device, epochs: int, ewc=None) -> float:
    if epochs <= 0:
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
            loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
    return float(np.mean(losses)) if losses else float("nan")


def make_loader(x, y, batch_size=64, shuffle=True):
    if x.numel() == 0:
        return None
    return DataLoader(TensorDataset(x.detach().cpu(), y.detach().cpu()),
                      batch_size=batch_size, shuffle=shuffle)


def parse_args():
    p = argparse.ArgumentParser(description="Controlled MNIST continual semi-supervised experiment")
    p.add_argument("--method", choices=["naive", "centroid", "centroid_ewc", "centroid_replay", "centroid_replay_ewc"], default="centroid_ewc")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ewc-lambda", type=float, default=50.0)
    p.add_argument("--threshold", type=float, default=0.90)
    p.add_argument("--replay-capacity", type=int, default=1000)
    p.add_argument("--replay-samples", type=int, default=128)
    p.add_argument("--online-consolidate-every", type=int, default=5,
                   help="Re-estimate online Fisher from trusted replay pseudo-labels every N batches; 0 disables")
    p.add_argument("--output-dir", default="results/single")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = ExperimentConfig(seed=args.seed, ewc_lambda=args.ewc_lambda,
                           confidence_threshold=args.threshold)
    set_global_seed(cfg.seed)
    device = choose_device(cfg.device)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    stream = FrozenMNISTStream(cfg.data_root, cfg.seed, cfg.initial_per_class,
                               cfg.stream_batches, cfg.stream_batch_size,
                               cfg.dominant_fraction)
    distribution = stream.distribution_table()
    distribution.to_csv(out / f"distribution_seed{cfg.seed}.csv", index=False)

    model = SimpleCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    initial_train = stream.initial_loader(cfg.train_batch_size)
    initial_eval = stream.initial_eval_loader(cfg.test_batch_size)
    test_loader = stream.test_loader(cfg.test_batch_size)
    train_epochs(model, initial_train, optimizer, device, cfg.initial_epochs)

    labeler = CentroidRefinedPseudoLabeler(model, device, cfg.confidence_threshold,
                                           cfg.centroid_weight, cfg.centroid_temperature)
    labeler.fit_reference_centroids(initial_eval)
    reference_centroids = labeler.centroids.detach().clone()

    uses_ewc = "ewc" in args.method
    uses_replay = "replay" in args.method
    ewc = OnlineEWC(model, device, cfg.ewc_lambda, cfg.online_ewc_gamma) if uses_ewc else None
    if ewc:
        ewc.consolidate(initial_eval, cfg.fisher_samples, use_true_labels=True)
    replay = ReservoirReplayBuffer(args.replay_capacity, cfg.seed) if uses_replay else None

    initial_acc = accuracy(model, test_loader, device)
    class_hist = [class_accuracy(model, test_loader, device)]
    rows = [{"batch": -1, "method": args.method, "seed": cfg.seed,
             "test_accuracy": initial_acc, "pseudo_precision": np.nan,
             "pseudo_coverage": np.nan, "agreement": np.nan,
             "distribution_tv": 0.0, "feature_centroid_drift": 0.0,
             "parameter_change": 0.0, "training_loss": np.nan,
             "buffer_size": 0, "buffer_replaced": 0}]
    previous = snapshot(model)

    for batch in stream.batches():
        if args.method == "naive":
            model.eval()
            with torch.no_grad():
                x = batch.images.to(device)
                probs = torch.softmax(model(x), dim=1)
                conf, pred = probs.max(1)
                mask = conf >= cfg.confidence_threshold
                accepted_x, accepted_y = x[mask], pred[mask]
                coverage = float(mask.float().mean())
                precision = float((pred[mask].cpu() == batch.hidden_labels[mask]).float().mean()) if mask.any() else np.nan
                agreement = np.nan
        else:
            pl = labeler.generate(batch.images, batch.hidden_labels)
            accepted_x, accepted_y = pl.accepted_images, pl.pseudo_labels
            coverage, precision, agreement = pl.coverage, pl.precision, pl.classifier_centroid_agreement

        replaced = 0
        train_x, train_y = accepted_x, accepted_y
        if replay is not None and accepted_x.numel() > 0:
            stats = replay.add_batch(accepted_x, accepted_y)
            replaced = stats.replaced
            old = replay.sample(args.replay_samples, device)
            if old is not None:
                rx, ry = old
                train_x = torch.cat([accepted_x, rx], dim=0)
                train_y = torch.cat([accepted_y, ry], dim=0)

        train_loader = make_loader(train_x, train_y, cfg.train_batch_size)
        loss = train_epochs(model, train_loader, optimizer, device, cfg.online_epochs, ewc) if train_loader else np.nan

        # Feature-centroid drift: compare current true-class batch centroids to fixed trusted centroids.
        with torch.no_grad():
            feats = torch.nn.functional.normalize(model.forward_features(batch.images.to(device)), dim=1)
            drifts = []
            for c in range(10):
                m = batch.hidden_labels.to(device) == c
                if m.any():
                    current = torch.nn.functional.normalize(feats[m].mean(0, keepdim=True), dim=1).squeeze(0)
                    drifts.append(float(torch.norm(current - reference_centroids[c], p=2)))
            centroid_drift = float(np.mean(drifts)) if drifts else np.nan

        # Optional online EWC refresh from high-confidence replay pseudo-labels.
        if ewc and replay and args.online_consolidate_every > 0 and (batch.batch_id + 1) % args.online_consolidate_every == 0:
            sample = replay.sample(min(cfg.fisher_samples, len(replay)), device)
            if sample is not None:
                fx, fy = sample
                fisher_loader = make_loader(fx, fy, cfg.train_batch_size, shuffle=False)
                ewc.consolidate(fisher_loader, min(cfg.fisher_samples, len(replay)), use_true_labels=True)

        change = parameter_change(previous, model); previous = snapshot(model)
        test_acc = accuracy(model, test_loader, device)
        class_hist.append(class_accuracy(model, test_loader, device))
        dist_row = distribution.iloc[batch.batch_id]
        rows.append({"batch": batch.batch_id, "method": args.method, "seed": cfg.seed,
                     "test_accuracy": test_acc, "pseudo_precision": precision,
                     "pseudo_coverage": coverage, "agreement": agreement,
                     "distribution_tv": float(dist_row.total_variation_from_previous),
                     "feature_centroid_drift": centroid_drift,
                     "parameter_change": change, "training_loss": loss,
                     "buffer_size": len(replay) if replay else 0,
                     "buffer_replaced": replaced})

    df = pd.DataFrame(rows)
    stem = f"{args.method}_seed{cfg.seed}_lam{cfg.ewc_lambda:g}_thr{cfg.confidence_threshold:g}"
    df.to_csv(out / f"metrics_{stem}.csv", index=False)
    class_df = pd.DataFrame(class_hist, columns=[f"class_{c}_accuracy" for c in range(10)])
    class_df.insert(0, "batch", [-1] + list(range(cfg.stream_batches)))
    class_df.to_csv(out / f"class_accuracy_{stem}.csv", index=False)
    summary = {
        "method": args.method, "seed": cfg.seed,
        "initial_accuracy": float(df.iloc[0].test_accuracy),
        "final_accuracy": float(df.iloc[-1].test_accuracy),
        "mean_stream_accuracy": float(df[df.batch >= 0].test_accuracy.mean()),
        "classwise_forgetting": forgetting_from_history(class_hist),
        "mean_pseudo_precision": float(df.pseudo_precision.mean(skipna=True)),
        "mean_pseudo_coverage": float(df.pseudo_coverage.mean(skipna=True)),
        "mean_distribution_tv": float(df.distribution_tv.mean()),
        "mean_feature_centroid_drift": float(df.feature_centroid_drift.mean(skipna=True)),
        "mean_parameter_change": float(df.parameter_change.mean()),
        "final_buffer_size": int(df.iloc[-1].buffer_size),
    }
    if ewc: summary.update(ewc.fisher_summary())
    with open(out / f"summary_{stem}.json", "w") as f: json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
