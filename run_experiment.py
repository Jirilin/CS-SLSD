from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from config import ExperimentConfig
from seed_utils import set_global_seed
from models import SimpleCNN
from mnist_stream import FrozenMNISTStream
from centroid_pseudolabel import CentroidRefinedPseudoLabeler
from ewc import OnlineEWC
from metrics import accuracy, parameter_change, snapshot


def choose_device(name: str):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def train_initial(model, loader, optimizer, device, epochs):
    for _ in range(epochs):
        model.train()
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(images), labels)
            loss.backward()
            optimizer.step()


def train_pseudo_batch(model, images, labels, optimizer, device, epochs, ewc=None):
    if len(images) == 0:
        return float("nan")
    loader = DataLoader(TensorDataset(images.detach().cpu(), labels.detach().cpu()),
                        batch_size=64, shuffle=True)
    losses = []
    for _ in range(epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            if ewc is not None:
                loss = loss + ewc.penalty()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--method", choices=["naive", "centroid", "centroid_ewc"],
                        default="centroid_ewc")
    parser.add_argument("--ewc-lambda", type=float, default=50.0)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    cfg = ExperimentConfig(seed=args.seed, ewc_lambda=args.ewc_lambda,
                           confidence_threshold=args.threshold)
    set_global_seed(cfg.seed)
    device = choose_device(cfg.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stream = FrozenMNISTStream(cfg.data_root, cfg.seed, cfg.initial_per_class,
                               cfg.stream_batches, cfg.stream_batch_size,
                               cfg.dominant_fraction)
    dist = stream.distribution_table()
    dist.to_csv(output_dir / f"distribution_seed{cfg.seed}.csv", index=False)

    model = SimpleCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    initial_loader = stream.initial_loader(cfg.train_batch_size)
    initial_eval_loader = stream.initial_eval_loader(cfg.test_batch_size)
    test_loader = stream.test_loader(cfg.test_batch_size)
    train_initial(model, initial_loader, optimizer, device, cfg.initial_epochs)

    labeler = CentroidRefinedPseudoLabeler(
        model, device, cfg.confidence_threshold,
        cfg.centroid_weight, cfg.centroid_temperature)
    labeler.fit_reference_centroids(initial_eval_loader)

    ewc = None
    if args.method == "centroid_ewc":
        ewc = OnlineEWC(model, device, cfg.ewc_lambda, cfg.online_ewc_gamma)
        ewc.consolidate(initial_eval_loader, cfg.fisher_samples, use_true_labels=True)

    initial_accuracy = accuracy(model, test_loader, device)
    rows = [{"batch": -1, "method": args.method, "seed": cfg.seed,
             "test_accuracy": initial_accuracy, "pseudo_precision": np.nan,
             "pseudo_coverage": np.nan, "agreement": np.nan,
             "parameter_change": 0.0, "training_loss": np.nan,
             "ewc_lambda": cfg.ewc_lambda, "threshold": cfg.confidence_threshold}]

    previous = snapshot(model)
    for stream_batch in stream.batches():
        if args.method == "naive":
            model.eval()
            with torch.no_grad():
                images_device = stream_batch.images.to(device)
                probs = torch.softmax(model(images_device), dim=1)
                conf, pred = probs.max(dim=1)
                mask = conf >= cfg.confidence_threshold
                accepted_images = images_device[mask]
                pseudo_labels = pred[mask]
                coverage = mask.float().mean().item()
                precision = (pred[mask].cpu() == stream_batch.hidden_labels[mask]).float().mean().item() if mask.any() else np.nan
                agreement = np.nan
        else:
            out = labeler.generate(stream_batch.images, stream_batch.hidden_labels)
            accepted_images, pseudo_labels = out.accepted_images, out.pseudo_labels
            coverage, precision, agreement = out.coverage, out.precision, out.classifier_centroid_agreement

        loss = train_pseudo_batch(model, accepted_images, pseudo_labels, optimizer,
                                  device, cfg.online_epochs, ewc)
        change = parameter_change(previous, model)
        previous = snapshot(model)
        test_acc = accuracy(model, test_loader, device)

        rows.append({"batch": stream_batch.batch_id, "method": args.method,
                     "seed": cfg.seed, "test_accuracy": test_acc,
                     "pseudo_precision": precision, "pseudo_coverage": coverage,
                     "agreement": agreement, "parameter_change": change,
                     "training_loss": loss, "ewc_lambda": cfg.ewc_lambda,
                     "threshold": cfg.confidence_threshold})

    result = pd.DataFrame(rows)
    stem = f"{args.method}_seed{cfg.seed}_lam{cfg.ewc_lambda:g}_thr{cfg.confidence_threshold:g}"
    result.to_csv(output_dir / f"metrics_{stem}.csv", index=False)
    summary = {
        "method": args.method,
        "seed": cfg.seed,
        "initial_accuracy": float(result.iloc[0].test_accuracy),
        "final_accuracy": float(result.iloc[-1].test_accuracy),
        "mean_stream_accuracy": float(result[result.batch >= 0].test_accuracy.mean()),
        "forgetting_proxy": float(result.iloc[0].test_accuracy - result.iloc[-1].test_accuracy),
        "mean_pseudo_precision": float(result.pseudo_precision.mean(skipna=True)),
        "mean_pseudo_coverage": float(result.pseudo_coverage.mean(skipna=True)),
        "mean_parameter_change": float(result.parameter_change.mean()),
    }
    if ewc is not None:
        summary.update(ewc.fisher_summary())
    with open(output_dir / f"summary_{stem}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
