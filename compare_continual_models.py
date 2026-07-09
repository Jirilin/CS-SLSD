import argparse
import csv
import os
import random
import time
from collections import deque
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Models
# -----------------------------
class SmallCNN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# -----------------------------
# Dataset loading
# -----------------------------
def get_dataset(name: str, data_dir: str = "./data"):
    name = name.upper()
    if name in ["MNIST", "FASHIONMNIST", "KMNIST"]:
        tfm = transforms.Compose([transforms.ToTensor()])
        if name == "MNIST":
            train = datasets.MNIST(data_dir, train=True, download=True, transform=tfm)
            test = datasets.MNIST(data_dir, train=False, download=True, transform=tfm)
        elif name == "FASHIONMNIST":
            train = datasets.FashionMNIST(data_dir, train=True, download=True, transform=tfm)
            test = datasets.FashionMNIST(data_dir, train=False, download=True, transform=tfm)
        else:
            train = datasets.KMNIST(data_dir, train=True, download=True, transform=tfm)
            test = datasets.KMNIST(data_dir, train=False, download=True, transform=tfm)
        return train, test, 1, 10

    if name == "CIFAR10":
        tfm = transforms.Compose([transforms.ToTensor()])
        train = datasets.CIFAR10(data_dir, train=True, download=True, transform=tfm)
        test = datasets.CIFAR10(data_dir, train=False, download=True, transform=tfm)
        return train, test, 3, 10

    if name == "SVHN":
        tfm = transforms.Compose([transforms.ToTensor()])
        train = datasets.SVHN(data_dir, split="train", download=True, transform=tfm)
        test = datasets.SVHN(data_dir, split="test", download=True, transform=tfm)
        return train, test, 3, 10

    raise ValueError(f"Unsupported dataset: {name}")


def get_targets(dataset) -> np.ndarray:
    if hasattr(dataset, "targets"):
        return np.array(dataset.targets)
    if hasattr(dataset, "labels"):
        return np.array(dataset.labels)
    raise ValueError("Dataset has no targets/labels field")


# -----------------------------
# Stream creation
# -----------------------------
def create_stream_indices(dataset, labelled_per_class: int, stream_batches: int, quick: bool):
    targets = get_targets(dataset)
    classes = sorted(np.unique(targets).tolist())

    labelled_indices = []
    remaining_indices = []

    for c in classes:
        idx = np.where(targets == c)[0].tolist()
        random.shuffle(idx)
        labelled_indices.extend(idx[:labelled_per_class])
        remaining_indices.extend(idx[labelled_per_class:])

    random.shuffle(remaining_indices)
    if quick:
        remaining_indices = remaining_indices[: min(len(remaining_indices), 6000)]

    stream = np.array_split(remaining_indices, stream_batches)
    stream = [list(x) for x in stream if len(x) > 0]
    return labelled_indices, stream


# -----------------------------
# Pseudo-label generation
# -----------------------------
@torch.no_grad()
def pseudo_label_batch(model, loader, device, threshold: float):
    model.eval()
    xs, ys = [], []
    total, kept = 0, 0
    confidences = []

    for x, _ in loader:
        x = x.to(device)
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        mask = conf >= threshold
        total += x.size(0)
        kept += int(mask.sum().item())
        confidences.extend(conf.detach().cpu().numpy().tolist())
        if mask.any():
            xs.append(x[mask].cpu())
            ys.append(pred[mask].cpu())

    if not xs:
        return None, None, kept, total, float(np.mean(confidences)) if confidences else 0.0
    return torch.cat(xs), torch.cat(ys), kept, total, float(np.mean(confidences))


# -----------------------------
# FIFO replay buffer
# -----------------------------
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.dropped = 0

    def add_batch(self, x: torch.Tensor, y: torch.Tensor):
        for xi, yi in zip(x, y):
            if len(self.buffer) == self.capacity:
                self.dropped += 1
            self.buffer.append((xi.clone(), yi.clone()))

    def sample(self, n: int):
        if len(self.buffer) == 0:
            return None, None
        n = min(n, len(self.buffer))
        batch = random.sample(list(self.buffer), n)
        xs, ys = zip(*batch)
        return torch.stack(xs), torch.stack(ys).long()

    def __len__(self):
        return len(self.buffer)


# -----------------------------
# EWC continual learning model
# -----------------------------
class EWC:
    def __init__(self, model: nn.Module, dataloader: DataLoader, device, lambda_ewc: float = 100.0):
        self.model = model
        self.device = device
        self.lambda_ewc = lambda_ewc
        self.params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        self.fisher = self._estimate_fisher(dataloader)

    def _estimate_fisher(self, dataloader):
        fisher = {n: torch.zeros_like(p, device=self.device) for n, p in self.model.named_parameters() if p.requires_grad}
        self.model.eval()
        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            self.model.zero_grad()
            loss = F.cross_entropy(self.model(x), y)
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2
        for n in fisher:
            fisher[n] /= max(1, len(dataloader))
        return fisher

    def penalty(self):
        loss = 0.0
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                loss += (self.fisher[n] * (p - self.params[n]) ** 2).sum()
        return self.lambda_ewc * loss


# -----------------------------
# Training and evaluation
# -----------------------------
def train_loader(model, loader, optimizer, device, epochs=1, ewc: EWC = None):
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            if ewc is not None:
                loss = loss + ewc.penalty()
            loss.backward()
            optimizer.step()


@torch.no_grad()
def accuracy(model, loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += y.size(0)
    return correct / max(1, total)


@torch.no_grad()
def pseudo_label_precision(model, loader, device, threshold: float) -> float:
    model.eval()
    correct, kept = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        probs = F.softmax(model(x), dim=1)
        conf, pred = probs.max(dim=1)
        mask = conf >= threshold
        if mask.any():
            correct += int((pred[mask] == y[mask]).sum().item())
            kept += int(mask.sum().item())
    return correct / kept if kept else 0.0


def make_loader(dataset, indices, batch_size, shuffle=True):
    return DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=shuffle, num_workers=0)


def make_tensor_loader(x, y, batch_size, shuffle=True):
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle, num_workers=0)


# -----------------------------
# Experiment runner
# -----------------------------
def run_method(method: str, args) -> List[Dict]:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    train_ds, test_ds, in_ch, num_classes = get_dataset(args.dataset, args.data_dir)
    labelled_idx, stream_indices = create_stream_indices(train_ds, args.labelled_per_class, args.stream_batches, args.quick)

    model = SmallCNN(in_ch, num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    initial_loader = make_loader(train_ds, labelled_idx, args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    start = time.time()
    train_loader(model, initial_loader, opt, device, epochs=args.epochs_initial)
    initial_acc = accuracy(model, test_loader, device)

    buffer = ReplayBuffer(args.buffer_size)
    ewc_obj = None
    if method == "ewc":
        ewc_obj = EWC(model, initial_loader, device, lambda_ewc=args.ewc_lambda)

    rows = [{
        "method": method,
        "dataset": args.dataset,
        "batch": 0,
        "test_accuracy": initial_acc,
        "forgetting_from_initial": 0.0,
        "pseudo_kept": 0,
        "pseudo_total": 0,
        "pseudo_keep_rate": 0.0,
        "pseudo_precision": 0.0,
        "mean_confidence": 0.0,
        "buffer_size": len(buffer),
        "buffer_dropped": buffer.dropped,
        "seconds": round(time.time() - start, 2),
    }]

    if method == "offline":
        return rows

    for batch_no, idx in enumerate(stream_indices, start=1):
        stream_loader = make_loader(train_ds, idx, args.batch_size, shuffle=False)
        px, py, kept, total, mean_conf = pseudo_label_batch(model, stream_loader, device, args.threshold)

        if px is not None:
            train_x, train_y = px, py

            if method == "replay":
                rx, ry = buffer.sample(args.replay_sample_size)
                if rx is not None:
                    train_x = torch.cat([train_x, rx], dim=0)
                    train_y = torch.cat([train_y, ry], dim=0)
                buffer.add_batch(px, py)

            update_loader = make_tensor_loader(train_x, train_y, args.batch_size, shuffle=True)
            train_loader(model, update_loader, opt, device, epochs=args.epochs_stream, ewc=ewc_obj if method == "ewc" else None)

        acc = accuracy(model, test_loader, device)
        pll_precision = pseudo_label_precision(model, stream_loader, device, args.threshold)
        rows.append({
            "method": method,
            "dataset": args.dataset,
            "batch": batch_no,
            "test_accuracy": acc,
            "forgetting_from_initial": max(0.0, initial_acc - acc),
            "pseudo_kept": kept,
            "pseudo_total": total,
            "pseudo_keep_rate": kept / max(1, total),
            "pseudo_precision": pll_precision,
            "mean_confidence": mean_conf,
            "buffer_size": len(buffer),
            "buffer_dropped": buffer.dropped,
            "seconds": round(time.time() - start, 2),
        })
        print(f"[{method.upper()}] batch {batch_no:02d}: acc={acc:.4f}, kept={kept}/{total}, buffer={len(buffer)}, dropped={buffer.dropped}")

    return rows


def save_csv(rows: List[Dict], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="MNIST", choices=["MNIST", "FashionMNIST", "KMNIST", "CIFAR10", "SVHN"])
    parser.add_argument("--method", default="all", choices=["offline", "naive", "replay", "ewc", "all"])
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--out", default="results/continual_comparison.csv")
    parser.add_argument("--labelled-per-class", type=int, default=100)
    parser.add_argument("--stream-batches", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs-initial", type=int, default=2)
    parser.add_argument("--epochs-stream", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--buffer-size", type=int, default=2000)
    parser.add_argument("--replay-sample-size", type=int, default=256)
    parser.add_argument("--ewc-lambda", type=float, default=30.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true", help="Use fewer stream samples for a fast presentation demo")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    methods = ["offline", "naive", "replay", "ewc"] if args.method == "all" else [args.method]

    all_rows = []
    for m in methods:
        print(f"\nRunning method={m}, dataset={args.dataset}")
        all_rows.extend(run_method(m, args))

    save_csv(all_rows, args.out)
    print(f"\nSaved results to: {args.out}")
    print("Presentation message: compare accuracy, forgetting, pseudo-label keep rate, and buffer_dropped.")


if __name__ == "__main__":
    main()
