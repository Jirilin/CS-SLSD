from __future__ import annotations
import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import time
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms


def set_global_seed(seed: int = 42) -> None:
    """Sets global random seeds for exact experiment reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def choose_device(requested_device: str = "auto") -> torch.device:
    """Selects the best available compute device (CUDA, Apple Silicon MPS, or CPU)."""
    if requested_device is None or requested_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(requested_device)


class SafeAdaptiveAvgPool2d(nn.Module):
    """MPS-safe adaptive average pooling layer to prevent hardware backend issues."""
    def __init__(self, output_size):
        super().__init__()
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = tuple(output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.output_size == (1, 1):
            return x.mean(dim=(-2, -1), keepdim=True)
        H_in, W_in = x.shape[-2:]
        H_out, W_out = self.output_size
        if x.device.type == "mps" and (H_in % H_out != 0 or W_in % W_out != 0):
            return F.adaptive_avg_pool2d(x.cpu(), self.output_size).to(x.device)
        return F.adaptive_avg_pool2d(x, self.output_size)


class VisionCNN(nn.Module):
    """Feature-extracting CNN architecture for image classification."""
    def __init__(self, in_channels: int = 1, num_classes: int = 10, feature_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),
            SafeAdaptiveAvgPool2d((1, 1)),
        )
        self.projector = nn.Identity()
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        if feat.dim() > 2:
            feat = torch.flatten(feat, 1)
        return self.projector(feat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))


SimpleCNN = VisionCNN


@dataclass
class StreamBatch:
    images: torch.Tensor
    hidden_labels: torch.Tensor  # Evaluation only
    batch_id: int
    dominant_classes: Tuple[int, int]


class FrozenMNISTStream:
    PAIR_SCHEDULE = [
        (0, 1), (0, 1), (2, 3), (2, 3), (4, 5),
        (4, 5), (6, 7), (6, 7), (8, 9), (8, 9),
        (0, 2), (1, 3), (4, 6), (5, 7), (8, 0),
        (9, 1), (2, 4), (3, 5), (6, 8), (7, 9),
    ]

    def __init__(
        self,
        data_root: str,
        seed: int,
        initial_per_class: int,
        stream_batches: int,
        stream_batch_size: int,
        dominant_fraction: float,
    ):
        if stream_batches > len(self.PAIR_SCHEDULE):
            raise ValueError(f"stream_batches must be <= {len(self.PAIR_SCHEDULE)}")
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        self.dataset = datasets.MNIST(data_root, train=True, download=True, transform=transform)
        self.test_dataset = datasets.MNIST(data_root, train=False, download=True, transform=transform)
        self.rng = np.random.default_rng(seed)
        self.stream_batches = stream_batches
        self.stream_batch_size = stream_batch_size
        self.dominant_fraction = dominant_fraction

        targets = np.asarray(self.dataset.targets)
        by_class = {c: np.where(targets == c)[0].tolist() for c in range(10)}
        for c in by_class:
            self.rng.shuffle(by_class[c])

        labelled = []
        pools = {}
        for c in range(10):
            labelled.extend(by_class[c][:initial_per_class])
            pools[c] = by_class[c][initial_per_class:]
        self.labelled_indices = labelled
        self.pools = pools
        self._batches = self._build_batches()

    def _take(self, cls: int, count: int) -> List[int]:
        available = self.pools[cls]
        if len(available) < count:
            raise RuntimeError(f"Not enough remaining MNIST samples for class {cls}.")
        chosen = available[:count]
        del available[:count]
        return chosen

    def _build_batches(self) -> List[Tuple[List[int], Tuple[int, int]]]:
        batches = []
        dominant_total = int(round(self.stream_batch_size * self.dominant_fraction))
        background_total = self.stream_batch_size - dominant_total
        for pair in self.PAIR_SCHEDULE[:self.stream_batches]:
            counts = {c: 0 for c in range(10)}
            counts[pair[0]] += dominant_total // 2
            counts[pair[1]] += dominant_total - counts[pair[0]]
            others = [c for c in range(10) if c not in pair]
            for i in range(background_total):
                counts[others[i % len(others)]] += 1
            indices = []
            for c, n in counts.items():
                indices.extend(self._take(c, n))
            self.rng.shuffle(indices)
            batches.append((indices, pair))
        return batches

    def initial_loader(self, batch_size: int) -> DataLoader:
        return DataLoader(Subset(self.dataset, self.labelled_indices), batch_size=batch_size, shuffle=True)

    def initial_eval_loader(self, batch_size: int) -> DataLoader:
        return DataLoader(Subset(self.dataset, self.labelled_indices), batch_size=batch_size, shuffle=False)

    def test_loader(self, batch_size: int) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

    def batches(self):
        for batch_id, (indices, pair) in enumerate(self._batches):
            images, labels = zip(*(self.dataset[i] for i in indices))
            yield StreamBatch(torch.stack(images), torch.tensor(labels), batch_id, pair)

    def distribution_table(self) -> pd.DataFrame:
        rows = []
        previous = None
        for batch_id, (indices, pair) in enumerate(self._batches):
            labels = np.asarray([int(self.dataset.targets[i]) for i in indices])
            proportions = np.asarray([(labels == c).mean() for c in range(10)])
            tv = 0.0 if previous is None else 0.5 * np.abs(proportions - previous).sum()
            row = {"batch": batch_id, "dominant_classes": f"{pair[0]},{pair[1]}", "total_variation_from_previous": tv}
            row.update({f"class_{c}_proportion": proportions[c] for c in range(10)})
            rows.append(row)
            previous = proportions
        return pd.DataFrame(rows)

    def class_group_test_loaders(self, batch_size: int) -> Dict[int, DataLoader]:
        targets = np.asarray(self.test_dataset.targets)
        loaders = {}
        for task_id, pair in enumerate([(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]):
            indices = np.where(np.isin(targets, pair))[0].tolist()
            loaders[task_id] = DataLoader(Subset(self.test_dataset, indices), batch_size=batch_size, shuffle=False)
        return loaders


@dataclass
class ReplayStats:
    added: int
    replaced: int
    current_size: int
    total_seen: int = 0


class ReservoirReplayBuffer:
    """Reservoir sampling memory buffer for rehearsal-based continual learning."""
    def __init__(self, capacity: int, seed: int = 0):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.rng = random.Random(seed)
        self.images: list[torch.Tensor] = []
        self.labels: list[int] = []
        self.seen = 0

    def __len__(self) -> int:
        return len(self.images)

    def add_batch(self, images: torch.Tensor, labels: torch.Tensor) -> ReplayStats:
        if len(images) != len(labels):
            raise ValueError("images and labels must contain the same number of samples")
        added = replaced = 0
        for image, label in zip(images.detach().cpu(), labels.detach().cpu()):
            self.seen += 1
            if len(self.images) < self.capacity:
                self.images.append(image.clone())
                self.labels.append(int(label))
                added += 1
            else:
                j = self.rng.randrange(self.seen)
                if j < self.capacity:
                    self.images[j] = image.clone()
                    self.labels[j] = int(label)
                    replaced += 1
        return ReplayStats(added, replaced, len(self.images), self.seen)

    def sample(self, n: int, device: torch.device) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if not self.images:
            return None
        n = min(n, len(self.images))
        idx = self.rng.sample(range(len(self.images)), n)
        x = torch.stack([self.images[i] for i in idx]).to(device)
        y = torch.tensor([self.labels[i] for i in idx], dtype=torch.long, device=device)
        return x, y


@dataclass
class PseudoLabelOutput:
    accepted_images: torch.Tensor
    pseudo_labels: torch.Tensor
    coverage: float
    precision: float
    classifier_centroid_agreement: float
    accepted_count: int
    rejected_count: int
    mean_accepted_confidence: float


class CentroidRefinedPseudoLabeler:
    """Refines pseudo-labels by checking feature distance to reference centroids."""
    def __init__(self, model: nn.Module, device: torch.device, num_classes: int = 10,
                 confidence_threshold: float = 0.75, centroid_weight: float = 0.3, centroid_temperature: float = 1.0):
        self.model = model
        self.device = device
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.centroid_weight = centroid_weight
        self.centroid_temperature = centroid_temperature
        self.centroids: Optional[torch.Tensor] = None

    def fit_reference_centroids(self, loader: DataLoader):
        self.model.eval()
        sum_feats = None
        counts = torch.zeros(self.num_classes, device=self.device)
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                feats = F.normalize(self.model.forward_features(x), dim=1)
                if sum_feats is None:
                    sum_feats = torch.zeros(self.num_classes, feats.shape[1], device=self.device)
                for c in range(self.num_classes):
                    mask = (y == c)
                    if mask.any():
                        sum_feats[c] += feats[mask].sum(dim=0)
                        counts[c] += mask.sum()
        for c in range(self.num_classes):
            if counts[c] > 0:
                sum_feats[c] /= counts[c]
        self.centroids = F.normalize(sum_feats, dim=1)

    def generate(self, images: torch.Tensor, hidden_labels: Optional[torch.Tensor] = None) -> PseudoLabelOutput:
        self.model.eval()
        with torch.no_grad():
            x = images.to(self.device)
            logits = self.model(x)
            probs = F.softmax(logits, dim=1)
            clf_conf, clf_pred = probs.max(dim=1)

            feats = F.normalize(self.model.forward_features(x), dim=1)
            centroid_sims = torch.matmul(feats, self.centroids.T)
            centroid_probs = F.softmax(centroid_sims / self.centroid_temperature, dim=1)
            cent_conf, cent_pred = centroid_probs.max(dim=1)

            combined_score = (1.0 - self.centroid_weight) * clf_conf + self.centroid_weight * cent_conf
            agreement_mask = (clf_pred == cent_pred)
            mask = (combined_score >= self.confidence_threshold) & agreement_mask

            accepted_images = x[mask]
            accepted_labels = clf_pred[mask]
            
            total = len(images)
            accepted_count = int(mask.sum().item())
            rejected_count = total - accepted_count
            coverage = float(accepted_count / total) if total > 0 else 0.0

            agreement = float(agreement_mask.float().mean().item()) if total > 0 else 0.0
            mean_conf = float(combined_score[mask].mean().item()) if accepted_count > 0 else float("nan")

            precision = float("nan")
            if accepted_count > 0 and hidden_labels is not None:
                hidden_cpu = hidden_labels.cpu()
                pred_cpu = accepted_labels.cpu()
                mask_cpu = mask.cpu()
                precision = float((pred_cpu == hidden_cpu[mask_cpu]).float().mean().item())

            return PseudoLabelOutput(
                accepted_images=accepted_images,
                pseudo_labels=accepted_labels,
                coverage=coverage,
                precision=precision,
                classifier_centroid_agreement=agreement,
                accepted_count=accepted_count,
                rejected_count=rejected_count,
                mean_accepted_confidence=mean_conf
            )


class OnlineEWC:
    """Calculates Fisher Information to penalize important parameter movements."""
    def __init__(self, model: nn.Module, device: torch.device, ewc_lambda: float = 100.0, gamma: float = 0.9):
        self.model = model
        self.device = device
        self.ewc_lambda = ewc_lambda
        self.gamma = gamma
        self.fisher: Dict[str, torch.Tensor] = {}
        self.optpar: Dict[str, torch.Tensor] = {}

    def consolidate(self, loader: DataLoader, num_samples: int = 200, use_true_labels: bool = True):
        self.model.eval()
        new_fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}
        self.optpar = {n: p.detach().clone() for n, p in self.model.named_parameters() if p.requires_grad}

        count = 0
        for x, y in loader:
            if count >= num_samples:
                break
            x = x.to(self.device)
            bs = x.size(0)
            for i in range(bs):
                xi = x[i:i+1]
                self.model.zero_grad()
                output = self.model(xi)
                if use_true_labels and y is not None:
                    label = y[i:i+1].to(self.device)
                else:
                    label = output.argmax(dim=1)
                
                loss = F.cross_entropy(output, label)
                loss.backward()

                for n, p in self.model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        new_fisher[n] += p.grad.data ** 2
                count += 1
                if count >= num_samples:
                    break

        if count > 0:
            for n in new_fisher:
                new_fisher[n] /= count

        if not self.fisher:
            self.fisher = new_fisher
        else:
            for n in self.fisher:
                self.fisher[n] = self.gamma * self.fisher[n] + (1 - self.gamma) * new_fisher[n]

    def penalty(self) -> torch.Tensor:
        loss = torch.tensor(0.0, device=self.device)
        if not self.fisher or not self.optpar:
            return loss
        for n, p in self.model.named_parameters():
            if n in self.fisher and n in self.optpar:
                loss += (self.fisher[n] * (p - self.optpar[n]) ** 2).sum()
        return (self.ewc_lambda / 2.0) * loss

    def fisher_summary(self) -> Dict[str, float]:
        if not self.fisher:
            return {"fisher_mean": 0.0, "fisher_max": 0.0, "fisher_nonzero_fraction": 0.0}
        all_f = torch.cat([f.view(-1) for f in self.fisher.values()])
        return {
            "fisher_mean": float(all_f.mean().item()),
            "fisher_max": float(all_f.max().item()),
            "fisher_nonzero_fraction": float((all_f > 1e-8).float().mean().item()),
        }


def train_epochs(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                 device: torch.device, epochs: int, ewc: Optional[OnlineEWC] = None) -> float:
    model.train()
    total_loss = 0.0
    steps = 0
    for epoch in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, y)
            if ewc is not None:
                loss += ewc.penalty()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            steps += 1
    return total_loss / steps if steps > 0 else float("nan")


def accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += len(y)
    return float(correct / total) if total > 0 else 0.0


def class_accuracy(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int = 10) -> List[float]:
    model.eval()
    correct = [0] * num_classes
    total = [0] * num_classes
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            for c in range(num_classes):
                m = (y == c)
                if m.any():
                    correct[c] += int((pred[m] == c).sum().item())
                    total[c] += int(m.sum().item())
    return [float(correct[c] / total[c]) if total[c] > 0 else 0.0 for c in range(num_classes)]


def safe_class_accuracy(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int = 10) -> List[float]:
    return class_accuracy(model, loader, device, num_classes)


def task_accuracy_vector(model: nn.Module, task_loaders: Dict[int, DataLoader], device: torch.device) -> List[float]:
    accs = []
    for task_id in sorted(task_loaders.keys()):
        accs.append(accuracy(model, task_loaders[task_id], device))
    return accs


def snapshot(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def parameter_change(prev: Optional[Dict[str, torch.Tensor]], model: nn.Module) -> float:
    if prev is None:
        return 0.0
    diff = 0.0
    count = 0
    for n, p in model.named_parameters():
        if n in prev:
            diff += float(torch.norm(p - prev[n], p=2).cpu().item())
            count += 1
    return diff / count if count > 0 else 0.0


def make_loader(x: Optional[torch.Tensor], y: Optional[torch.Tensor], batch_size: int, shuffle: bool = True) -> Optional[DataLoader]:
    if x is None or y is None or len(x) == 0:
        return None
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)


def forgetting_from_history(class_hist: List[List[float]]) -> float:
    if not class_hist or len(class_hist) <= 1:
        return 0.0
    mat = np.array(class_hist)
    max_accs = np.max(mat, axis=0)
    final_accs = mat[-1]
    return float(np.mean(max_accs - final_accs))


def average_incremental_accuracy(task_hist: List[List[float]]) -> float:
    if not task_hist:
        return 0.0
    return float(np.mean(task_hist))


def backward_transfer_proxy(task_hist: List[List[float]]) -> float:
    if not task_hist or len(task_hist) <= 1:
        return 0.0
    init_task = np.array(task_hist[0])
    final_task = np.array(task_hist[-1])
    return float(np.mean(final_task - init_task))


def confidence_pseudo_labels(model: nn.Module, images: torch.Tensor, hidden_labels: torch.Tensor,
                             device: torch.device, threshold: float) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
    model.eval()
    with torch.no_grad():
        x = images.to(device)
        probs = torch.softmax(model(x), dim=1)
        conf, pred = probs.max(1)
        mask = conf >= threshold
        accepted_x, accepted_y = x[mask], pred[mask]
        coverage = float(mask.float().mean().item())
        precision = float("nan")
        if mask.any() and hidden_labels is not None:
            mask_cpu = mask.cpu()
            hidden_cpu = hidden_labels.cpu()
            precision = float((pred[mask].cpu() == hidden_cpu[mask_cpu]).float().mean().item())
    return accepted_x, accepted_y, coverage, precision


@dataclass
class Config:
    dataset: str = "mnist"
    seed: int = 0
    data_root: str = "./data"
    output_dir: str = "./results"
    initial_per_class: int = 100
    stream_batches: int = 20
    stream_batch_size: int = 256
    dominant_fraction: float = 0.7
    initial_epochs: int = 5
    online_epochs: int = 1
    lr: float = 0.01
    weight_decay: float = 1e-4
    confidence_threshold: float = 0.75  # Set to 0.75 for active default pseudo-labeling
    centroid_weight: float = 0.3
    centroid_temperature: float = 1.0
    replay_capacity: float = 500
    replay_samples: int = 64
    ewc_lambda: float = 100.0
    online_ewc_gamma: float = 0.9
    fisher_samples: int = 200
    online_consolidate_every: int = 1
    eval_batch_size: int = 256
    train_batch_size: int = 64


def parse_args():
    p = argparse.ArgumentParser(description="Continual Semi-Supervised Learning Framework")
    p.add_argument("--dataset", default="mnist", choices=["mnist"], help="Dataset to run experiments on (default: mnist)")
    p.add_argument("--method", default="proposed", choices=["naive", "ewc", "replay", "centroid", "proposed", "offline"],
                   help="Ablation method selection (default: proposed)")
    p.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    p.add_argument("--threshold", type=float, default=0.75, help="Confidence threshold for pseudo-labeling (default: 0.75)")
    p.add_argument("--ewc-lambda", type=float, default=100.0, help="EWC regularization weight")
    p.add_argument("--data-root", default="./data", help="Root directory for dataset storage")
    p.add_argument("--output-dir", default="./results", help="Directory for output metric export")
    p.add_argument("--device", default="auto", help="Execution compute device (auto, cuda, mps, cpu)")
    p.add_argument("--smoke", action="store_true", help="Execute rapid 2-batch smoke test")
    return p.parse_args()


def main():
    args = parse_args()
    set_global_seed(args.seed)
    device = choose_device(args.device)
    print(f"Executing experiment: dataset={args.dataset}, method={args.method}, seed={args.seed}, device={device}")

    cfg = Config(
        dataset=args.dataset,
        seed=args.seed,
        data_root=args.data_root,
        output_dir=args.output_dir,
        confidence_threshold=args.threshold,
        ewc_lambda=args.ewc_lambda,
    )

    if args.smoke:
        cfg.stream_batches = 2
        cfg.initial_epochs = 1
        cfg.online_epochs = 1
        cfg.fisher_samples = 20

    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    stream = FrozenMNISTStream(
        data_root=cfg.data_root,
        seed=cfg.seed,
        initial_per_class=cfg.initial_per_class,
        stream_batches=cfg.stream_batches,
        stream_batch_size=cfg.stream_batch_size,
        dominant_fraction=cfg.dominant_fraction,
    )

    distribution = stream.distribution_table()
    distribution.to_csv(output / f"distribution_seed{cfg.seed}.csv", index=False)

    num_classes = 10
    model = VisionCNN(in_channels=1, num_classes=num_classes).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=0.9)

    initial_train = stream.initial_loader(cfg.train_batch_size)
    initial_eval = stream.initial_eval_loader(cfg.eval_batch_size)
    test_loader = stream.test_loader(cfg.eval_batch_size)
    task_loaders = stream.class_group_test_loaders(cfg.eval_batch_size)

    start_time = time.perf_counter()
    train_epochs(model, initial_train, optimizer, device, cfg.initial_epochs)

    labeler = CentroidRefinedPseudoLabeler(
        model, device, num_classes,
        cfg.confidence_threshold, cfg.centroid_weight, cfg.centroid_temperature
    )
    labeler.fit_reference_centroids(initial_eval)
    reference_centroids = labeler.centroids.detach().clone().to(device)

    uses_replay = args.method in ["replay", "proposed"]
    uses_ewc = args.method in ["ewc", "proposed"]
    uses_centroid = args.method in ["centroid", "proposed"]
    
    replay = ReservoirReplayBuffer(int(cfg.replay_capacity), cfg.seed) if uses_replay else None
    ewc = OnlineEWC(model, device, cfg.ewc_lambda, cfg.online_ewc_gamma) if uses_ewc else None
    if ewc is not None:
        ewc.consolidate(initial_eval, cfg.fisher_samples, use_true_labels=True)

    initial_acc = accuracy(model, test_loader, device)
    class_hist = [safe_class_accuracy(model, test_loader, device, num_classes)]
    task_hist = [task_accuracy_vector(model, task_loaders, device)]

    rows = [{
        "dataset": "MNIST", "batch": -1, "method": args.method, "seed": cfg.seed,
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
            features = F.normalize(model.forward_features(batch.images.to(device)), dim=1)
            drifts = []
            hidden = batch.hidden_labels.to(device)
            for c in range(num_classes):
                m = (hidden == c)
                if m.any():
                    current = F.normalize(features[m].mean(0, keepdim=True), dim=1).squeeze(0)
                    ref_c = reference_centroids[c].to(device)
                    drifts.append(float(torch.norm(current - ref_c, p=2).cpu()))
            centroid_drift = float(np.mean(drifts)) if drifts else float("nan")

        if args.method in ["ewc", "proposed"] and ewc is not None and replay is not None and len(replay) > 0 and \
                cfg.online_consolidate_every > 0 and (batch.batch_id + 1) % cfg.online_consolidate_every == 0:
            sample = replay.sample(min(cfg.fisher_samples, len(replay)), device)
            if sample is not None:
                fx, fy = sample
                fisher_loader = make_loader(fx, fy, cfg.train_batch_size, shuffle=False)
                ewc.consolidate(fisher_loader, min(cfg.fisher_samples, len(replay)), use_true_labels=True)

        change = parameter_change(previous, model)
        previous = snapshot(model)
        current_acc = accuracy(model, test_loader, device)
        class_hist.append(safe_class_accuracy(model, test_loader, device, num_classes))
        task_hist.append(task_accuracy_vector(model, task_loaders, device))
        dist_row = distribution.iloc[batch.batch_id]

        rows.append({
            "dataset": cfg.dataset, "batch": batch.batch_id, "method": args.method, "seed": cfg.seed,
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
        print(f"Batch {batch.batch_id:2d} | Acc: {current_acc*100:.2f}% | Coverage: {coverage*100:.1f}% | "
              f"Prec: {precision*100 if not np.isnan(precision) else 0.0:.1f}% | Buffer: {len(replay) if replay else 0}")

    total_seconds = time.perf_counter() - start_time
    df = pd.DataFrame(rows)
    stem = f"{cfg.dataset}_{args.method}_seed{cfg.seed}_lam{cfg.ewc_lambda}_thr{cfg.confidence_threshold}"
    df.to_csv(output / f"metrics_{stem}.csv", index=False)

    class_df = pd.DataFrame(class_hist, columns=[f"class_{c}_accuracy" for c in range(num_classes)])
    class_df.insert(0, "batch", [-1] + list(range(cfg.stream_batches)))
    class_df.to_csv(output / f"class_accuracy_{stem}.csv", index=False)

    task_df = pd.DataFrame(task_hist, columns=[f"task_{i}_accuracy" for i in range(5)])
    task_df.insert(0, "batch", [-1] + list(range(cfg.stream_batches)))
    task_df.to_csv(output / f"task_accuracy_{stem}.csv", index=False)

    summary = {
        "dataset": cfg.dataset,
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
        "mean_feature_centroid_drift": float(df.feature_centroid_drift.mean(skipna=True)) if df.feature_centroid_drift.notna().any() else float("nan"),
        "mean_parameter_change": float(df.parameter_change.mean()),
        "mean_batch_seconds": float(df[df.batch >= 0].batch_seconds.mean()),
        "total_seconds": float(total_seconds),
        "final_buffer_size": int(df.iloc[-1].buffer_size),
        "ewc_lambda": cfg.ewc_lambda,
        "threshold": cfg.confidence_threshold,
    }
    if ewc is not None:
        summary.update(ewc.fisher_summary())

    json_summary = {
        k: (None if isinstance(v, float) and (np.isnan(v) or np.isinf(v)) else v)
        for k, v in summary.items()
    }
    with open(output / f"summary_{stem}.json", "w") as f:
        json.dump(json_summary, f, indent=2)
    
    print("\n--- Summary Results ---")
    print(json.dumps(json_summary, indent=2))


if __name__ == "__main__":
    main()