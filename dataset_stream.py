from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


@dataclass
class DatasetSpec:
    name: str
    in_channels: int
    num_classes: int


@dataclass
class StreamBatch:
    images: torch.Tensor
    hidden_labels: torch.Tensor  # evaluation only; learner never trains on these labels
    batch_id: int
    dominant_classes: Tuple[int, int]


PAIR_SCHEDULE = [
    (0, 1), (0, 1), (2, 3), (2, 3), (4, 5),
    (4, 5), (6, 7), (6, 7), (8, 9), (8, 9),
    (0, 2), (1, 3), (4, 6), (5, 7), (8, 0),
    (9, 1), (2, 4), (3, 5), (6, 8), (7, 9),
]


def _dataset_bundle(name: str, root: str):
    name = name.lower()
    if name == "mnist":
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train = datasets.MNIST(root, train=True, download=True, transform=tfm)
        test = datasets.MNIST(root, train=False, download=True, transform=tfm)
        return train, test, DatasetSpec("mnist", 1, 10)
    if name == "cifar10":
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        train = datasets.CIFAR10(root, train=True, download=True, transform=tfm)
        test = datasets.CIFAR10(root, train=False, download=True, transform=tfm)
        return train, test, DatasetSpec("cifar10", 3, 10)
    if name == "svhn":
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ])
        train = datasets.SVHN(root, split="train", download=True, transform=tfm)
        test = datasets.SVHN(root, split="test", download=True, transform=tfm)
        return train, test, DatasetSpec("svhn", 3, 10)
    raise ValueError("dataset must be one of: mnist, cifar10, svhn")


def _targets(dataset) -> np.ndarray:
    if hasattr(dataset, "targets"):
        values = dataset.targets
    elif hasattr(dataset, "labels"):
        values = dataset.labels
    else:
        raise AttributeError("Dataset has no targets/labels attribute")
    if torch.is_tensor(values):
        values = values.cpu().numpy()
    return np.asarray(values, dtype=np.int64)


class ControlledVisionStream:
    """Controlled class-prior stream shared by MNIST, CIFAR-10 and SVHN.

    The dataset remains a static benchmark, but we impose a deterministic stream.
    Each batch has a known dominant class pair, making distribution change measurable.
    """

    def __init__(self, dataset: str, data_root: str, seed: int, initial_per_class: int,
                 stream_batches: int, stream_batch_size: int, dominant_fraction: float):
        if stream_batches > len(PAIR_SCHEDULE):
            raise ValueError("stream_batches must be <= %d" % len(PAIR_SCHEDULE))
        if not 0.5 <= dominant_fraction < 1.0:
            raise ValueError("dominant_fraction should be in [0.5, 1.0)")

        self.dataset, self.test_dataset, self.spec = _dataset_bundle(dataset, data_root)
        self.rng = np.random.default_rng(seed)
        self.stream_batches = stream_batches
        self.stream_batch_size = stream_batch_size
        self.dominant_fraction = dominant_fraction

        targets = _targets(self.dataset)
        by_class: Dict[int, List[int]] = {
            c: np.where(targets == c)[0].tolist() for c in range(self.spec.num_classes)
        }
        for c in by_class:
            self.rng.shuffle(by_class[c])

        self.labelled_indices: List[int] = []
        self.pools: Dict[int, List[int]] = {}
        for c in range(self.spec.num_classes):
            if len(by_class[c]) <= initial_per_class:
                raise RuntimeError("Not enough samples for class %d" % c)
            self.labelled_indices.extend(by_class[c][:initial_per_class])
            self.pools[c] = by_class[c][initial_per_class:]
        self._batches = self._build_batches()

    def _take(self, cls: int, count: int) -> List[int]:
        available = self.pools[cls]
        if len(available) < count:
            raise RuntimeError(
                "%s: not enough remaining samples for class %d; reduce stream size" %
                (self.spec.name, cls)
            )
        chosen = available[:count]
        del available[:count]
        return chosen

    def _build_batches(self):
        batches = []
        dominant_total = int(round(self.stream_batch_size * self.dominant_fraction))
        background_total = self.stream_batch_size - dominant_total
        for pair in PAIR_SCHEDULE[:self.stream_batches]:
            counts = {c: 0 for c in range(self.spec.num_classes)}
            counts[pair[0]] += dominant_total // 2
            counts[pair[1]] += dominant_total - counts[pair[0]]
            others = [c for c in range(self.spec.num_classes) if c not in pair]
            for i in range(background_total):
                counts[others[i % len(others)]] += 1
            indices: List[int] = []
            for c, n in counts.items():
                indices.extend(self._take(c, n))
            self.rng.shuffle(indices)
            batches.append((indices, pair))
        return batches

    def initial_loader(self, batch_size: int, shuffle: bool = True) -> DataLoader:
        return DataLoader(Subset(self.dataset, self.labelled_indices), batch_size=batch_size, shuffle=shuffle)

    def initial_eval_loader(self, batch_size: int) -> DataLoader:
        return self.initial_loader(batch_size, shuffle=False)

    def test_loader(self, batch_size: int) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

    def batches(self):
        for batch_id, (indices, pair) in enumerate(self._batches):
            images, labels = zip(*(self.dataset[i] for i in indices))
            yield StreamBatch(torch.stack(images), torch.tensor(labels, dtype=torch.long), batch_id, pair)

    def distribution_table(self) -> pd.DataFrame:
        target_array = _targets(self.dataset)
        rows = []
        previous = None
        for batch_id, (indices, pair) in enumerate(self._batches):
            labels = target_array[np.asarray(indices)]
            proportions = np.asarray([(labels == c).mean() for c in range(self.spec.num_classes)])
            tv = 0.0 if previous is None else 0.5 * np.abs(proportions - previous).sum()
            row = {
                "dataset": self.spec.name,
                "batch": batch_id,
                "dominant_classes": "%d,%d" % pair,
                "total_variation_from_previous": float(tv),
            }
            row.update({"class_%d_proportion" % c: float(proportions[c]) for c in range(self.spec.num_classes)})
            rows.append(row)
            previous = proportions
        return pd.DataFrame(rows)

    def class_group_test_loaders(self, batch_size: int):
        target_array = _targets(self.test_dataset)
        loaders = {}
        groups = [(0,1), (2,3), (4,5), (6,7), (8,9)]
        for task_id, pair in enumerate(groups):
            indices = np.where(np.isin(target_array, pair))[0].tolist()
            loaders[task_id] = DataLoader(Subset(self.test_dataset, indices), batch_size=batch_size, shuffle=False)
        return loaders
