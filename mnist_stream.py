from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


@dataclass
class StreamBatch:
    images: torch.Tensor
    hidden_labels: torch.Tensor  # used only for evaluation, never for training
    batch_id: int
    dominant_classes: Tuple[int, int]


class FrozenMNISTStream:
        PAIR_SCHEDULE = [
        (0, 1), (0, 1), (2, 3), (2, 3), (4, 5),
        (4, 5), (6, 7), (6, 7), (8, 9), (8, 9),
        (0, 2), (1, 3), (4, 6), (5, 7), (8, 0),
        (9, 1), (2, 4), (3, 5), (6, 8), (7, 9),
    ]

    def __init__(self, data_root: str, seed: int, initial_per_class: int,
                 stream_batches: int, stream_batch_size: int,
                 dominant_fraction: float):
        if stream_batches > len(self.PAIR_SCHEDULE):
            raise ValueError(f"stream_batches must be <= {len(self.PAIR_SCHEDULE)}")
        if not 0.5 <= dominant_fraction < 1.0:
            raise ValueError("dominant_fraction should be in [0.5, 1.0).")

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
        return DataLoader(Subset(self.dataset, self.labelled_indices), batch_size=batch_size,
                          shuffle=True)

    def initial_eval_loader(self, batch_size: int) -> DataLoader:
        return DataLoader(Subset(self.dataset, self.labelled_indices), batch_size=batch_size,
                          shuffle=False)

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
            row = {"batch": batch_id, "dominant_classes": f"{pair[0]},{pair[1]}",
                   "total_variation_from_previous": tv}
            row.update({f"class_{c}_proportion": proportions[c] for c in range(10)})
            rows.append(row)
            previous = proportions
        return pd.DataFrame(rows)

    def class_group_test_loaders(self, batch_size: int):
                targets = np.asarray(self.test_dataset.targets)
        loaders = {}
        for task_id, pair in enumerate([(0,1),(2,3),(4,5),(6,7),(8,9)]):
            indices = np.where(np.isin(targets, pair))[0].tolist()
            loaders[task_id] = DataLoader(
                Subset(self.test_dataset, indices), batch_size=batch_size, shuffle=False
            )
        return loaders
