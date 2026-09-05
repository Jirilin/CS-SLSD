from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import random
from typing import Optional

import torch


@dataclass
class ReplayStats:
    added: int
    replaced: int
    current_size: int
    total_seen: int


class ReservoirReplayBuffer:

    def __init__(self, capacity: int, seed: int = 0) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")

        self.capacity = int(capacity)
        self.rng = random.Random(seed)
        self.images: list[torch.Tensor] = []
        self.labels: list[int] = []
        self.seen = 0

    def __len__(self) -> int:
        return len(self.images)

    @property
    def is_full(self) -> bool:
        return len(self.images) >= self.capacity

    def add_batch(self, images: torch.Tensor, labels: torch.Tensor) -> ReplayStats:
        
        if len(images) != len(labels):
            raise ValueError("images and labels must contain the same number of samples")

        added = 0
        replaced = 0

        for image, label in zip(images.detach().cpu(), labels.detach().cpu()):
            self.seen += 1

            if len(self.images) < self.capacity:
                self.images.append(image.clone())
                self.labels.append(int(label))
                added += 1
                continue

            # Algorithm R: replace an existing element with probability capacity / total_seen. Otherwise the new sample is discarded.
            replacement_index = self.rng.randrange(self.seen)
            if replacement_index < self.capacity:
                self.images[replacement_index] = image.clone()
                self.labels[replacement_index] = int(label)
                replaced += 1

        return ReplayStats(
            added=added,
            replaced=replaced,
            current_size=len(self.images),
            total_seen=self.seen,
        )

    def sample(
        self,
        n: int,
        device: torch.device,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        
        if n <= 0:
            return None
        if not self.images:
            return None

        n = min(int(n), len(self.images))
        indices = self.rng.sample(range(len(self.images)), n)
        images = torch.stack([self.images[i] for i in indices]).to(device)
        labels = torch.tensor(
            [self.labels[i] for i in indices],
            dtype=torch.long,
            device=device,
        )
        return images, labels

    def class_histogram(self) -> dict[int, int]:
        return dict(sorted(Counter(self.labels).items()))
