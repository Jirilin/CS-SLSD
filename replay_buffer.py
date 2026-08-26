from __future__ import annotations
from dataclasses import dataclass
import random
import torch

@dataclass
class ReplayStats:
    added: int
    replaced: int
    current_size: int
    total_seen: int = 0


class ReservoirReplayBuffer:
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

    @property
    def is_full(self) -> bool:
        
        return len(self.images) >= self.capacity

    def class_histogram(self) -> dict[int, int]:
        
        counts: dict[int, int] = {}
        for label in self.labels:
            counts[label] = counts.get(label, 0) + 1
        return counts

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

    def sample(self, n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor] | None:
        
        if not self.images:
            return None
        n = min(n, len(self.images))
        idx = self.rng.sample(range(len(self.images)), n)
        x = torch.stack([self.images[i] for i in idx]).to(device)
        y = torch.tensor([self.labels[i] for i in idx], dtype=torch.long, device=device)
        return x, y