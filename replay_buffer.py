from __future__ import annotations
from dataclasses import dataclass
import random
import torch

@dataclass
class ReplayStats:
    added: int
    replaced: int
    current_size: int

class ReservoirReplayBuffer:
    
    def __init__(self, capacity: int, seed: int = 0):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.rng = random.Random(seed)
        self.images = []
        self.labels = []
        self.seen = 0

    def __len__(self):
        return len(self.images)

    def add_batch(self, images: torch.Tensor, labels: torch.Tensor) -> ReplayStats:
        added = 0
        replaced = 0
        for image, label in zip(images.detach().cpu(), labels.detach().cpu()):
            self.seen += 1
            if len(self.images) < self.capacity:
                self.images.append(image.clone())
                self.labels.append(int(label))
                added += 1
            else:
                idx = self.rng.randrange(self.seen)
                if idx < self.capacity:
                    self.images[idx] = image.clone()
                    self.labels[idx] = int(label)
                    replaced += 1
        return ReplayStats(added, replaced, len(self.images))

    def sample(self, n: int, device: torch.device):
        if not self.images:
            return None
        n = min(n, len(self.images))
        selected = self.rng.sample(range(len(self.images)), n)
        x = torch.stack([self.images[i] for i in selected]).to(device)
        y = torch.tensor([self.labels[i] for i in selected], dtype=torch.long, device=device)
        return x, y
