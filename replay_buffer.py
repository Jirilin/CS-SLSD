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

        self.images: list[torch.Tensor] = []
        self.labels: list[int] = []

        # Total number of samples observed since the buffer was created.
        self.seen = 0

    def __len__(self) -> int:
        """Return the current number of stored samples."""

        return len(self.images)

    def add_batch(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> ReplayStats:
        

        added = 0
        replaced = 0

        cpu_images = images.detach().cpu()
        cpu_labels = labels.detach().cpu()

        for image, label in zip(cpu_images, cpu_labels):
            self.seen += 1

            # Buffer still has free space.
            if len(self.images) < self.capacity:
                self.images.append(image.clone())
                self.labels.append(int(label))
                added += 1

            # Buffer is full: use reservoir replacement.
            else:
                replacement_index = self.rng.randrange(self.seen)

                if replacement_index < self.capacity:
                    self.images[replacement_index] = image.clone()
                    self.labels[replacement_index] = int(label)
                    replaced += 1

        return ReplayStats(
            added=added,
            replaced=replaced,
            current_size=len(self.images),
        )

    def sample(
        self,
        n: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        

        if not self.images:
            return None

        n = min(n, len(self.images))

        selected_indices = self.rng.sample(
            range(len(self.images)),
            n,
        )

        sampled_images = torch.stack(
            [self.images[index] for index in selected_indices]
        ).to(device)

        sampled_labels = torch.tensor(
            [self.labels[index] for index in selected_indices],
            dtype=torch.long,
            device=device,
        )

        return sampled_images, sampled_labels