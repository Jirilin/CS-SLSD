import random
import torch
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity=1000, device="cpu"):
        self.capacity = capacity
        self.device = device
        self.buffer = deque(maxlen=capacity)
        self.total_added = 0
        self.total_evicted = 0

    def add(self, images, labels):
        for i in range(len(images)):
            if len(self.buffer) == self.capacity:
                self.total_evicted += 1
            self.buffer.append((images[i].detach().cpu(), labels[i].detach().cpu()))
            self.total_added += 1

    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return None, None
        batch = list(self.buffer) if len(self.buffer) < batch_size else random.sample(self.buffer, batch_size)
        images, labels = zip(*batch)
        return torch.stack(images).to(self.device), torch.tensor(labels, device=self.device)

    def stats(self):
        return {"size": len(self.buffer), "capacity": self.capacity, "added": self.total_added, "evicted_oldest": self.total_evicted}

    def __len__(self):
        return len(self.buffer)
