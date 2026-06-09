
import random
import torch
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity=500, device='cpu'):
        self.capacity = capacity
        self.device = device
        self.buffer = deque(maxlen=capacity)
    
    def add(self, images, labels):
        """
        Add a batch of (images, labels) to the buffer.
        Args:
            images (torch.Tensor): Shape (B, C, H, W)
            labels (torch.Tensor): Shape (B,)
        """
        for i in range(len(images)):
            self.buffer.append((images[i].cpu(), labels[i].cpu()))
    
    def sample(self, batch_size):
        """
        Randomly sample a batch from the buffer.
        Returns:
            images (torch.Tensor), labels (torch.Tensor)
        """
        if len(self.buffer) < batch_size:
            # If buffer not full enough, return all
            batch = list(self.buffer)
        else:
            batch = random.sample(self.buffer, batch_size)
        images, labels = zip(*batch)
        images = torch.stack(images).to(self.device)
        labels = torch.tensor(labels).to(self.device)
        return images, labels
    
    def __len__(self):
        return len(self.buffer)
    
    def clear(self):
        self.buffer.clear()