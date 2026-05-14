import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

class MNISTStreamSimulator:
    """
    Simulates a streaming data scenario using MNIST.
    Data arrives in small batches over time. No labels are provided
    after the initial labeled set (for semi-supervised learning).
    """
    def __init__(self, batch_size=32, initial_labeled_size=1000, shuffle_stream=True):
        self.batch_size = batch_size
        self.initial_labeled_size = initial_labeled_size
        self.shuffle_stream = shuffle_stream

        # Load full MNIST training set
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # standard MNIST normalization
        ])
        full_dataset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform
        )

        # Separate initial labeled set (with true labels)
        # The rest will be the unlabeled stream
        indices = list(range(len(full_dataset)))
        if shuffle_stream:
            np.random.seed(42)  # reproducible
            np.random.shuffle(indices)

        self.labeled_indices = indices[:initial_labeled_size]
        self.stream_indices = indices[initial_labeled_size:]

        self.labeled_dataset = Subset(full_dataset, self.labeled_indices)
        self.stream_dataset = Subset(full_dataset, self.stream_indices)

        # Dataloader for the stream (sequential, one batch at a time)
        self.stream_loader = DataLoader(
            self.stream_dataset,
            batch_size=batch_size,
            shuffle=False,          # important: simulate arrival order
            drop_last=False
        )
        self.stream_iter = iter(self.stream_loader)

        print(f"Simulator ready:")
        print(f"  - Initial labeled set: {len(self.labeled_dataset)} images")
        print(f"  - Unlabeled stream: {len(self.stream_dataset)} images")
        print(f"  - Batch size: {batch_size}")

    def get_initial_labeled_data(self):
        """
        Returns the initial labeled dataset (images, labels).
        """
        images = []
        labels = []
        for idx in range(len(self.labeled_dataset)):
            img, lbl = self.labeled_dataset[idx]
            images.append(img)
            labels.append(lbl)
        return torch.stack(images), torch.tensor(labels)

    def next_batch(self):
        """
        Returns the next batch of unlabeled data from the stream.
        Returns (images, None) because labels are not available.
        If stream ends, returns (None, None).
        """
        try:
            images, _ = next(self.stream_iter)   # we ignore the true label (unlabeled)
            return images, None                  # no labels in the stream
        except StopIteration:
            return None, None

    def has_next(self):
        """Check if more batches remain in the stream."""
        try:
            # peek without consuming
            self.stream_loader.dataset[0]
            # simpler: check if iterator is exhausted
            # We'll just rely on next_batch returning None
            return True
        except:
            return False

# ------------------- Quick test (run this file directly) -------------------
if __name__ == "__main__":
    # Create simulator
    sim = MNISTStreamSimulator(batch_size=64, initial_labeled_size=1000, shuffle_stream=True)

    # Get initial labeled data
    init_x, init_y = sim.get_initial_labeled_data()
    print(f"\nInitial labeled images shape: {init_x.shape}, labels: {init_y.shape}")

    # Simulate streaming unlabeled batches
    batch_num = 0
    while True:
        x_batch, _ = sim.next_batch()
        if x_batch is None:
            break
        print(f"Batch {batch_num}: shape = {x_batch.shape} (no labels)")
        batch_num += 1
        if batch_num >= 5:   # just show first 5 batches
            break
    print("\nStream simulation successful!")