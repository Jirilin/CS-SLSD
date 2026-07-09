# Supports: MNIST, FashionMNIST, KMNIST, CIFAR10, SVHN.
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

DATASET_INFO = {
    "mnist":        {"class": datasets.MNIST,        "channels": 1, "image_size": 28, "mean": (0.1307,), "std": (0.3081,)},
    "fashionmnist": {"class": datasets.FashionMNIST, "channels": 1, "image_size": 28, "mean": (0.2860,), "std": (0.3530,)},
    "kmnist":       {"class": datasets.KMNIST,       "channels": 1, "image_size": 28, "mean": (0.1918,), "std": (0.3483,)},
    "cifar10":      {"class": datasets.CIFAR10,      "channels": 3, "image_size": 32, "mean": (0.4914,0.4822,0.4465), "std": (0.2470,0.2435,0.2616)},
    "svhn":         {"class": datasets.SVHN,         "channels": 3, "image_size": 32, "mean": (0.4377,0.4438,0.4728), "std": (0.1980,0.2010,0.1970)},
}

def _targets(dataset):
    if hasattr(dataset, "targets"):
        return np.array(dataset.targets)
    if hasattr(dataset, "labels"):
        return np.array(dataset.labels)
    raise AttributeError("Cannot find target labels")

def load_dataset(dataset_name="MNIST", root="./data", train=True, download=True):
    key = dataset_name.lower()
    if key not in DATASET_INFO:
        raise ValueError(f"Unsupported dataset {dataset_name}. Choose {list(DATASET_INFO.keys())}")
    info = DATASET_INFO[key]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(info["mean"], info["std"]),
    ])
    if key == "svhn":
        split = "train" if train else "test"
        return info["class"](root=root, split=split, download=download, transform=transform)
    return info["class"](root=root, train=train, download=download, transform=transform)

class ContinualStreamSimulator:
    def __init__(self, dataset_name="MNIST", batch_size=64, initial_labeled_size=1000,
                 max_stream_samples=5000, distribution="class_incremental", seed=42, root="./data"):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.initial_labeled_size = initial_labeled_size
        self.max_stream_samples = max_stream_samples
        self.distribution = distribution
        self.seed = seed
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

        self.train_dataset = load_dataset(dataset_name, root=root, train=True, download=True)
        self.test_dataset = load_dataset(dataset_name, root=root, train=False, download=True)
        targets = _targets(self.train_dataset)

        if distribution == "class_incremental":
            # Stream ordered by class groups: 0-1, 2-3, ..., 8-9.
            ordered = []
            for cls_group in [(0,1),(2,3),(4,5),(6,7),(8,9)]:
                cls_idx = np.where(np.isin(targets, cls_group))[0].tolist()
                random.shuffle(cls_idx)
                ordered.extend(cls_idx)
            indices = ordered
        elif distribution == "random_iid":
            indices = list(range(len(self.train_dataset)))
            random.shuffle(indices)
        else:
            raise ValueError("distribution must be 'class_incremental' or 'random_iid'")

        self.labeled_indices = indices[:initial_labeled_size]
        stream_indices = indices[initial_labeled_size:initial_labeled_size + max_stream_samples]
        self.labeled_dataset = Subset(self.train_dataset, self.labeled_indices)
        self.stream_dataset = Subset(self.train_dataset, stream_indices)
        self.stream_loader = DataLoader(self.stream_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=256, shuffle=False)

    def get_initial_labeled_loader(self):
        return DataLoader(self.labeled_dataset, batch_size=self.batch_size, shuffle=True)

    def get_stream_loader(self):
        return self.stream_loader

    def get_test_loader(self):
        return self.test_loader

    def describe(self):
        return {
            "dataset": self.dataset_name,
            "distribution": self.distribution,
            "initial_labeled": len(self.labeled_dataset),
            "stream_samples": len(self.stream_dataset),
            "batch_size": self.batch_size,
        }
