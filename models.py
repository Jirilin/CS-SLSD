import torch
import torch.nn as nn
import torch.nn.functional as F

class FlexibleCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10, image_size=28):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        reduced = image_size // 4
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * reduced * reduced, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

class SimpleCNN(FlexibleCNN):
    """Backward-compatible MNIST model used by older files."""
    def __init__(self):
        super().__init__(in_channels=1, num_classes=10, image_size=28)

def build_model(dataset_name: str):
    name = dataset_name.lower()
    if name in ["mnist", "fashionmnist", "kmnist"]:
        return FlexibleCNN(in_channels=1, image_size=28)
    if name in ["cifar10", "svhn"]:
        return FlexibleCNN(in_channels=3, image_size=32)
    raise ValueError(f"Unsupported dataset: {dataset_name}")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    for ds in ["MNIST", "CIFAR10", "SVHN"]:
        m = build_model(ds)
        x = torch.randn(2, 1, 28, 28) if ds == "MNIST" else torch.randn(2, 3, 32, 32)
        print(ds, m(x).shape, count_parameters(m))
