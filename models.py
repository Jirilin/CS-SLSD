"""
models.py - CNN architecture for MNIST classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    A small CNN suitable for MNIST (28x28 grayscale images).
    Outputs logits for 10 classes.
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # First convolutional layer: 1 input channel -> 32 output channels
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # Second convolutional layer: 32 -> 64 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Max pooling layer (2x2)
        self.pool = nn.MaxPool2d(2, 2)
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)   # after two poolings, 28x28 -> 7x7
        self.fc2 = nn.Linear(128, 10)
        # Dropout for regularization
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # Conv block 1: conv -> ReLU -> pool
        x = self.pool(F.relu(self.conv1(x)))
        # Conv block 2: conv -> ReLU -> pool
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten: from (batch, 64, 7, 7) to (batch, 64*7*7)
        x = x.view(-1, 64 * 7 * 7)
        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def count_parameters(model):
    """Utility: count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Quick test
if __name__ == "__main__":
    model = SimpleCNN()
    print(f"Model has {count_parameters(model):,} trainable parameters")
    
    # Test forward pass
    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape} (10 classes)")