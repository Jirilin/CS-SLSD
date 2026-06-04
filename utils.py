"""
utils.py - Helper functions for training and evaluation
"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def train_epoch(model, dataloader, optimizer, device='cpu'):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / total
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def evaluate(model, dataloader, device='cpu'):
    """
    Evaluate model on test data.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader with (images, labels)
        device: 'cpu' or 'cuda'
    
    Returns:
        accuracy: float
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100 * correct / total

def generate_pseudolabels(model, images, confidence_threshold=0.9, device='cpu'):
    model.eval()
    with torch.no_grad():
        images = images.to(device)
        outputs = model(images)
        probabilities = F.softmax(outputs, dim=1)
        max_probs, predictions = torch.max(probabilities, dim=1)
        
        # Only keep predictions above threshold
        mask = max_probs >= confidence_threshold
        pseudolabels = predictions[mask]
    
    return pseudolabels, mask

def get_test_loader(batch_size=128):
    """Load MNIST test set for evaluation"""
    import torchvision
    import torchvision.transforms as transforms
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    return torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )