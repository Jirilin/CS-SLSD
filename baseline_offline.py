"""
baseline_offline.py - Train once on initial labeled data only.
No streaming updates. This is the "lower bound" baseline.
"""
import torch
import torch.optim as optim
from stream_simulator import MNISTStreamSimulator
from models import SimpleCNN
from utils import train_epoch, evaluate, get_test_loader

def main():
    # Configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    INITIAL_LABELED_SIZE = 1000
    BATCH_SIZE = 64
    EPOCHS = 10
    LEARNING_RATE = 0.001
    
    print("=" * 50)
    print("BASELINE 1: OFFLINE-ONLY (no streaming updates)")
    print("=" * 50)
    print(f"Device: {DEVICE}")
    print(f"Initial labeled samples: {INITIAL_LABELED_SIZE}")
    print(f"Training epochs: {EPOCHS}")
    
    # Create simulator
    sim = MNISTStreamSimulator(
        batch_size=BATCH_SIZE,
        initial_labeled_size=INITIAL_LABELED_SIZE,
        shuffle_stream=True
    )
    
    # Get initial labeled data
    init_images, init_labels = sim.get_initial_labeled_data()
    print(f"\nInitial labeled data: {init_images.shape}")
    
    # Create DataLoader for initial labeled data
    init_dataset = torch.utils.data.TensorDataset(init_images, init_labels)
    init_loader = torch.utils.data.DataLoader(
        init_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    
    # Create model and optimizer
    model = SimpleCNN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Get test loader
    test_loader = get_test_loader()
    
    # Evaluate before training (random initialization)
    print("\n--- Before any training ---")
    initial_test_acc = evaluate(model, test_loader, DEVICE)
    print(f"Test accuracy (random init): {initial_test_acc:.2f}%")
    
    # Train on initial labeled data only
    print(f"\n--- Training for {EPOCHS} epochs ---")
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, init_loader, optimizer, DEVICE)
        test_acc = evaluate(model, test_loader, DEVICE)
        print(f"Epoch {epoch+1:2d}/{EPOCHS} | Train Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")
    
    # Final evaluation
    final_test_acc = evaluate(model, test_loader, DEVICE)
    print("\n" + "=" * 50)
    print("BASELINE 1 RESULTS (Offline-only)")
    print(f"Final test accuracy: {final_test_acc:.2f}%")
    print("=" * 50)
    
    # Save model for later comparison
    torch.save(model.state_dict(), 'baseline_offline_model.pth')
    print("\nModel saved to 'baseline_offline_model.pth'")

if __name__ == "__main__":
    main()