"""
baseline_naive_online.py - Naive online learning.
Updates model on each unlabeled batch using its own predictions as pseudolabels.
This often leads to catastrophic forgetting.
"""
import torch
import torch.optim as optim
from stream_simulator import MNISTStreamSimulator
from models import SimpleCNN
from utils import train_epoch, evaluate, get_test_loader, generate_pseudolabels
import matplotlib.pyplot as plt

def main():
    # Configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    INITIAL_LABELED_SIZE = 1000
    BATCH_SIZE = 64
    INITIAL_EPOCHS = 10          # epochs on initial labeled data
    ONLINE_EPOCHS_PER_BATCH = 3  # how many epochs to train on each batch
    LEARNING_RATE = 0.001
    CONFIDENCE_THRESHOLD = 0.9   # only use confident pseudolabels
    
    print("=" * 50)
    print("BASELINE 2: NAIVE ONLINE (fine-tune on each batch)")
    print("=" * 50)
    print(f"Device: {DEVICE}")
    print(f"Initial labeled samples: {INITIAL_LABELED_SIZE}")
    print(f"Initial training epochs: {INITIAL_EPOCHS}")
    print(f"Online epochs per batch: {ONLINE_EPOCHS_PER_BATCH}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    
    # Create simulator
    sim = MNISTStreamSimulator(
        batch_size=BATCH_SIZE,
        initial_labeled_size=INITIAL_LABELED_SIZE,
        shuffle_stream=True
    )
    
    # Get initial labeled data
    init_images, init_labels = sim.get_initial_labeled_data()
    init_dataset = torch.utils.data.TensorDataset(init_images, init_labels)
    init_loader = torch.utils.data.DataLoader(
        init_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    
    # Get test loader
    test_loader = get_test_loader()
    
    # Create model and optimizer
    model = SimpleCNN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # ========== PHASE 1: Train on initial labeled data ==========
    print("\n--- PHASE 1: Initial training on labeled data ---")
    for epoch in range(INITIAL_EPOCHS):
        train_loss, train_acc = train_epoch(model, init_loader, optimizer, DEVICE)
        test_acc = evaluate(model, test_loader, DEVICE)
        print(f"Initial Epoch {epoch+1}/{INITIAL_EPOCHS} | "
              f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")
    
    # Track accuracy over time
    accuracies = [evaluate(model, test_loader, DEVICE)]
    
    # ========== PHASE 2: Naive online streaming ==========
    print("\n--- PHASE 2: Naive online streaming updates ---")
    batch_num = 0
    total_pseudolabeled = 0
    
    while True:
        # Get next unlabeled batch from stream
        images, _ = sim.next_batch()
        if images is None:
            break
        
        # Generate pseudolabels from current model
        pseudolabels, mask = generate_pseudolabels(
            model, images, CONFIDENCE_THRESHOLD, DEVICE
        )
        
        num_confident = mask.sum().item()
        total_pseudolabeled += num_confident
        
        if num_confident > 0:
            # Extract confident samples and their pseudolabels
            confident_images = images[mask]
            # Create a small dataset from this batch
            pseudo_dataset = torch.utils.data.TensorDataset(
                confident_images, pseudolabels
            )
            pseudo_loader = torch.utils.data.DataLoader(
                pseudo_dataset, batch_size=min(32, num_confident), shuffle=True
            )
            
            # Fine-tune on this batch (critical: this causes forgetting!)
            # We use a new optimizer for each batch to simulate "naive" update
            online_optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE * 0.1)
            
            for _ in range(ONLINE_EPOCHS_PER_BATCH):
                for batch_images, batch_labels in pseudo_loader:
                    batch_images, batch_labels = batch_images.to(DEVICE), batch_labels.to(DEVICE)
                    outputs = model(batch_images)
                    loss = torch.nn.functional.cross_entropy(outputs, batch_labels)
                    online_optimizer.zero_grad()
                    loss.backward()
                    online_optimizer.step()
        
        # Evaluate after each batch
        current_acc = evaluate(model, test_loader, DEVICE)
        accuracies.append(current_acc)
        
        print(f"Batch {batch_num:3d} | Confident: {num_confident:3d}/{BATCH_SIZE:3d} | "
              f"Test Acc: {current_acc:.2f}%")
        
        batch_num += 1
        if batch_num >= 20:  # Limit for demonstration (shows forgetting)
            print("\n(Stopping after 20 batches for demonstration)")
            break
    
    # ========== Final Results ==========
    print("\n" + "=" * 50)
    print("BASELINE 2 RESULTS (Naive Online)")
    print(f"Initial test accuracy: {accuracies[0]:.2f}%")
    print(f"Final test accuracy: {accuracies[-1]:.2f}%")
    
    # Show accuracy drop
    if accuracies[-1] < accuracies[0]:
        drop = accuracies[0] - accuracies[-1]
        print(f"⚠️ Accuracy dropped by {drop:.2f}% (catastrophic forgetting!)")
    else:
        print("✓ Accuracy did not drop (unusual for naive online)")
    
    print(f"Total pseudolabeled samples used: {total_pseudolabeled}")
    print("=" * 50)
    
    # Save model
    torch.save(model.state_dict(), 'baseline_naive_model.pth')
    print("\nModel saved to 'baseline_naive_model.pth'")

# After running, plot accuracy over time
if 'accuracies' in locals():
    plt.plot(accuracies)
    plt.xlabel('Batch Number')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Catastrophic Forgetting in Naive Online Learning')
    plt.grid(True)
    plt.savefig('forgetting_curve.png')
    plt.show()
else:
    print("Error: 'accuracies' list was never created. Check if training completed successfully.")

if __name__ == "__main__":
    main()