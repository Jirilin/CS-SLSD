
import torch
import torch.optim as optim
from stream_simulator import MNISTStreamSimulator
from models import SimpleCNN
from utils import evaluate, get_test_loader, generate_pseudolabels
from replay_buffer import ReplayBuffer

def train_on_batch(model, images, labels, optimizer, device='cpu'):
    # Single training step on a batch of (images, labels)
    model.train()
    images, labels = images.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs = model(images)
    loss = torch.nn.functional.cross_entropy(outputs, labels)
    loss.backward()
    optimizer.step()
    return loss.item()

def main():
    # ========== Configuration ==========
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    INITIAL_LABELED_SIZE = 1000
    BATCH_SIZE = 64
    INITIAL_EPOCHS = 10
    ONLINE_EPOCHS_PER_BATCH = 3      # number of training epochs per incoming batch
    REPLAY_BATCH_SIZE = 32            # how many samples to take from replay buffer
    BUFFER_CAPACITY = 500             # max size of replay buffer
    CONFIDENCE_THRESHOLD = 0.9
    LEARNING_RATE = 0.001
    
    print("=" * 60)
    print("EXPERIENCE REPLAY + PSEUDO-LABELS (Improved Continual Learning)")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Buffer capacity: {BUFFER_CAPACITY}")
    print(f"Replay batch size: {REPLAY_BATCH_SIZE}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    
    # ========== Setup stream simulator ==========
    sim = MNISTStreamSimulator(
        batch_size=BATCH_SIZE,
        initial_labeled_size=INITIAL_LABELED_SIZE,
        shuffle_stream=True
    )
    
    # ========== Initial labelled data ==========
    init_images, init_labels = sim.get_initial_labeled_data()
    init_dataset = torch.utils.data.TensorDataset(init_images, init_labels)
    init_loader = torch.utils.data.DataLoader(init_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # ========== Test loader ==========
    test_loader = get_test_loader()
    
    # ========== Model & Optimizer ==========
    model = SimpleCNN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # ========== Replay buffer ==========
    replay_buffer = ReplayBuffer(capacity=BUFFER_CAPACITY, device=DEVICE)
    
    # ========== Phase 1: Train on initial labelled data ==========
    print("\n--- Phase 1: Initial training on labelled data ---")
    for epoch in range(INITIAL_EPOCHS):
        total_loss = 0
        for batch_x, batch_y in init_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            out = model(batch_x)
            loss = torch.nn.functional.cross_entropy(out, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        test_acc = evaluate(model, test_loader, DEVICE)
        print(f"Epoch {epoch+1:2d}/{INITIAL_EPOCHS} | Loss: {total_loss/len(init_loader):.4f} | Test Acc: {test_acc:.2f}%")
    
    # Track accuracy
    accuracies = [evaluate(model, test_loader, DEVICE)]
    
    # ========== Phase 2: Streaming with experience replay ==========
    print("\n--- Phase 2: Streaming updates with replay ---")
    batch_num = 0
    total_pseudo_used = 0
    
    while True:
        # Get next unlabeled batch from stream
        images, _ = sim.next_batch()
        if images is None:
            break
        
        # Generate pseudolabels (Algorithm 1)
        pseudolabels, mask = generate_pseudolabels(model, images, CONFIDENCE_THRESHOLD, DEVICE)
        confident_images = images[mask]
        num_confident = len(confident_images)
        total_pseudo_used += num_confident
        
        # Add confident samples to replay buffer
        if num_confident > 0:
            replay_buffer.add(confident_images, pseudolabels)
        
        # Train on mixed batch: current confident samples + replay sample
        # We'll train for several epochs on the current batch (but each epoch uses fresh replay sample)
        for _ in range(ONLINE_EPOCHS_PER_BATCH):
            # Create a training batch: first use current confident samples
            if num_confident > 0:
                # Use current batch as part of training
                current_images = confident_images
                current_labels = pseudolabels
                # Also sample from replay buffer (if buffer has enough)
                if len(replay_buffer) >= REPLAY_BATCH_SIZE:
                    replay_images, replay_labels = replay_buffer.sample(REPLAY_BATCH_SIZE)
                    # Concatenate current and replay
                    combined_images = torch.cat([current_images, replay_images], dim=0)
                    combined_labels = torch.cat([current_labels, replay_labels], dim=0)
                else:
                    combined_images = current_images
                    combined_labels = current_labels
            else:
                # No confident samples this batch – use only replay buffer
                if len(replay_buffer) >= REPLAY_BATCH_SIZE:
                    combined_images, combined_labels = replay_buffer.sample(REPLAY_BATCH_SIZE)
                else:
                    # Nothing to train on – skip
                    continue
            
            # Train on combined batch
            train_on_batch(model, combined_images, combined_labels, optimizer, DEVICE)
        
        # Evaluate after each stream batch
        current_acc = evaluate(model, test_loader, DEVICE)
        accuracies.append(current_acc)
        
        print(f"Batch {batch_num:3d} | Confident: {num_confident:3d}/{BATCH_SIZE:3d} | "
              f"Buffer size: {len(replay_buffer):3d} | Test Acc: {current_acc:.2f}%")
        
        batch_num += 1
        if batch_num >= 20:  # Limit for demonstration
            print("\n(Stopping after 20 batches for demonstration)")
            break
    
    # ========== Final Results ==========
    print("\n" + "=" * 60)
    print("EXPERIENCE REPLAY RESULTS")
    print(f"Initial test accuracy: {accuracies[0]:.2f}%")
    print(f"Final test accuracy: {accuracies[-1]:.2f}%")
    if accuracies[-1] < accuracies[0]:
        drop = accuracies[0] - accuracies[-1]
        print(f"Forgetting drop: {drop:.2f}%")
    else:
        print("No forgetting detected (accuracy improved or stayed same).")
    print(f"Total pseudo-labeled samples added to buffer: {total_pseudo_used}")
    print("=" * 60)
    
    # Save model
    torch.save(model.state_dict(), 'baseline_replay_model.pth')
    print("\nModel saved to 'baseline_replay_model.pth'")

if __name__ == "__main__":
    main()