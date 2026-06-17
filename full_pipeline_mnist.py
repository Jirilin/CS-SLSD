import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
from stream_simulator import MNISTStreamSimulator
from models import SimpleCNN
from utils import evaluate, get_test_loader, generate_pseudolabels
from replay_buffer import ReplayBuffer

# ==================== CONFIGURATION ====================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64
INITIAL_LABELED_SIZE = 1000
INITIAL_EPOCHS = 10
ONLINE_EPOCHS_PER_BATCH = 3
REPLAY_BATCH_SIZE = 32
BUFFER_CAPACITY = 500
CONFIDENCE_THRESHOLD = 0.9
LEARNING_RATE = 0.001
STREAM_BATCHES = 20  # Number of batches to process (for demonstration)
NUM_SEEDS = 1        # For reproducibility; can increase for final experiments

def train_on_batch(model, images, labels, optimizer):
    model.train()
    images, labels = images.to(DEVICE), labels.to(DEVICE)
    optimizer.zero_grad()
    loss = torch.nn.functional.cross_entropy(model(images), labels)
    loss.backward()
    optimizer.step()
    return loss.item()

def run_offline_on_stream(model, test_loader):
    # Offline: no updates, just evaluate on the stream.
    acc = evaluate(model, test_loader, DEVICE)
    return [acc] * STREAM_BATCHES

def run_naive_on_stream(model, sim, test_loader):
    # Naive online: update on each batch using pseudo-labels (no replay).
    accuracies = []
    sim.stream_iter = iter(sim.stream_loader)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE * 0.1)
    for _ in range(STREAM_BATCHES):
        images, _ = sim.next_batch()
        if images is None:
            break
        pseudolabels, mask = generate_pseudolabels(model, images, CONFIDENCE_THRESHOLD, DEVICE)
        confident_images = images[mask]
        if len(confident_images) > 0:
            for _ in range(ONLINE_EPOCHS_PER_BATCH):
                train_on_batch(model, confident_images, pseudolabels, optimizer)
        accuracies.append(evaluate(model, test_loader, DEVICE))
    return accuracies

def run_replay_on_stream(model, sim, test_loader):
    # Experience replay: buffer + current batch.
    accuracies = []
    buffer = ReplayBuffer(capacity=BUFFER_CAPACITY, device=DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE * 0.1)
    sim.stream_iter = iter(sim.stream_loader)
    for _ in range(STREAM_BATCHES):
        images, _ = sim.next_batch()
        if images is None:
            break
        pseudolabels, mask = generate_pseudolabels(model, images, CONFIDENCE_THRESHOLD, DEVICE)
        confident_images = images[mask]
        if len(confident_images) > 0:
            buffer.add(confident_images, pseudolabels)
            for _ in range(ONLINE_EPOCHS_PER_BATCH):
                current = confident_images
                current_labels = pseudolabels
                if len(buffer) >= REPLAY_BATCH_SIZE:
                    replay_imgs, replay_lbls = buffer.sample(REPLAY_BATCH_SIZE)
                    combined_imgs = torch.cat([current, replay_imgs], dim=0)
                    combined_lbls = torch.cat([current_labels, replay_lbls], dim=0)
                else:
                    combined_imgs = current
                    combined_lbls = current_labels
                train_on_batch(model, combined_imgs, combined_lbls, optimizer)
        accuracies.append(evaluate(model, test_loader, DEVICE))
    return accuracies

def run_full_pipeline(model, sim, test_loader):
    
    accuracies = []
    losses = []
    confident_ratios = []
    buffer_sizes = []
    times = []
    
    buffer = ReplayBuffer(capacity=BUFFER_CAPACITY, device=DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE * 0.1)
    sim.stream_iter = iter(sim.stream_loader)
    
    for _ in range(STREAM_BATCHES):
        start_time = time.time()
        images, _ = sim.next_batch()
        if images is None:
            break
        
        # Generate pseudolabels
        pseudolabels, mask = generate_pseudolabels(model, images, CONFIDENCE_THRESHOLD, DEVICE)
        confident_images = images[mask]
        confident_ratio = len(confident_images) / len(images)
        
        # Add to buffer
        if len(confident_images) > 0:
            buffer.add(confident_images, pseudolabels)
        
        # Train on mixed batch
        batch_loss = 0.0
        if len(confident_images) > 0:
            for epoch in range(ONLINE_EPOCHS_PER_BATCH):
                current = confident_images
                current_labels = pseudolabels
                if len(buffer) >= REPLAY_BATCH_SIZE:
                    replay_imgs, replay_lbls = buffer.sample(REPLAY_BATCH_SIZE)
                    combined_imgs = torch.cat([current, replay_imgs], dim=0)
                    combined_lbls = torch.cat([current_labels, replay_lbls], dim=0)
                else:
                    combined_imgs = current
                    combined_lbls = current_labels
                loss_val = train_on_batch(model, combined_imgs, combined_lbls, optimizer)
                batch_loss += loss_val
        else:
            # No confident samples: train only on replay if available
            if len(buffer) >= REPLAY_BATCH_SIZE:
                replay_imgs, replay_lbls = buffer.sample(REPLAY_BATCH_SIZE)
                for _ in range(ONLINE_EPOCHS_PER_BATCH):
                    loss_val = train_on_batch(model, replay_imgs, replay_lbls, optimizer)
                    batch_loss += loss_val
        
        # Log metrics
        acc = evaluate(model, test_loader, DEVICE)
        accuracies.append(acc)
        losses.append(batch_loss / ONLINE_EPOCHS_PER_BATCH if ONLINE_EPOCHS_PER_BATCH > 0 else 0)
        confident_ratios.append(confident_ratio)
        buffer_sizes.append(len(buffer))
        times.append(time.time() - start_time)
        
        # Print progress
        print(f"Batch {len(accuracies)-1:3d} | Acc: {acc:.2f}% | Conf: {confident_ratio:.2f} | "
              f"Buffer: {len(buffer):3d} | Loss: {losses[-1]:.4f}")
    
    return accuracies, losses, confident_ratios, buffer_sizes, times

# ==================== COMPARISON FUNCTION ====================
def compare_all_methods():

    print("=" * 70)
    print("COMPARISON: Offline vs Naive vs Replay vs Full Pipeline")
    print("=" * 70)
    
    test_loader = get_test_loader()
    
    # Create a fresh simulator with fixed seed for fairness
    np.random.seed(42)
    sim = MNISTStreamSimulator(
        batch_size=BATCH_SIZE,
        initial_labeled_size=INITIAL_LABELED_SIZE,
        shuffle_stream=True
    )
    
    results = {}
    
    # 1. Offline
    print("\nLoading Offline model...")
    model = SimpleCNN().to(DEVICE)
    model.load_state_dict(torch.load('baseline_offline_model.pth', map_location=DEVICE))
    accs_offline = run_offline_on_stream(model, test_loader)
    results['Offline'] = accs_offline
    
    # 2. Naive Online
    print("\nRunning Naive Online...")
    np.random.seed(42)  # Reset simulator
    sim2 = MNISTStreamSimulator(
        batch_size=BATCH_SIZE,
        initial_labeled_size=INITIAL_LABELED_SIZE,
        shuffle_stream=True
    )
    model = SimpleCNN().to(DEVICE)
    model.load_state_dict(torch.load('baseline_naive_model.pth', map_location=DEVICE))
    accs_naive = run_naive_on_stream(model, sim2, test_loader)
    results['Naive Online'] = accs_naive
    
    # 3. Experience Replay
    print("\nRunning Experience Replay...")
    np.random.seed(42)
    sim3 = MNISTStreamSimulator(
        batch_size=BATCH_SIZE,
        initial_labeled_size=INITIAL_LABELED_SIZE,
        shuffle_stream=True
    )
    model = SimpleCNN().to(DEVICE)
    model.load_state_dict(torch.load('baseline_replay_model.pth', map_location=DEVICE))
    accs_replay = run_replay_on_stream(model, sim3, test_loader)
    results['Experience Replay'] = accs_replay
    
    # 4. Full Pipeline (proposed)
    print("\nRunning Full Pipeline (with detailed logging)...")
    np.random.seed(42)
    sim4 = MNISTStreamSimulator(
        batch_size=BATCH_SIZE,
        initial_labeled_size=INITIAL_LABELED_SIZE,
        shuffle_stream=True
    )
    model = SimpleCNN().to(DEVICE)
    
    print("   Initialising Full Pipeline model...")
    model = SimpleCNN().to(DEVICE)
    # Train on initial labelled data (same as baselines)
    init_images, init_labels = sim4.get_initial_labeled_data()
    init_dataset = torch.utils.data.TensorDataset(init_images, init_labels)
    init_loader = torch.utils.data.DataLoader(init_dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    for epoch in range(INITIAL_EPOCHS):
        for bx, by in init_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(bx), by)
            loss.backward()
            optimizer.step()
    print(f"   Initial Test Acc: {evaluate(model, test_loader, DEVICE):.2f}%")
    
    accs_full, losses, conf_ratios, buf_sizes, times = run_full_pipeline(model, sim4, test_loader)
    results['Full Pipeline (Ours)'] = accs_full
    
    # Compute Metrics and Print Summary Table
    print("\n" + "=" * 70)
    print("SUMMARY METRICS")
    print("=" * 70)
    metrics = {}
    for name, accs in results.items():
        first = accs[0]
        last = accs[-1]
        avg = np.mean(accs)
        std = np.std(accs)
        forgetting = first - last
        metrics[name] = {
            'First Acc': first,
            'Last Acc': last,
            'Avg Acc': avg,
            'Std': std,
            'Forgetting': forgetting
        }
    
    # Print table
    print(f"{'Method':<20} {'First':<8} {'Last':<8} {'Avg':<8} {'Std':<8} {'Forgetting':<10}")
    print("-" * 70)
    for name, m in metrics.items():
        print(f"{name:<20} {m['First Acc']:<8.2f} {m['Last Acc']:<8.2f} {m['Avg Acc']:<8.2f} {m['Std']:<8.2f} {m['Forgetting']:<10.2f}")
        
    # ---- Save metrics to CSV ----
    df = pd.DataFrame({
        'Batch': list(range(len(accs_full))),
        'Accuracy': accs_full,
        'Loss': losses,
        'Confident_Ratio': conf_ratios,
        'Buffer_Size': buf_sizes,
        'Time_Seconds': times
    })
    df.to_csv('metrics_full_pipeline.csv', index=False)
    print("\nMetrics saved to 'metrics_full_pipeline.csv'")
    
    # ---- Plotting ----
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for name, accs in results.items():
        plt.plot(accs, label=name, marker='o', markersize=3)
    plt.xlabel('Stream Batch')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Accuracy over Stream Batches')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    methods = list(metrics.keys())
    forgetting_vals = [metrics[m]['Forgetting'] for m in methods]
    colors = ['gray', 'red', 'orange', 'green']
    bars = plt.bar(methods, forgetting_vals, color=colors)
    plt.ylabel('Forgetting (First - Last Accuracy %)')
    plt.title('Catastrophic Forgetting Comparison')
    for bar, val in zip(bars, forgetting_vals):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, f'{val:.2f}', ha='center')
    plt.xticks(rotation=15)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('pipeline_comparison.png', dpi=150)
    plt.show()
    print("Plots saved to 'pipeline_comparison.png'")

if __name__ == "__main__":
    compare_all_methods()