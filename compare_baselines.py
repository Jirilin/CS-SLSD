"""
compare_baselines.py
Loads saved models and compares their streaming performance on the same data stream.
Generates accuracy-over-time plots and metrics.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from stream_simulator import MNISTStreamSimulator
from models import SimpleCNN
from utils import evaluate, get_test_loader, generate_pseudolabels
from replay_buffer import ReplayBuffer

# ------------------- Configuration -------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64
INITIAL_LABELED_SIZE = 1000
CONFIDENCE_THRESHOLD = 0.9
BUFFER_CAPACITY = 500
REPLAY_BATCH_SIZE = 32
ONLINE_EPOCHS_PER_BATCH = 3
LEARNING_RATE = 0.001

def train_on_batch(model, images, labels, optimizer):
    model.train()
    images, labels = images.to(DEVICE), labels.to(DEVICE)
    optimizer.zero_grad()
    loss = torch.nn.functional.cross_entropy(model(images), labels)
    loss.backward()
    optimizer.step()

def run_offline(model, test_loader):
    """Offline model never updates; just evaluate once."""
    return evaluate(model, test_loader, DEVICE)

def run_naive_online(model, sim, test_loader):
    """
    Naive online: update on each batch using pseudo-labels (no replay).
    Returns (accuracies_list, initial_accuracy)
    """
    accuracies = []
    # Evaluate BEFORE any streaming update
    initial_acc = evaluate(model, test_loader, DEVICE)
    accuracies.append(initial_acc)
    
    sim.stream_iter = iter(sim.stream_loader)
    batch_num = 0
    while True:
        images, _ = sim.next_batch()
        if images is None:
            break
        # Generate pseudo-labels
        pseudolabels, mask = generate_pseudolabels(model, images, CONFIDENCE_THRESHOLD, DEVICE)
        confident_images = images[mask]
        if len(confident_images) > 0:
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE * 0.1)
            for _ in range(ONLINE_EPOCHS_PER_BATCH):
                train_on_batch(model, confident_images, pseudolabels, optimizer)
        acc = evaluate(model, test_loader, DEVICE)
        accuracies.append(acc)
        batch_num += 1
        if batch_num >= 20:
            break
    return accuracies, initial_acc

def run_experience_replay(model, sim, test_loader):
    """
    Experience replay: store past samples and mix with current batch.
    Returns (accuracies_list, initial_accuracy)
    """
    accuracies = []
    # Evaluate BEFORE any streaming update
    initial_acc = evaluate(model, test_loader, DEVICE)
    accuracies.append(initial_acc)
    
    sim.stream_iter = iter(sim.stream_loader)
    replay_buffer = ReplayBuffer(capacity=BUFFER_CAPACITY, device=DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE * 0.1)
    batch_num = 0
    while True:
        images, _ = sim.next_batch()
        if images is None:
            break
        # Generate pseudo-labels
        pseudolabels, mask = generate_pseudolabels(model, images, CONFIDENCE_THRESHOLD, DEVICE)
        confident_images = images[mask]
        if len(confident_images) > 0:
            replay_buffer.add(confident_images, pseudolabels)
            for _ in range(ONLINE_EPOCHS_PER_BATCH):
                current_images = confident_images
                current_labels = pseudolabels
                if len(replay_buffer) >= REPLAY_BATCH_SIZE:
                    replay_imgs, replay_lbls = replay_buffer.sample(REPLAY_BATCH_SIZE)
                    combined_imgs = torch.cat([current_images, replay_imgs], dim=0)
                    combined_lbls = torch.cat([current_labels, replay_lbls], dim=0)
                else:
                    combined_imgs = current_images
                    combined_lbls = current_labels
                train_on_batch(model, combined_imgs, combined_lbls, optimizer)
        acc = evaluate(model, test_loader, DEVICE)
        accuracies.append(acc)
        batch_num += 1
        if batch_num >= 20:
            break
    return accuracies, initial_acc

# ------------------- Main -------------------
def main():
    print("Loading test set...")
    test_loader = get_test_loader()
    
    # Create a fresh stream simulator with fixed seed for reproducibility
    np.random.seed(42)
    sim = MNISTStreamSimulator(
        batch_size=BATCH_SIZE,
        initial_labeled_size=INITIAL_LABELED_SIZE,
        shuffle_stream=True
    )
    
    # ---- Offline model ----
    print("Loading offline model...")
    model_offline = SimpleCNN().to(DEVICE)
    # Suppress the FutureWarning by explicitly using weights_only=False (safe as we trust our own files)
    state_dict = torch.load('baseline_offline_model.pth', map_location=DEVICE, weights_only=False)
    model_offline.load_state_dict(state_dict)
    offline_acc = run_offline(model_offline, test_loader)
    print(f"Offline final accuracy: {offline_acc:.2f}%")
    # For offline, accuracy is constant for all batches (including initial)
    offline_accs = [offline_acc] * 21  # 1 initial + 20 batches
    
    # ---- Naive online model ----
    print("Running naive online simulation...")
    model_naive = SimpleCNN().to(DEVICE)
    model_naive.load_state_dict(torch.load('baseline_naive_model.pth', map_location=DEVICE, weights_only=False))
    np.random.seed(42)
    sim2 = MNISTStreamSimulator(
        batch_size=BATCH_SIZE,
        initial_labeled_size=INITIAL_LABELED_SIZE,
        shuffle_stream=True
    )
    naive_accs, naive_initial = run_naive_online(model_naive, sim2, test_loader)
    
    # ---- Experience replay model ----
    print("Running experience replay simulation...")
    model_replay = SimpleCNN().to(DEVICE)
    model_replay.load_state_dict(torch.load('baseline_replay_model.pth', map_location=DEVICE, weights_only=False))
    np.random.seed(42)
    sim3 = MNISTStreamSimulator(
        batch_size=BATCH_SIZE,
        initial_labeled_size=INITIAL_LABELED_SIZE,
        shuffle_stream=True
    )
    replay_accs, replay_initial = run_experience_replay(model_replay, sim3, test_loader)
    
    # ---- Compute metrics (using correct initial accuracy) ----
    metrics = {
        'Offline': {
            'final_acc': offline_accs[-1],
            'avg_acc': np.mean(offline_accs),
            'std_acc': np.std(offline_accs),
            'forgetting': 0.0
        },
        'Naive Online': {
            'final_acc': naive_accs[-1],
            'avg_acc': np.mean(naive_accs),
            'std_acc': np.std(naive_accs),
            'forgetting': naive_initial - naive_accs[-1]   # positive = forgetting
        },
        'Experience Replay': {
            'final_acc': replay_accs[-1],
            'avg_acc': np.mean(replay_accs),
            'std_acc': np.std(replay_accs),
            'forgetting': replay_initial - replay_accs[-1]
        }
    }
    
    # Print metrics table
    print("\n" + "="*70)
    print("COMPARISON METRICS (over 20 streaming batches)")
    print("="*70)
    print(f"{'Model':<18} {'Final Acc':<12} {'Avg Acc':<12} {'Std Dev':<12} {'Forgetting':<12}")
    print("-"*70)
    for name, m in metrics.items():
        print(f"{name:<18} {m['final_acc']:<12.2f} {m['avg_acc']:<12.2f} {m['std_acc']:<12.2f} {m['forgetting']:<12.2f}")
    
    # ---- Plotting accuracy over time ----
    plt.figure(figsize=(10, 6))
    # Number of points: initial + 20 batches = 21
    batches = list(range(0, len(naive_accs)))  # 0 = initial, 1..20 = after each batch
    plt.plot(batches, offline_accs[:len(batches)], 'k--', label='Offline (no update)', linewidth=2)
    plt.plot(batches, naive_accs, 'r-', label='Naive Online', linewidth=2, marker='o', markersize=4)
    plt.plot(batches, replay_accs, 'g-', label='Experience Replay', linewidth=2, marker='s', markersize=4)
    plt.xlabel('Stream Batch Number (0 = initial)', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.title('Comparison of Continual Learning Baselines', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(70, 100)
    plt.tight_layout()
    plt.savefig('baseline_comparison.png', dpi=150)
    plt.show()
    
    # ---- Plot forgetting bars ----
    plt.figure(figsize=(8, 5))
    models = list(metrics.keys())
    forgetting_vals = [metrics[m]['forgetting'] for m in models]
    colors = ['gray', 'red', 'green']
    bars = plt.bar(models, forgetting_vals, color=colors)
    plt.ylabel('Forgetting (Initial - Final Accuracy %)', fontsize=12)
    plt.title('Catastrophic Forgetting Comparison', fontsize=14)
    for bar, val in zip(bars, forgetting_vals):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.2f}', ha='center')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('forgetting_comparison.png', dpi=150)
    plt.show()
    
    print("\nPlots saved as 'baseline_comparison.png' and 'forgetting_comparison.png'")

if __name__ == "__main__":
    main()