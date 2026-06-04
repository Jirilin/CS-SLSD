# Test pseudolabel generation
import torch
from stream_simulator import MNISTStreamSimulator
from models import SimpleCNN
from utils import generate_pseudolabels, get_test_loader, evaluate

# Setup
device = 'cpu'
sim = MNISTStreamSimulator(batch_size=64, initial_labeled_size=1000)
model = SimpleCNN().to(device)

# Load a pre‑trained model (from baseline_offline.pth)
state_dict = torch.load('baseline_offline_model.pth', map_location=device, weights_only=True)
model.load_state_dict(state_dict)
model.eval()

# Get one unlabeled batch
images, _ = sim.next_batch()

# Generate pseudolabels with threshold 0.9
pseudolabels, mask = generate_pseudolabels(model, images, confidence_threshold=0.9, device=device)

print(f"Batch size: {len(images)}")
print(f"Confident samples: {mask.sum().item()}")
print(f"Pseudolabels: {pseudolabels[:10]}")  # show first 10