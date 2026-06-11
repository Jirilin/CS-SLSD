import torch
from models import SimpleCNN
from utils import get_test_loader, evaluate

model = SimpleCNN()
model.load_state_dict(torch.load('baseline_replay_model.pth'))
test_loader = get_test_loader()
acc = evaluate(model, test_loader)
print(f"Model accuracy: {acc:.2f}%")