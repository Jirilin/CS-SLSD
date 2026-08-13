import torch
from models import VisionCNN
from replay_buffer import ReservoirReplayBuffer

def test_model_shapes_mnist_and_rgb():
    m1 = VisionCNN(1, 10)
    m3 = VisionCNN(3, 10)
    assert m1(torch.randn(4, 1, 28, 28)).shape == (4, 10)
    assert m3(torch.randn(4, 3, 32, 32)).shape == (4, 10)
    assert m1.forward_features(torch.randn(4, 1, 28, 28)).shape[1] == 128

def test_replay_capacity():
    b = ReservoirReplayBuffer(10, seed=0)
    for _ in range(5):
        b.add_batch(torch.randn(8, 1, 28, 28), torch.randint(0, 10, (8,)))
    assert len(b) == 10
