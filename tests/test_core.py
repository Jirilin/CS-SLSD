import torch

from models import VisionCNN
from replay_buffer import ReservoirReplayBuffer
from centroid_pseudolabel import CentroidRefinedPseudoLabeler


def test_model_shapes_mnist_and_rgb():
    m1 = VisionCNN(1, 10)
    m3 = VisionCNN(3, 10)
    assert m1(torch.randn(4, 1, 28, 28)).shape == (4, 10)
    assert m3(torch.randn(4, 3, 32, 32)).shape == (4, 10)
    assert m1.forward_features(torch.randn(4, 1, 28, 28)).shape[1] == 128


def test_replay_capacity_and_diagnostics():
    buffer = ReservoirReplayBuffer(10, seed=0)
    for _ in range(5):
        stats = buffer.add_batch(
            torch.randn(8, 1, 28, 28),
            torch.randint(0, 10, (8,)),
        )
    assert len(buffer) == 10
    assert buffer.is_full
    assert stats.total_seen == 40
    assert sum(buffer.class_histogram().values()) == 10


def test_pseudolabel_parameter_validation():
    model = VisionCNN(1, 10)
    try:
        CentroidRefinedPseudoLabeler(model, torch.device("cpu"), 10, threshold=1.1)
        assert False, "invalid threshold should raise"
    except ValueError:
        pass
