import torch
from replay_buffer import ReservoirReplayBuffer
from models import SimpleCNN


def test_model_shapes():
    model=SimpleCNN(); x=torch.randn(4,1,28,28)
    assert model(x).shape==(4,10)
    assert model.forward_features(x).shape==(4,128)


def test_replay_capacity():
    b=ReservoirReplayBuffer(10,seed=0)
    b.add_batch(torch.randn(25,1,28,28),torch.arange(25)%10)
    assert len(b)==10
