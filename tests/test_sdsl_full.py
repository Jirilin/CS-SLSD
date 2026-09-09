import torch
from torch.utils.data import DataLoader, TensorDataset

from sdsl_model import SDSLVisionNet
from sdsl_pseudolabel import SDSLRobustPseudoLabeler
from gradient_subspace import GradientSubspaceMemory
from flat_minimax import SDSLFlatMinimaxSolver


def test_sdsl_model_shapes():
    m = SDSLVisionNet(1, 10)
    x = torch.randn(4, 1, 28, 28)
    assert m(x).shape == (4, 10)
    assert m.forward_features(x).shape == (4, 128)


def test_three_stage_pseudolabel_generation():
    torch.manual_seed(0)
    m = SDSLVisionNet(1, 10)
    xs = torch.randn(40, 1, 28, 28)
    ys = torch.arange(40) % 10
    loader = DataLoader(TensorDataset(xs, ys), batch_size=20)
    labeler = SDSLRobustPseudoLabeler(m, torch.device("cpu"), 10, semantic_rank=5,
                                     centroid_iterations=2, threshold=0.0)
    labeler.fit_invariant_semantics(loader)
    result = labeler.generate(torch.randn(16, 1, 28, 28), torch.randint(0, 10, (16,)))
    assert result.accepted_count == 16
    assert result.images.shape[0] == 16
    assert 0.0 <= result.coverage <= 1.0


def test_flat_minimax_step_changes_parameters():
    torch.manual_seed(0)
    m = SDSLVisionNet(1, 10)
    memory = GradientSubspaceMemory(max_rank=3)
    solver = SDSLFlatMinimaxSolver(m, memory, replay_lr=1e-3, ascent_lr=1e-3,
                                   ascent_steps=1, max_perturb_norm=0.1)
    x = torch.randn(8, 1, 28, 28)
    y = torch.randint(0, 10, (8,))
    before = torch.cat([p.detach().flatten() for p in m.parameters()])
    stats = solver.step(x, y)
    after = torch.cat([p.detach().flatten() for p in m.parameters()])
    assert torch.norm(after - before) > 0
    assert stats.perturb_norm >= 0
    assert stats.worst_loss == stats.worst_loss
