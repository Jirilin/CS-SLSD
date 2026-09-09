from dataclasses import dataclass


@dataclass(frozen=True)
class SDSLConfig:
    dataset: str = "mnist"
    seed: int = 0
    data_root: str = "./data"
    initial_per_class: int = 100
    stream_batches: int = 20
    stream_batch_size: int = 256
    dominant_fraction: float = 0.70
    train_batch_size: int = 64
    test_batch_size: int = 256
    initial_epochs: int = 5
    generation_warmup_epochs: int = 1
    replay_epochs: int = 1
    initial_lr: float = 1e-3
    warmup_lr: float = 2e-4
    replay_lr: float = 5e-4
    semantic_rank: int = 5
    semantic_blend: float = 0.70
    centroid_iterations: int = 5
    centroid_temperature: float = 0.20
    classifier_weight: float = 0.35
    pseudo_threshold: float = 0.0
    min_accept_fraction: float = 0.25
    lookback_limit: int = 256
    subspace_rank: int = 8
    subspace_batches: int = 2
    ascent_steps: int = 2
    ascent_lr: float = 0.02
    max_perturb_norm: float = 0.25
    gradient_clip: float = 5.0
    weight_decay: float = 1e-4
    device: str = "auto"
