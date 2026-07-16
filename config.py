from dataclasses import dataclass

@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 0
    data_root: str = "./data"
    initial_per_class: int = 100
    stream_batches: int = 20
    stream_batch_size: int = 256
    dominant_fraction: float = 0.70
    train_batch_size: int = 64
    test_batch_size: int = 256
    initial_epochs: int = 5
    online_epochs: int = 1
    learning_rate: float = 1e-3
    confidence_threshold: float = 0.90
    centroid_weight: float = 0.40
    centroid_temperature: float = 1.0
    ewc_lambda: float = 50.0
    fisher_samples: int = 1000
    online_ewc_gamma: float = 0.95
    device: str = "auto"
