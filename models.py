from __future__ import annotations

from sdsl_model import SDSLVisionNet


class VisionCNN(SDSLVisionNet):
    """Alias of the final SDSL backbone for fair baseline comparisons."""

    def __init__(self, in_channels: int, num_classes: int = 10, feature_dim: int = 128):
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            feature_dim=feature_dim,
        )
