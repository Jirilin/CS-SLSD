from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from typing import Optional
import torch.nn.functional as F


@dataclass
class PseudoLabelResult:
    accepted_images: torch.Tensor
    pseudo_labels: torch.Tensor
    coverage: float
    precision: float
    classifier_centroid_agreement: float


class CentroidRefinedPseudoLabeler:

    def __init__(self, model, device, num_classes: int, threshold: float = 0.90,
                 centroid_weight: float = 0.40, temperature: float = 1.0):
        self.model = model
        self.device = device
        self.num_classes = num_classes
        self.threshold = threshold
        self.centroid_weight = centroid_weight
        self.temperature = temperature
        self.centroids = None

    @torch.no_grad()
    def fit_reference_centroids(self, loader) -> None:
        self.model.eval()
        sums = None
        counts = torch.zeros(self.num_classes, device=self.device)
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            features = F.normalize(self.model.forward_features(x), dim=1)
            if sums is None:
                sums = torch.zeros(self.num_classes, features.size(1), device=self.device)
            for c in range(self.num_classes):
                mask = y == c
                if mask.any():
                    sums[c] += features[mask].sum(0)
                    counts[c] += mask.sum()
        if sums is None or (counts == 0).any():
            raise RuntimeError("Reference centroid fitting requires labelled samples for every class")
        self.centroids = F.normalize(sums / counts.unsqueeze(1), dim=1)

    @torch.no_grad()
    def generate(self, images: torch.Tensor, hidden_labels: Optional[torch.Tensor] = None) -> PseudoLabelResult:
        if self.centroids is None:
            raise RuntimeError("Call fit_reference_centroids() first")
        self.model.eval()
        x = images.to(self.device)
        logits = self.model(x)
        classifier_probs = torch.softmax(logits, dim=1)
        classifier_conf, classifier_pred = classifier_probs.max(1)

        features = F.normalize(self.model.forward_features(x), dim=1)
        similarities = features @ self.centroids.T
        centroid_probs = torch.softmax(similarities / self.temperature, dim=1)
        _, centroid_pred = centroid_probs.max(1)

        combined = (1.0 - self.centroid_weight) * classifier_probs + self.centroid_weight * centroid_probs
        combined_conf, combined_pred = combined.max(1)
        agreement_mask = classifier_pred == centroid_pred
        mask = agreement_mask & (combined_conf >= self.threshold)
        accepted_x = x[mask]
        accepted_y = combined_pred[mask]
        coverage = float(mask.float().mean().item())
        agreement = float(agreement_mask.float().mean().item())

        precision = float("nan")
        if hidden_labels is not None and mask.any():
            truth = hidden_labels.to(self.device)
            precision = float((accepted_y == truth[mask]).float().mean().item())
        return PseudoLabelResult(accepted_x, accepted_y, coverage, precision, agreement)
