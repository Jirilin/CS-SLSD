from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class PseudoLabelResult:
    
    accepted_images: torch.Tensor
    pseudo_labels: torch.Tensor
    coverage: float
    precision: float
    classifier_centroid_agreement: float
    accepted_count: int
    rejected_count: int
    mean_accepted_confidence: float


class CentroidRefinedPseudoLabeler:
    
    def __init__(
        self,
        model,
        device: torch.device,
        num_classes: int,
        threshold: float = 0.90,
        centroid_weight: float = 0.40,
        temperature: float = 1.0,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        if not 0.0 <= centroid_weight <= 1.0:
            raise ValueError("centroid_weight must be between 0 and 1")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.model = model
        self.device = device
        self.num_classes = num_classes
        self.threshold = threshold
        self.centroid_weight = centroid_weight
        self.temperature = temperature
        self.centroids: Optional[torch.Tensor] = None

    @torch.no_grad()
    def fit_reference_centroids(self, loader) -> None:
        
        self.model.eval()
        sums = None
        counts = torch.zeros(self.num_classes, device=self.device)

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            features = F.normalize(self.model.forward_features(images), dim=1)

            if sums is None:
                sums = torch.zeros(
                    self.num_classes,
                    features.size(1),
                    device=self.device,
                )

            for class_id in range(self.num_classes):
                class_mask = labels == class_id
                if class_mask.any():
                    sums[class_id] += features[class_mask].sum(dim=0)
                    counts[class_id] += class_mask.sum()

        if sums is None or (counts == 0).any():
            missing = torch.where(counts == 0)[0].tolist()
            raise RuntimeError(
                
                f"class; missing classes: {missing}"
            )

        self.centroids = F.normalize(sums / counts.unsqueeze(1), dim=1)

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        hidden_labels: Optional[torch.Tensor] = None,
    ) -> PseudoLabelResult:
        
        if self.centroids is None:
            raise RuntimeError("Call fit_reference_centroids() before generate().")

        self.model.eval()
        x = images.to(self.device)

        # One feature extraction pass is enough for both classifier and centroid
        # decisions. This is cleaner and faster than forwarding through the CNN
        # twice.
        raw_features = self.model.forward_features(x)
        logits = self.model.classifier(raw_features)
        classifier_probs = torch.softmax(logits, dim=1)
        _, classifier_pred = classifier_probs.max(dim=1)

        features = F.normalize(raw_features, dim=1)
        similarities = features @ self.centroids.T
        centroid_probs = torch.softmax(similarities / self.temperature, dim=1)
        _, centroid_pred = centroid_probs.max(dim=1)

        combined_probs = (
            (1.0 - self.centroid_weight) * classifier_probs
            + self.centroid_weight * centroid_probs
        )
        combined_conf, combined_pred = combined_probs.max(dim=1)

        agreement_mask = classifier_pred == centroid_pred
        acceptance_mask = agreement_mask & (combined_conf >= self.threshold)

        accepted_x = x[acceptance_mask]
        accepted_y = combined_pred[acceptance_mask]
        accepted_count = int(acceptance_mask.sum().item())
        rejected_count = int(len(x) - accepted_count)

        coverage = float(acceptance_mask.float().mean().item()) if len(x) else 0.0
        agreement = float(agreement_mask.float().mean().item()) if len(x) else 0.0
        mean_confidence = (
            float(combined_conf[acceptance_mask].mean().item())
            if acceptance_mask.any()
            else float("nan")
        )

        precision = float("nan")
        if hidden_labels is not None and acceptance_mask.any():
            truth = hidden_labels.to(self.device)
            precision = float(
                (accepted_y == truth[acceptance_mask]).float().mean().item()
            )

        return PseudoLabelResult(
            accepted_images=accepted_x,
            pseudo_labels=accepted_y,
            coverage=coverage,
            precision=precision,
            classifier_centroid_agreement=agreement,
            accepted_count=accepted_count,
            rejected_count=rejected_count,
            mean_accepted_confidence=mean_confidence,
        )
