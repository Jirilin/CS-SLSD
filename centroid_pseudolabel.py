from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


@dataclass
class PseudoLabelOutput:
    accepted_images: torch.Tensor
    pseudo_labels: torch.Tensor
    mask: torch.Tensor
    confidence: torch.Tensor
    precision: float
    coverage: float
    classifier_centroid_agreement: float


class CentroidRefinedPseudoLabeler:
    """
    Combines classifier probabilities with distances to trusted class centroids.

    Invariant semantics are represented by centroids computed only from the
    original labelled set. They stay fixed during the streaming experiment.
    """

    def __init__(self, model, device, confidence_threshold=0.9,
                 centroid_weight=0.4, temperature=1.0):
        self.model = model
        self.device = device
        self.threshold = confidence_threshold
        self.centroid_weight = centroid_weight
        self.temperature = temperature
        self.centroids = None

    @torch.no_grad()
    def fit_reference_centroids(self, labelled_loader: DataLoader, num_classes=10):
        self.model.eval()
        sums = None
        counts = torch.zeros(num_classes, device=self.device)
        for images, labels in labelled_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            features = F.normalize(self.model.forward_features(images), dim=1)
            if sums is None:
                sums = torch.zeros(num_classes, features.size(1), device=self.device)
            for c in range(num_classes):
                mask = labels == c
                if mask.any():
                    sums[c] += features[mask].sum(0)
                    counts[c] += mask.sum()
        if (counts == 0).any():
            raise RuntimeError("Every class needs at least one labelled sample.")
        self.centroids = F.normalize(sums / counts.unsqueeze(1), dim=1)

    @torch.no_grad()
    def generate(self, images: torch.Tensor, hidden_true_labels: torch.Tensor | None = None):
        if self.centroids is None:
            raise RuntimeError("Call fit_reference_centroids() first.")
        self.model.eval()
        images_device = images.to(self.device)
        logits = self.model(images_device)
        classifier_probs = F.softmax(logits, dim=1)
        classifier_conf, classifier_pred = classifier_probs.max(dim=1)

        features = F.normalize(self.model.forward_features(images_device), dim=1)
        cosine_similarity = features @ self.centroids.t()
        centroid_probs = F.softmax(cosine_similarity / self.temperature, dim=1)
        centroid_pred = centroid_probs.argmax(dim=1)

        combined = (1.0 - self.centroid_weight) * classifier_probs + self.centroid_weight * centroid_probs
        combined_conf, combined_pred = combined.max(dim=1)
        agreement = classifier_pred.eq(centroid_pred)
        mask = agreement & combined_conf.ge(self.threshold)

        coverage = mask.float().mean().item()
        agreement_rate = agreement.float().mean().item()
        precision = float("nan")
        if hidden_true_labels is not None and mask.any():
            true = hidden_true_labels.to(self.device)
            precision = combined_pred[mask].eq(true[mask]).float().mean().item()

        return PseudoLabelOutput(
            accepted_images=images_device[mask],
            pseudo_labels=combined_pred[mask],
            mask=mask,
            confidence=combined_conf,
            precision=precision,
            coverage=coverage,
            classifier_centroid_agreement=agreement_rate,
        )
