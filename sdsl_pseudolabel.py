from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class SDSLPseudoLabelResult:
    images: torch.Tensor
    labels: torch.Tensor
    confidence: torch.Tensor
    coverage: float
    precision: float
    initial_precision: float
    cluster_precision: float
    semantic_precision: float
    classifier_cluster_agreement: float
    semantic_shift: float
    accepted_count: int


class SDSLRobustPseudoLabeler:

    def __init__(
        self,
        model,
        device: torch.device,
        num_classes: int,
        semantic_rank: int = 5,
        semantic_blend: float = 0.70,
        centroid_iterations: int = 5,
        temperature: float = 0.20,
        classifier_weight: float = 0.35,
        threshold: float = 0.0,
        min_accept_fraction: float = 0.25,
    ) -> None:
        if not 1 <= semantic_rank <= num_classes:
            raise ValueError("semantic_rank must be between 1 and num_classes")
        if not 0.0 <= semantic_blend <= 1.0:
            raise ValueError("semantic_blend must be in [0,1]")
        if not 0.0 <= classifier_weight <= 1.0:
            raise ValueError("classifier_weight must be in [0,1]")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0,1]")
        if not 0.0 <= min_accept_fraction <= 1.0:
            raise ValueError("min_accept_fraction must be in [0,1]")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.model = model
        self.device = device
        self.num_classes = num_classes
        self.semantic_rank = semantic_rank
        self.semantic_blend = semantic_blend
        self.centroid_iterations = centroid_iterations
        self.temperature = temperature
        self.classifier_weight = classifier_weight
        self.threshold = threshold
        self.min_accept_fraction = min_accept_fraction
        self.gold_centroids: Optional[torch.Tensor] = None
        self.label_embedding: Optional[torch.Tensor] = None

    @torch.no_grad()
    def fit_invariant_semantics(self, loader) -> None:
        self.model.eval()
        sums = None
        counts = torch.zeros(self.num_classes, device=self.device)
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            z = F.normalize(self.model.forward_features(x), dim=1)
            if sums is None:
                sums = torch.zeros(self.num_classes, z.size(1), device=self.device)
            for c in range(self.num_classes):
                mask = y == c
                if mask.any():
                    sums[c] += z[mask].sum(0)
                    counts[c] += mask.sum()
        if sums is None or (counts == 0).any():
            raise RuntimeError("trusted initial set must contain every class")
        u0 = F.normalize(sums / counts[:, None], dim=1)  # C x D
        self.gold_centroids = u0

        # U0^T = P S V^T. Rows of V^T are invariant class-semantic coordinates.
        # Keeping rank < C captures stable class relationships rather than an identity map.
        _, _, vh = torch.linalg.svd(u0.T, full_matrices=False)
        rank = min(self.semantic_rank, vh.size(0))
        self.label_embedding = vh[:rank].contiguous()  # r x C

    @staticmethod
    def _precision(pred: torch.Tensor, truth: Optional[torch.Tensor]) -> float:
        if truth is None or pred.numel() == 0:
            return float("nan")
        return float((pred == truth).float().mean().item())

    def _soft_initial_centroids(self, z: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        # Paper Eq. (3): probability-weighted current-stream centroid initialisation.
        weights = probs.T  # C x N
        denom = weights.sum(1, keepdim=True).clamp_min(1e-8)
        centroids = (weights @ z) / denom
        return F.normalize(centroids, dim=1)

    def _cluster(self, z: torch.Tensor, centroids: torch.Tensor):
        labels = None
        for _ in range(self.centroid_iterations):
            similarities = z @ centroids.T
            new_labels = similarities.argmax(1)
            new_centroids = centroids.clone()
            for c in range(self.num_classes):
                mask = new_labels == c
                if mask.any():
                    new_centroids[c] = F.normalize(z[mask].mean(0), dim=0)
            new_centroids = F.normalize(new_centroids, dim=1)
            if labels is not None and torch.equal(new_labels, labels):
                centroids = new_centroids
                labels = new_labels
                break
            labels = new_labels
            centroids = new_centroids
        return labels, centroids

    def _semantic_reconstruct(self, current_centroids: torch.Tensor) -> torch.Tensor:
        if self.label_embedding is None:
            raise RuntimeError("call fit_invariant_semantics() first")
        v = self.label_embedding  # r x C
        u = current_centroids.T   # D x C
        gram_inv = torch.linalg.pinv(v @ v.T)
        h_t = u @ v.T @ gram_inv  # D x r
        reconstructed = (h_t @ v).T  # C x D
        reconstructed = F.normalize(reconstructed, dim=1)
        blended = F.normalize(
            (1.0 - self.semantic_blend) * current_centroids
            + self.semantic_blend * reconstructed,
            dim=1,
        )
        return blended

    @torch.no_grad()
    def generate(self, images: torch.Tensor, hidden_labels: Optional[torch.Tensor] = None):
        if self.gold_centroids is None or self.label_embedding is None:
            raise RuntimeError("call fit_invariant_semantics() before generate()")

        self.model.eval()
        x = images.to(self.device)
        truth = hidden_labels.to(self.device) if hidden_labels is not None else None
        raw = self.model.forward_features(x)
        z = F.normalize(raw, dim=1)
        logits = self.model.classifier(raw)
        p_cls = torch.softmax(logits, dim=1)
        initial_pred = p_cls.argmax(1)

        # Stage 2: current-stream unsupervised structure.
        centroids0 = self._soft_initial_centroids(z, p_cls)
        cluster_pred, centroids = self._cluster(z, centroids0)

        # Stage 3: invariant label semantics learned only from gold data at t=0.
        semantic_centroids = self._semantic_reconstruct(centroids)
        semantic_logits = (z @ semantic_centroids.T) / self.temperature
        p_sem = torch.softmax(semantic_logits, dim=1)
        semantic_pred = p_sem.argmax(1)

        # Final pseudo-label distribution: primarily semantics, with classifier support.
        p_final = self.classifier_weight * p_cls + (1.0 - self.classifier_weight) * p_sem
        conf, final_pred = p_final.max(1)

        mask = conf >= self.threshold
        # Avoid a dead stream when a conservative threshold is used: retain the most
        # confident fraction, but never use hidden ground truth for this decision.
        minimum = int(round(self.min_accept_fraction * len(x)))
        if mask.sum().item() < minimum and minimum > 0:
            top = torch.topk(conf, k=min(minimum, len(x))).indices
            mask = torch.zeros_like(mask)
            mask[top] = True

        accepted_x = x[mask]
        accepted_y = final_pred[mask]
        accepted_conf = conf[mask]
        coverage = float(mask.float().mean().item()) if len(x) else 0.0
        precision = self._precision(accepted_y, truth[mask] if truth is not None else None)
        agreement = float((initial_pred == cluster_pred).float().mean().item())
        semantic_shift = float(torch.norm(semantic_centroids - centroids, p=2, dim=1).mean().item())

        return SDSLPseudoLabelResult(
            images=accepted_x,
            labels=accepted_y,
            confidence=accepted_conf,
            coverage=coverage,
            precision=precision,
            initial_precision=self._precision(initial_pred, truth),
            cluster_precision=self._precision(cluster_pred, truth),
            semantic_precision=self._precision(semantic_pred, truth),
            classifier_cluster_agreement=agreement,
            semantic_shift=semantic_shift,
            accepted_count=int(mask.sum().item()),
        )
