from __future__ import annotations

from typing import Iterable, List

import torch
import torch.nn.functional as F


class ParameterVectorizer:

    def __init__(self, model):
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.sizes = [p.numel() for p in self.params]

    def parameters_vector(self) -> torch.Tensor:
        return torch.cat([p.detach().reshape(-1) for p in self.params])

    def gradients_vector(self, allow_none: bool = True) -> torch.Tensor:
        chunks = []
        for p in self.params:
            if p.grad is None:
                if not allow_none:
                    raise RuntimeError("missing gradient")
                chunks.append(torch.zeros_like(p).reshape(-1))
            else:
                chunks.append(p.grad.detach().reshape(-1))
        return torch.cat(chunks)

    @torch.no_grad()
    def set_parameters_vector(self, vector: torch.Tensor) -> None:
        offset = 0
        for p, n in zip(self.params, self.sizes):
            p.copy_(vector[offset:offset+n].view_as(p))
            offset += n
        if offset != vector.numel():
            raise ValueError("vector has incorrect length")

    def set_gradients_vector(self, vector: torch.Tensor) -> None:
        offset = 0
        for p, n in zip(self.params, self.sizes):
            g = vector[offset:offset+n].view_as(p)
            if p.grad is None:
                p.grad = g.clone()
            else:
                p.grad.copy_(g)
            offset += n


class GradientSubspaceMemory:
    
    def __init__(self, max_rank: int = 8, eps: float = 1e-8):
        if max_rank <= 0:
            raise ValueError("max_rank must be positive")
        self.max_rank = max_rank
        self.eps = eps
        self._basis: List[torch.Tensor] = []

    def __len__(self):
        return len(self._basis)

    def add(self, vector: torch.Tensor) -> bool:
        v = vector.detach().clone()
        if not torch.isfinite(v).all():
            return False
        for b in self._basis:
            v -= torch.dot(v, b) * b
        norm = torch.norm(v)
        if norm <= self.eps:
            return False
        v /= norm
        if len(self._basis) >= self.max_rank:
            self._basis.pop(0)
            # Re-orthogonalise against retained basis after eviction.
            for b in self._basis:
                v -= torch.dot(v, b) * b
            norm = torch.norm(v)
            if norm <= self.eps:
                return False
            v /= norm
        self._basis.append(v)
        return True

    def project(self, vector: torch.Tensor) -> torch.Tensor:
        if not self._basis:
            return torch.zeros_like(vector)
        result = torch.zeros_like(vector)
        for b in self._basis:
            result += torch.dot(vector, b) * b
        return result

    def orthogonal(self, vector: torch.Tensor) -> torch.Tensor:
        return vector - self.project(vector)

    def projection_fraction(self, vector: torch.Tensor) -> float:
        denom = torch.norm(vector).item() + self.eps
        return float(torch.norm(self.project(vector)).item() / denom)


def gradient_vector_on_loader(model, loader, device, vectorizer: ParameterVectorizer,
                              max_batches: int = 2) -> torch.Tensor:
    
    model.train()
    collected = []
    for batch_idx, (x, y) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        collected.append(vectorizer.gradients_vector())
    model.zero_grad(set_to_none=True)
    if not collected:
        return torch.zeros_like(vectorizer.parameters_vector())
    return torch.stack(collected).mean(0)
