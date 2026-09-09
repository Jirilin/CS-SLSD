from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from gradient_subspace import ParameterVectorizer, GradientSubspaceMemory


@dataclass
class FlatStepStats:
    clean_loss: float
    worst_loss: float
    sharpness_gap: float
    perturb_norm: float
    projected_gradient_fraction: float
    update_norm: float


class SDSLFlatMinimaxSolver:
    
    def __init__(self, model, memory: GradientSubspaceMemory, replay_lr: float = 5e-4,
                 ascent_lr: float = 0.02, ascent_steps: int = 2,
                 max_perturb_norm: float = 0.25, gradient_clip: float = 5.0,
                 weight_decay: float = 1e-4):
        if replay_lr <= 0 or ascent_lr <= 0:
            raise ValueError("learning rates must be positive")
        if ascent_steps < 1:
            raise ValueError("ascent_steps must be >= 1")
        self.model = model
        self.memory = memory
        self.vectorizer = ParameterVectorizer(model)
        self.replay_lr = replay_lr
        self.ascent_lr = ascent_lr
        self.ascent_steps = ascent_steps
        self.max_perturb_norm = max_perturb_norm
        self.gradient_clip = gradient_clip
        self.weight_decay = weight_decay

    def _loss_and_grad(self, x, y):
        self.model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(self.model(x), y)
        loss.backward()
        g = self.vectorizer.gradients_vector()
        return loss, g

    @torch.no_grad()
    def _clip_vector(self, v: torch.Tensor, max_norm: float) -> torch.Tensor:
        norm = torch.norm(v)
        if max_norm > 0 and norm > max_norm:
            v = v * (max_norm / (norm + 1e-12))
        return v

    def step(self, x, y) -> FlatStepStats:
        self.model.train()
        theta = self.vectorizer.parameters_vector()

        # Clean loss is diagnostic only.
        clean_loss, _ = self._loss_and_grad(x, y)
        clean_value = float(clean_loss.detach().item())
        self.model.zero_grad(set_to_none=True)

        xi = torch.zeros_like(theta)
        for _ in range(self.ascent_steps):
            with torch.no_grad():
                self.vectorizer.set_parameters_vector(theta + xi)
            _, g = self._loss_and_grad(x, y)
            ascent = self.memory.project(g)
            # When no old subspace exists yet, use the full gradient for the first
            # flatness search; from the next state onwards projection is active.
            if len(self.memory) == 0:
                ascent = g
            xi = xi + self.ascent_lr * ascent
            xi = self._clip_vector(xi, self.max_perturb_norm)

        with torch.no_grad():
            self.vectorizer.set_parameters_vector(theta + xi)
        worst_loss, g_worst = self._loss_and_grad(x, y)
        worst_value = float(worst_loss.detach().item())

        # Restore theta before the outer minimisation.
        with torch.no_grad():
            self.vectorizer.set_parameters_vector(theta)

        projected_fraction = self.memory.projection_fraction(g_worst)
        if len(self.memory) > 0:
            update_grad = self.memory.orthogonal(g_worst)
        else:
            update_grad = g_worst

        if self.weight_decay > 0:
            update_grad = update_grad + self.weight_decay * theta
        update_grad = self._clip_vector(update_grad, self.gradient_clip)
        delta = -self.replay_lr * update_grad
        with torch.no_grad():
            self.vectorizer.set_parameters_vector(theta + delta)
        self.model.zero_grad(set_to_none=True)

        return FlatStepStats(
            clean_loss=clean_value,
            worst_loss=worst_value,
            sharpness_gap=worst_value - clean_value,
            perturb_norm=float(torch.norm(xi).item()),
            projected_gradient_fraction=projected_fraction,
            update_norm=float(torch.norm(delta).item()),
        )
