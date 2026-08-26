from __future__ import annotations
from typing import Dict
import torch
import torch.nn.functional as F


class OnlineEWC:

    def __init__(self, model, device, strength: float = 50.0, gamma: float = 0.95):
        self.model = model
        self.device = device
        self.strength = strength
        self.gamma = gamma
        self.fisher: Dict[str, torch.Tensor] = {}
        self.reference: Dict[str, torch.Tensor] = {}

    def consolidate(self, loader, max_samples: int, use_true_labels: bool = True) -> None:
        self.model.eval()
        current = {
            name: torch.zeros_like(param, device=self.device)
            for name, param in self.model.named_parameters() if param.requires_grad
        }
        processed = 0
        for x, y in loader:
            for i in range(x.size(0)):
                if processed >= max_samples:
                    break
                xi = x[i:i+1].to(self.device)
                yi = y[i:i+1].to(self.device)
                self.model.zero_grad(set_to_none=True)
                logits = self.model(xi)
                if not use_true_labels:
                    yi = logits.detach().argmax(1)
                log_prob = F.log_softmax(logits, dim=1)[0, yi.item()]
                (-log_prob).backward()
                for name, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        current[name] += param.grad.detach().pow(2)
                processed += 1
            if processed >= max_samples:
                break
        if processed == 0:
            raise RuntimeError("Fisher estimation received no samples")
        for name in current:
            current[name] /= float(processed)
        if self.fisher:
            self.fisher = {
                name: self.gamma * self.fisher[name] + current[name]
                for name in current
            }
        else:
            self.fisher = current
        self.reference = {
            name: param.detach().clone()
            for name, param in self.model.named_parameters() if param.requires_grad
        }

    def penalty(self) -> torch.Tensor:
        if not self.fisher:
            return torch.zeros((), device=self.device)
        total = torch.zeros((), device=self.device)
        for name, param in self.model.named_parameters():
            if name in self.fisher:
                total += (self.fisher[name] * (param - self.reference[name]).pow(2)).sum()
        return 0.5 * self.strength * total

    def fisher_summary(self):
        if not self.fisher:
            return {"fisher_mean": float("nan"), "fisher_max": float("nan")}
        values = torch.cat([v.detach().flatten().cpu() for v in self.fisher.values()])
        return {"fisher_mean": float(values.mean()), "fisher_max": float(values.max())}
