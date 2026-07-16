from __future__ import annotations
from typing import Dict
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


class OnlineEWC:
    """
    Diagonal online EWC with empirical Fisher estimation.

    Fisher is averaged over individual samples (up to max_samples). The model is
    placed in evaluation mode so dropout does not add noise. Fisher values can
    be accumulated online: F_total <- gamma * F_old + F_current.
    """

    def __init__(self, model, device, ewc_lambda=50.0, gamma=0.95):
        self.model = model
        self.device = device
        self.ewc_lambda = ewc_lambda
        self.gamma = gamma
        self.fisher: Dict[str, torch.Tensor] = {}
        self.reference: Dict[str, torch.Tensor] = {}

    def estimate_fisher(self, loader: DataLoader, max_samples: int = 1000,
                        use_true_labels: bool = True) -> Dict[str, torch.Tensor]:
        self.model.eval()
        fisher = {n: torch.zeros_like(p, device=self.device)
                  for n, p in self.model.named_parameters() if p.requires_grad}
        seen = 0

        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            for i in range(images.size(0)):
                if seen >= max_samples:
                    break
                self.model.zero_grad(set_to_none=True)
                logits = self.model(images[i:i+1])
                target = labels[i:i+1] if use_true_labels else logits.argmax(dim=1)
                log_likelihood = -F.cross_entropy(logits, target, reduction="sum")
                log_likelihood.backward()
                for name, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        fisher[name] += param.grad.detach().pow(2)
                seen += 1
            if seen >= max_samples:
                break

        if seen == 0:
            raise RuntimeError("Cannot estimate Fisher from an empty loader.")
        for name in fisher:
            fisher[name] /= float(seen)
        return fisher

    def consolidate(self, loader: DataLoader, max_samples: int = 1000,
                    use_true_labels: bool = True) -> None:
        current = self.estimate_fisher(loader, max_samples, use_true_labels)
        if not self.fisher:
            self.fisher = current
        else:
            self.fisher = {name: self.gamma * self.fisher[name] + current[name]
                           for name in current}
        self.reference = {name: param.detach().clone()
                          for name, param in self.model.named_parameters() if param.requires_grad}

    def penalty(self) -> torch.Tensor:
        if not self.fisher:
            return torch.zeros((), device=self.device)
        total = torch.zeros((), device=self.device)
        for name, param in self.model.named_parameters():
            if name in self.fisher:
                total += (self.fisher[name] * (param - self.reference[name]).pow(2)).sum()
        return 0.5 * self.ewc_lambda * total

    def fisher_summary(self):
        if not self.fisher:
            return {"fisher_mean": 0.0, "fisher_max": 0.0}
        values = torch.cat([v.flatten() for v in self.fisher.values()])
        return {"fisher_mean": values.mean().item(), "fisher_max": values.max().item()}
