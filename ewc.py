from __future__ import annotations
from collections import OrderedDict
import torch
import torch.nn.functional as F


class OnlineEWC:
    def __init__(self, model, device, strength: float = 50.0, gamma: float = 0.95):
        self.model = model
        self.device = device
        self.strength = float(strength)
        self.gamma = float(gamma)
        self.means: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.fisher: OrderedDict[str, torch.Tensor] = OrderedDict()

    def _trainable(self):
        return [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]

    def estimate_fisher(self, loader, max_samples: int = 1000,
                        use_true_labels: bool = True) -> OrderedDict[str, torch.Tensor]:
        self.model.eval()
        fisher = OrderedDict((n, torch.zeros_like(p, device=self.device)) for n, p in self._trainable())
        processed = 0
        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            for i in range(images.size(0)):
                if processed >= max_samples:
                    break
                self.model.zero_grad(set_to_none=True)
                logits = self.model(images[i:i+1])
                target = labels[i:i+1] if use_true_labels else logits.argmax(dim=1)
                log_likelihood = F.log_softmax(logits, dim=1).gather(1, target.view(-1, 1)).mean()
                log_likelihood.backward()
                for name, parameter in self._trainable():
                    if parameter.grad is not None:
                        fisher[name] += parameter.grad.detach().pow(2)
                processed += 1
            if processed >= max_samples:
                break
        if processed == 0:
            raise RuntimeError("No samples were available for Fisher estimation")
        for name in fisher:
            fisher[name] /= processed
        return fisher

    def consolidate(self, loader, max_samples: int = 1000,
                    use_true_labels: bool = True) -> None:
        latest = self.estimate_fisher(loader, max_samples, use_true_labels)
        current_means = OrderedDict((n, p.detach().clone()) for n, p in self._trainable())
        if not self.fisher:
            self.fisher = latest
        else:
            self.fisher = OrderedDict(
                (n, self.gamma * self.fisher[n] + latest[n]) for n in latest
            )
        self.means = current_means

    def penalty(self) -> torch.Tensor:
        if not self.fisher:
            return torch.zeros((), device=self.device)
        loss = torch.zeros((), device=self.device)
        params = dict(self._trainable())
        for name, importance in self.fisher.items():
            loss += (importance * (params[name] - self.means[name]).pow(2)).sum()
        return 0.5 * self.strength * loss

    def fisher_summary(self) -> dict[str, float]:
        if not self.fisher:
            return {"fisher_mean": 0.0, "fisher_max": 0.0, "fisher_nonzero_fraction": 0.0}
        flat = torch.cat([v.flatten() for v in self.fisher.values()])
        return {
            "fisher_mean": float(flat.mean().item()),
            "fisher_max": float(flat.max().item()),
            "fisher_nonzero_fraction": float((flat > 0).float().mean().item()),
        }
