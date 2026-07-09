# Elastic Weight Consolidation baseline.
import torch
import torch.nn.functional as F

class EWC:
    def __init__(self, model, dataloader, device="cpu", fisher_batches=10):
        self.device = device
        self.params = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        self.fisher = {n: torch.zeros_like(p, device=device) for n, p in model.named_parameters() if p.requires_grad}
        model.eval()
        count = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            model.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    self.fisher[n] += p.grad.detach() ** 2
            count += 1
            if count >= fisher_batches:
                break
        for n in self.fisher:
            self.fisher[n] /= max(count, 1)

    def penalty(self, model):
        loss = 0.0
        for n, p in model.named_parameters():
            if p.requires_grad:
                loss += (self.fisher[n] * (p - self.params[n]) ** 2).sum()
        return loss
