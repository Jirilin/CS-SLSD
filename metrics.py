from __future__ import annotations
from collections import OrderedDict
import numpy as np
import torch


@torch.no_grad()
def accuracy(model, loader, device) -> float:
    model.eval(); correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += int((pred == y).sum())
        total += y.numel()
    return correct / max(total, 1)


@torch.no_grad()
def class_accuracy(model, loader, device, num_classes: int = 10) -> list[float]:
    model.eval()
    correct = np.zeros(num_classes, dtype=np.int64)
    total = np.zeros(num_classes, dtype=np.int64)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        for c in range(num_classes):
            mask = y == c
            total[c] += int(mask.sum())
            correct[c] += int((pred[mask] == y[mask]).sum())
    return [float(correct[c] / total[c]) if total[c] else float("nan") for c in range(num_classes)]


def snapshot(model) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict((n, p.detach().cpu().clone()) for n, p in model.named_parameters())


def parameter_change(previous, model, eps: float = 1e-12) -> float:
    num = den = 0.0
    for name, p in model.named_parameters():
        old = previous[name].to(p.device)
        num += float((p.detach() - old).pow(2).sum())
        den += float(old.pow(2).sum())
    return (num ** 0.5) / (den ** 0.5 + eps)


def forgetting_from_history(history: list[list[float]]) -> float:

    if len(history) < 2:
        return 0.0
    arr = np.asarray(history, dtype=float)
    best = np.nanmax(arr[:-1], axis=0)
    final = arr[-1]
    return float(np.nanmean(np.maximum(0.0, best - final)))


def task_accuracy_vector(model, task_loaders, device):
    return {task_id: accuracy(model, loader, device) for task_id, loader in task_loaders.items()}


def average_incremental_accuracy(task_history):
    if not task_history:
        return float("nan")
    values = [v for row in task_history for v in row.values()]
    return float(np.mean(values)) if values else float("nan")


def backward_transfer_proxy(task_history):
    
    if len(task_history) < 2:
        return float("nan")
    first, final = task_history[0], task_history[-1]
    return float(np.mean([final[k] - first[k] for k in first]))


def stability_plasticity_score(final_accuracy, forgetting, parameter_change):
    
    return float(final_accuracy - forgetting - 0.1 * parameter_change)
