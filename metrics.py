from __future__ import annotations
import numpy as np
import torch


@torch.no_grad()
def accuracy(model, loader, device) -> float:
    model.eval(); correct = 0; total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += int((pred == y).sum())
        total += y.numel()
    return correct / max(total, 1)


@torch.no_grad()
def class_accuracy(model, loader, device, num_classes: int = 10):
    model.eval(); correct = np.zeros(num_classes); total = np.zeros(num_classes)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        for c in range(num_classes):
            m = y == c
            total[c] += int(m.sum())
            correct[c] += int((pred[m] == y[m]).sum())
    return np.divide(correct, total, out=np.zeros_like(correct), where=total > 0)


def task_accuracy_vector(model, task_loaders, device):
    return [accuracy(model, task_loaders[k], device) for k in sorted(task_loaders)]


def snapshot(model):
    return {k: v.detach().clone().cpu() for k, v in model.state_dict().items() if torch.is_floating_point(v)}


def parameter_change(previous, model) -> float:
    numerator = 0.0; denominator = 0.0
    for name, tensor in model.state_dict().items():
        if name in previous and torch.is_floating_point(tensor):
            now = tensor.detach().cpu()
            numerator += float((now - previous[name]).pow(2).sum())
            denominator += float(previous[name].pow(2).sum())
    return (numerator ** 0.5) / (denominator ** 0.5 + 1e-12)


def forgetting_from_history(history) -> float:
    arr = np.asarray(history, dtype=float)
    if arr.shape[0] < 2:
        return 0.0
    best = np.nanmax(arr[:-1], axis=0)
    final = arr[-1]
    return float(np.nanmean(np.maximum(best - final, 0.0)))


def average_incremental_accuracy(task_history) -> float:
    arr = np.asarray(task_history, dtype=float)
    return float(np.nanmean(arr))


def backward_transfer_proxy(task_history) -> float:
    arr = np.asarray(task_history, dtype=float)
    if arr.shape[0] < 2:
        return 0.0
    return float(np.nanmean(arr[-1] - arr[0]))
