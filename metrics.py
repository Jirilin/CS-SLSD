import torch


@torch.no_grad()
def accuracy(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        pred = model(images).argmax(dim=1)
        correct += pred.eq(labels).sum().item()
        total += labels.numel()
    return correct / total


def parameter_change(previous, model) -> float:
    numerator = denominator = 0.0
    with torch.no_grad():
        for name, param in model.named_parameters():
            old = previous[name].to(param.device)
            numerator += torch.sum((param - old) ** 2).item()
            denominator += torch.sum(old ** 2).item()
    return (numerator ** 0.5) / (denominator ** 0.5 + 1e-12)


def snapshot(model):
    return {name: param.detach().cpu().clone() for name, param in model.named_parameters()}
