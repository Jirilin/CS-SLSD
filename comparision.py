import argparse, copy, time
import pandas as pd
import torch
import torch.nn.functional as F
from torch import optim
import matplotlib.pyplot as plt

from models import build_model
from dataset_factory import ContinualStreamSimulator
from replay_buffer import ReplayBuffer
from ewc import EWC


def train_supervised(model, loader, optimizer, device, epochs=1, ewc=None, ewc_lambda=100.0):
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            if ewc is not None:
                loss = loss + ewc_lambda * ewc.penalty(model)
            loss.backward()
            optimizer.step()


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / max(total, 1)


def make_pseudo_labels(model, x, threshold, device):
    model.eval()
    with torch.no_grad():
        logits = model(x.to(device))
        probs = torch.softmax(logits, dim=1)
        conf, pseudo = probs.max(dim=1)
        mask = conf >= threshold
    return x[mask.cpu()], pseudo[mask].detach().cpu(), conf[mask].detach().cpu()


def train_on_tensors(model, x, y, optimizer, device, ewc=None, ewc_lambda=100.0):
    if len(x) == 0:
        return
    model.train()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    loss = F.cross_entropy(model(x), y)
    if ewc is not None:
        loss = loss + ewc_lambda * ewc.penalty(model)
    loss.backward()
    optimizer.step()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="MNIST", choices=["MNIST", "FashionMNIST", "KMNIST", "CIFAR10", "SVHN"])
    parser.add_argument("--distribution", default="class_incremental", choices=["class_incremental", "random_iid"])
    parser.add_argument("--initial", type=int, default=1000)
    parser.add_argument("--stream-samples", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--buffer-size", type=int, default=1000)
    parser.add_argument("--ewc-lambda", type=float, default=100.0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    sim = ContinualStreamSimulator(args.dataset, args.batch_size, args.initial, args.stream_samples, args.distribution)
    print("Stream setup:", sim.describe())

    init_loader = sim.get_initial_labeled_loader()
    test_loader = sim.get_test_loader()
    base_model = build_model(args.dataset).to(device)
    opt = optim.Adam(base_model.parameters(), lr=1e-3)
    print("Training initial supervised model...")
    train_supervised(base_model, init_loader, opt, device, epochs=2)
    print("Initial accuracy:", evaluate(base_model, test_loader, device))

    naive = copy.deepcopy(base_model)
    replay = copy.deepcopy(base_model)
    ewc_model = copy.deepcopy(base_model)
    opt_naive = optim.Adam(naive.parameters(), lr=5e-4)
    opt_replay = optim.Adam(replay.parameters(), lr=5e-4)
    opt_ewc = optim.Adam(ewc_model.parameters(), lr=5e-4)
    ewc_obj = EWC(ewc_model, init_loader, device=device)
    buffer = ReplayBuffer(capacity=args.buffer_size, device=device)

    records = []
    t0 = time.time()
    for step, (x_stream, true_y_hidden) in enumerate(sim.get_stream_loader(), start=1):
        if step > args.batches:
            break

        # Naive online: train only on confident pseudo-labels from the current batch.
        x_n, y_n, conf_n = make_pseudo_labels(naive, x_stream, args.threshold, device)
        train_on_tensors(naive, x_n, y_n, opt_naive, device)

        # Replay: train on current confident pseudo-labels + old samples from buffer.
        x_r, y_r, conf_r = make_pseudo_labels(replay, x_stream, args.threshold, device)
        if len(x_r) > 0:
            buffer.add(x_r, y_r)
            bx, by = buffer.sample(args.batch_size)
            if bx is not None:
                x_comb = torch.cat([x_r.to(device), bx], dim=0)
                y_comb = torch.cat([y_r.to(device), by], dim=0)
            else:
                x_comb, y_comb = x_r.to(device), y_r.to(device)
            train_on_tensors(replay, x_comb, y_comb, opt_replay, device)

        # EWC: train on current pseudo-labels, but penalise movement away from initial important weights.
        x_e, y_e, conf_e = make_pseudo_labels(ewc_model, x_stream, args.threshold, device)
        train_on_tensors(ewc_model, x_e, y_e, opt_ewc, device, ewc=ewc_obj, ewc_lambda=args.ewc_lambda)

        if step == 1 or step % 5 == 0 or step == args.batches:
            row = {
                "batch": step,
                "naive_acc": evaluate(naive, test_loader, device),
                "replay_acc": evaluate(replay, test_loader, device),
                "ewc_acc": evaluate(ewc_model, test_loader, device),
                "naive_pseudo_count": len(x_n),
                "replay_pseudo_count": len(x_r),
                "ewc_pseudo_count": len(x_e),
                "buffer_size": buffer.stats()["size"],
                "buffer_evicted_oldest": buffer.stats()["evicted_oldest"],
            }
            records.append(row)
            print(row)

    df = pd.DataFrame(records)
    csv_name = f"today_results_{args.dataset}_{args.distribution}.csv"
    png_name = f"today_comparison_{args.dataset}_{args.distribution}.png"
    df.to_csv(csv_name, index=False)

    plt.figure(figsize=(8,5))
    plt.plot(df["batch"], df["naive_acc"], marker="o", label="Naive Online")
    plt.plot(df["batch"], df["replay_acc"], marker="o", label="Replay Buffer")
    plt.plot(df["batch"], df["ewc_acc"], marker="o", label="EWC")
    plt.xlabel("Stream batch")
    plt.ylabel("Test accuracy (%)")
    plt.title(f"Continual learning comparison - {args.dataset} ({args.distribution})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(png_name, dpi=150)
    print(f"\nSaved: {csv_name}")
    print(f"Saved: {png_name}")
    print(f"Buffer final stats: {buffer.stats()}")
    print(f"Runtime: {time.time()-t0:.1f} seconds")

if __name__ == "__main__":
    main()
