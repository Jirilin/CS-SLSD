from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def add_box(ax, x, y, w, h, text):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fill=False)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->", mutation_scale=12))


def main():
    out = Path("results/final_figures")
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7)
    ax.axis("off")

    boxes = [
        (0.4, 5.3, 2.1, 0.8, "Initial trusted\nlabelled set"),
        (3.1, 5.3, 2.1, 0.8, "Train base CNN +\nreference centroids"),
        (6.0, 5.3, 2.1, 0.8, "Incoming unlabelled\nstream batch"),
        (8.6, 5.3, 2.0, 0.8, "Measure distribution\nchange"),
        (5.9, 3.4, 2.3, 0.9, "Pseudo-label generator\nconfidence + centroid"),
        (8.6, 3.4, 2.0, 0.9, "Accept / reject\nstream samples"),
        (5.8, 1.4, 2.5, 0.9, "Continual update\nReplay + Online EWC"),
        (2.7, 1.4, 2.2, 0.9, "Evaluation\naccuracy / forgetting"),
        (0.2, 1.4, 1.8, 0.9, "Next stream\nbatch"),
    ]
    for b in boxes:
        add_box(ax, *b)

    arrow(ax, 2.5, 5.7, 3.1, 5.7)
    arrow(ax, 5.2, 5.7, 6.0, 5.7)
    arrow(ax, 8.1, 5.7, 8.6, 5.7)
    arrow(ax, 7.0, 5.3, 7.0, 4.3)
    arrow(ax, 8.2, 3.85, 8.6, 3.85)
    arrow(ax, 9.6, 3.4, 7.3, 2.3)
    arrow(ax, 5.8, 1.85, 4.9, 1.85)
    arrow(ax, 2.7, 1.85, 2.0, 1.85)
    arrow(ax, 1.1, 2.3, 6.5, 5.3)

    ax.set_title("Continual Semi-Supervised Learning from Streaming Data — Experimental Pipeline", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "pipeline_diagram.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(out / "pipeline_diagram.png")


if __name__ == "__main__":
    main()
