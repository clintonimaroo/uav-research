from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUTPUT_DIR = Path("navigation/comparison_results/paper1_fire_density_final/figures")


def box(ax, xy, width, height, label, face, edge="#333333", fontsize=12, weight="bold"):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.03,rounding_size=0.035",
        facecolor=face,
        edgecolor=edge,
        linewidth=2.0,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        weight=weight,
        wrap=True,
    )
    return patch


def arrow(ax, start, end, label=None, rad=0.0, linestyle="-", color="#111111"):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=2.2,
        color=color,
        linestyle=linestyle,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(patch)
    if label:
        ax.text(
            (start[0] + end[0]) / 2,
            (start[1] + end[1]) / 2 + 0.035,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
            color="#222222",
        )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(15, 7.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.93, "System Architecture", ha="center", va="center", fontsize=30, weight="bold")

    box(ax, (0.04, 0.55), 0.15, 0.13, "UAV Camera /\nAerial Imagery", "#EFE7DC")
    box(ax, (0.25, 0.56), 0.16, 0.11, "Image\nPreprocessing\nresize / normalize / tile", "#EFE7DC", fontsize=10)
    box(ax, (0.47, 0.56), 0.16, 0.11, "EfficientNet-B0\nCNN\n5-class classifier", "#6C5840", edge="#6C5840", fontsize=11, weight="bold")
    box(ax, (0.47, 0.33), 0.23, 0.13, "Hazard Localization /\nRisk Map Generation\nprobability + confidence", "#FFFFFF", edge="#6C5840", fontsize=11)
    box(ax, (0.78, 0.48), 0.17, 0.22, "Navigation\nPPO Path Planning", "#D8E6F3", edge="#111111", fontsize=12)
    box(ax, (0.78, 0.25), 0.17, 0.12, "Routing Decision\nNext Safe Move", "#EFE7DC", fontsize=11)
    box(ax, (0.39, 0.10), 0.22, 0.10, "UAV Execution\nApply Movement Command", "#FFFFFF", edge="#111111", fontsize=12)
    box(ax, (0.08, 0.18), 0.18, 0.08, "Next Camera Frame\nnew sensor input", "#FFFFFF", edge="#777777", fontsize=10, weight="normal")

    ax.text(
        0.63,
        0.64,
        "Detected classes:\nfire\ncollapsed building\nflooded areas\ntraffic incident\nnormal",
        ha="left",
        va="center",
        fontsize=11,
        linespacing=1.35,
    )

    arrow(ax, (0.19, 0.615), (0.25, 0.615))
    arrow(ax, (0.41, 0.615), (0.47, 0.615))
    arrow(ax, (0.55, 0.56), (0.55, 0.46))
    arrow(ax, (0.70, 0.395), (0.78, 0.55))
    arrow(ax, (0.865, 0.48), (0.865, 0.37))
    arrow(ax, (0.78, 0.31), (0.61, 0.15))
    arrow(ax, (0.39, 0.15), (0.26, 0.22), linestyle="--", color="#555555")

    ax.text(
        0.5,
        0.035,
        "Execution applies the routing command to move the UAV. The next camera frame is noted as a new sensor input, not a processing output back to aerial imagery.",
        ha="center",
        va="center",
        fontsize=10,
        color="#333333",
    )

    fig.tight_layout()
    png = OUTPUT_DIR / "paper1_system_architecture_updated.png"
    pdf = OUTPUT_DIR / "paper1_system_architecture_updated.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
