from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from uav_environment import UAVNavigationEnv
from uav_navigation_system import UAVNavigationSystem


MAP_SPECS = [
    {
        "slug": "light_sparse",
        "label": "Light/Sparse Fire",
        "legend": "Light/Sparse",
        "profile": "fire_light",
        "seed": 20260530,
        "source_figure": "navigation/comparison_results/paper1_fire_density_final/figures/paper1_light_sparse_fire_baseline_50x50_environment.png",
    },
    {
        "slug": "moderate",
        "label": "Moderate Fire",
        "legend": "Moderate",
        "profile": "fire_moderate",
        "seed": 20360530,
        "source_figure": "navigation/comparison_results/paper1_fire_density_final/figures/paper1_moderate_fire_baseline_50x50_environment.png",
    },
    {
        "slug": "dense_heavy",
        "label": "Dense/Heavy Fire",
        "legend": "Dense/Heavy",
        "profile": "fire_dense",
        "seed": 20460530,
        "source_figure": "navigation/comparison_results/paper1_fire_density_final/figures/paper1_dense_heavy_fire_baseline_50x50_environment.png",
    },
]


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def ensure_dirs(output_dir: Path) -> dict[str, Path]:
    paths = {
        "maps": output_dir / "maps",
        "runs": output_dir / "runs",
        "figures": output_dir / "figures",
        "metrics": output_dir / "metrics",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def map_package_path(maps_dir: Path, spec: dict[str, object]) -> Path:
    return maps_dir / f"paper1_exact_{spec['slug']}_map.npz"


def map_metadata_path(package_path: Path) -> Path:
    return package_path.with_suffix(".json")


def write_map_visual(metadata: dict[str, object], aerial_image: np.ndarray, gt_hazard_map: np.ndarray, classifier_hazard_map: np.ndarray, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(aerial_image)
    axes[0].set_title(str(metadata["map_label"]))
    axes[0].axis("off")

    im0 = axes[1].imshow(gt_hazard_map, cmap="Reds", vmin=0, vmax=1)
    axes[1].set_title("Ground-Truth Fire Risk")
    axes[1].set_xlim(-0.5, int(metadata["grid_size"]) - 0.5)
    axes[1].set_ylim(int(metadata["grid_size"]) - 0.5, -0.5)
    fig.colorbar(im0, ax=axes[1], fraction=0.046, pad=0.04)

    im1 = axes[2].imshow(classifier_hazard_map, cmap="Reds", vmin=0, vmax=1)
    axes[2].scatter([2], [2], c="lime", s=65, edgecolors="black", label="Start")
    axes[2].scatter([int(metadata["grid_size"]) - 3], [int(metadata["grid_size"]) - 3], c="yellow", s=90, marker="*", edgecolors="black", label="Goal")
    axes[2].set_title("Classifier-Derived Risk Map")
    axes[2].set_xlim(-0.5, int(metadata["grid_size"]) - 0.5)
    axes[2].set_ylim(int(metadata["grid_size"]) - 0.5, -0.5)
    axes[2].legend(loc="upper right", fontsize=8)
    fig.colorbar(im1, ax=axes[2], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_exact_map(args: argparse.Namespace, spec: dict[str, object], maps_dir: Path) -> Path:
    package_path = map_package_path(maps_dir, spec)
    metadata_path = map_metadata_path(package_path)
    if package_path.exists() and metadata_path.exists() and not args.force_maps:
        return package_path

    env = UAVNavigationEnv(
        grid_size=args.grid,
        max_steps=args.max_steps,
        classifier_path=str(resolve_path(args.classifier)),
        cache_imagery=True,
        observation_radius=args.observation_radius,
        confidence_decay=args.confidence_decay,
        termination_threshold=args.termination_threshold,
        perception_noise_std=args.perception_noise,
        lightweight_info=False,
        aerial_cell_px=args.aerial_cell_px,
        quiet_scene=True,
        scene_profile=str(spec["profile"]),
        scene_seed=int(spec["seed"]),
    )
    classifier_hazard_map, confidence_map = env.aerial_processor.process_aerial_image_grid(
        env.aerial_image,
        grid_size=args.grid,
        batch_size=args.classification_batch_size,
    )

    metadata = {
        "map_label": spec["label"],
        "legend_label": spec["legend"],
        "scene_profile": spec["profile"],
        "scene_seed": int(spec["seed"]),
        "grid_size": int(args.grid),
        "max_steps": int(args.max_steps),
        "aerial_cell_px": int(args.aerial_cell_px),
        "source_figure": str(resolve_path(spec["source_figure"])),
        "map_source": "recreated_exact_paper1_fire_density_episode_1_from_profile_and_seed",
        "map_package_path": str(package_path.resolve()),
        "created_at": datetime.utcnow().isoformat(),
        "classifier_checkpoint": str(resolve_path(args.classifier)),
        "disaster_locations": env.disaster_locations,
        "fire_zone_count": len([d for d in env.disaster_locations if d.get("type") == "fire"]),
        "max_ground_truth_hazard": float(np.max(env.gt_hazard_map)),
        "max_classifier_hazard": float(np.max(classifier_hazard_map)),
    }

    np.savez_compressed(
        package_path,
        aerial_image=env.aerial_image.astype(np.uint8),
        gt_hazard_map=env.gt_hazard_map.astype(np.float32),
        classifier_hazard_map=classifier_hazard_map.astype(np.float32),
        confidence_map=confidence_map.astype(np.float32),
        disaster_locations_json=np.array(json.dumps(env.disaster_locations)),
    )
    metadata_path.write_text(json.dumps(metadata, indent=2))
    write_map_visual(
        metadata,
        env.aerial_image,
        env.gt_hazard_map,
        classifier_hazard_map,
        maps_dir / f"paper1_exact_{spec['slug']}_map_visual.png",
    )
    return package_path


def read_training_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open() as handle:
        return list(csv.DictReader(handle))


def row_count(csv_path: Path) -> int:
    if not csv_path.exists():
        return 0
    return max(0, sum(1 for _ in csv_path.open()) - 1)


def find_existing_training_csv(run_dir: Path, expected_episodes: int) -> Path | None:
    candidates = sorted(run_dir.glob("training_episode_metrics_*.csv"))
    for candidate in reversed(candidates):
        if row_count(candidate) == expected_episodes:
            return candidate
    return None


def run_fire_density_training(args: argparse.Namespace, spec: dict[str, object], package_path: Path, runs_dir: Path) -> Path:
    run_dir = runs_dir / str(spec["slug"])
    run_dir.mkdir(parents=True, exist_ok=True)
    existing = find_existing_training_csv(run_dir, args.episodes)
    if existing and not args.force_training:
        return existing

    torch.manual_seed(args.training_seed + int(spec["seed"]) % 1000)
    np.random.seed(args.training_seed + int(spec["seed"]) % 1000)

    system = UAVNavigationSystem(
        grid_size=args.grid,
        max_episode_steps=args.max_steps,
        cache_imagery=True,
        classifier_path=str(resolve_path(args.classifier)),
        fixed_map_path=str(package_path),
        checkpoint_dir=str(run_dir),
        quiet_scene=True,
    )
    system.train_navigation_system(
        total_episodes=args.episodes,
        update_frequency=args.update_freq,
        save_frequency=args.save_freq,
        log_frequency=args.log_freq,
        episode_offset=0,
        resume_checkpoint=None,
        save_success_visualizations=False,
    )
    return Path(system.training_csv_path)


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    smoothed = np.zeros_like(values, dtype=float)
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        smoothed[idx] = float(np.mean(values[start:idx + 1]))
    return smoothed


def plot_metric(training_csvs: dict[str, Path], specs: list[dict[str, object]], output_dir: Path, metric: str, ylabel: str, title: str, output_name: str, smooth_window: int) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = {
        "light_sparse": "#2F6DAE",
        "moderate": "#E66101",
        "dense_heavy": "#4C9F38",
    }
    for spec in specs:
        rows = read_training_rows(training_csvs[str(spec["slug"])])
        episodes = np.asarray([int(float(row["episode"])) for row in rows], dtype=int)
        values = np.asarray([float(row.get(metric) or row.get("navigation_efficiency") or 0.0) for row in rows], dtype=float)
        smoothed = rolling_mean(values, smooth_window)
        ax.plot(
            episodes,
            smoothed,
            linewidth=2.4,
            color=colors[str(spec["slug"])],
            label=str(spec["legend"]),
        )
    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.set_xlim(left=0)
    if metric == "success":
        ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    png_path = output_dir / f"{output_name}.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{output_name}.pdf", bbox_inches="tight")
    plt.close(fig)


def write_summary(
    output_dir: Path,
    specs: list[dict[str, object]],
    map_paths: dict[str, Path],
    training_csvs: dict[str, Path],
    args: argparse.Namespace,
) -> None:
    rows = []
    for spec in specs:
        csv_path = training_csvs[str(spec["slug"])]
        data = read_training_rows(csv_path)
        success = np.asarray([float(row["success"]) for row in data], dtype=float)
        reward = np.asarray([float(row["reward"]) for row in data], dtype=float)
        path_eff = np.asarray([float(row.get("path_efficiency") or row.get("navigation_efficiency") or 0.0) for row in data], dtype=float)
        rows.append({
            "map": spec["label"],
            "profile": spec["profile"],
            "seed": spec["seed"],
            "episodes": len(data),
            "last_100_success_rate": float(np.mean(success[-100:])) if len(success) else 0.0,
            "last_100_reward": float(np.mean(reward[-100:])) if len(reward) else 0.0,
            "last_100_path_efficiency": float(np.mean(path_eff[-100:])) if len(path_eff) else 0.0,
            "map_package": str(map_paths[str(spec["slug"])]),
            "training_csv": str(csv_path),
        })

    metrics_path = output_dir / "metrics" / "paper1_fire_density_training_summary.csv"
    with metrics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    markdown = [
        "# Paper 1 Fire-Density PPO Training Summary",
        "",
        "These runs use controlled Paper 1 fire-density maps recreated from the original profile and seed used for the earlier environment figures. Training loads the saved map package so the same map condition is used throughout each run.",
        "",
        f"- Episodes per map: {args.episodes}",
        f"- Grid size: {args.grid} x {args.grid}",
        f"- Smoothing window: {args.smooth_window}",
        "",
        "| Map | Profile | Seed | Episodes | Last-100 Success | Last-100 Reward | Last-100 Path Efficiency |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        markdown.append(
            f"| {row['map']} | {row['profile']} | {row['seed']} | {row['episodes']} | "
            f"{row['last_100_success_rate']:.3f} | {row['last_100_reward']:.3f} | {row['last_100_path_efficiency']:.3f} |"
        )
    markdown.extend([
        "",
        "## Key Output Files",
        "",
        "- `figures/paper1_fire_density_reward_vs_episode.png`",
        "- `figures/paper1_fire_density_success_rate_vs_episode.png`",
        "- `figures/paper1_fire_density_path_efficiency_vs_episode.png`",
        "- `metrics/paper1_fire_density_training_summary.csv`",
    ])
    (output_dir / "PAPER1_FIRE_DENSITY_TRAINING_SUMMARY.md").write_text("\n".join(markdown) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Paper 1 fire-density maps and train PPO on each controlled map condition.")
    parser.add_argument("--output-dir", default="navigation/comparison_results/paper1_fire_density_training_3000")
    parser.add_argument("--classifier", default="checkpoints/checkpoint.pth")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--grid", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--aerial-cell-px", type=int, default=8)
    parser.add_argument("--observation-radius", type=int, default=2)
    parser.add_argument("--confidence-decay", type=float, default=0.95)
    parser.add_argument("--termination-threshold", type=float, default=0.9)
    parser.add_argument("--perception-noise", type=float, default=0.0)
    parser.add_argument("--classification-batch-size", type=int, default=128)
    parser.add_argument("--update-freq", type=int, default=32)
    parser.add_argument("--save-freq", type=int, default=500)
    parser.add_argument("--log-freq", type=int, default=100)
    parser.add_argument("--smooth-window", type=int, default=100)
    parser.add_argument("--training-seed", type=int, default=20260618)
    parser.add_argument("--only-generate-maps", action="store_true")
    parser.add_argument("--force-maps", action="store_true")
    parser.add_argument("--force-training", action="store_true")
    parser.add_argument(
        "--map-slugs",
        default="",
        help="Comma-separated subset to run: light_sparse,moderate,dense_heavy. Defaults to all maps.",
    )
    args = parser.parse_args()

    output_dir = resolve_path(args.output_dir)
    paths = ensure_dirs(output_dir)
    requested_slugs = [slug.strip() for slug in args.map_slugs.split(",") if slug.strip()]
    selected_specs = MAP_SPECS
    if requested_slugs:
        valid_slugs = {str(spec["slug"]) for spec in MAP_SPECS}
        invalid_slugs = sorted(set(requested_slugs) - valid_slugs)
        if invalid_slugs:
            raise ValueError(f"Unknown map slug(s): {', '.join(invalid_slugs)}")
        selected_specs = [spec for spec in MAP_SPECS if str(spec["slug"]) in requested_slugs]

    map_paths = {
        str(spec["slug"]): generate_exact_map(args, spec, paths["maps"])
        for spec in selected_specs
    }
    if args.only_generate_maps:
        print(json.dumps({key: str(value) for key, value in map_paths.items()}, indent=2))
        return

    training_csvs = {
        str(spec["slug"]): run_fire_density_training(args, spec, map_paths[str(spec["slug"])], paths["runs"])
        for spec in selected_specs
    }

    plot_metric(
        training_csvs,
        selected_specs,
        paths["figures"],
        "reward",
        "Reward",
        "Reward vs Episode by Fire-Density Map",
        "paper1_fire_density_reward_vs_episode",
        args.smooth_window,
    )
    plot_metric(
        training_csvs,
        selected_specs,
        paths["figures"],
        "success",
        "Success Rate",
        "Success Rate vs Episode by Fire-Density Map",
        "paper1_fire_density_success_rate_vs_episode",
        args.smooth_window,
    )
    plot_metric(
        training_csvs,
        selected_specs,
        paths["figures"],
        "path_efficiency",
        "Path Efficiency",
        "Path Efficiency vs Episode by Fire-Density Map",
        "paper1_fire_density_path_efficiency_vs_episode",
        args.smooth_window,
    )
    write_summary(output_dir, selected_specs, map_paths, training_csvs, args)

    print(json.dumps({
        "output_dir": str(output_dir),
        "map_packages": {key: str(value) for key, value in map_paths.items()},
        "training_csvs": {key: str(value) for key, value in training_csvs.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
