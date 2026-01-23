#!/usr/bin/env python3
"""
Visualize event count statistics from pre-computed JSON data.

Usage:
    CLI: python -m utils.event_count_visualize /path/to/event_counts.json
    API: from utils.event_count_visualize import visualize_event_counts
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def load_event_counts(json_path: str) -> dict:
    """Load event counts from JSON file."""
    json_path = Path(json_path).expanduser().resolve()
    
    if not json_path.exists():
        raise FileNotFoundError(f"Event counts file not found: {json_path}")
    
    with open(json_path, 'r') as f:
        return json.load(f)


def compute_statistics(counts: list) -> dict:
    """Compute statistics for a list of event counts."""
    if not counts:
        return {}
    
    counts_array = np.array(counts)
    return {
        "num_files": len(counts),
        "total_events": int(counts_array.sum()),
        "mean": float(counts_array.mean()),
        "median": float(np.median(counts_array)),
        "std": float(counts_array.std()),
        "min": int(counts_array.min()),
        "max": int(counts_array.max()),
        "percentile_25": float(np.percentile(counts_array, 25)),
        "percentile_75": float(np.percentile(counts_array, 75))
    }


def print_statistics(stats: dict, label: str = "Overall"):
    """Print statistics in formatted output."""
    if not stats:
        print(f"\n{label}: No data")
        return
    
    print(f"\n{'='*60}")
    print(f"{label} Statistics")
    print(f"{'='*60}")
    print(f"Number of files: {stats['num_files']:,}")
    print(f"Total events: {stats['total_events']:,}")
    print(f"Mean events per file: {stats['mean']:,.2f}")
    print(f"Median events per file: {stats['median']:,.2f}")
    print(f"Std deviation: {stats['std']:,.2f}")
    print(f"Min events: {stats['min']:,}")
    print(f"Max events: {stats['max']:,}")
    print(f"25th percentile: {stats['percentile_25']:,.2f}")
    print(f"75th percentile: {stats['percentile_75']:,.2f}")


def visualize_event_counts(
    json_path: str,
    output_dir: Optional[str] = None,
    show_plots: bool = False
) -> dict:
    """
    Generate statistics and visualizations from event count data.
    
    Args:
        json_path: Path to the event_counts.json file
        output_dir: Directory for output plots. If None, saves next to JSON file.
        show_plots: Whether to display plots interactively
    
    Returns:
        Dictionary containing all computed statistics
    """
    json_path = Path(json_path).expanduser().resolve()
    data = load_event_counts(json_path)
    
    # Default output directory
    if output_dir is None:
        output_dir = json_path.parent
    else:
        output_dir = Path(output_dir).expanduser().resolve()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract counts by category
    all_counts = [f["count"] for f in data["files"]]
    
    town_counts = defaultdict(list)
    split_counts = defaultdict(list)
    
    for f in data["files"]:
        town_counts[f["town"]].append(f["count"])
        split_counts[f["split"]].append(f["count"])
    
    # Compute statistics
    stats = {
        "overall": compute_statistics(all_counts),
        "by_split": {k: compute_statistics(v) for k, v in split_counts.items()},
        "by_town": {k: compute_statistics(v) for k, v in town_counts.items()}
    }
    
    # Print statistics
    print_statistics(stats["overall"], "Overall Dataset")
    
    for split_name in sorted(stats["by_split"].keys()):
        print_statistics(stats["by_split"][split_name], f"{split_name.capitalize()} Split")
    
    print(f"\n{'='*60}")
    print("Per-Town Breakdown")
    print(f"{'='*60}")
    for town_name in sorted(stats["by_town"].keys()):
        print_statistics(stats["by_town"][town_name], f"Town: {town_name}")
    
    # Generate visualizations
    print(f"\n{'='*60}")
    print("Creating visualizations...")
    print(f"{'='*60}")
    
    # 1. Overall histogram
    _plot_histogram(
        all_counts,
        "Overall Event Count Distribution",
        output_dir / "histogram_overall.png"
    )
    
    # 2. Per-town histograms
    _plot_per_town_histograms(
        town_counts,
        output_dir / "histogram_per_town.png"
    )
    
    # 3. Town comparison bar chart
    _plot_town_comparison(
        town_counts,
        output_dir / "town_comparison.png"
    )
    
    if show_plots:
        plt.show()
    
    print(f"\n{'='*60}")
    print("Visualization complete!")
    print(f"{'='*60}")
    
    return stats


def _plot_histogram(counts: list, title: str, output_path: Path):
    """Create and save a histogram of event counts."""
    counts_array = np.array(counts)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(counts_array, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Number of Events', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    stats_text = (f'n={len(counts_array):,}\n'
                  f'mean={counts_array.mean():,.0f}\n'
                  f'median={np.median(counts_array):,.0f}\n'
                  f'std={counts_array.std():,.0f}')
    ax.text(0.98, 0.97, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def _plot_per_town_histograms(town_counts: dict, output_path: Path):
    """Create per-town histograms in a grid."""
    num_towns = len(town_counts)
    cols = min(4, num_towns)
    rows = (num_towns + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if num_towns == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (town_name, counts) in enumerate(sorted(town_counts.items())):
        counts_array = np.array(counts)
        axes[idx].hist(counts_array, bins=30, edgecolor='black', alpha=0.7)
        axes[idx].set_xlabel('Number of Events', fontsize=10)
        axes[idx].set_ylabel('Frequency', fontsize=10)
        axes[idx].set_title(f'{town_name} (n={len(counts):,})',
                           fontsize=12, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)
        
        mean_val = counts_array.mean()
        axes[idx].axvline(mean_val, color='red', linestyle='--',
                         linewidth=2, label=f'Mean: {mean_val:,.0f}')
        axes[idx].legend(fontsize=8)
    
    # Hide unused subplots
    for idx in range(len(town_counts), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def _plot_town_comparison(town_counts: dict, output_path: Path):
    """Create bar chart comparing towns."""
    towns = sorted(town_counts.keys())
    means = [np.mean(town_counts[town]) for town in towns]
    stds = [np.std(town_counts[town]) for town in towns]
    file_counts = [len(town_counts[town]) for town in towns]
    
    fig, ax = plt.subplots(figsize=(max(10, len(towns) * 1.5), 8))
    x = np.arange(len(towns))
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7, edgecolor='black')
    
    colors = plt.cm.viridis(np.array(means) / max(means))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax.set_xlabel('Town', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Number of Events', fontsize=12, fontweight='bold')
    ax.set_title('Average Event Count per Town', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(towns, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, count in zip(bars, file_counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'n={count:,}',
                ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize event count statistics from JSON data"
    )
    parser.add_argument(
        "json_path",
        type=str,
        help="Path to event_counts.json file"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Output directory for plots (default: same as JSON file)"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively"
    )
    
    args = parser.parse_args()
    visualize_event_counts(args.json_path, args.output_dir, args.show)


if __name__ == "__main__":
    main()
