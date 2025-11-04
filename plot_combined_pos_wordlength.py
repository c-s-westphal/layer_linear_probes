#!/usr/bin/env python3
"""
Publication-quality linear probe visualization: GPT-2 and Gemma.

Creates a 4-subplot figure showing:
- GPT-2: POS | Word Length (narrower, ~12 layers)
- Gemma: POS | Word Length (wider, ~25 layers, roughly 2x width)

Displays accuracy across layers with 65% confidence intervals.
Designed to fit ICML paper width (~2x current figure width).
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import scipy.stats as stats
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Set up Times New Roman font for publication quality
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'


def load_results(csv_path: str, task: str, method: str = 'random') -> pd.DataFrame:
    """Load linear probe results from CSV for a specific task.

    Args:
        csv_path: Path to results CSV file
        task: Task name (e.g., 'pos', 'word_length')
        method: Method name (default: 'random')

    Returns:
        DataFrame filtered for the specified task and method
    """
    if not Path(csv_path).exists():
        print(f"Warning: {csv_path} not found")
        return None

    df = pd.read_csv(csv_path)
    # Filter for task and method
    df = df[(df['task'] == task) & (df['method'] == method)]
    print(f"Loaded {len(df)} results for task='{task}', method='{method}' from {csv_path}")
    return df


def compute_layer_statistics(
    results_df: pd.DataFrame,
    metric_col: str,
    confidence: float = 0.65,
    as_percentage: bool = True
) -> Tuple[List[int], List[float], List[float]]:
    """Compute mean and confidence interval for each layer.

    Args:
        results_df: DataFrame with results
        metric_col: Column name to analyze (e.g., 'accuracy', 'mutual_information')
        confidence: Confidence level for interval (default: 0.65 for 65% CI)
        as_percentage: If True, multiply values by 100

    Returns:
        (layers, means, ci_errors)
    """
    if results_df is None or len(results_df) == 0:
        return [], [], []

    # Get unique layers from data
    layer_list = sorted(results_df['layer'].unique())

    means = []
    cis = []

    for layer in layer_list:
        layer_df = results_df[results_df['layer'] == layer]
        values = layer_df[metric_col].values

        # Convert to percentage if requested
        if as_percentage:
            values = values * 100

        # Calculate mean and CI
        mean = values.mean()
        if len(values) > 1:
            ci = stats.t.interval(
                confidence=confidence,
                df=len(values)-1,
                loc=mean,
                scale=stats.sem(values)
            )
            ci_error = mean - ci[0]  # Error bar size (symmetric)
        else:
            ci_error = 0  # No error bar if only one sample

        means.append(mean)
        cis.append(ci_error)

    return layer_list, means, cis


def plot_combined_figure(
    gpt2_csv: str,
    gemma_csv: str,
    output_path: Path,
    metric_col: str = 'accuracy',
    confidence: float = 0.65
):
    """Create combined 4-subplot publication-quality figure.

    Args:
        gpt2_csv: Path to GPT-2 results CSV
        gemma_csv: Path to Gemma results CSV
        output_path: Path to save plot (without extension)
        metric_col: Metric column to plot ('accuracy' or 'mutual_information')
        confidence: Confidence level for error bars (default: 0.65)
    """
    # Load data for all tasks
    print("\nLoading data...")
    gpt2_pos = load_results(gpt2_csv, 'pos')
    gpt2_wl = load_results(gpt2_csv, 'word_length')
    gemma_pos = load_results(gemma_csv, 'pos')
    gemma_wl = load_results(gemma_csv, 'word_length')

    # Compute statistics
    print("\nComputing statistics...")
    gpt2_pos_stats = compute_layer_statistics(gpt2_pos, metric_col, confidence)
    gpt2_wl_stats = compute_layer_statistics(gpt2_wl, metric_col, confidence)
    gemma_pos_stats = compute_layer_statistics(gemma_pos, metric_col, confidence)
    gemma_wl_stats = compute_layer_statistics(gemma_wl, metric_col, confidence)

    # Create figure with GridSpec for custom width ratios
    # Width ratio: GPT-2 has ~12 layers, Gemma has ~25 layers (approx 2x)
    fig = plt.figure(figsize=(20, 4))  # Wider figure for ICML paper
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 2, 2], wspace=0.15)

    ax1 = fig.add_subplot(gs[0])  # GPT-2 POS
    ax2 = fig.add_subplot(gs[1])  # GPT-2 Word Length
    ax3 = fig.add_subplot(gs[2])  # Gemma POS
    ax4 = fig.add_subplot(gs[3])  # Gemma Word Length

    # Color scheme
    gpt2_color = '#4DBBD5'  # Cyan for GPT-2
    gemma_color = '#90EE90'  # Light green for Gemma
    error_color = 'black'
    error_kw = {'linewidth': 2.0, 'capsize': 6, 'capthick': 2.0}

    # Y-axis label (only on leftmost subplot) - always use Accuracy (%)
    ylabel = 'Accuracy (%)'

    # Helper function to compute individual y-axis range for each subplot
    def compute_y_limits(layers, means, cis, padding_percent=4):
        """Compute y-axis limits as [min(mean-ci), max(mean+ci)] with padding, rounded to integers"""
        if len(means) == 0:
            return 0, 100
        lower_bounds = [m - c for m, c in zip(means, cis)]
        upper_bounds = [m + c for m, c in zip(means, cis)]
        y_min = min(lower_bounds)
        y_max = max(upper_bounds)
        # Add 4% padding
        y_range = y_max - y_min
        padding = y_range * (padding_percent / 100)
        # Round to integers - floor for min, ceil for max to ensure data fits
        import math
        y_min_padded = math.floor(y_min - padding)
        y_max_padded = math.ceil(y_max + padding)
        return y_min_padded, y_max_padded

    # ===================== SUBPLOT 1: GPT-2 POS =====================
    if gpt2_pos is not None and len(gpt2_pos_stats[0]) > 0:
        layers, means, cis = gpt2_pos_stats
        ax1.bar(layers, means, yerr=cis, alpha=0.8,
                color=gpt2_color, ecolor=error_color,
                edgecolor='black', linewidth=1.0, error_kw=error_kw)

        ax1.set_xlabel('Layer', fontsize=20, fontweight='normal')
        ax1.set_ylabel(ylabel, fontsize=20, fontweight='normal')
        ax1.set_title('Part-of-Speech', fontsize=20, fontweight='bold', pad=15)
        # Show every other layer for GPT-2
        ax1.set_xticks([l for l in layers if l % 2 == 1])
        # Individual y-axis range for this subplot
        y_min, y_max = compute_y_limits(layers, means, cis)
        ax1.set_ylim(y_min, y_max)
        # Integer y-ticks only with limited number of ticks
        ax1.yaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=5))
        ax1.tick_params(axis='both', which='major', labelsize=18)
        ax1.grid(False)
        ax1.set_axisbelow(True)
        ax1.yaxis.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, zorder=1)

        for spine in ax1.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.5)
        ax1.set_facecolor('white')

    # ===================== SUBPLOT 2: GPT-2 Word Length =====================
    if gpt2_wl is not None and len(gpt2_wl_stats[0]) > 0:
        layers, means, cis = gpt2_wl_stats
        ax2.bar(layers, means, yerr=cis, alpha=0.8,
                color=gpt2_color, ecolor=error_color,
                edgecolor='black', linewidth=1.0, error_kw=error_kw)

        ax2.set_xlabel('Layer', fontsize=20, fontweight='normal')
        ax2.set_title('Word Length', fontsize=20, fontweight='bold', pad=15)
        # Show every other layer for GPT-2
        ax2.set_xticks([l for l in layers if l % 2 == 1])
        # Individual y-axis range for this subplot
        y_min, y_max = compute_y_limits(layers, means, cis)
        ax2.set_ylim(y_min, y_max)
        # Integer y-ticks only with limited number of ticks
        ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=5))
        ax2.tick_params(axis='both', which='major', labelsize=18)
        ax2.grid(False)
        ax2.set_axisbelow(True)
        ax2.yaxis.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, zorder=1)

        for spine in ax2.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.5)
        ax2.set_facecolor('white')

    # ===================== SUBPLOT 3: Gemma POS =====================
    if gemma_pos is not None and len(gemma_pos_stats[0]) > 0:
        layers, means, cis = gemma_pos_stats
        ax3.bar(layers, means, yerr=cis, alpha=0.8,
                color=gemma_color, ecolor=error_color,
                edgecolor='black', linewidth=1.0, error_kw=error_kw)

        ax3.set_xlabel('Layer', fontsize=20, fontweight='normal')
        ax3.set_title('Part-of-Speech', fontsize=20, fontweight='bold', pad=15)
        ax3.set_xticks([1, 5, 10, 15, 20, 25])
        # Individual y-axis range for this subplot
        y_min, y_max = compute_y_limits(layers, means, cis)
        ax3.set_ylim(y_min, y_max)
        # Integer y-ticks only with limited number of ticks
        ax3.yaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=5))
        ax3.tick_params(axis='both', which='major', labelsize=18)
        ax3.grid(False)
        ax3.set_axisbelow(True)
        ax3.yaxis.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, zorder=1)

        for spine in ax3.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.5)
        ax3.set_facecolor('white')

    # ===================== SUBPLOT 4: Gemma Word Length =====================
    if gemma_wl is not None and len(gemma_wl_stats[0]) > 0:
        layers, means, cis = gemma_wl_stats
        ax4.bar(layers, means, yerr=cis, alpha=0.8,
                color=gemma_color, ecolor=error_color,
                edgecolor='black', linewidth=1.0, error_kw=error_kw)

        ax4.set_xlabel('Layer', fontsize=20, fontweight='normal')
        ax4.set_title('Word Length', fontsize=20, fontweight='bold', pad=15)
        ax4.set_xticks([1, 5, 10, 15, 20, 25])
        # Individual y-axis range for this subplot
        y_min, y_max = compute_y_limits(layers, means, cis)
        ax4.set_ylim(y_min, y_max)
        # Integer y-ticks only with limited number of ticks
        ax4.yaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=5))
        ax4.tick_params(axis='both', which='major', labelsize=18)
        ax4.grid(False)
        ax4.set_axisbelow(True)
        ax4.yaxis.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, zorder=1)

        for spine in ax4.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.5)
        ax4.set_facecolor('white')

        # Add legend in top-left corner
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=gpt2_color, edgecolor='black', label='GPT-2'),
            Patch(facecolor=gemma_color, edgecolor='black', label='Gemma-2-2b')
        ]
        ax4.legend(handles=legend_elements, loc='upper left', fontsize=16,
                  frameon=True, fancybox=False, edgecolor='black', framealpha=1)

    # Set background
    fig.patch.set_facecolor('white')

    # Save as PNG and PDF
    output_path_png = Path(str(output_path).replace('.png', '') + '.png')
    plt.savefig(output_path_png, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path_png}")

    output_path_pdf = Path(str(output_path).replace('.png', '') + '.pdf')
    plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path_pdf}")

    plt.close()


def main():
    """Generate publication-quality linear probe visualization."""
    parser = argparse.ArgumentParser(
        description='Create publication-quality GPT-2 and Gemma linear probe plots'
    )
    parser.add_argument('--gpt2_csv', type=str,
                       default='outputs/linear_probe_pca_gpt2/raw_results.csv',
                       help='Path to GPT-2 results CSV')
    parser.add_argument('--gemma_csv', type=str,
                       default='outputs/linear_probe_pca_gemma/raw_results.csv',
                       help='Path to Gemma results CSV')
    parser.add_argument('--output_dir', type=str, default='outputs/combined_plots',
                       help='Directory to save plots')
    parser.add_argument('--metric', type=str, default='accuracy',
                       choices=['accuracy', 'mutual_information', 'f1_score'],
                       help='Metric to plot')
    parser.add_argument('--confidence', type=float, default=0.65,
                       help='Confidence level for error bars (default: 0.65 for 65%% CI)')
    args = parser.parse_args()

    print("="*70)
    print("Publication-Quality Linear Probe Visualization")
    print("4-Panel Figure: POS and Word Length for GPT-2 and Gemma")
    print("="*70)

    # Create plots directory
    plots_dir = Path(args.output_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Generate plot
    metric_name = args.metric.replace('_', ' ').title()
    output_file = plots_dir / f'combined_pos_wordlength_{args.metric}'

    print(f"\nGenerating combined publication-quality plot...")
    print(f"Metric: {metric_name}")
    print(f"Confidence: {args.confidence*100:.0f}%")
    plot_combined_figure(args.gpt2_csv, args.gemma_csv, output_file,
                        args.metric, args.confidence)

    print("\n" + "="*70)
    print("Publication-quality plots generated successfully!")
    print(f"PNG (600 DPI): {output_file}.png")
    print(f"PDF: {output_file}.pdf")
    print("="*70)


if __name__ == '__main__':
    main()
