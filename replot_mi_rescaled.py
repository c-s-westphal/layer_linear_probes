#!/usr/bin/env python3
"""
Re-create MI plots with rescaled y-axis: [min(mean - error_bar), max(mean + error_bar)]

Reads the raw_results.csv from both GPT-2 and Gemma experiments and regenerates
only the Mutual Information plots with tighter y-axis scaling.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import scipy.stats as stats


def create_mi_plot_rescaled(df, metric, ylabel, title, output_path, logger=None):
    """
    Create bar plot with rescaled y-axis based on data range.

    Y-axis range: [min(mean - stderr), max(mean + stderr)]
    """
    layers = sorted(df['layer'].unique())

    # Compute mean and 95% CI for each layer
    means = []
    ci_lows = []
    ci_highs = []

    for layer in layers:
        layer_df = df[df['layer'] == layer]
        values = layer_df[metric].values

        mean = np.mean(values)
        std = np.std(values, ddof=1)
        n = len(values)

        # Handle edge cases
        if n <= 1 or std == 0:
            # No variance, use mean with zero error bars
            ci_lows.append(0)
            ci_highs.append(0)
        else:
            # 95% confidence interval using t-distribution
            ci = stats.t.interval(0.95, n-1, loc=mean, scale=std/np.sqrt(n))
            ci_lows.append(mean - ci[0])  # Error bar below mean
            ci_highs.append(ci[1] - mean)  # Error bar above mean

        means.append(mean)

    means = np.array(means)
    ci_lows = np.array(ci_lows)
    ci_highs = np.array(ci_highs)

    # Calculate y-axis range: [min(mean - error), max(mean + error)]
    y_min = np.min(means - ci_lows)
    y_max = np.max(means + ci_highs)

    # Handle edge case where all values are identical
    if np.isnan(y_min) or np.isnan(y_max) or y_min == y_max:
        # Use simple range around mean
        y_min = np.min(means) * 0.95
        y_max = np.max(means) * 1.05
    else:
        # Add small padding (5% of range)
        y_range = y_max - y_min
        y_min = y_min - 0.05 * y_range
        y_max = y_max + 0.05 * y_range

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(layers))
    bars = ax.bar(x, means, yerr=[ci_lows, ci_highs],
                   capsize=5, alpha=0.8, color='steelblue',
                   error_kw={'linewidth': 2, 'ecolor': 'black'})

    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in layers])
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Set rescaled y-axis
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    if logger:
        logger.info(f"  Saved (rescaled): {output_path.name} [y-axis: {y_min:.4f} to {y_max:.4f}]")
    else:
        print(f"  Saved (rescaled): {output_path.name} [y-axis: {y_min:.4f} to {y_max:.4f}]")


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate MI plots with rescaled y-axis"
    )
    parser.add_argument(
        "--gpt2_dir",
        type=str,
        default="outputs/linear_probe_pca_gpt2",
        help="GPT-2 results directory"
    )
    parser.add_argument(
        "--gemma_dir",
        type=str,
        default="outputs/linear_probe_pca_gemma",
        help="Gemma results directory"
    )
    args = parser.parse_args()

    # Process GPT-2 results
    gpt2_dir = Path(args.gpt2_dir)
    if gpt2_dir.exists():
        print(f"\n{'='*80}")
        print(f"Processing GPT-2 results: {gpt2_dir}")
        print(f"{'='*80}")

        csv_path = gpt2_dir / "raw_results.csv"
        if not csv_path.exists():
            print(f"ERROR: {csv_path} not found!")
        else:
            results_df = pd.read_csv(csv_path)
            plots_dir = gpt2_dir / "plots_rescaled"
            plots_dir.mkdir(exist_ok=True)

            # Get all task names
            tasks = results_df['task'].unique()

            for task in tasks:
                task_df = results_df[results_df['task'] == task]

                # PCA MI plot
                pca_df = task_df[task_df['method'] == 'pca']
                if len(pca_df) > 0:
                    print(f"\n{task.upper()} - PCA method:")
                    create_mi_plot_rescaled(
                        pca_df,
                        'mutual_information',
                        'Mutual Information',
                        f'{task.upper()} (PCA): Mutual Information Across Layers',
                        plots_dir / f'{task}_pca_mutual_information_rescaled.png'
                    )

                # Random baseline MI plot
                random_df = task_df[task_df['method'] == 'random']
                if len(random_df) > 0:
                    print(f"\n{task.upper()} - Random baseline:")
                    create_mi_plot_rescaled(
                        random_df,
                        'mutual_information',
                        'Mutual Information',
                        f'{task.upper()} (Random): Mutual Information Across Layers',
                        plots_dir / f'{task}_random_mutual_information_rescaled.png'
                    )

            print(f"\nGPT-2 rescaled plots saved to: {plots_dir}")
    else:
        print(f"GPT-2 directory not found: {gpt2_dir}")

    # Process Gemma results
    gemma_dir = Path(args.gemma_dir)
    if gemma_dir.exists():
        print(f"\n{'='*80}")
        print(f"Processing Gemma results: {gemma_dir}")
        print(f"{'='*80}")

        csv_path = gemma_dir / "raw_results.csv"
        if not csv_path.exists():
            print(f"ERROR: {csv_path} not found!")
        else:
            results_df = pd.read_csv(csv_path)
            plots_dir = gemma_dir / "plots_rescaled"
            plots_dir.mkdir(exist_ok=True)

            # Get all task names
            tasks = results_df['task'].unique()

            for task in tasks:
                task_df = results_df[results_df['task'] == task]

                # PCA MI plot
                pca_df = task_df[task_df['method'] == 'pca']
                if len(pca_df) > 0:
                    print(f"\n{task.upper()} - PCA method:")
                    create_mi_plot_rescaled(
                        pca_df,
                        'mutual_information',
                        'Mutual Information',
                        f'{task.upper()} (PCA): Mutual Information Across Layers',
                        plots_dir / f'{task}_pca_mutual_information_rescaled.png'
                    )

                # Random baseline MI plot
                random_df = task_df[task_df['method'] == 'random']
                if len(random_df) > 0:
                    print(f"\n{task.upper()} - Random baseline:")
                    create_mi_plot_rescaled(
                        random_df,
                        'mutual_information',
                        'Mutual Information',
                        f'{task.upper()} (Random): Mutual Information Across Layers',
                        plots_dir / f'{task}_random_mutual_information_rescaled.png'
                    )

            print(f"\nGemma rescaled plots saved to: {plots_dir}")
    else:
        print(f"Gemma directory not found: {gemma_dir}")

    print(f"\n{'='*80}")
    print("DONE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
