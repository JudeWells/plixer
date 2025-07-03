#!/usr/bin/env python
"""
retro_tanimoto_hist.py – Re-create the Tanimoto similarity histogram in a
retro-terminal aesthetic.

Usage
-----
$ python scripts/retro_tanimoto_hist.py \
    --csv evaluation_results/bubba_zjhnye4j_2025-05-11_highPropPoc2Mol/combined_model_results_backup.csv \
    --output tanimoto_similarity_retro.png

The script will produce a PNG with a transparent background by default.  The
style mimics a green-on-black terminal with magenta highlights.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# -----------------------------------------------------------------------------
# Retro-terminal style configuration
# -----------------------------------------------------------------------------
# Primary colours (bright lime green and deep-sky blue)
TERM_GREEN = "#00FF00"
TERM_BLUE = "#00BFFF"  # decoy colour

mpl.rcParams.update({
    # Fonts (prefer Courier New for full retro vibes)
    "font.family": "Courier New",
    "font.monospace": [
        "Courier New",
        "DejaVu Sans Mono",
        "Liberation Mono",
        "Consolas",
    ],
    # Background / foreground
    "figure.facecolor": "none",  # keep figure transparent
    "axes.facecolor": "black",
    "savefig.facecolor": "none",
    "savefig.edgecolor": "none",
    # Text & axes colours
    "text.color": TERM_GREEN,
    "axes.labelcolor": TERM_GREEN,
    "xtick.color": TERM_GREEN,
    "ytick.color": TERM_GREEN,
    "axes.edgecolor": TERM_GREEN,
    # Grid
    "grid.color": "#00AA00",
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

# Apply seaborn dark theme but override with our rcParams afterwards so that
# things like legend frame alpha are nicer.
sns.set_theme(style="dark", rc={"axes.facecolor": "black"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retro-style Tanimoto similarity histogram")
    parser.add_argument("--csv", type=Path, default=Path("evaluation_results/bubba_zjhnye4j_2025-05-11_highPropPoc2Mol/combined_model_results_backup.csv"), help="Path to combined_model_results_backup.csv")
    parser.add_argument("--output", type=Path, default=Path("tanimoto_similarity_retro.png"), help="Output image path")
    parser.add_argument("--dpi", type=int, default=300, help="Image resolution in DPI")
    return parser.parse_args()


def main():  # noqa: D103
    args = parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"CSV file not found: {args.csv}")

    df = pd.read_csv(args.csv)

    # Ensure the required columns exist
    if not {"tanimoto_similarity", "decoy_tanimoto_similarity"}.issubset(df.columns):
        raise ValueError(
            "CSV must contain 'tanimoto_similarity' and 'decoy_tanimoto_similarity' columns."
        )

    # Drop NaNs
    sample_vals = df["tanimoto_similarity"].dropna()
    decoy_vals = df["decoy_tanimoto_similarity"].dropna()

    # Choose bins based on combined data range
    combined = pd.concat([sample_vals, decoy_vals])
    n_bins = max(10, int(np.sqrt(len(combined))))
    bin_edges = np.linspace(combined.min(), combined.max(), n_bins + 1)

    fig, ax = plt.subplots(figsize=(8, 5), facecolor="none")

    # Histograms
    sns.histplot(
        sample_vals,
        bins=bin_edges,
        color=TERM_GREEN,
        alpha=0.5,
        label="tanimoto_similarity",
        ax=ax,
    )
    # Plot decoys **after** samples so they appear on top
    sns.histplot(
        decoy_vals,
        bins=bin_edges,
        color=TERM_BLUE,
        alpha=0.5,
        label="decoy_tanimoto_similarity",
        ax=ax,
    )

    # Mean indicators (dashed vertical lines)
    ax.axvline(sample_vals.mean(), color=TERM_GREEN, linestyle="--")
    ax.axvline(decoy_vals.mean(), color=TERM_BLUE, linestyle="--")

    # Labels & title
    ax.set_title("Tanimoto similarity", color=TERM_GREEN, pad=15)
    ax.set_xlabel("tanimoto_similarity")
    ax.set_ylabel("Count")

    # Legend styling – transparent background
    legend = ax.legend(frameon=False)
    for text in legend.get_texts():
        text.set_color(TERM_GREEN)

    # Tight layout & save
    fig.tight_layout()
    fig.savefig(args.output, dpi=args.dpi, transparent=True)
    print(f"Histogram saved to → {args.output}")


if __name__ == "__main__":
    main() 