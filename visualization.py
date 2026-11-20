"""Plotting helpers for term structure, signal, and backtest output."""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_term_structure_snapshot(
    prices: pd.DataFrame,
    dates: Sequence,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot the term structure on selected dates."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    tenors = np.asarray(prices.columns, dtype=float)
    for d in dates:
        if d not in prices.index:
            d = prices.index[prices.index.get_indexer([d], method="nearest")[0]]
        ax.plot(tenors, prices.loc[d].values, marker="o", label=str(d.date()))
    ax.set_xlabel("Tenor (months)")
    ax.set_ylabel("Futures price ($/bbl)")
    ax.set_title("Crude oil term structure")
    ax.legend()
    ax.grid(alpha=0.3)
    return ax


def plot_signal(signal: pd.DataFrame, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Plot the rolling z-score of the backwardation signal."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    z = signal["z"]
    ax.plot(z.index, z.values, color="navy", lw=1.0, label="z-score")
    ax.axhline(0, color="black", lw=0.5)
    ax.axhline(1, color="green", ls="--", lw=0.7, label="long threshold")
    ax.axhline(-1, color="red", ls="--", lw=0.7, label="contango band")
    ax.fill_between(z.index, 0, z.values, where=z.values > 0,
                    color="green", alpha=0.15)
    ax.fill_between(z.index, 0, z.values, where=z.values < 0,
                    color="red", alpha=0.15)
    ax.set_title("Backwardation signal (rolling z-score)")
    ax.set_ylabel("z-score")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    return ax


def plot_backtest(result, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Plot strategy vs benchmark equity curves."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    ax.plot(result.equity.index, result.equity.values,
            label="Strategy", color="navy")
    ax.plot(result.benchmark.index, result.benchmark.values,
            label="Buy & hold front", color="grey", alpha=0.7)
    ax.set_yscale("log")
    ax.set_title("Equity curve (log scale)")
    ax.set_ylabel("Equity (base 1.0)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    return ax


def plot_heatmap(prices: pd.DataFrame, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Heatmap of the row-demeaned log-price curve over time.

    Red bands = backwardation (near-tenor log-prices above the mean for
    that day), blue bands = contango.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    log_p = np.log(prices)
    centred = log_p.sub(log_p.mean(axis=1), axis=0)
    im = ax.imshow(
        centred.T.values,
        aspect="auto",
        origin="lower",
        cmap="RdBu_r",
        extent=[0, len(prices),
                float(prices.columns.min()), float(prices.columns.max())],
        vmin=-0.10, vmax=0.10,
    )
    ax.set_ylabel("Tenor (months)")
    ax.set_xlabel("Time index")
    ax.set_title("Term structure shape (log-price minus row mean)")
    plt.colorbar(im, ax=ax, label="log-deviation")
    return ax
