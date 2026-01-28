"""
Plotting utilities for sensitivity analysis and results visualization.

This module provides functions for creating comparative plots across
different mechanisms (CE, CCE, Selfish FI).
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_three_curves(x, y_cce, y_ce, y_fi, xlabel, ylabel, title=None):
    """
    Plot three curves comparing CCE, CE, and Selfish FI benchmarks.
    
    Parameters:
    -----------
    x : array-like
        X-axis values
    y_cce : array-like
        Y-axis values for CCE
    y_ce : array-like
        Y-axis values for CE
    y_fi : array-like
        Y-axis values for Selfish FI
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    title : str, optional
        Plot title
    """
    plt.figure()
    plt.plot(x, y_fi, marker="^", label="Selfish FI (observable)")
    plt.plot(x, y_ce, marker="o", label="CE")
    plt.plot(x, y_cce, marker="s", label="CCE")
    plt.xlabel(xlabel, fontsize=13)
    plt.ylabel(ylabel, fontsize=13)
    if title is not None:
        plt.title(title, fontsize=13)
    plt.grid(True, linewidth=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()
