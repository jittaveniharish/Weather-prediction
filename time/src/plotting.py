"""
Visualization helpers.
"""
from __future__ import annotations

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


def comparison_plot(
    timestamps: np.ndarray,
    actual: np.ndarray,
    baseline: np.ndarray,
    corrected: np.ndarray,
    metrics: Dict[str, float],
) -> None:
    plt.figure(figsize=(12, 5))
    plt.plot(timestamps, actual, label="Observed", marker="o")
    plt.plot(timestamps, baseline, label="STM Forecast", linestyle="--")
    plt.plot(timestamps, corrected, label="RL-Adjusted", linestyle="-")
    metric_text = "\n".join(
        f"{name.upper()}: {value:.3f}"
        for name, value in metrics.items()
        if name in ("mae", "rmse", "accuracy")
    )
    plt.gca().text(
        0.02,
        0.95,
        metric_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    plt.title("Temperature Forecast Comparison")
    plt.xlabel("Timestamp")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()
    plt.close()


