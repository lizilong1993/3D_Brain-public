from __future__ import annotations

import math

import numpy as np


def summarize_scalar(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0.0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    arr = np.array(values, dtype=np.float64)
    return {
        "n": float(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def bootstrap_mean_interval(values: list[float], num_samples: int = 500, alpha: float = 0.05) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    arr = np.array(values, dtype=np.float64)
    rng = np.random.default_rng(42)
    draws = []
    for _ in range(num_samples):
        sample = rng.choice(arr, size=arr.size, replace=True)
        draws.append(float(sample.mean()))
    draws_arr = np.sort(np.array(draws, dtype=np.float64))
    low_idx = max(0, int(math.floor((alpha / 2.0) * (draws_arr.size - 1))))
    high_idx = min(draws_arr.size - 1, int(math.ceil((1.0 - alpha / 2.0) * (draws_arr.size - 1))))
    return {
        "mean": float(arr.mean()),
        "ci_low": float(draws_arr[low_idx]),
        "ci_high": float(draws_arr[high_idx]),
    }
