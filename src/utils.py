from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def ensure_dir(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def tensor_to_float(value: torch.Tensor | float) -> float:
    if isinstance(value, float):
        return value
    return float(value.detach().cpu().item())


def safe_std(values: list[float]) -> float:
    if not values:
        return 0.0
    arr = np.array(values, dtype=np.float64)
    return float(arr.std())


def softmax_np(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x)


def pairwise_path_length_or_large(path_lengths: dict[int, dict[int, float]], src: int, dst: int, large: float = 1e6) -> float:
    return float(path_lengths.get(src, {}).get(dst, large))


def entropy_from_counts(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    return float(-(probs * np.log(probs + 1e-12)).sum())


def euclidean_norm(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.sqrt(torch.sum(x * x, dim=dim) + 1e-12)


def quantize_to_bins(values: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    distances = torch.abs(values.unsqueeze(-1) - bins.view(1, 1, -1))
    return torch.argmin(distances, dim=-1)


def clipped_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = torch.clamp(mask.sum(), min=1.0)
    return (values * mask).sum() / denom


def make_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def running_average(old: float, new: float, momentum: float) -> float:
    return momentum * old + (1.0 - momentum) * new


def count_degrees(edge_index: torch.Tensor, num_nodes: int) -> tuple[torch.Tensor, torch.Tensor]:
    out_degree = torch.zeros(num_nodes, dtype=torch.long, device=edge_index.device)
    in_degree = torch.zeros(num_nodes, dtype=torch.long, device=edge_index.device)
    ones = torch.ones(edge_index.shape[1], dtype=torch.long, device=edge_index.device)
    out_degree.index_add_(0, edge_index[0], ones)
    in_degree.index_add_(0, edge_index[1], ones)
    return in_degree, out_degree


def degree_limits_by_type(node_types: torch.Tensor, surface_limit: int, interior_limit: int) -> torch.Tensor:
    limits = torch.full_like(node_types, interior_limit, dtype=torch.long)
    limits[node_types == 0] = surface_limit
    return limits


def stable_correlation(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_centered = x - x.mean(dim=0, keepdim=True)
    y_centered = y - y.mean(dim=0, keepdim=True)
    numerator = torch.sum(x_centered * y_centered, dim=0)
    denominator = torch.sqrt(torch.sum(x_centered * x_centered, dim=0) * torch.sum(y_centered * y_centered, dim=0) + 1e-12)
    return numerator / denominator
