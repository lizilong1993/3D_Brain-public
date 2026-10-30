"""
Consolidated null-model random region samplers for spatial neural network audits.

Provides four matched control types:
  - same_size:       same number of nodes, random selection
  - degree_matched:  same size, matched mean undirected degree
  - surface_fraction_matched: same size, matched surface/interior fraction
  - distance_matched: same size, matched mean pairwise spatial distance

These samplers are designed to plug into the six-gate causal audit protocol
(compute_routeA_causal_dispersion_metrics.py) as well as the Claim 2b
fixed-measurement gate (claim2_fixed_measurement.py).

Usage:
    from src.null_model_samplers import build_null_samplers, sample_null_regions

    samplers = build_null_samplers(
        target_size=8,
        target_degree_mean=3.5,
        target_surface_fraction=0.25,
        target_mean_pairwise_distance=1.2,
        node_types=node_types_array,
        degree=degree_array,
        coords=coords_tensor,
        rng=np.random.default_rng(42),
    )

    # Draw 500 samples per type
    null_regions = sample_null_regions(
        samplers,
        n_per_type=500,
        types=["same_size", "degree_matched", "surface_fraction_matched", "distance_matched"],
    )
"""

from __future__ import annotations

import numpy as np
from typing import Callable

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

Sampler = Callable[[], list[int]]


# ---------------------------------------------------------------------------
# Individual samplers
# ---------------------------------------------------------------------------

def sample_same_size(
    size: int,
    num_nodes: int,
    rng: np.random.Generator,
) -> list[int]:
    """Sample *size* nodes uniformly without replacement from [0, num_nodes)."""
    size = max(1, min(size, num_nodes))
    return sorted(int(i) for i in rng.choice(np.arange(num_nodes), size=size, replace=False).tolist())


def sample_degree_matched(
    size: int,
    target_degree_mean: float,
    degree: np.ndarray,
    rng: np.random.Generator,
    attempts: int = 256,
) -> list[int]:
    """Sample *size* nodes whose mean undirected degree is within tolerance of *target_degree_mean*.

    Falls back to same-size when no acceptable sample is found within *attempts*.
    """
    size = max(1, min(size, degree.shape[0]))
    best_nodes = rng.choice(np.arange(degree.shape[0]), size=size, replace=False)
    best_distance = abs(float(np.mean(degree[best_nodes])) - target_degree_mean)

    for _ in range(attempts - 1):
        candidates = rng.choice(np.arange(degree.shape[0]), size=size, replace=False)
        mean_deg = float(np.mean(degree[candidates]))
        distance = abs(mean_deg - target_degree_mean)
        if distance < best_distance:
            best_distance = distance
            best_nodes = candidates

    return sorted(int(i) for i in best_nodes.tolist())


def sample_surface_fraction_matched(
    size: int,
    target_surface_fraction: float,
    node_types: np.ndarray,
    rng: np.random.Generator,
) -> list[int]:
    """Sample *size* nodes matching the target surface fraction (node_types == 0 is surface)."""
    size = max(1, min(size, node_types.shape[0]))
    surface_ids = np.where(node_types == 0)[0]
    interior_ids = np.where(node_types != 0)[0]

    n_surface = int(round(size * target_surface_fraction))
    n_surface = min(n_surface, surface_ids.shape[0], size)
    n_interior = min(size - n_surface, interior_ids.shape[0])

    # Patch deficit
    deficit = size - (n_surface + n_interior)
    if deficit > 0:
        extra_from_surface = min(deficit, max(0, surface_ids.shape[0] - n_surface))
        n_surface += extra_from_surface
        deficit -= extra_from_surface
        n_interior = min(size - n_surface, interior_ids.shape[0])

    chosen: list[int] = []
    if n_surface > 0:
        chosen.extend(rng.choice(surface_ids, size=n_surface, replace=False).tolist())
    if n_interior > 0:
        chosen.extend(rng.choice(interior_ids, size=n_interior, replace=False).tolist())

    return sorted(int(i) for i in chosen[:size])


def sample_distance_matched(
    size: int,
    target_mean_pairwise_distance: float,
    coords: np.ndarray,
    rng: np.random.Generator,
    attempts: int = 512,
) -> list[int]:
    """Sample *size* nodes whose mean pairwise Euclidean distance approximates *target_mean_pairwise_distance*.

    ``coords`` should have shape [num_nodes, spatial_dim] (typically 3).
    Falls back to same-size when *attempts* is exhausted.
    """
    num_nodes = coords.shape[0]
    size = max(1, min(size, num_nodes))

    def _mean_pairwise(indices: np.ndarray) -> float:
        pts = coords[indices]  # [size, D]
        # Full pairwise distances, upper triangle only
        n = pts.shape[0]
        if n < 2:
            return 0.0
        total = 0.0
        count = 0
        for i in range(n):
            di = pts[i] - pts[i + 1 :]
            total += float(np.linalg.norm(di, axis=1).sum())
            count += n - i - 1
        return total / max(count, 1)

    best_indices = rng.choice(np.arange(num_nodes), size=size, replace=False)
    best_distance = abs(_mean_pairwise(best_indices) - target_mean_pairwise_distance)

    for _ in range(attempts - 1):
        candidates = rng.choice(np.arange(num_nodes), size=size, replace=False)
        mpd = _mean_pairwise(candidates)
        dist = abs(mpd - target_mean_pairwise_distance)
        if dist < best_distance:
            best_distance = dist
            best_indices = candidates

    return sorted(int(i) for i in best_indices.tolist())


# ---------------------------------------------------------------------------
# Convenience builders
# ---------------------------------------------------------------------------

def build_null_samplers(
    target_size: int,
    target_degree_mean: float,
    target_surface_fraction: float,
    target_mean_pairwise_distance: float,
    node_types: np.ndarray,
    degree: np.ndarray,
    coords: np.ndarray,
    rng: np.random.Generator,
    eligible_ids: np.ndarray | None = None,
) -> dict[str, Sampler]:
    """Return a dict of named sampler callables keyed by null-model type.

    The returned samplers are zero-argument closures that draw one random region
    and return the sorted node-id list.
    """
    ids = np.arange(node_types.shape[0]) if eligible_ids is None else np.asarray(eligible_ids, dtype=np.int64)
    local_types, local_degree, local_coords = node_types[ids], degree[ids], coords[ids]
    def remap(draw):
        return [int(ids[i]) for i in draw]
    return {
        "same_size": lambda: remap(sample_same_size(target_size, ids.shape[0], rng)),
        "degree_matched": lambda: remap(sample_degree_matched(
            target_size, target_degree_mean, local_degree, rng
        )),
        "surface_fraction_matched": lambda: remap(sample_surface_fraction_matched(
            target_size, target_surface_fraction, local_types, rng
        )),
        "distance_matched": lambda: remap(sample_distance_matched(
            target_size, target_mean_pairwise_distance, local_coords, rng
        )),
    }


def sample_null_regions(
    samplers: dict[str, Sampler],
    n_per_type: int,
    types: list[str] | None = None,
) -> dict[str, list[list[int]]]:
    """Draw *n_per_type* regions for each null type in *types* (default: all).

    Returns ``{type_name: [[node_ids], ...]}``.
    """
    if types is None:
        types = list(samplers.keys())
    return {t: [samplers[t]() for _ in range(n_per_type)] for t in types if t in samplers}


# ---------------------------------------------------------------------------
# Utility: compute candidate region statistics from snapshot data
# ---------------------------------------------------------------------------

def _undirected_degree(edge_index, num_nodes: int) -> np.ndarray:
    """Compute undirected degree from edge_index (supports both torch.Tensor and numpy arrays)."""
    deg = np.zeros(num_nodes, dtype=np.float64)
    if torch is not None and isinstance(edge_index, torch.Tensor):
        src = edge_index[0].cpu().numpy()
        dst = edge_index[1].cpu().numpy()
    else:
        src = np.asarray(edge_index[0])
        dst = np.asarray(edge_index[1])
    for s, d in zip(src, dst):
        deg[int(s)] += 1.0
        deg[int(d)] += 1.0
    return deg


def compute_candidate_stats(
    candidate_nodes: list[int],
    node_types: np.ndarray,
    degree: np.ndarray,
    coords: np.ndarray,
) -> dict:
    """Compute target statistics for a candidate region (size, degree, surface fraction, mean pairwise distance)."""
    arr = np.array(candidate_nodes, dtype=np.int64)
    size = arr.shape[0]
    surface_fraction = float(np.mean(node_types[arr] == 0)) if size > 0 else 0.0
    degree_mean = float(np.mean(degree[arr])) if size > 0 else 0.0
    # Mean pairwise distance
    if size >= 2:
        pts = coords[arr]
        total = 0.0
        count = 0
        for i in range(size):
            di = pts[i] - pts[i + 1 :]
            total += float(np.linalg.norm(di, axis=1).sum())
            count += size - i - 1
        mpd = total / max(count, 1)
    else:
        mpd = 0.0
    return {
        "size": size,
        "degree_mean": degree_mean,
        "surface_fraction": surface_fraction,
        "mean_pairwise_distance": mpd,
    }


def load_snapshot_stats(snapshot_dir: str) -> dict[str, np.ndarray]:
    """Load node_types, degree, and coords from a snapshot directory.

    Requires torch to be installed.
    """
    if torch is None:
        raise ImportError("torch is required to load snapshot stats")
    node_types = torch.load(f"{snapshot_dir}/node_types.pt", map_location="cpu").detach().cpu().numpy()
    edge_index = torch.load(f"{snapshot_dir}/edge_index.pt", map_location="cpu")
    coords = torch.load(f"{snapshot_dir}/coords.pt", map_location="cpu").detach().cpu().numpy()
    degree = _undirected_degree(edge_index, node_types.shape[0])
    return {
        "node_types": node_types,
        "degree": degree,
        "coords": coords,
    }
