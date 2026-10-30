from __future__ import annotations

import math

import torch

from .utils import count_degrees, degree_limits_by_type


def build_node_coordinates(num_surface: int, num_interior: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    surface_coords = fibonacci_sphere(num_surface, radius=1.0, device=device)
    interior_coords = sample_interior_nodes(num_interior, max_radius=0.92, device=device)
    coords = torch.cat([surface_coords, interior_coords], dim=0)
    node_types = torch.cat(
        [
            torch.zeros(num_surface, dtype=torch.long, device=device),
            torch.ones(num_interior, dtype=torch.long, device=device),
        ],
        dim=0,
    )
    return coords, node_types


def fibonacci_sphere(num_points: int, radius: float, device: torch.device) -> torch.Tensor:
    points = []
    phi = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(num_points):
        y = 1 - (i / float(max(num_points - 1, 1))) * 2
        radius_xy = math.sqrt(max(0.0, 1 - y * y))
        theta = phi * i
        x = math.cos(theta) * radius_xy
        z = math.sin(theta) * radius_xy
        points.append([radius * x, radius * y, radius * z])
    return torch.tensor(points, dtype=torch.float32, device=device)


def sample_interior_nodes(num_points: int, max_radius: float, device: torch.device) -> torch.Tensor:
    directions = torch.randn(num_points, 3, device=device)
    directions = directions / torch.sqrt(torch.sum(directions * directions, dim=1, keepdim=True) + 1e-12)
    u = torch.rand(num_points, 1, device=device)
    radii = torch.pow(u, 1.0 / 3.0) * max_radius
    return directions * radii


def compute_pairwise_distances(coords: torch.Tensor) -> torch.Tensor:
    delta = coords[:, None, :] - coords[None, :, :]
    return torch.sqrt(torch.sum(delta * delta, dim=-1) + 1e-12)


def _sample_edges_for_type(
    source_ids: torch.Tensor,
    target_ids: torch.Tensor,
    num_edges: int,
    distances: torch.Tensor,
    selected: set[tuple[int, int]],
    degree_limits: torch.Tensor,
) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    num_nodes = distances.shape[0]
    current_in = torch.zeros(num_nodes, dtype=torch.long, device=distances.device)
    current_out = torch.zeros(num_nodes, dtype=torch.long, device=distances.device)

    if selected:
        selected_tensor = torch.tensor(list(selected), dtype=torch.long, device=distances.device)
        ones = torch.ones(selected_tensor.shape[0], dtype=torch.long, device=distances.device)
        current_out.index_add_(0, selected_tensor[:, 0], ones)
        current_in.index_add_(0, selected_tensor[:, 1], ones)

    source_grid = source_ids.repeat_interleave(target_ids.numel())
    target_grid = target_ids.repeat(source_ids.numel())
    valid_mask = source_grid != target_grid
    source_grid = source_grid[valid_mask]
    target_grid = target_grid[valid_mask]
    edge_scores = torch.exp(-4.0 * distances[source_grid, target_grid])
    order = torch.multinomial(edge_scores, num_samples=edge_scores.numel(), replacement=False)

    for idx in order.tolist():
        if len(edges) >= num_edges:
            break
        src = int(source_grid[idx].item())
        dst = int(target_grid[idx].item())
        if (src, dst) in selected:
            continue
        if current_out[src] >= degree_limits[src] or current_in[dst] >= degree_limits[dst]:
            continue
        selected.add((src, dst))
        edges.append((src, dst))
        current_out[src] += 1
        current_in[dst] += 1
    return edges


def build_initial_sparse_graph(
    coords: torch.Tensor,
    node_types: torch.Tensor,
    edge_budget: int,
    degree_limits: dict[str, int],
    geometry_constrained: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    distances = compute_pairwise_distances(coords)
    surface_ids = torch.where(node_types == 0)[0]
    interior_ids = torch.where(node_types == 1)[0]
    if geometry_constrained:
        limit_tensor = degree_limits_by_type(node_types, degree_limits["surface"], degree_limits["interior"])
    else:
        uniform_limit = max(int(degree_limits["surface"]), int(degree_limits["interior"]))
        limit_tensor = torch.full_like(node_types, uniform_limit, dtype=torch.long)

    selected: set[tuple[int, int]] = set()
    edges: list[tuple[int, int]] = []
    if geometry_constrained:
        base_quotas = {"ss": 200, "si": 160, "is": 160, "ii": 440}
        scale = edge_budget / float(sum(base_quotas.values()))
        quotas = {key: max(1, int(round(value * scale))) for key, value in base_quotas.items()}
        quota_sum = sum(quotas.values())
        quotas["ii"] += edge_budget - quota_sum
        edges.extend(_sample_edges_for_type(surface_ids, surface_ids, quotas["ss"], distances, selected, limit_tensor))
        edges.extend(_sample_edges_for_type(surface_ids, interior_ids, quotas["si"], distances, selected, limit_tensor))
        edges.extend(_sample_edges_for_type(interior_ids, surface_ids, quotas["is"], distances, selected, limit_tensor))
        edges.extend(_sample_edges_for_type(interior_ids, interior_ids, quotas["ii"], distances, selected, limit_tensor))
    else:
        all_ids = torch.arange(coords.shape[0], device=coords.device)
        edges.extend(_sample_edges_for_type(all_ids, all_ids, edge_budget, distances * 0.0, selected, limit_tensor))

    if len(edges) != edge_budget:
        # Adjust budget to actual created edge count (degree limits may prevent full fill)
        print(f"  Graph edge count mismatch: created {len(edges)}, requested {edge_budget}. Adjusting budget.")
        edge_budget_actual = len(edges)

    edge_index = torch.tensor(edges, dtype=torch.long, device=coords.device).T
    actual_count = edge_index.shape[1]
    edge_weight = torch.randn(actual_count, dtype=torch.float32, device=coords.device) * 0.05
    return edge_index, edge_weight
