from __future__ import annotations

import csv
from pathlib import Path

import torch

from .utils import count_degrees, degree_limits_by_type, stable_correlation


def compute_activity_corr(model) -> torch.Tensor:
    if model.activity_buffer is None or model.activity_buffer.shape[0] < 4:
        return torch.zeros(model.edge_index.shape[1], device=model.edge_weight.device)
    src_activity = model.activity_buffer[:, model.edge_index[0]]
    dst_activity = model.activity_buffer[:, model.edge_index[1]]
    return stable_correlation(src_activity, dst_activity)


def compute_edge_scores(model, grad_ema: torch.Tensor | None, activity_corr: torch.Tensor | None) -> torch.Tensor:
    weight_term = torch.abs(model.edge_weight)
    if grad_ema is None:
        grad_term = torch.zeros_like(weight_term)
    else:
        grad_term = grad_ema
    if activity_corr is None:
        corr_term = torch.zeros_like(weight_term)
    else:
        corr_term = torch.relu(activity_corr)
    weight_term = weight_term / torch.clamp(weight_term.max(), min=1e-8)
    grad_term = grad_term / torch.clamp(grad_term.max(), min=1e-8)
    corr_term = corr_term / torch.clamp(corr_term.max(), min=1e-8)
    return 0.5 * weight_term + 0.3 * grad_term + 0.2 * corr_term


def prune_edges(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    scores: torch.Tensor,
    num_prune: int,
    protected_nodes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    keep_mask = torch.ones(edge_weight.shape[0], dtype=torch.bool, device=edge_weight.device)
    protected_set = set(protected_nodes.tolist())
    order = torch.argsort(scores)
    pruned_indices: list[int] = []
    for idx in order.tolist():
        if len(pruned_indices) >= num_prune:
            break
        src = int(edge_index[0, idx].item())
        dst = int(edge_index[1, idx].item())
        if src in protected_set or dst in protected_set:
            continue
        keep_mask[idx] = False
        pruned_indices.append(idx)
    return edge_index[:, keep_mask], edge_weight[keep_mask], pruned_indices


def sample_regrow_edges(
    model,
    num_new_edges: int,
    distance_matrix: torch.Tensor,
    activity_corr: torch.Tensor | None,
    degree_limits: torch.Tensor,
    geometry_constrained: bool = True,
) -> torch.Tensor:
    if num_new_edges <= 0:
        return torch.empty((2, 0), dtype=torch.long, device=model.edge_index.device)
    existing = set(map(tuple, model.edge_index.T.tolist()))
    device = model.edge_index.device
    num_nodes = model.num_nodes
    in_deg, out_deg = count_degrees(model.edge_index, num_nodes)
    new_edges: list[list[int]] = []

    candidates = []
    scores = []
    for src in range(num_nodes):
        for dst in range(num_nodes):
            if src == dst or (src, dst) in existing:
                continue
            if out_deg[src] >= degree_limits[src] or in_deg[dst] >= degree_limits[dst]:
                continue
            corr = 0.0
            if model.activity_buffer is not None and model.activity_buffer.shape[0] > 3:
                corr = float(torch.relu(stable_correlation(model.activity_buffer[:, src:src + 1], model.activity_buffer[:, dst:dst + 1])).item())
            if geometry_constrained:
                src_type = int(model.node_types[src].item())
                dst_type = int(model.node_types[dst].item())
                if src_type == 1 and dst_type == 1:
                    type_bias = 1.3
                elif src_type == 1 and dst_type == 0:
                    type_bias = 1.0
                elif src_type == 0 and dst_type == 1:
                    type_bias = 0.9
                else:
                    type_bias = 0.6
                distance_bias = float(torch.exp(-4.0 * distance_matrix[src, dst]).item())
            else:
                type_bias = 1.0
                distance_bias = 1.0
            score = distance_bias * (1.0 + corr) * type_bias
            candidates.append((src, dst))
            scores.append(score)
    if not candidates:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    score_tensor = torch.tensor(scores, dtype=torch.float32, device=device)
    sample_count = min(num_new_edges, len(candidates))
    if sample_count <= 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    chosen = torch.multinomial(score_tensor, num_samples=sample_count, replacement=False)
    for idx in chosen.tolist():
        src, dst = candidates[idx]
        new_edges.append([src, dst])
        out_deg[src] += 1
        in_deg[dst] += 1
    return torch.tensor(new_edges, dtype=torch.long, device=device).T if new_edges else torch.empty((2, 0), dtype=torch.long, device=device)


def initialize_new_edge_weights(num_new_edges: int, device: torch.device) -> torch.Tensor:
    return torch.randn(num_new_edges, device=device) * 0.05


def append_rewire_log(log_path: Path, row: dict[str, float]) -> None:
    file_exists = log_path.exists()
    with open(log_path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def apply_rewiring(model, config: dict, grad_ema: torch.Tensor | None = None, log_path: Path | None = None) -> dict[str, float]:
    num_edges = model.edge_weight.shape[0]
    num_prune = max(1, int(num_edges * config["rewire_fraction"]))
    activity_corr = compute_activity_corr(model)
    scores = compute_edge_scores(model, grad_ema, activity_corr)
    protected_nodes = torch.cat([model.input_node_ids, model.pred_readout_ids, model.ctrl_readout_ids]).unique()
    pruned_edge_index, pruned_edge_weight, pruned_indices = prune_edges(model.edge_index, model.edge_weight.detach(), scores, num_prune, protected_nodes)

    model.edge_index = pruned_edge_index
    model.edge_weight = torch.nn.Parameter(pruned_edge_weight.clone())

    geometry_constrained = bool(config.get("geometry_constrained", True))
    if geometry_constrained:
        degree_limits = degree_limits_by_type(model.node_types, config["surface_degree_limit"], config["interior_degree_limit"])
    else:
        uniform_limit = max(int(config["surface_degree_limit"]), int(config["interior_degree_limit"]))
        degree_limits = torch.full_like(model.node_types, uniform_limit, dtype=torch.long)
    delta = model.coords[:, None, :] - model.coords[None, :, :]
    distance_matrix = torch.sqrt(torch.sum(delta * delta, dim=-1) + 1e-12)
    new_edges = sample_regrow_edges(
        model,
        len(pruned_indices),
        distance_matrix,
        activity_corr,
        degree_limits,
        geometry_constrained=geometry_constrained,
    )
    new_weights = initialize_new_edge_weights(new_edges.shape[1], model.edge_weight.device)

    if new_edges.numel() > 0:
        model.edge_index = torch.cat([model.edge_index, new_edges], dim=1)
        model.edge_weight = torch.nn.Parameter(torch.cat([model.edge_weight.detach(), new_weights], dim=0))

    metrics = {
        "num_pruned": float(len(pruned_indices)),
        "num_added": float(new_edges.shape[1]),
        "edge_budget": float(model.edge_weight.shape[0]),
        "mean_score": float(scores.mean().item()),
    }
    if log_path is not None:
        append_rewire_log(log_path, metrics)
    return metrics
