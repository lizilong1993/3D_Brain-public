from __future__ import annotations

import copy

import torch

from .evaluate import evaluate_model, evaluate_model_on_batches


def recompute_outputs_from_hidden(model, hidden_traj: torch.Tensor) -> dict[str, torch.Tensor]:
    pred_logits = model.pred_head(hidden_traj[:, :, model.pred_readout_ids])
    mem_logits = model.mem_head(hidden_traj)
    # Use strategy-aware control readout if available
    if hasattr(model, "ctrl_readout_from_hidden"):
        ctrl_logits = model.ctrl_readout_from_hidden(hidden_traj)
    elif hasattr(model, "ctrl_readout_ids"):
        ctrl_logits = model.ctrl_head(hidden_traj[:, :, model.ctrl_readout_ids])
    else:
        ctrl_logits = model.ctrl_head(hidden_traj)
    return {
        "pred_logits": pred_logits,
        "mem_logits": mem_logits,
        "ctrl_logits": ctrl_logits,
        "hidden_traj": hidden_traj,
    }


def project_out_subspace(hidden_traj: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    if basis.numel() == 0:
        return hidden_traj
    flat_hidden = hidden_traj.reshape(-1, hidden_traj.shape[-1])
    projection = flat_hidden @ basis @ basis.T
    return (flat_hidden - projection).reshape_as(hidden_traj)


def inject_subspace_noise(hidden_traj: torch.Tensor, basis: torch.Tensor, noise_scale: float) -> torch.Tensor:
    if basis.numel() == 0:
        return hidden_traj
    flat_hidden = hidden_traj.reshape(-1, hidden_traj.shape[-1])
    coeff = torch.randn(flat_hidden.shape[0], basis.shape[1], device=flat_hidden.device) * noise_scale
    noise = coeff @ basis.T
    return (flat_hidden + noise).reshape_as(hidden_traj)


@torch.no_grad()
def run_hidden_trajectory_perturbation(model, batches: list[dict[str, torch.Tensor]], perturb_fn) -> dict[str, float]:
    original_forward = model.forward

    def perturbed_forward(obs, mem_query, danger_flag):
        outputs = original_forward(obs, mem_query, danger_flag)
        hidden_traj = perturb_fn(outputs["hidden_traj"])
        return recompute_outputs_from_hidden(model, hidden_traj)

    model.forward = perturbed_forward  # type: ignore[assignment]
    try:
        return evaluate_model_on_batches(model, batches)
    finally:
        model.forward = original_forward  # type: ignore[assignment]


@torch.no_grad()
def run_node_lesion(model, region_nodes: list[int], generator, num_batches: int, config: dict, cached_batches: list[dict[str, torch.Tensor]] | None = None) -> dict[str, float]:
    """Run a recurrent node lesion, masking nodes after every hidden update.

    The previous implementation zeroed a completed trajectory, which changed only
    readout values and left downstream recurrent dynamics intact.
    """
    if not hasattr(model, "active_node_mask"):
        raise TypeError("model does not expose recurrent node masking")
    device = next(model.parameters()).device
    width = getattr(model, "num_nodes", getattr(model, "hidden_dim", None))
    if width is None:
        raise TypeError("model does not expose recurrent state width")
    mask = torch.ones(width, dtype=torch.bool, device=device)
    mask[torch.as_tensor(region_nodes, dtype=torch.long, device=device)] = False
    previous_mask = model.active_node_mask
    model.active_node_mask = mask
    try:
        if cached_batches is not None:
            return evaluate_model_on_batches(model, cached_batches)
        return evaluate_model(model, generator, num_batches, config)
    finally:
        model.active_node_mask = previous_mask


@torch.no_grad()
def run_interface_lesion(model, region_nodes: list[int], generator, num_batches: int, config: dict) -> dict[str, float]:
    region_tensor = torch.tensor(region_nodes, device=model.edge_index.device)
    surface_mask = model.node_types == 0
    surface_nodes = torch.where(surface_mask)[0]
    src_in_region = torch.isin(model.edge_index[0], region_tensor)
    dst_in_region = torch.isin(model.edge_index[1], region_tensor)
    src_surface = torch.isin(model.edge_index[0], surface_nodes)
    dst_surface = torch.isin(model.edge_index[1], surface_nodes)
    keep_mask = ~((src_in_region & dst_surface) | (dst_in_region & src_surface))

    backup_edge_index = model.edge_index.clone()
    backup_edge_weight = model.edge_weight.detach().clone()
    model.edge_index = model.edge_index[:, keep_mask]
    model.edge_weight = torch.nn.Parameter(model.edge_weight.detach()[keep_mask].clone())
    try:
        return evaluate_model(model, generator, num_batches, config)
    finally:
        model.edge_index = backup_edge_index
        model.edge_weight = torch.nn.Parameter(backup_edge_weight)


def compute_lesion_specialization(delta_pred: float, delta_mem: float, delta_ctrl: float, target_task: str) -> float:
    if target_task == "ctrl":
        return delta_ctrl - 0.5 * (delta_pred + delta_mem)
    if target_task == "mem":
        return delta_mem - 0.5 * (delta_pred + delta_ctrl)
    return delta_pred - 0.5 * (delta_mem + delta_ctrl)


def summarize_lesion_effects(baseline: dict[str, float], lesioned: dict[str, float]) -> dict[str, float]:
    delta_pred = lesioned["Pred-MSE"] - baseline["Pred-MSE"]
    delta_mem = baseline["Mem-Acc"] - lesioned["Mem-Acc"]
    delta_ctrl = baseline["Ctrl-Acc"] - lesioned["Ctrl-Acc"]
    return {
        "Delta Pred": delta_pred,
        "Delta Mem": delta_mem,
        "Delta Ctrl": delta_ctrl,
        "LS_ctrl": compute_lesion_specialization(delta_pred, delta_mem, delta_ctrl, "ctrl"),
        "LS_mem": compute_lesion_specialization(delta_pred, delta_mem, delta_ctrl, "mem"),
        "LS_pred": compute_lesion_specialization(delta_pred, delta_mem, delta_ctrl, "pred"),
    }


def run_region_lesion_matrix(model, region_map: dict[int, int], generator, config: dict, cached_batches: list[dict[str, torch.Tensor]] | None = None) -> dict:
    if cached_batches is not None:
        baseline = evaluate_model_on_batches(model, cached_batches)
    else:
        baseline = evaluate_model(model, generator, config["validation_batches"], config)
    matrix = {}
    unique_regions = sorted(set(region_map.values()))
    for region_id in unique_regions:
        nodes = [node for node, rid in region_map.items() if rid == region_id]
        lesioned = run_node_lesion(model, nodes, generator, config["validation_batches"], config, cached_batches=cached_batches)
        matrix[region_id] = summarize_lesion_effects(baseline, lesioned)
    return matrix
