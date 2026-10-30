from __future__ import annotations

import torch
import torch.nn.functional as F


def prediction_loss(pred_logits: torch.Tensor, pred_target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred_logits, pred_target)


def memory_loss(mem_logits: torch.Tensor, mem_target: torch.Tensor, mem_mask: torch.Tensor) -> torch.Tensor:
    flat_logits = mem_logits.reshape(-1, mem_logits.shape[-1])
    flat_target = mem_target.reshape(-1)
    flat_mask = mem_mask.reshape(-1)
    losses = F.cross_entropy(flat_logits, flat_target, reduction="none")
    return (losses * flat_mask).sum() / torch.clamp(flat_mask.sum(), min=1.0)


def control_loss(ctrl_logits: torch.Tensor, ctrl_target: torch.Tensor, ctrl_mask: torch.Tensor) -> torch.Tensor:
    flat_logits = ctrl_logits.reshape(-1, ctrl_logits.shape[-1])
    flat_target = ctrl_target.reshape(-1)
    flat_mask = ctrl_mask.reshape(-1)
    losses = F.cross_entropy(flat_logits, flat_target, reduction="none")
    return (losses * flat_mask).sum() / torch.clamp(flat_mask.sum(), min=1.0)


def wire_length_penalty(edge_index: torch.Tensor, edge_weight: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    src = coords[edge_index[0]]
    dst = coords[edge_index[1]]
    distances = torch.sqrt(torch.sum((src - dst) ** 2, dim=-1) + 1e-12)
    return torch.mean(torch.abs(edge_weight) * distances)


def surface_shortcut_penalty(edge_index: torch.Tensor, edge_weight: torch.Tensor, node_types: torch.Tensor) -> torch.Tensor:
    is_surface_edge = (node_types[edge_index[0]] == 0) & (node_types[edge_index[1]] == 0)
    if is_surface_edge.sum() == 0:
        return torch.zeros((), device=edge_weight.device)
    return torch.mean(torch.abs(edge_weight[is_surface_edge]))


def total_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], model, config: dict) -> dict[str, torch.Tensor]:
    loss_pred = prediction_loss(outputs["pred_logits"], batch["pred_target"])
    loss_mem = memory_loss(outputs["mem_logits"], batch["mem_target"], batch["mem_mask"])
    if config.get("enable_control_loss", True):
        loss_ctrl = control_loss(outputs["ctrl_logits"], batch["ctrl_target"], batch["ctrl_mask"])
    else:
        loss_ctrl = torch.zeros((), device=loss_pred.device)
    if hasattr(model, "edge_index") and hasattr(model, "edge_weight") and hasattr(model, "coords"):
        loss_len = wire_length_penalty(model.edge_index, model.edge_weight, model.coords)
    else:
        loss_len = torch.zeros((), device=loss_pred.device)
    if hasattr(model, "edge_index") and hasattr(model, "edge_weight") and hasattr(model, "node_types"):
        loss_surf = surface_shortcut_penalty(model.edge_index, model.edge_weight, model.node_types)
    else:
        loss_surf = torch.zeros((), device=loss_pred.device)

    loss_total = (
        config["lambda_pred"] * loss_pred
        + config["lambda_mem"] * loss_mem
        + config["lambda_ctrl"] * loss_ctrl
        + config["lambda_len"] * loss_len
        + config["lambda_surf"] * loss_surf
    )

    # Adapter convergence regularizer (Phase-6)
    loss_conv_overlap = torch.zeros((), device=loss_pred.device)
    loss_conv_entropy = torch.zeros((), device=loss_pred.device)
    if hasattr(model, "adapter_convergence_regularizer") and model.adapter_convergence_regularizer:
        conv_losses = model.compute_adapter_convergence_loss()
        loss_conv_overlap = model.adapter_channel_diversity_weight * conv_losses["channel_overlap"]
        loss_conv_entropy = model.adapter_entropy_floor_weight * conv_losses["entropy_shortfall"]
        if not model.adapter_regularizer_diagnostic_only:
            loss_total = loss_total + loss_conv_overlap + loss_conv_entropy

    return {
        "loss_total": loss_total,
        "loss_pred": loss_pred,
        "loss_mem": loss_mem,
        "loss_ctrl": loss_ctrl,
        "loss_len": loss_len,
        "loss_surf": loss_surf,
        "loss_conv_overlap": loss_conv_overlap,
        "loss_conv_entropy": loss_conv_entropy,
    }
