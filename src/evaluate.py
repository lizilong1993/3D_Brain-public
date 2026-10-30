from __future__ import annotations

import torch


def compute_prediction_metrics(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, float]:
    mse = torch.mean((outputs["pred_logits"] - batch["pred_target"]) ** 2)
    return {"Pred-MSE": float(mse.item())}


def compute_memory_metrics(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, float]:
    preds = torch.argmax(outputs["mem_logits"], dim=-1)
    mask = batch["mem_mask"] > 0
    if mask.sum() == 0:
        acc = 0.0
    else:
        acc = float((preds[mask] == batch["mem_target"][mask]).float().mean().item())
    return {"Mem-Acc": acc}


def compute_control_metrics(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, float]:
    preds = torch.argmax(outputs["ctrl_logits"], dim=-1)
    mask = batch["ctrl_mask"] > 0
    if mask.sum() == 0:
        return {"Ctrl-Acc": 0.0}
    acc = float((preds[mask] == batch["ctrl_target"][mask]).float().mean().item())
    return {"Ctrl-Acc": acc}


def compute_ctrl_stability(state_c: torch.Tensor, ctrl_logits: torch.Tensor, ctrl_mask: torch.Tensor) -> dict[str, float]:
    excess = torch.clamp(torch.abs(state_c.squeeze(-1)) - 1.0, min=0.0)
    stability = 1.0 - float(excess.mean().item())
    mask = ctrl_mask > 0
    if mask.sum() == 0:
        ttc = 0.0
    else:
        danger_steps = torch.where(mask)
        ttc = float(danger_steps[1].float().mean().item())
    return {"Ctrl-Stability": stability, "Time-to-correct": ttc}


@torch.no_grad()
def evaluate_model_on_batches(model, batches: list[dict[str, torch.Tensor]]) -> dict[str, float]:
    model.eval()
    metric_sums: dict[str, float] = {}
    if not batches:
        model.train()
        return metric_sums
    for batch in batches:
        outputs = model(batch["obs"], batch["mem_query"], batch["danger_flag"])
        metrics = {}
        metrics.update(compute_prediction_metrics(outputs, batch))
        metrics.update(compute_memory_metrics(outputs, batch))
        metrics.update(compute_control_metrics(outputs, batch))
        metrics.update(compute_ctrl_stability(batch["state_c"], outputs["ctrl_logits"], batch["ctrl_mask"]))
        for key, value in metrics.items():
            metric_sums[key] = metric_sums.get(key, 0.0) + value
    model.train()
    return {key: value / len(batches) for key, value in metric_sums.items()}


@torch.no_grad()
def evaluate_model(model, generator, num_batches: int, config: dict) -> dict[str, float]:
    batches = generator.sample_dataset(num_batches, config["batch_size"])
    return evaluate_model_on_batches(model, batches)
