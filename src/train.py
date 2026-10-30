from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch

from .analysis_regions import build_analysis_graph, compute_region_statistics, detect_regions_louvain, identify_control_region, threshold_graph_by_weight
from .data_generator import SequenceWorldGenerator
from .evaluate import evaluate_model
from .losses import total_loss
from .model_registry import build_model
from .rewiring import apply_rewiring
from .task_ecologies import resolve_ecology_config
from .utils import ensure_dir, load_yaml, make_device, merge_dict, set_seed, write_json, write_yaml


def load_config(path: str) -> dict:
    config_path = Path(path).resolve()
    project_root = Path(__file__).resolve().parents[1]
    base_path = project_root / "configs" / "default.yaml"
    base = load_yaml(base_path)
    override = load_yaml(config_path)
    return resolve_ecology_config(merge_dict(base, override))


def build_generator(config: dict, device: torch.device) -> SequenceWorldGenerator:
    return SequenceWorldGenerator(
        seq_len=config["seq_len"],
        obs_noise_std=config["obs_noise_std"],
        memory_delay=config["memory_delay"],
        danger_threshold=config["danger_threshold"],
        world_type=config["world_type"],
        control_action_bins=config["control_action_bins"],
        device=device,
        cue_probability=config["cue_probability"],
        shock_probability=config["shock_probability"],
        shock_scale_low=config["shock_scale_low"],
        shock_scale_high=config["shock_scale_high"],
        control_drift=config["control_drift"],
        control_gain=config["control_gain"],
        control_noise_std=config["control_noise_std"],
    )


def build_model_and_data(config: dict) -> tuple:
    device = make_device(config["device"])
    model = build_model(config, device)
    train_generator = build_generator(config, device)
    val_generator = build_generator(config, device)
    return model, train_generator, val_generator, device


def train_one_step(model, batch: dict, optimizer, config: dict) -> dict[str, float]:
    optimizer.zero_grad()
    outputs = model(batch["obs"], batch["mem_query"], batch["danger_flag"])
    losses = total_loss(outputs, batch, model, config)
    losses["loss_total"].backward()
    optimizer.step()
    return {key: float(value.detach().cpu().item()) for key, value in losses.items()}


def run_validation(model, val_generator, config: dict) -> dict[str, float]:
    return evaluate_model(model, val_generator, config["validation_batches"], config)


def save_checkpoint(run_dir: Path, step: int, model, optimizer, metrics: dict) -> None:
    ckpt_dir = ensure_dir(run_dir / "checkpoints")
    torch.save(
        {
            "step": step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metrics": metrics,
        },
        ckpt_dir / f"step_{step:06d}.pt",
    )


def load_checkpoint(model, optimizer, checkpoint_path: str | None) -> int:
    if not checkpoint_path:
        return 0
    payload = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(payload["model_state"], strict=False)
    optimizer.load_state_dict(payload["optimizer_state"])
    return int(payload.get("step", 0))


def save_structure_snapshot(run_dir: Path, step: int, model, metrics: dict) -> None:
    snapshot_dir = ensure_dir(run_dir / "snapshots" / f"step_{step:06d}")
    torch.save(model.edge_index.detach().cpu(), snapshot_dir / "edge_index.pt")
    torch.save(model.edge_weight.detach().cpu(), snapshot_dir / "edge_weight.pt")
    if hasattr(model, "coords"):
        torch.save(model.coords.detach().cpu(), snapshot_dir / "coords.pt")
        torch.save(model.node_types.detach().cpu(), snapshot_dir / "node_types.pt")
    write_json(snapshot_dir / "metrics.json", metrics)


def append_train_log(log_path: Path, row: dict[str, float]) -> None:
    file_exists = log_path.exists()
    with open(log_path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _git_commit(project_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_run_manifest(run_dir: Path, config_path: str, config: dict) -> None:
    project_root = Path(__file__).resolve().parents[1]
    resolved_config = run_dir / "resolved_config.yaml"
    write_yaml(resolved_config, config)
    config_hash = hashlib.sha256(resolved_config.read_bytes()).hexdigest()
    manifest = {
        "config_path": str(Path(config_path).resolve()),
        "resolved_config": str(resolved_config),
        "resolved_config_sha256": config_hash,
        "seed": config.get("seed"),
        "git_commit": _git_commit(project_root),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": config.get("device"),
    }
    with open(run_dir / "run_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def train_run(config_path: str, run_dir: str, seed_override: int | None = None) -> None:
    config = load_config(config_path)
    if seed_override is not None:
        config["seed"] = seed_override
    set_seed(config["seed"])
    run_dir_path = ensure_dir(run_dir)
    _write_run_manifest(run_dir_path, config_path, config)
    model, train_generator, val_generator, _device = build_model_and_data(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    start_step = load_checkpoint(model, optimizer, config.get("resume_from_checkpoint"))
    train_log_path = run_dir_path / "train_log.csv"
    rewire_log_path = run_dir_path / "rewire_logs.csv"

    for local_step in range(1, config["train_steps"] + 1):
        step = start_step + local_step
        batch = train_generator.sample_batch(config["batch_size"])
        loss_metrics = train_one_step(model, batch, optimizer, config)
        row = {"step": float(step), **loss_metrics}
        append_train_log(train_log_path, row)

        if hasattr(model, "edge_weight") and step > config["warmup_steps"] and step % config["rewire_every"] == 0 and config.get("model_type", "main") != "moe":
            apply_rewiring(model, config, log_path=rewire_log_path)

        if step % config["val_interval"] == 0:
            val_metrics = run_validation(model, val_generator, config)
            append_train_log(train_log_path, {"step": float(step), **val_metrics})

        if step % config["checkpoint_interval"] == 0:
            val_metrics = run_validation(model, val_generator, config)
            save_checkpoint(run_dir_path, step, model, optimizer, val_metrics)
            if hasattr(model, "edge_weight"):
                save_structure_snapshot(run_dir_path, step, model, val_metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train demand-driven 3D regionalization models.")
    parser.add_argument("--config", required=True, help="Path to YAML config override file.")
    parser.add_argument("--run-dir", help="Directory to save logs and checkpoints.")
    parser.add_argument("--output-dir", help="Alias for --run-dir.")
    parser.add_argument("--seed", type=int, help="Override the seed from the YAML config.")
    args = parser.parse_args()
    run_dir = args.output_dir or args.run_dir
    if not run_dir:
        raise ValueError("one of --run-dir or --output-dir is required")
    train_run(args.config, run_dir, seed_override=args.seed)



if __name__ == "__main__":
    main()
