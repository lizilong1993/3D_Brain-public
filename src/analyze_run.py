from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from .evaluate import evaluate_model
from .analysis_regions import (
    build_analysis_graph,
    attach_activity_statistics,
    compare_region_maps,
    compute_region_statistics,
    detect_regions_louvain,
    detect_regions_spectral,
    identify_task_regions_nonlesion,
    predict_task_regions_without_lesion,
    identify_control_region,
    threshold_graph_by_weight,
)
from .graph_metrics import build_directed_analysis_graph, compute_graph_metrics
from .lesion import run_region_lesion_matrix
from .train import build_model_and_data, load_config
from .utils import ensure_dir, write_json
from .visualize import plot_graph_3d, plot_lesion_heatmap


def _load_latest_snapshot(run_dir: Path) -> Path:
    snapshot_root = run_dir / "snapshots"
    snapshots = sorted(snapshot_root.glob("step_*"))
    if not snapshots:
        raise FileNotFoundError(f"No snapshots found under {snapshot_root}")
    return snapshots[-1]


def _write_activity_feature_artifacts(analysis_dir: Path, region_stats: dict[int, dict], activity_available: bool) -> tuple[str, str]:
    activity_keys = [
        "mean_activation",
        "activation_std",
        "danger_flag_response",
        "ctrl_mask_correlation",
        "mem_mask_correlation",
        "pred_mask_correlation",
        "prediction_target_correlation",
        "prediction_activity_response",
        "temporal_autocorrelation",
        "activity_features_available",
    ]
    payload = {
        "activity_features_available": bool(activity_available),
        "regions": {
            str(region_id): {key: stats.get(key, 0.0 if key != "activity_features_available" else bool(activity_available)) for key in activity_keys}
            for region_id, stats in region_stats.items()
        },
    }
    json_path = analysis_dir / "activity_features.json"
    csv_path = analysis_dir / "region_activity_features.csv"
    write_json(json_path, payload)
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        fieldnames = ["region_id"] + activity_keys
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for region_id, stats in region_stats.items():
            writer.writerow({"region_id": int(region_id), **{key: stats.get(key, 0.0 if key != "activity_features_available" else False) for key in activity_keys}})
    return str(json_path), str(csv_path)


def analyze_run(config_path: str, run_dir: str) -> dict:
    config = load_config(config_path)
    run_dir_path = Path(run_dir)
    analysis_dir = ensure_dir(run_dir_path / "analysis")
    model, _train_gen, val_gen, _device = build_model_and_data(config)

    latest_ckpt = sorted((run_dir_path / "checkpoints").glob("step_*.pt"))[-1]
    ckpt = torch.load(latest_ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()

    if config.get("model_type") == "moe":
        metrics = evaluate_model(model, val_gen, config["validation_batches"], config)
        expert_usage = None
        batch = val_gen.sample_batch(config["batch_size"])
        outputs = model(batch["obs"], batch["mem_query"], batch["danger_flag"])
        if "expert_usage" in outputs:
            usage = outputs["expert_usage"].detach().cpu().tolist()
            expert_usage = usage
        results = {
            "config_path": config_path,
            "latest_checkpoint": str(latest_ckpt),
            "model_type": "moe",
            "metrics": metrics,
            "expert_usage": expert_usage,
        }
        write_json(analysis_dir / "results.json", results)
        return results

    latest_snapshot = _load_latest_snapshot(run_dir_path)
    edge_index = torch.load(latest_snapshot / "edge_index.pt", map_location="cpu")
    edge_weight = torch.load(latest_snapshot / "edge_weight.pt", map_location="cpu")
    coords = torch.load(latest_snapshot / "coords.pt", map_location="cpu")
    node_types = torch.load(latest_snapshot / "node_types.pt", map_location="cpu")

    edge_index_thr, edge_weight_thr = threshold_graph_by_weight(edge_index, edge_weight, config["analysis_weight_quantile"])
    graph = build_analysis_graph(edge_index_thr, edge_weight_thr)
    directed_graph = build_directed_analysis_graph(edge_index_thr, edge_weight_thr)
    region_map = detect_regions_louvain(graph)
    spectral_map = detect_regions_spectral(graph, len(set(region_map.values())))
    region_stats = compute_region_statistics(graph, region_map, coords, node_types, model, edge_index_thr)
    graph_metrics = compute_graph_metrics(graph, coords, node_types, region_map, directed_graph, model)
    ari = compare_region_maps(region_map, spectral_map)
    lesion_batches = val_gen.sample_dataset(config["validation_batches"], config["batch_size"])
    lesion_matrix = run_region_lesion_matrix(model, region_map, val_gen, config, cached_batches=lesion_batches)
    ctrl_region_id, ctrl_region_evidence = identify_control_region(region_stats, lesion_matrix)
    batch = val_gen.sample_batch(config["batch_size"])
    outputs = model(batch["obs"], batch["mem_query"], batch["danger_flag"])
    region_stats, activity_features_available = attach_activity_statistics(
        region_stats,
        region_map,
        outputs.get("hidden_traj", torch.empty(0)),
        batch,
        pred_logits=outputs.get("pred_logits"),
    )
    predicted_regions_nonlesion = predict_task_regions_without_lesion(region_stats, feature_set="structure_only")
    activity_features_path, region_activity_csv_path = _write_activity_feature_artifacts(analysis_dir, region_stats, activity_features_available)

    results = {
        "config_path": config_path,
        "latest_checkpoint": str(latest_ckpt),
        "model_type": config.get("model_type", "main"),
        "latest_snapshot": str(latest_snapshot),
        "num_regions": len(set(region_map.values())),
        "control_region_id": ctrl_region_id,
        "control_region_evidence": ctrl_region_evidence,
        "community_consistency_ari": ari,
        "activity_features_available": activity_features_available,
        "activity_features_path": activity_features_path,
        "region_activity_features_csv": region_activity_csv_path,
        "graph_metrics": graph_metrics,
        "predicted_regions_nonlesion": predicted_regions_nonlesion,
        "region_stats": region_stats,
        "lesion_matrix": lesion_matrix,
    }
    write_json(analysis_dir / "results.json", results)
    write_json(analysis_dir / "regions.json", {str(k): int(v) for k, v in region_map.items()})
    plot_graph_3d(coords, edge_index_thr, edge_weight_thr, node_types, region_map, analysis_dir / "graph_regions.png")
    plot_lesion_heatmap(lesion_matrix, analysis_dir / "lesion_heatmap.png")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a completed run.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    results = analyze_run(args.config, args.run_dir)
    if results.get("model_type") == "moe":
        print(f"model_type=moe expert_usage={results.get('expert_usage')}")
    else:
        print(f"regions={results['num_regions']} control_region_id={results['control_region_id']}")


if __name__ == "__main__":
    main()
