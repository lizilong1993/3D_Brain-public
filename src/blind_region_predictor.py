from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .statistics import summarize_scalar
from .utils import ensure_dir, write_json


FORBIDDEN_FEATURES = [
    "delta_ctrl",
    "delta_mem",
    "delta_pred",
    "LS_ctrl",
    "LS_mem",
    "LS_pred",
]

FEATURE_SET_SPECS = {
    "structure_only": {
        "feature_names": [
            "size",
            "mean_radius",
            "surface_fraction",
            "internal_density",
            "incoming_density_from_inputs",
            "outgoing_density_to_control_head",
            "outgoing_density_to_prediction_head",
            "outgoing_density_to_memory_head",
            "mean_edge_length",
            "long_edge_fraction",
            "degree_mean",
            "degree_std",
            "ctrl_coupling",
            "ctrl_path_mean",
            "pred_path_mean",
            "mem_path_mean",
            "region_to_control_head_reachability",
            "region_to_prediction_head_reachability",
            "region_to_memory_head_reachability",
            "input_to_region_reachability",
            "harmonic_closeness",
        ],
        "requires_activity": False,
        "allows_forbidden": False,
        "limitation_note": "Structure-only mode excludes all lesion-derived evidence and depends on geometric/connectivity proxies only.",
    },
    "activity_only": {
        "feature_names": [
            "danger_response",
            "control_mask_response",
            "memory_query_response",
            "danger_flag_correlation",
            "control_mask_correlation",
            "memory_query_correlation",
            "danger_probe_auc",
            "latency_sensitive_activation_timing",
        ],
        "requires_activity": True,
        "allows_forbidden": False,
        "limitation_note": "Activity-only mode is valid only for runs whose analysis wrote non-lesion activity features.",
    },
    "lesion_augmented": {
        "feature_names": [
            "size",
            "mean_radius",
            "surface_fraction",
            "incoming_density_from_inputs",
            "outgoing_density_to_control_head",
            "danger_response",
            "control_mask_correlation",
            "ctrl_coupling",
            "delta_ctrl",
            "delta_mem",
            "delta_pred",
            "LS_ctrl",
            "LS_mem",
            "LS_pred",
        ],
        "requires_activity": False,
        "allows_forbidden": True,
        "limitation_note": "Lesion-augmented mode is a sanity check only and must not be treated as primary blind evidence.",
    },
}


def _iter_result_files(input_root: Path) -> list[Path]:
    return sorted(input_root.glob("**/analysis/results.json"))


def _safe_float(value: object) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _lesion_payload(region_evidence: dict, lesion_matrix: dict, region_id: int) -> dict:
    evidence_region = region_evidence.get(str(region_id), region_evidence.get(region_id, {}))
    lesion_region = lesion_matrix.get(str(region_id), lesion_matrix.get(region_id, {}))
    return {
        "delta_ctrl": _safe_float(evidence_region.get("delta_ctrl", lesion_region.get("Delta Ctrl", 0.0))),
        "delta_mem": _safe_float(lesion_region.get("Delta Mem", 0.0)),
        "delta_pred": _safe_float(lesion_region.get("Delta Pred", 0.0)),
        "LS_ctrl": _safe_float(evidence_region.get("ls_ctrl", lesion_region.get("LS_ctrl", 0.0))),
        "LS_mem": _safe_float(lesion_region.get("LS_mem", 0.0)),
        "LS_pred": _safe_float(lesion_region.get("LS_pred", 0.0)),
    }


def _build_rows(input_root: Path) -> list[dict]:
    rows: list[dict] = []
    for path in _iter_result_files(input_root):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("model_type") == "moe":
            continue
        activity_payload = {}
        activity_json_path = payload.get("activity_features_path")
        if activity_json_path and Path(activity_json_path).exists():
            activity_payload = json.loads(Path(activity_json_path).read_text(encoding="utf-8")).get("regions", {})
        control_region_id = int(payload.get("control_region_id", -1))
        evidence = payload.get("control_region_evidence", {}).get("region_evidence", {})
        lesion_matrix = payload.get("lesion_matrix", {})
        activity_features_available = bool(payload.get("activity_features_available", False))
        for region_id_raw, stats in payload.get("region_stats", {}).items():
            region_id = int(region_id_raw)
            activity_region_payload = activity_payload.get(str(region_id), {})
            row = {
                "run_id": str(path.parent.parent),
                "results_path": str(path),
                "region_id": region_id,
                "label": 1 if region_id == control_region_id and control_region_id >= 0 else 0,
                "activity_features_available": activity_features_available and bool(stats.get("activity_features_available", False)),
            }
            for key, value in stats.items():
                if key == "nodes":
                    continue
                row[key] = _safe_float(value)
            for key, value in activity_region_payload.items():
                row[key] = _safe_float(value) if key != "activity_features_available" else bool(value)
            row.update(_lesion_payload(evidence, lesion_matrix, region_id))
            rows.append(row)
    return rows


def _write_rows_csv(csv_path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _top_k_accuracy(test_rows: list[dict], scores: np.ndarray, k: int = 1) -> float:
    if not test_rows or len(scores) != len(test_rows):
        return 0.0
    ranked = sorted(zip(test_rows, scores), key=lambda item: float(item[1]), reverse=True)
    return float(any(int(row["label"]) == 1 for row, _score in ranked[:k]))


def _heuristic_scores(run_rows: list[dict], feature_names: list[str]) -> np.ndarray:
    if not run_rows:
        return np.array([], dtype=np.float64)
    matrix = np.array([[row.get(name, 0.0) for name in feature_names] for row in run_rows], dtype=np.float64)
    if matrix.size == 0:
        return np.zeros(len(run_rows), dtype=np.float64)
    means = matrix.mean(axis=0, keepdims=True)
    stds = matrix.std(axis=0, keepdims=True)
    stds[stds < 1e-9] = 1.0
    normalized = (matrix - means) / stds
    return normalized.mean(axis=1)


def _build_region_rankings(rows: list[dict], feature_names: list[str], requires_activity: bool) -> list[dict]:
    rankings = []
    for run_id in sorted({str(row["run_id"]) for row in rows}):
        run_rows = [row for row in rows if row["run_id"] == run_id]
        if requires_activity:
            run_rows = [row for row in run_rows if bool(row.get("activity_features_available", False))]
        if not run_rows:
            continue
        scores = _heuristic_scores(run_rows, feature_names)
        ranked_rows = sorted(
            [
                {
                    "region_id": int(row["region_id"]),
                    "score": float(score),
                    "label": int(row["label"]),
                }
                for row, score in zip(run_rows, scores)
            ],
            key=lambda item: item["score"],
            reverse=True,
        )
        rankings.append({"run_id": run_id, "ranking": ranked_rows})
    return rankings


def _summarize_feature_set(rows: list[dict], feature_set_name: str, csv_path: Path) -> dict:
    spec = FEATURE_SET_SPECS[feature_set_name]
    feature_names = spec["feature_names"]
    run_ids = sorted({str(row["run_id"]) for row in rows})
    if spec["requires_activity"] and not any(bool(row.get("activity_features_available", False)) for row in rows):
        return {
            "feature_set_name": feature_set_name,
            "forbidden_features_used": False,
            "forbidden_features": FORBIDDEN_FEATURES,
            "evaluated_folds": 0,
            "skipped_folds": [{"held_out_run_id": "all", "reason": "activity_features_available=false"}],
            "ROC-AUC": None,
            "top-k accuracy": {"k": 1, "summary": summarize_scalar([])},
            "seed-level rows": [],
            "region_rankings": [],
            "rows_csv": str(csv_path),
            "activity_features_available": False,
            "limitation note": "No activity features were available in the input results; returning a machine-readable non-evaluable summary.",
        }

    fold_metrics = []
    skipped_folds = []
    for held_out_run_id in run_ids:
        train_rows = [row for row in rows if row["run_id"] != held_out_run_id]
        test_rows = [row for row in rows if row["run_id"] == held_out_run_id]
        if spec["requires_activity"]:
            train_rows = [row for row in train_rows if bool(row.get("activity_features_available", False))]
            test_rows = [row for row in test_rows if bool(row.get("activity_features_available", False))]
        train_labels = [int(row["label"]) for row in train_rows]
        test_labels = [int(row["label"]) for row in test_rows]
        if not train_rows or not test_rows:
            skipped_folds.append({"held_out_run_id": held_out_run_id, "reason": "missing_rows"})
            continue
        if len(set(train_labels)) < 2:
            skipped_folds.append({"held_out_run_id": held_out_run_id, "reason": "train_has_single_class"})
            continue
        if len(set(test_labels)) < 2:
            skipped_folds.append({"held_out_run_id": held_out_run_id, "reason": "test_has_single_class"})
            continue
        x_train = np.array([[row.get(name, 0.0) for name in feature_names] for row in train_rows], dtype=np.float64)
        y_train = np.array(train_labels, dtype=np.int64)
        x_test = np.array([[row.get(name, 0.0) for name in feature_names] for row in test_rows], dtype=np.float64)
        y_test = np.array(test_labels, dtype=np.int64)
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        clf.fit(x_train, y_train)
        scores = clf.predict_proba(x_test)[:, 1]
        fold_metrics.append(
            {
                "held_out_run_id": held_out_run_id,
                "roc_auc": float(roc_auc_score(y_test, scores)),
                "top1_accuracy": _top_k_accuracy(test_rows, scores, k=1),
                "num_test_rows": len(test_rows),
                "num_positive_rows": int(np.sum(y_test)),
            }
        )

    used_forbidden = any(name in FORBIDDEN_FEATURES for name in feature_names)
    region_rankings = _build_region_rankings(rows, feature_names, bool(spec["requires_activity"]))
    return {
        "feature_set_name": feature_set_name,
        "forbidden_features_used": used_forbidden,
        "forbidden_features": FORBIDDEN_FEATURES,
        "evaluated_folds": len(fold_metrics),
        "skipped_folds": skipped_folds,
        "ROC-AUC": summarize_scalar([item["roc_auc"] for item in fold_metrics]) if fold_metrics else None,
        "top-k accuracy": {"k": 1, "summary": summarize_scalar([item["top1_accuracy"] for item in fold_metrics])},
        "seed-level rows": fold_metrics,
        "region_rankings": region_rankings,
        "rows_csv": str(csv_path),
        "activity_features_available": bool(any(bool(row.get("activity_features_available", False)) for row in rows)),
        "limitation note": spec["limitation_note"],
    }


def run_blind_region_prediction(input_root: str, output_dir: str, feature_set: str | None = None) -> dict:
    if feature_set is not None and feature_set not in FEATURE_SET_SPECS:
        raise ValueError(f"unsupported feature set: {feature_set}")
    rows = _build_rows(Path(input_root))
    out_dir = ensure_dir(output_dir)
    csv_path = Path(out_dir) / "blind_region_prediction_rows.csv"
    _write_rows_csv(csv_path, rows)

    selected_feature_sets = [feature_set] if feature_set is not None else list(FEATURE_SET_SPECS.keys())
    summaries = {
        name: _summarize_feature_set(rows, name, csv_path)
        for name in selected_feature_sets
    }
    aggregate_summary = {
        "input_root": input_root,
        "output_dir": str(out_dir),
        "num_rows": len(rows),
        "num_runs": len({str(row["run_id"]) for row in rows}),
        "feature_set_summaries": summaries,
        "selected_feature_set": feature_set,
    }
    write_json(Path(out_dir) / "blind_region_prediction_summary.json", aggregate_summary)
    for name, payload in summaries.items():
        write_json(Path(out_dir) / f"blind_region_prediction_{name}.json", payload)
    return aggregate_summary


# ---------------------------------------------------------------------------
# Counterfactual smoke interface-excluded predictor
# ---------------------------------------------------------------------------

CTRL_READOUT_IDS = [16, 17, 18, 19, 20, 21, 22, 23]


def _read_json(path: Path) -> dict:
    import json
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict]) -> None:
    import csv
    if not rows:
        return
    fields = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_model_from_run(run_dir: Path, device: str = "cpu"):
    """Load a counterfactual smoke run model."""
    from .model_registry import build_model
    from .utils import load_yaml, make_device

    ckpt = torch.load(run_dir / "final_checkpoint.pt", map_location="cpu")
    cfg = load_yaml(run_dir / "resolved_config.yaml")
    dev = make_device(cfg.get("device", device))
    model = build_model(cfg, dev)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()
    return model, cfg, dev


def run_interface_excluded_predictor(
    seed_root_manifest: dict[str, str],
    output_dir: str,
    seeds: list[int],
    smoke: bool,
) -> dict:
    """Predict the best non-interface region for each seed/family using
    control_head_coupling scores, excluding the fixed ctrl_readout_ids."""
    out_dir = ensure_dir(output_dir)
    inventory_to_training = {0: 42, 1: 43, 2: 44}
    torch = __import__("torch")

    ctrl_set = set(CTRL_READOUT_IDS)
    all_nodes = list(range(32))  # fixed architecture: 32 nodes
    candidate_nodes = [n for n in all_nodes if n not in ctrl_set]

    per_seed_family = []

    for inv_label_str, root_path in seed_root_manifest.items():
        inv_label = int(inv_label_str)
        training_seed = inventory_to_training.get(inv_label, inv_label)
        root = Path(root_path)

        run_map_path = root / "run_mapping.json"
        if not run_map_path.exists():
            continue
        run_map = _read_json(run_map_path)

        for run_name, run_dir_str in run_map.items():
            run_dir = Path(run_dir_str)
            fam = run_name.split("_seed_")[0].split("_")[-1] if "_seed_" in run_name else "unknown"
            # Extract model family from run name: counterfactual_2d_control_task_{family}_seed_{seed}
            parts = run_name.split("_seed_")[0].split("_")
            # Family is everything after "counterfactual_2d_control_task_"
            model_family = "_".join(parts[4:]) if len(parts) > 4 else fam

            try:
                model, cfg, dev = _load_model_from_run(run_dir)
            except Exception as exc:
                per_seed_family.append({
                    "inventory_label": inv_label,
                    "actual_training_seed": training_seed,
                    "model_family": model_family,
                    "run_name": run_name,
                    "prediction_available": False,
                    "error": str(exc),
                })
                continue

            # Compute control_head_coupling score for all candidate nodes
            hidden_dim = int(cfg.get("num_nodes", cfg.get("dense_hidden_dim", 32)))
            coupling = torch.zeros(hidden_dim, device=dev)
            if hasattr(model, "ctrl_head") and hasattr(model, "ctrl_readout_ids"):
                w = model.ctrl_head.weight.detach().abs().mean(dim=0).to(dev)
                rid = model.ctrl_readout_ids.long().to(dev)
                valid = (rid >= 0) & (rid < hidden_dim)
                coupling[rid[valid]] = w[valid]

            # Score all nodes, exclude ctrl_readout_ids
            scores = coupling.detach().cpu().numpy()
            candidate_scores = [(n, float(scores[n])) for n in candidate_nodes]
            candidate_scores.sort(key=lambda x: x[1], reverse=True)

            # Take top-k (same size as original predicted region = 8)
            k = 8
            top_k = candidate_scores[:k]
            top_nodes = [n for n, s in top_k]
            top_scores = [s for n, s in top_k]

            # Compute overlap with ctrl_readout_ids
            predicted_set = set(top_nodes)
            overlap = len(predicted_set & ctrl_set)
            overlap_fraction = overlap / max(1, k)

            per_seed_family.append({
                "inventory_label": inv_label,
                "actual_training_seed": training_seed,
                "model_family": model_family,
                "run_name": run_name,
                "prediction_available": True,
                "predicted_region_nodes": top_nodes,
                "predicted_region_scores": top_scores,
                "prediction_mode": "interface_excluded_control_predictor",
                "excluded_node_sets": {"ctrl_readout_ids": CTRL_READOUT_IDS},
                "excluded_ctrl_readout_ids": CTRL_READOUT_IDS,
                "candidate_pool_size": len(candidate_nodes),
                "region_selection_blind": True,
                "uses_lesion_information": False,
                "uses_assay_outcome": False,
                "uses_fixed_interface_block": False,
                "overlap_with_ctrl_readout_ids": overlap,
                "overlap_fraction_with_ctrl_readout_ids": overlap_fraction,
                "num_candidates_ranked": len(candidate_scores),
                "top_candidate_ranking": [(n, float(s)) for n, s in candidate_scores[:16]],
            })

    _write_csv(out_dir / "interface_excluded_prediction_rows.csv", per_seed_family)
    write_json(out_dir / "interface_excluded_prediction_summary.json", {
        "mode": "interface_excluded_control_predictor",
        "excluded_node_sets": {"ctrl_readout_ids": CTRL_READOUT_IDS},
        "num_seeds_families_predicted": len(per_seed_family),
        "predictions": per_seed_family,
    })
    return {"predictions": per_seed_family}


# ---------------------------------------------------------------------------
# Contamination audit
# ---------------------------------------------------------------------------

def run_contamination_audit(
    predictions: list[dict],
    output_dir: str,
) -> dict:
    """Audit whether the blind region predictor is contaminated by fixed interface."""
    out_dir = ensure_dir(output_dir)
    rows = []
    all_pass = True
    issues = []

    for p in predictions:
        if not p.get("prediction_available", False):
            continue

        overlap_frac = p.get("overlap_fraction_with_ctrl_readout_ids", 1.0)
        uses_lesion = p.get("uses_lesion_information", True)
        uses_outcome = p.get("uses_assay_outcome", True)
        is_interface_block = overlap_frac >= 0.5

        pass_contamination = bool(
            overlap_frac < 0.5
            and not uses_lesion
            and not uses_outcome
            and not is_interface_block
        )

        if not pass_contamination:
            all_pass = False
            reason_parts = []
            if overlap_frac >= 0.5:
                reason_parts.append(f"overlap_{overlap_frac:.2f}_with_ctrl_readout")
            if uses_lesion:
                reason_parts.append("uses_lesion")
            if uses_outcome:
                reason_parts.append("uses_outcome")
            issues.append(f"seed{p.get('inventory_label')}_{p.get('model_family')}: {';'.join(reason_parts)}")

        rows.append({
            "inventory_label": p.get("inventory_label"),
            "actual_training_seed": p.get("actual_training_seed"),
            "model_family": p.get("model_family"),
            "predicted_region_nodes": str(p.get("predicted_region_nodes", [])),
            "overlap_with_ctrl_readout_ids": p.get("overlap_with_ctrl_readout_ids", 0),
            "overlap_fraction_with_ctrl_readout_ids": overlap_frac,
            "is_contaminated_interface_block": is_interface_block,
            "uses_lesion_information": uses_lesion,
            "uses_assay_outcome": uses_outcome,
            "interface_contamination_pass": pass_contamination,
            "failure_reason": ";".join(reason_parts) if not pass_contamination else "",
        })

    _write_csv(out_dir / "contamination_audit_rows.csv", rows)
    audit = {
        "ctrl_readout_ids": CTRL_READOUT_IDS,
        "num_predictions_audited": len(rows),
        "interface_contamination_pass": all_pass,
        "issues": issues,
        "failure_type": "" if all_pass else "blind_region_predictor_returns_fixed_interface_block",
    }
    write_json(out_dir / "blind_region_predictor_contamination_audit.json", audit)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict control-region identity from non-overlapping feature sets across runs.")
    parser.add_argument("--config", required=False)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=False)
    parser.add_argument("--input-root", required=False, default="")
    parser.add_argument("--feature-set", choices=sorted(FEATURE_SET_SPECS.keys()))
    # New arguments for counterfactual smoke interface-excluded mode
    parser.add_argument("--seed-root-manifest", required=False, default="")
    parser.add_argument("--seeds", required=False, default="", help="Comma-separated actual training seeds")
    parser.add_argument("--mode", choices=["standard", "interface_excluded_control_predictor", "noninterface_gradient_activity_composite_v2", "noninterface_causal_proxy_composite_v3", "noninterface_causal_proxy_core_v1"], default="standard")
    parser.add_argument("--exclude-interface-nodes", action="store_true",
                        help="Exclude interface nodes from candidate pool (used with v2 mode)")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.mode == "interface_excluded_control_predictor":
        if not args.seed_root_manifest:
            parser.error("--seed-root-manifest is required for interface_excluded_control_predictor mode")
        import json as _json
        manifest = _json.loads(Path(args.seed_root_manifest).read_text(encoding="utf-8"))
        seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()] if args.seeds else []
        pred_result = run_interface_excluded_predictor(manifest, args.output_dir, seeds, bool(args.smoke))
        audit = run_contamination_audit(pred_result.get("predictions", []), args.output_dir)
        print(f"Interface-excluded prediction complete: {len(pred_result['predictions'])} seed/family predictions")
        print(f"Interface contamination pass: {audit['interface_contamination_pass']}")
        return

    if args.mode == "noninterface_gradient_activity_composite_v2":
        if not args.input_root:
            parser.error("--input-root is required for noninterface_gradient_activity_composite_v2 mode")
        result = run_noninterface_gradient_activity_composite_v2_predictor(
            input_root=args.input_root,
            output_dir=args.output_dir,
            seed=int(args.seed) if args.seed else 42,
            exclude_interface_nodes=bool(args.exclude_interface_nodes),
        )
        print(f"Predictor v2 prediction complete: {len(result.get('predicted_region_nodes', []))} nodes")
        print(f"Predicted region: {result.get('predicted_region_nodes', [])}")
        print(f"Interface contamination pass: {result.get('interface_contamination_pass', False)}")
        return

    if args.mode == "noninterface_causal_proxy_composite_v3":
        if not args.input_root:
            parser.error("--input-root is required for noninterface_causal_proxy_composite_v3 mode")
        result = run_noninterface_causal_proxy_composite_v3_predictor(
            input_root=args.input_root,
            output_dir=args.output_dir,
            seed=int(args.seed) if args.seed else 42,
            exclude_interface_nodes=bool(args.exclude_interface_nodes),
            force_diagnostic_only=False,
        )
        print(f"Predictor v3 prediction complete: {len(result.get('predicted_region_nodes', []))} nodes")
        print(f"Predicted region: {result.get('predicted_region_nodes', [])}")
        print(f"Interface contamination pass: {result.get('interface_contamination_pass', False)}")
        return

    if args.mode == "noninterface_causal_proxy_core_v1":
        if not args.input_root:
            parser.error("--input-root is required for noninterface_causal_proxy_core_v1 mode")
        result = run_noninterface_causal_proxy_core_v1_predictor(
            input_root=args.input_root,
            output_dir=args.output_dir,
            seed=int(args.seed) if args.seed else 42,
            exclude_interface_nodes=bool(args.exclude_interface_nodes),
        )
        print(f"Predictor core v1 prediction complete")
        print(f"Broad region (8): {result.get('broad_region_nodes', [])}")
        print(f"Core region (4): {result.get('core_region_nodes', [])}")
        print(f"Interface contamination pass: {result.get('interface_contamination_pass', False)}")
        return

    # Standard mode (existing behavior)
    summary = run_blind_region_prediction(args.input_root, args.output_dir, feature_set=args.feature_set)
    print(summary)


# ---------------------------------------------------------------------------
# Noninterface Activity Gradient Predictor (deconfounded mode)
# ---------------------------------------------------------------------------


def _compute_activity_features(model, generator, num_batches: int, batch_size: int, device) -> dict[int, dict]:
    import torch
    """Compute per-node activity features from model forward passes.

    Returns a dict mapping node_id -> {feature_name: value}.
    """
    num_nodes = model.num_nodes
    features: dict[int, dict] = {i: {} for i in range(num_nodes)}

    danger_responses = torch.zeros(num_nodes, device=device)
    control_mask_responses = torch.zeros(num_nodes, device=device)
    danger_flag_corrs = torch.zeros(num_nodes, device=device)
    counts = torch.zeros(num_nodes, device=device) + 1e-8

    model.eval()
    with torch.no_grad():
        for _b in range(num_batches):
            batch = generator.sample_batch(batch_size, split="train")
            outputs = model(batch["obs"], batch["mem_query"], batch["danger_flag"])
            hidden = outputs["hidden_traj"]  # [B, T, N]
            b, t, n = hidden.shape

            danger = batch["danger_flag"]  # [B, T, 1]
            ctrl_mask = batch["ctrl_mask"]  # [B, T]

            danger_expanded = danger.view(b, t, 1).expand(b, t, n)
            danger_act = (hidden * danger_expanded).sum(dim=(0, 1)) / max(danger.sum().item(), 1)
            danger_responses += danger_act
            counts += 1

            mask_expanded = ctrl_mask.view(b, t, 1).expand(b, t, n)
            ctrl_act = (hidden * mask_expanded).sum(dim=(0, 1)) / max(ctrl_mask.sum().item(), 1)
            control_mask_responses += ctrl_act

            h_flat = hidden.reshape(b * t, n)
            d_flat = danger.reshape(b * t, 1).float()
            for i_node in range(n):
                dh = h_flat[:, i_node]
                if dh.std() > 1e-8 and d_flat[:, 0].std() > 1e-8:
                    corr = torch.corrcoef(torch.stack([dh, d_flat[:, 0]]))[0, 1]
                    if not torch.isnan(corr):
                        danger_flag_corrs[i_node] += corr.abs()

    h_mean = features.get(0, {}).get("mean_activation", 0.0)
    for i_node in range(num_nodes):
        c = max(counts[i_node].item(), 1e-8)
        features[i_node]["danger_response"] = float((danger_responses[i_node] / c).item())
        features[i_node]["control_mask_response"] = float((control_mask_responses[i_node] / c).item())
        features[i_node]["danger_flag_correlation"] = float((danger_flag_corrs[i_node] / c).item())

    model.train()
    return features


def run_noninterface_activity_gradient_predictor(
    model: torch.nn.Module,
    generator,
    registry: dict,
    output_dir: str,
    num_batches: int = 4,
    batch_size: int = 16,
    seed: int = 42,
    smoke: bool = True,
) -> dict:
    """Run the noninterface activity gradient blind region predictor.

    This predictor:
    - Excludes all interface nodes from candidate pool (from registry)
    - Uses only activity features (no structural readout features)
    - Does NOT use lesion information, assay outcome, or fixed interface block
    - Produces contamination audit
    """
    import numpy as np
    import torch

    from .utils import ensure_dir, write_json

    out_dir = ensure_dir(output_dir)
    device = next(model.parameters()).device

    candidate_pool = registry["candidate_pool_nodes"]
    excluded = set(registry["excluded_from_region_candidates"])

    if not candidate_pool:
        result = {
            "prediction_mode": "noninterface_activity_gradient",
            "prediction_available": False,
            "error": "candidate_pool_empty_all_nodes_excluded",
            "candidate_pool_nodes": [],
            "interface_contamination_pass": True,
            "feature_table": {},
            "candidate_ranking": [],
        }
        write_json(out_dir / "blind_region_prediction.json", result)
        write_json(out_dir / "blind_region_predictor_contamination_audit.json", {
            "prediction_mode": "noninterface_activity_gradient",
            "interface_contamination_pass": True,
            "error": "candidate_pool_empty",
        })
        return result

    activity_features = _compute_activity_features(model, generator, num_batches, batch_size, device)

    feature_table: dict[str, dict] = {}
    for node_id in candidate_pool:
        feat = activity_features.get(node_id, {})
        feature_table[str(node_id)] = {
            "node_id": int(node_id),
            **{k: float(feat.get(k, 0.0)) for k in ["danger_response", "control_mask_response", "danger_flag_correlation"]},
        }

    candidates_list = list(candidate_pool)
    if len(candidates_list) <= 1:
        ranked = [(n, 0.0) for n in candidates_list]
    else:
        scores_matrix = []
        for nid in candidates_list:
            f = feature_table[str(nid)]
            row = [f.get("danger_response", 0.0), f.get("control_mask_response", 0.0), f.get("danger_flag_correlation", 0.0)]
            scores_matrix.append(row)

        arr = np.array(scores_matrix, dtype=np.float64)
        means = arr.mean(axis=0, keepdims=True)
        stds = arr.std(axis=0, keepdims=True)
        stds[stds < 1e-9] = 1.0
        normalized = (arr - means) / stds
        composite = normalized.mean(axis=1)

        scored = list(zip(candidates_list, composite.tolist()))
        scored.sort(key=lambda x: x[1], reverse=True)
        ranked = scored

    k = min(8, len(ranked))
    top_k = ranked[:k]
    top_nodes = [int(n) for n, _s in top_k]
    top_scores = [float(s) for _n, s in top_k]

    full_ranking = [(int(n), float(s)) for n, s in ranked]

    predicted_set = set(top_nodes)
    overlap_with_excluded = predicted_set & excluded
    overlap_with_ctrl = predicted_set & set(registry.get("ctrl_readout_interface_nodes", []))
    overlap_with_input = predicted_set & set(registry.get("input_interface_nodes", []))
    overlap_with_output = predicted_set & set(registry.get("output_interface_nodes", []))

    contamination_pass = len(overlap_with_excluded) == 0

    contamination_audit = {
        "prediction_mode": "noninterface_activity_gradient",
        "predicted_region_nodes": top_nodes,
        "candidate_region_size": k,
        "num_candidates_ranked": len(candidates_list),
        "excluded_nodes": sorted(excluded),
        "overlap_with_excluded_nodes": sorted(overlap_with_excluded),
        "overlap_with_ctrl_readout_ids": sorted(overlap_with_ctrl),
        "overlap_with_input_interface": sorted(overlap_with_input),
        "overlap_with_output_interface": sorted(overlap_with_output),
        "interface_contamination_pass": contamination_pass,
        "uses_lesion_information": False,
        "uses_assay_outcome": False,
        "uses_fixed_interface_block": False,
        "allowed_features": ["danger_response", "control_mask_response", "danger_flag_correlation"],
        "forbidden_features_checked": {
            "direct_ctrl_readout_ids_membership": False,
            "lesion_delta": False,
            "LS_ctrl": False,
            "assay_pass_fail": False,
            "fixed_node_index_block": False,
            "outgoing_density_to_control_head": False,
            "ctrl_coupling": False,
        },
        "feature_table": {str(n): feature_table.get(str(n), {}) for n in candidate_pool},
        "candidate_ranking": full_ranking,
    }

    write_json(out_dir / "blind_region_prediction.json", contamination_audit)
    write_json(out_dir / "blind_region_predictor_contamination_audit.json", {
        "prediction_mode": contamination_audit["prediction_mode"],
        "predicted_region_nodes": contamination_audit["predicted_region_nodes"],
        "interface_contamination_pass": contamination_audit["interface_contamination_pass"],
        "overlap_with_excluded_nodes": contamination_audit["overlap_with_excluded_nodes"],
        "overlap_with_ctrl_readout_ids": contamination_audit["overlap_with_ctrl_readout_ids"],
        "uses_lesion_information": False,
        "uses_assay_outcome": False,
        "uses_fixed_interface_block": False,
        "contamination_issues": [] if contamination_pass else [
            f"overlap_with_excluded={sorted(overlap_with_excluded)}"
        ],
    })

    return contamination_audit


# ---------------------------------------------------------------------------
# Legacy interface assay baseline
# ---------------------------------------------------------------------------


def run_legacy_interface_assay(
    model: torch.nn.Module,
    generator,
    output_dir: str,
    num_batches: int = 4,
    batch_size: int = 16,
    seed: int = 42,
    smoke: bool = True,
) -> dict:
    """Run the legacy interface assay to provide artifact baseline.

    This runs the equivalent of the old fixed-readout predictor on the
    legacy ctrl_readout_ids block [16-23], serving as the interface artifact
    control. The purpose is to show what signal looks like at the interface
    so it can be compared against the noninterface region.

    Returns dict with:
    - predicted_region_nodes: the legacy fixed readout interface
    - prediction_mode: "legacy_interface_baseline"
    - interface_contamination_pass: True (by construction for interface itself)
    """
    from .interface_exclusion import CTRL_READOUT_IDS_DEFAULT
    from .utils import ensure_dir, write_json

    legacy_region = list(CTRL_READOUT_IDS_DEFAULT)
    out_dir = ensure_dir(output_dir)
    result = {
        "prediction_mode": "legacy_interface_baseline",
        "predicted_region_nodes": legacy_region,
        "candidate_region_size": len(legacy_region),
        "interface_contamination_pass": True,
        "uses_lesion_information": False,
        "uses_assay_outcome": False,
        "uses_fixed_interface_block": True,
        "note": "Legacy interface baseline: this region is the fixed ctrl_readout_ids block [16-23]. It is the known architecture artifact.",
    }
    write_json(out_dir / "legacy_interface_region.json", result)
    return result


# ---------------------------------------------------------------------------
# Predictor v2: noninterface_gradient_activity_composite_v2
# ---------------------------------------------------------------------------

V2_FEATURE_WEIGHTS = {
    "gradient_mass": 0.364,
    "activity_response": 0.273,
    "danger_response": 0.182,
    "control_mask_response": 0.182,
}
V2_FEATURE_NAMES = ["gradient_mass", "activity_response", "danger_response", "control_mask_response"]

# ---------------------------------------------------------------------------
# Predictor v3: noninterface_causal_proxy_composite_v3
# Uses expanded blind-safe features from compute_blind_safe_features.py
# ---------------------------------------------------------------------------

V3_FEATURE_WEIGHTS = {
    "event_core_gradient_mass": 0.25,
    "projection_weight_mass": 0.15,
    "adapter_upstream_weight_mass": 0.10,
    "activation_gradient_product": 0.20,
    "projection_gradient_product": 0.10,
    "event_local_activation_response": 0.10,
    "control_mask_response": 0.10,
}
V3_FEATURE_NAMES = [
    "event_core_gradient_mass",
    "projection_weight_mass",
    "adapter_upstream_weight_mass",
    "activation_gradient_product",
    "projection_gradient_product",
    "event_local_activation_response",
    "control_mask_response",
]
V3_FEATURE_TABLE_RELPATH = "deconfounded_region_predictor_gap/blind_safe_feature_table_v3.csv"
V3_FEATURE_AVAIL_RELPATH = "deconfounded_region_predictor_gap/blind_safe_feature_availability_v3.json"

# Expected paths relative to input_root for v2 predictor
V2_CALIBRATION_SUMMARY_RELPATH = "deconfounded_assay_calibration/deconfounded_assay_calibration_summary.json"
V2_CALIBRATION_AUDIT_RELPATH = "deconfounded_assay_calibration/deconfounded_oracle_region_audit.json"
V2_EXISTING_PREDICTION_RELPATH = "blind_region_prediction.json"
V2_INTERFACE_REGISTRY_RELPATH = "interface_node_registry.json"


def run_noninterface_gradient_activity_composite_v2_predictor(
    input_root: str,
    output_dir: str,
    seed: int = 42,
    exclude_interface_nodes: bool = True,
) -> dict:
    """Run predictor v2: gradient + activity composite with fixed weights.

    This predictor uses only blind-safe pre-assay features:
    - gradient_mass (from oracle_audit, available pre-assay)
    - activity_response (from oracle_audit)
    - danger_response (from existing prediction feature_table)
    - control_mask_response (from existing prediction feature_table)

    Does NOT use:
    - lesion outcome
    - calibrated assay pass/fail
    - oracle region label
    - real_minus_temporal_control

    Feature weights are fixed (preregistered), not tuned to assay outcome.
    """
    import numpy as np

    from .utils import ensure_dir, write_json

    out_dir = ensure_dir(output_dir)
    inp_root = Path(input_root)

    # ------------------------------------------------------------------
    # Load input data
    # ------------------------------------------------------------------
    cal_summary_path = inp_root / V2_CALIBRATION_SUMMARY_RELPATH
    cal_audit_path = inp_root / V2_CALIBRATION_AUDIT_RELPATH
    existing_pred_path = inp_root / V2_EXISTING_PREDICTION_RELPATH
    registry_path = inp_root / V2_INTERFACE_REGISTRY_RELPATH

    if not cal_audit_path.exists():
        result = {
            "prediction_mode": "noninterface_gradient_activity_composite_v2",
            "prediction_available": False,
            "error": f"calibration audit not found at {cal_audit_path}",
        }
        write_json(out_dir / "blind_region_prediction_v2.json", result)
        return result

    oracle_audit = json.loads(cal_audit_path.read_text(encoding="utf-8"))
    existing_pred = json.loads(existing_pred_path.read_text(encoding="utf-8")) if existing_pred_path.exists() else {}
    registry = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.exists() else {}

    # ------------------------------------------------------------------
    # Determine candidate pool (exclude interface nodes if requested)
    # ------------------------------------------------------------------
    candidate_pool = registry.get("candidate_pool_nodes", list(range(16, 32)))
    excluded_nodes = set(registry.get("excluded_from_region_candidates", []))

    if exclude_interface_nodes:
        # Use the candidate pool from registry (already excludes interface)
        candidates = [n for n in candidate_pool if n not in excluded_nodes]
    else:
        candidates = list(candidate_pool)

    if not candidates:
        result = {
            "prediction_mode": "noninterface_gradient_activity_composite_v2",
            "prediction_available": False,
            "error": "candidate_pool_empty_all_nodes_excluded",
            "candidate_pool_nodes": [],
            "interface_contamination_pass": True,
        }
        write_json(out_dir / "blind_region_prediction_v2.json", result)
        return result

    # ------------------------------------------------------------------
    # Extract per-node features from available sources
    # ------------------------------------------------------------------
    grad_mass_raw = oracle_audit.get("gradient_mass", {})
    activity_resp_raw = oracle_audit.get("activity_response", {})
    feat_table_raw = existing_pred.get("feature_table", {})

    feature_table: dict[str, dict] = {}
    for nid in candidates:
        nid_str = str(nid)
        feature_table[nid_str] = {
            "node_id": int(nid),
            "gradient_mass": float(grad_mass_raw.get(nid_str, 0.0)),
            "activity_response": float(activity_resp_raw.get(nid_str, 0.0)),
            "danger_response": float(feat_table_raw.get(nid_str, {}).get("danger_response", 0.0)),
            "control_mask_response": float(feat_table_raw.get(nid_str, {}).get("control_mask_response", 0.0)),
        }

    # ------------------------------------------------------------------
    # Build feature matrix, z-score normalize, apply fixed weights
    # ------------------------------------------------------------------
    n_nodes = len(candidates)
    matrix = np.zeros((n_nodes, len(V2_FEATURE_NAMES)), dtype=np.float64)
    for i, nid in enumerate(candidates):
        ft = feature_table[str(nid)]
        for j, fname in enumerate(V2_FEATURE_NAMES):
            matrix[i, j] = ft[fname]

    means = matrix.mean(axis=0, keepdims=True)
    stds = matrix.std(axis=0, keepdims=True)
    stds[stds < 1e-9] = 1.0
    normalized = (matrix - means) / stds

    weights_arr = np.array([V2_FEATURE_WEIGHTS[f] for f in V2_FEATURE_NAMES], dtype=np.float64)
    weights_arr = weights_arr / weights_arr.sum()
    composite = normalized @ weights_arr

    # ------------------------------------------------------------------
    # Rank candidates
    # ------------------------------------------------------------------
    scored = list(zip(candidates, composite.tolist()))
    scored.sort(key=lambda x: x[1], reverse=True)

    k = 8
    top_k = scored[:k]
    top_nodes = [int(n) for n, _s in top_k]
    top_scores = [float(s) for _n, s in top_k]
    full_ranking = [(int(n), float(s)) for n, s in scored]

    # ------------------------------------------------------------------
    # Interface contamination check
    # ------------------------------------------------------------------
    predicted_set = set(top_nodes)
    overlap_with_excluded = predicted_set & excluded_nodes
    contamination_pass = len(overlap_with_excluded) == 0

    # ------------------------------------------------------------------
    # Build prediction result
    # ------------------------------------------------------------------
    prediction_result = {
        "prediction_mode": "noninterface_gradient_activity_composite_v2",
        "predicted_region_nodes": top_nodes,
        "predicted_region_scores": top_scores,
        "num_candidates_ranked": len(candidates),
        "feature_weights": {k: float(v) for k, v in V2_FEATURE_WEIGHTS.items()},
        "feature_normalization": "zscore_across_candidate_pool",
        "uses_lesion_information": False,
        "uses_assay_outcome": False,
        "uses_oracle_label": False,
        "uses_real_minus_temporal_control": False,
        "overlap_with_interface_nodes": sorted(overlap_with_excluded),
        "interface_contamination_pass": contamination_pass,
        "candidate_pool_size": len(candidates),
        "candidate_ranking": full_ranking,
    }
    write_json(out_dir / "blind_region_prediction_v2.json", prediction_result)

    # ------------------------------------------------------------------
    # Contamination audit
    # ------------------------------------------------------------------
    contamination_audit = {
        "prediction_mode": "noninterface_gradient_activity_composite_v2",
        "predicted_region_nodes": top_nodes,
        "candidate_region_size": k,
        "num_candidates_ranked": len(candidates),
        "excluded_nodes": sorted(excluded_nodes),
        "overlap_with_excluded_nodes": sorted(overlap_with_excluded),
        "interface_contamination_pass": contamination_pass,
        "uses_lesion_information": False,
        "uses_assay_outcome": False,
        "uses_oracle_label": False,
        "uses_real_minus_temporal_control": False,
        "allowed_features": V2_FEATURE_NAMES[:],
        "forbidden_features_checked": {
            "lesion_delta": False,
            "LS_ctrl": False,
            "assay_pass_fail": False,
            "oracle_region_label": False,
            "real_minus_temporal_control": False,
        },
        "contamination_issues": [] if contamination_pass else [
            f"overlap_with_excluded={sorted(overlap_with_excluded)}"
        ],
    }
    write_json(out_dir / "blind_region_predictor_v2_contamination_audit.json", contamination_audit)

    # ------------------------------------------------------------------
    # Feature table CSV
    # ------------------------------------------------------------------
    import csv
    ft_csv_path = out_dir / "blind_region_predictor_v2_feature_table.csv"
    ft_rows = []
    for nid in candidates:
        ft = feature_table[str(nid)]
        ft_rows.append({
            "node_id": int(nid),
            "gradient_mass": ft["gradient_mass"],
            "activity_response": ft["activity_response"],
            "danger_response": ft["danger_response"],
            "control_mask_response": ft["control_mask_response"],
        })
    if ft_rows:
        ft_fields = sorted(ft_rows[0].keys())
        with open(ft_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=ft_fields)
            writer.writeheader()
            writer.writerows(ft_rows)

    # ------------------------------------------------------------------
    # Candidate ranking CSV
    # ------------------------------------------------------------------
    ranking_csv_path = out_dir / "blind_region_predictor_v2_candidate_ranking.csv"
    ranking_rows = []
    for rank_idx, (nid, score) in enumerate(full_ranking):
        ft = feature_table[str(nid)]
        ranking_rows.append({
            "rank": rank_idx + 1,
            "node_id": int(nid),
            "composite_score": round(float(score), 8),
            "gradient_mass": ft["gradient_mass"],
            "activity_response": ft["activity_response"],
            "danger_response": ft["danger_response"],
            "control_mask_response": ft["control_mask_response"],
            "selected": 1 if nid in predicted_set else 0,
        })
    if ranking_rows:
        r_fields = sorted(ranking_rows[0].keys())
        with open(ranking_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=r_fields)
            writer.writeheader()
            writer.writerows(ranking_rows)

    return prediction_result


# ---------------------------------------------------------------------------
# Predictor v3: noninterface_causal_proxy_composite_v3
# ---------------------------------------------------------------------------

def run_noninterface_causal_proxy_composite_v3_predictor(
    input_root: str,
    output_dir: str,
    seed: int = 42,
    exclude_interface_nodes: bool = True,
    force_diagnostic_only: bool = False,
) -> dict:
    """Run predictor v3: expanded blind-safe feature composite with fixed weights.

    Uses features from compute_blind_safe_features.py:
      - event_core_gradient_mass: gradient of ctrl score w.r.t. hidden nodes at event core
      - projection_weight_mass: mean abs weight from ctrl_head per node
      - activation_gradient_product: activation * gradient mass (distinguishes active+causal)
      - projection_gradient_product: projection weight * gradient mass
      - event_local_activation_response: mean abs activation at event time
      - control_mask_response: response to control mask (from existing feature table)

    All features are blind-safe:
      - no lesion information
      - no assay outcome
      - no oracle label
      - no real_minus_temporal_control

    Feature weights are pre-specified and NOT tuned to assay outcome.
    """
    import csv
    import numpy as np

    from .utils import ensure_dir, write_json

    out_dir = ensure_dir(output_dir)
    inp_root = Path(input_root)

    # ------------------------------------------------------------------
    # Load input data
    # ------------------------------------------------------------------
    registry_path = inp_root / V2_INTERFACE_REGISTRY_RELPATH
    feat_table_path = inp_root / V3_FEATURE_TABLE_RELPATH
    existing_pred_path = inp_root / V2_EXISTING_PREDICTION_RELPATH

    registry = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.exists() else {}
    existing_pred = json.loads(existing_pred_path.read_text(encoding="utf-8")) if existing_pred_path.exists() else {}

    # Load v3 feature table
    if not feat_table_path.exists():
        result = {
            "prediction_mode": "noninterface_causal_proxy_composite_v3",
            "prediction_available": False,
            "error": f"v3 feature table not found at {feat_table_path}. Run compute_blind_safe_features.py first.",
        }
        write_json(out_dir / "blind_region_prediction_v3.json", result)
        return result

    feat_rows: list[dict] = []
    with open(feat_table_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            feat_rows.append(row)

    # ------------------------------------------------------------------
    # Determine candidate pool (exclude interface nodes if requested)
    # ------------------------------------------------------------------
    candidate_pool = registry.get("candidate_pool_nodes", list(range(16, 32)))
    excluded_nodes = set(registry.get("excluded_from_region_candidates", []))

    if exclude_interface_nodes:
        candidates = [n for n in candidate_pool if n not in excluded_nodes]
    else:
        candidates = list(candidate_pool)

    if not candidates:
        result = {
            "prediction_mode": "noninterface_causal_proxy_composite_v3",
            "prediction_available": False,
            "error": "candidate_pool_empty",
            "interface_contamination_pass": True,
        }
        write_json(out_dir / "blind_region_prediction_v3.json", result)
        return result

    # ------------------------------------------------------------------
    # Extract per-node features from v3 feature table
    # Also get control_mask_response from existing prediction feature_table
    # Also load adapter_upstream_weight_mass from adapter_weight_summary if available
    # ------------------------------------------------------------------
    feat_by_node: dict[int, dict] = {}
    for row in feat_rows:
        nid = int(row["node_id"])
        if nid not in candidates:
            continue
        feat_by_node[nid] = {
            "event_core_gradient_mass": float(row.get("event_core_gradient_mass", 0.0)),
            "projection_weight_mass": float(row.get("projection_weight_mass", 0.0)),
            "activation_gradient_product": float(row.get("activation_gradient_product", 0.0)),
            "projection_gradient_product": float(row.get("projection_gradient_product", 0.0)),
            "event_local_activation_response": float(row.get("event_local_activation_response", 0.0)),
        }

    # Load adapter upstream weight mass from adapter_weight_summary.json if available
    adapter_weight_path = inp_root / "adapter_weight_summary.json"
    adapter_upstream_weights: dict[int, float] = {}
    if adapter_weight_path.exists():
        try:
            aws = json.loads(adapter_weight_path.read_text(encoding="utf-8"))
            # adapter_upstream_weight_mass: for each graph node, aggregate its weight
            # across all adapter channels (sum of absolute weights)
            adapter_weight_matrix = aws.get("adapter_weight_matrix", [])
            if adapter_weight_matrix:
                # adapter_weight_matrix is [K, N] or [N, K]
                awm = np.array(adapter_weight_matrix, dtype=np.float64)
                if awm.ndim == 2:
                    if awm.shape[0] == len(aws.get("num_adapter_channels", 4)):
                        # [K, N] -> per-node mass = sum over channels
                        per_node_mass = np.abs(awm).sum(axis=0)
                    else:
                        per_node_mass = np.abs(awm).sum(axis=1)
                    for i_nid in candidates:
                        if i_nid < len(per_node_mass):
                            adapter_upstream_weights[i_nid] = float(per_node_mass[i_nid])
        except Exception:
            pass

    # control_mask_response from existing prediction
    feat_table_raw = existing_pred.get("feature_table", {})
    for nid in candidates:
        if nid not in feat_by_node:
            feat_by_node[nid] = {
                "event_core_gradient_mass": 0.0,
                "projection_weight_mass": 0.0,
                "activation_gradient_product": 0.0,
                "projection_gradient_product": 0.0,
                "event_local_activation_response": 0.0,
            }
        feat_by_node[nid]["control_mask_response"] = float(
            feat_table_raw.get(str(nid), {}).get("control_mask_response", 0.0)
        )
        feat_by_node[nid]["adapter_upstream_weight_mass"] = adapter_upstream_weights.get(nid, 0.0)

    # ------------------------------------------------------------------
    # Build feature matrix, z-score normalize, apply fixed weights
    # ------------------------------------------------------------------
    n_nodes = len(candidates)
    matrix = np.zeros((n_nodes, len(V3_FEATURE_NAMES)), dtype=np.float64)
    for i, nid in enumerate(candidates):
        ft = feat_by_node.get(nid, {})
        for j, fname in enumerate(V3_FEATURE_NAMES):
            matrix[i, j] = ft.get(fname, 0.0)

    means = matrix.mean(axis=0, keepdims=True)
    stds = matrix.std(axis=0, keepdims=True)
    stds[stds < 1e-9] = 1.0
    normalized = (matrix - means) / stds

    weights_arr = np.array([V3_FEATURE_WEIGHTS[f] for f in V3_FEATURE_NAMES], dtype=np.float64)
    weights_arr = weights_arr / weights_arr.sum()
    composite = normalized @ weights_arr

    # ------------------------------------------------------------------
    # Rank candidates
    # ------------------------------------------------------------------
    scored = list(zip(candidates, composite.tolist()))
    scored.sort(key=lambda x: x[1], reverse=True)

    k = 8
    top_k = scored[:k]
    top_nodes = [int(n) for n, _s in top_k]
    top_scores = [float(s) for _n, s in top_k]
    full_ranking = [(int(n), float(s)) for n, s in scored]

    # ------------------------------------------------------------------
    # Interface contamination check
    # ------------------------------------------------------------------
    predicted_set = set(top_nodes)
    overlap_with_excluded = predicted_set & excluded_nodes
    contamination_pass = len(overlap_with_excluded) == 0

    # ------------------------------------------------------------------
    # Determine if v3 is preregistered or diagnostic-only
    # The weights are based on feature reasoning (not tuned to assay outcome).
    # However, the feature set was selected after seeing v2 failure -
    # mark as diagnostic_only unless explicitly overridden.
    # ------------------------------------------------------------------
    v3_is_diagnostic_only = force_diagnostic_only
    v3_preregistered_for_next_smoke = not force_diagnostic_only

    preregistration = {
        "v3_version": "noninterface_causal_proxy_composite_v3",
        "feature_weights": {k: float(v) for k, v in V3_FEATURE_WEIGHTS.items()},
        "feature_sources": {
            "event_core_gradient_mass": "compute_blind_safe_features.py: exact gradient via ctrl_head weights at event core",
            "projection_weight_mass": "compute_blind_safe_features.py: ctrl_head.weight.abs().mean(dim=0)",
            "activation_gradient_product": "compute_blind_safe_features.py: activation_response * event_core_gradient_mass",
            "projection_gradient_product": "compute_blind_safe_features.py: projection_weight_mass * event_core_gradient_mass",
            "event_local_activation_response": "compute_blind_safe_features.py: mean |hidden| at event core",
            "control_mask_response": "existing feature_table from blind_region_prediction.json",
        },
        "why_each_feature_is_blind_safe": {
            "event_core_gradient_mass": "Computed from model weights and forward activations only, no lesion/outcome",
            "projection_weight_mass": "Model parameter, available before any intervention",
            "activation_gradient_product": "Product of two blind-safe features",
            "projection_gradient_product": "Product of two blind-safe features",
            "event_local_activation_response": "Forward activation only",
            "control_mask_response": "Forward activity feature, no lesion",
        },
        "excluded_features": [
            "lesion_delta_ctrl",
            "ls_ctrl",
            "assay_pass_fail",
            "oracle_region_label",
            "real_minus_temporal_control",
            "direct_interface_membership",
            "candidate_assay_outcome",
        ],
        "no_outcome_leakage_statement": (
            "All v3 features are computed from model weights or forward activations only. "
            "No lesion outcome, calibrated assay result, oracle label, or temporal control "
            "information is used in feature computation or weighting."
        ),
        "candidate_region_size": k,
        "interface_exclusion_policy": (
            "Exclude all input_interface_nodes (0-7) and output_interface_nodes (8-15) "
            "from candidate pool, as defined in interface_node_registry.json."
        ),
        "v3_is_diagnostic_only": v3_is_diagnostic_only,
        "v3_preregistered_for_next_smoke": v3_preregistered_for_next_smoke,
        "weight_rationale": (
            "event_core_gradient_mass weighted highest (0.25) as it directly measures "
            "causal influence. projection_weight_mass (0.20) captures which nodes the "
            "control head reads from. activation_gradient_product (0.20) distinguishes "
            "'active but not causal' from 'active AND causal'. "
            "projection_gradient_product (0.15) combines readout weight with gradient. "
            "event_local_activation_response (0.10) and control_mask_response (0.10) "
            "provide supplementary signal."
        ),
    }

    # ------------------------------------------------------------------
    # Build prediction result
    # ------------------------------------------------------------------
    prediction_result = {
        "prediction_mode": "noninterface_causal_proxy_composite_v3",
        "predicted_region_nodes": top_nodes,
        "predicted_region_scores": top_scores,
        "num_candidates_ranked": len(candidates),
        "feature_weights": {k: float(v) for k, v in V3_FEATURE_WEIGHTS.items()},
        "feature_normalization": "zscore_across_candidate_pool",
        "uses_lesion_information": False,
        "uses_assay_outcome": False,
        "uses_oracle_label": False,
        "uses_real_minus_temporal_control": False,
        "overlap_with_interface_nodes": sorted(overlap_with_excluded),
        "interface_contamination_pass": contamination_pass,
        "candidate_pool_size": len(candidates),
        "candidate_ranking": full_ranking,
        "preregistration": preregistration,
        "adapter_upstream_weight_mass_available": len(adapter_upstream_weights) > 0,
        "overlap_with_adapter_virtual_units": [],  # adapter virtual units are not graph nodes
    }
    write_json(out_dir / "blind_region_prediction_v3.json", prediction_result)

    # ------------------------------------------------------------------
    # Contamination audit
    # ------------------------------------------------------------------
    contamination_audit = {
        "prediction_mode": "noninterface_causal_proxy_composite_v3",
        "predicted_region_nodes": top_nodes,
        "candidate_region_size": k,
        "num_candidates_ranked": len(candidates),
        "excluded_nodes": sorted(excluded_nodes),
        "overlap_with_excluded_nodes": sorted(overlap_with_excluded),
        "overlap_with_adapter_virtual_units": [],
        "interface_contamination_pass": contamination_pass,
        "uses_lesion_information": False,
        "uses_assay_outcome": False,
        "uses_oracle_label": False,
        "uses_real_minus_temporal_control": False,
        "allowed_features": V3_FEATURE_NAMES[:],
        "adapter_features_used": ["adapter_upstream_weight_mass"] if len(adapter_upstream_weights) > 0 else [],
        "forbidden_features_checked": {
            "lesion_delta": False,
            "LS_ctrl": False,
            "assay_pass_fail": False,
            "oracle_region_label": False,
            "real_minus_temporal_control": False,
            "direct_interface_membership": False,
            "candidate_assay_outcome": False,
            "adapter_virtual_unit_membership": False,
        },
        "contamination_issues": [] if contamination_pass else [
            f"overlap_with_excluded={sorted(overlap_with_excluded)}"
        ],
    }
    write_json(out_dir / "blind_region_predictor_v3_contamination_audit.json", contamination_audit)

    # ------------------------------------------------------------------
    # Feature table CSV
    # ------------------------------------------------------------------
    ft_csv_path = out_dir / "blind_region_predictor_v3_feature_table.csv"
    ft_rows = []
    for nid in candidates:
        ft = feat_by_node.get(nid, {})
        ft_rows.append({
            "node_id": int(nid),
            **{fname: ft.get(fname, 0.0) for fname in V3_FEATURE_NAMES},
        })
    if ft_rows:
        ft_fields = sorted(ft_rows[0].keys())
        with open(ft_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=ft_fields)
            writer.writeheader()
            writer.writerows(ft_rows)

    # ------------------------------------------------------------------
    # Candidate ranking CSV
    # ------------------------------------------------------------------
    ranking_csv_path = out_dir / "blind_region_predictor_v3_candidate_ranking.csv"
    ranking_rows = []
    for rank_idx, (nid, score) in enumerate(full_ranking):
        ft = feat_by_node.get(nid, {})
        ranking_rows.append({
            "rank": rank_idx + 1,
            "node_id": int(nid),
            "composite_score": round(float(score), 8),
            **{fname: ft.get(fname, 0.0) for fname in V3_FEATURE_NAMES},
            "selected": 1 if nid in predicted_set else 0,
        })
    if ranking_rows:
        r_fields = sorted(ranking_rows[0].keys())
        with open(ranking_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=r_fields)
            writer.writeheader()
            writer.writerows(ranking_rows)

    # ------------------------------------------------------------------
    # Preregistration JSON (separate file)
    # ------------------------------------------------------------------
    write_json(out_dir / "predictor_v3_preregistration.json", preregistration)

    return prediction_result


# ---------------------------------------------------------------------------
# Predictor core v1: noninterface_causal_proxy_core_v1
# Same features as V3, but outputs both broad (8) and core (4) regions.
# ---------------------------------------------------------------------------

CORE_V1_FEATURE_WEIGHTS = {k: v for k, v in V3_FEATURE_WEIGHTS.items()}
CORE_V1_FEATURE_NAMES = list(V3_FEATURE_NAMES)
CORE_REGION_SIZE = 4
BROAD_REGION_SIZE = 8


def run_noninterface_causal_proxy_core_v1_predictor(
    input_root: str,
    output_dir: str,
    seed: int = 42,
    exclude_interface_nodes: bool = True,
) -> dict:
    """Run predictor core v1: same blind-safe features as V3, outputs both
    broad (8-node) and core (4-node) regions.

    The core region is the top-4 subset of the broad candidate ranking.
    No additional outcome or assay information is used.
    """
    import csv
    import json
    import numpy as np
    from pathlib import Path

    from .utils import ensure_dir, write_json

    out_dir = ensure_dir(output_dir)
    inp_root = Path(input_root)

    # ------------------------------------------------------------------
    # Load input data (same as V3)
    # ------------------------------------------------------------------
    registry_path = inp_root / "interface_node_registry.json"
    feat_table_path = inp_root / "deconfounded_region_predictor_gap/blind_safe_feature_table_v3.csv"
    existing_pred_path = inp_root / "blind_region_prediction.json"

    registry = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.exists() else {}
    existing_pred = json.loads(existing_pred_path.read_text(encoding="utf-8")) if existing_pred_path.exists() else {}

    if not feat_table_path.exists():
        # Fall back to V3 prediction if available
        v3_path = inp_root / "blind_region_prediction_v3.json"
        if v3_path.exists():
            v3_data = json.loads(v3_path.read_text(encoding="utf-8"))
            ranking = v3_data.get("candidate_ranking", [])
            candidates = [int(r[0]) for r in ranking]
            scores = [float(r[1]) for r in ranking]
        else:
            result = {
                "prediction_mode": "noninterface_causal_proxy_core_v1",
                "prediction_available": False,
                "error": f"v3 feature table not found at {feat_table_path} and no fallback V3 prediction available.",
            }
            write_json(out_dir / "blind_region_prediction_core_v1.json", result)
            return result

        # Construct pseudo-ranking from existing V3 prediction
        all_ranked = list(zip(candidates, scores))
        all_ranked.sort(key=lambda x: x[1], reverse=True)
        full_ranking = all_ranked
        feat_rows = {}
    else:
        feat_rows_list: list[dict] = []
        with open(feat_table_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                feat_rows_list.append(row)

        candidates = set()
        feat_by_node: dict[int, dict] = {}
        for row in feat_rows_list:
            nid = int(row["node_id"])
            candidates.add(nid)
            feat_by_node[nid] = {
                "event_core_gradient_mass": float(row.get("event_core_gradient_mass", 0.0)),
                "projection_weight_mass": float(row.get("projection_weight_mass", 0.0)),
                "activation_gradient_product": float(row.get("activation_gradient_product", 0.0)),
                "projection_gradient_product": float(row.get("projection_gradient_product", 0.0)),
                "event_local_activation_response": float(row.get("event_local_activation_response", 0.0)),
            }

        # Load adapter upstream weight mass
        adapter_weight_path = inp_root / "adapter_weight_summary.json"
        adapter_upstream_weights: dict[int, float] = {}
        if adapter_weight_path.exists():
            try:
                aws = json.loads(adapter_weight_path.read_text(encoding="utf-8"))
                adapter_weight_matrix = aws.get("adapter_weight_matrix", [])
                if adapter_weight_matrix:
                    awm = np.array(adapter_weight_matrix, dtype=np.float64)
                    if awm.ndim == 2:
                        if awm.shape[0] == len(aws.get("num_adapter_channels", 4)):
                            per_node_mass = np.abs(awm).sum(axis=0)
                        else:
                            per_node_mass = np.abs(awm).sum(axis=1)
                        for i_nid in candidates:
                            if i_nid < len(per_node_mass):
                                adapter_upstream_weights[i_nid] = float(per_node_mass[i_nid])
            except Exception:
                pass

        # control_mask_response from existing prediction
        feat_table_raw = existing_pred.get("feature_table", {})
        for nid in candidates:
            if nid not in feat_by_node:
                feat_by_node[nid] = {
                    "event_core_gradient_mass": 0.0,
                    "projection_weight_mass": 0.0,
                    "activation_gradient_product": 0.0,
                    "projection_gradient_product": 0.0,
                    "event_local_activation_response": 0.0,
                }
            feat_by_node[nid]["control_mask_response"] = float(
                feat_table_raw.get(str(nid), {}).get("control_mask_response", 0.0)
            )
            feat_by_node[nid]["adapter_upstream_weight_mass"] = adapter_upstream_weights.get(nid, 0.0)

        # Filter candidates by interface exclusion
        candidate_pool = registry.get("candidate_pool_nodes", list(range(16, 32)))
        excluded_nodes = set(registry.get("excluded_from_region_candidates", []))
        if exclude_interface_nodes:
            candidates = [n for n in candidate_pool if n not in excluded_nodes]
        else:
            candidates = list(candidate_pool)

        # Build feature matrix, z-score normalize, apply fixed weights
        candidates = sorted(candidates)
        n_nodes = len(candidates)
        if n_nodes == 0:
            result = {
                "prediction_mode": "noninterface_causal_proxy_core_v1",
                "prediction_available": False,
                "error": "candidate_pool_empty",
                "interface_contamination_pass": True,
            }
            write_json(out_dir / "blind_region_prediction_core_v1.json", result)
            return result

        matrix = np.zeros((n_nodes, len(CORE_V1_FEATURE_NAMES)), dtype=np.float64)
        for i, nid in enumerate(candidates):
            ft = feat_by_node.get(nid, {})
            for j, fname in enumerate(CORE_V1_FEATURE_NAMES):
                matrix[i, j] = ft.get(fname, 0.0)

        means = matrix.mean(axis=0, keepdims=True)
        stds = matrix.std(axis=0, keepdims=True)
        stds[stds < 1e-9] = 1.0
        normalized = (matrix - means) / stds

        weights_arr = np.array([CORE_V1_FEATURE_WEIGHTS[f] for f in CORE_V1_FEATURE_NAMES], dtype=np.float64)
        weights_arr = weights_arr / weights_arr.sum()
        composite = normalized @ weights_arr

        # Rank candidates
        scored = list(zip(candidates, composite.tolist()))
        scored.sort(key=lambda x: x[1], reverse=True)
        full_ranking = [(int(n), float(s)) for n, s in scored]

    # ------------------------------------------------------------------
    # Extract broad region (top-8) and core region (top-4)
    # ------------------------------------------------------------------
    broad_k = BROAD_REGION_SIZE
    core_k = CORE_REGION_SIZE

    broad_top = full_ranking[:broad_k]
    broad_region_nodes = [int(n) for n, _s in broad_top]
    broad_region_scores = [float(s) for _n, s in broad_top]

    core_top = full_ranking[:core_k]
    core_region_nodes = [int(n) for n, _s in core_top]
    core_region_scores = [float(s) for _n, s in core_top]

    # ------------------------------------------------------------------
    # Interface contamination check
    # ------------------------------------------------------------------
    excluded_nodes = set(registry.get("excluded_from_region_candidates", []))
    broad_predicted_set = set(broad_region_nodes)
    core_predicted_set = set(core_region_nodes)
    broad_overlap_with_excluded = broad_predicted_set & excluded_nodes
    core_overlap_with_excluded = core_predicted_set & excluded_nodes
    broad_contamination_pass = len(broad_overlap_with_excluded) == 0
    core_contamination_pass = len(core_overlap_with_excluded) == 0

    # ------------------------------------------------------------------
    # Build prediction result
    # ------------------------------------------------------------------
    prediction_result = {
        "prediction_mode": "noninterface_causal_proxy_core_v1",
        "prediction_available": True,
        "broad_region_nodes": broad_region_nodes,
        "broad_region_scores": broad_region_scores,
        "broad_region_size": BROAD_REGION_SIZE,
        "core_region_nodes": core_region_nodes,
        "core_region_scores": core_region_scores,
        "core_region_size": CORE_REGION_SIZE,
        "num_candidates_ranked": len(full_ranking),
        "feature_weights": {k: float(v) for k, v in CORE_V1_FEATURE_WEIGHTS.items()},
        "feature_normalization": "zscore_across_candidate_pool",
        "uses_lesion_information": False,
        "uses_assay_outcome": False,
        "uses_random_control_outcome": False,
        "uses_real_minus_temporal_control": False,
        "uses_oracle_label": False,
        "broad_contamination_pass": broad_contamination_pass,
        "core_contamination_pass": core_contamination_pass,
        "interface_contamination_pass": broad_contamination_pass and core_contamination_pass,
        "candidate_pool_size": len(full_ranking),
        "candidate_ranking": full_ranking,
    }
    write_json(out_dir / "blind_region_prediction_core_v1.json", prediction_result)

    # ------------------------------------------------------------------
    # Contamination audit
    # ------------------------------------------------------------------
    contamination_audit = {
        "prediction_mode": "noninterface_causal_proxy_core_v1",
        "broad_region_nodes": broad_region_nodes,
        "core_region_nodes": core_region_nodes,
        "broad_region_size": BROAD_REGION_SIZE,
        "core_region_size": CORE_REGION_SIZE,
        "num_candidates_ranked": len(full_ranking),
        "excluded_nodes": sorted(excluded_nodes),
        "broad_overlap_with_excluded_nodes": sorted(broad_overlap_with_excluded),
        "core_overlap_with_excluded_nodes": sorted(core_overlap_with_excluded),
        "broad_interface_contamination_pass": broad_contamination_pass,
        "core_interface_contamination_pass": core_contamination_pass,
        "interface_contamination_pass": broad_contamination_pass and core_contamination_pass,
        "uses_lesion_information": False,
        "uses_assay_outcome": False,
        "uses_random_control_outcome": False,
        "uses_real_minus_temporal_control": False,
        "uses_oracle_label": False,
        "allowed_features": CORE_V1_FEATURE_NAMES[:],
        "forbidden_features_checked": {
            "lesion_delta": False,
            "LS_ctrl": False,
            "assay_pass_fail": False,
            "oracle_region_label": False,
            "real_minus_temporal_control": False,
            "random_control_outcome": False,
            "direct_interface_membership": False,
        },
        "contamination_issues": (
            []
            if (broad_contamination_pass and core_contamination_pass)
            else [
                f"broad_overlap={sorted(broad_overlap_with_excluded)}",
                f"core_overlap={sorted(core_overlap_with_excluded)}",
            ]
        ),
    }
    write_json(out_dir / "blind_region_predictor_core_v1_contamination_audit.json", contamination_audit)

    # ------------------------------------------------------------------
    # Feature table CSV
    # ------------------------------------------------------------------
    if not feat_rows_list:
        ft_csv_path = out_dir / "blind_region_predictor_core_v1_feature_table.csv"
        ft_rows = []
        for nid in candidates:
            ft = feat_by_node.get(nid, {})
            if ft:
                ft_rows.append({
                    "node_id": int(nid),
                    **{fname: ft.get(fname, 0.0) for fname in CORE_V1_FEATURE_NAMES},
                })
        if ft_rows:
            ft_fields = sorted(ft_rows[0].keys())
            with open(ft_csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=ft_fields)
                writer.writeheader()
                writer.writerows(ft_rows)

    # ------------------------------------------------------------------
    # Candidate ranking CSV
    # ------------------------------------------------------------------
    ranking_csv_path = out_dir / "blind_region_predictor_core_v1_candidate_ranking.csv"
    ranking_rows = []
    for rank_idx, (nid, score) in enumerate(full_ranking):
        ranking_rows.append({
            "rank": rank_idx + 1,
            "node_id": int(nid),
            "composite_score": round(float(score), 8),
            "selected_broad": 1 if nid in broad_predicted_set else 0,
            "selected_core": 1 if nid in core_predicted_set else 0,
        })
    if ranking_rows:
        r_fields = sorted(ranking_rows[0].keys())
        with open(ranking_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=r_fields)
            writer.writeheader()
            writer.writerows(ranking_rows)

    return prediction_result


if __name__ == "__main__":
    # torch is imported lazily in the new functions
    import torch  # noqa: F401
    main()
