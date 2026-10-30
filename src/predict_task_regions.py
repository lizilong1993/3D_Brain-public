from __future__ import annotations

from pathlib import Path
import json

import numpy as np


FORBIDDEN_NONLESION_FEATURES = {
    "delta_pred",
    "delta_mem",
    "delta_ctrl",
    "LS_pred",
    "LS_mem",
    "LS_ctrl",
    "lesion_drop",
    "random_excess",
    "empirical_p_value",
    "post-lesion performance",
}

TASK_TO_DELTA_KEY = {
    "control": "Delta Ctrl",
    "memory": "Delta Mem",
    "prediction": "Delta Pred",
}

TASK_TO_LS_KEY = {
    "control": "LS_ctrl",
    "memory": "LS_mem",
    "prediction": "LS_pred",
}

TASK_FEATURE_SPECS = {
    "structure_only": {
        "control": {
            "weights": [
                ("outgoing_density_to_control_head", 0.26),
                ("ctrl_coupling", 0.18),
                ("region_to_control_head_reachability", 0.14),
                ("input_to_region_reachability", 0.12),
                ("internal_density", 0.1),
                ("harmonic_closeness", 0.08),
                ("surface_fraction", 0.06),
                ("compactness_proxy", 0.06),
            ],
            "required_keys": ["outgoing_density_to_control_head", "region_to_control_head_reachability"],
            "reason": "control_structure_features_missing",
        },
        "memory": {
            "weights": [
                ("outgoing_density_to_memory_head", 0.28),
                ("region_to_memory_head_reachability", 0.18),
                ("internal_density", 0.12),
                ("harmonic_closeness", 0.12),
                ("input_to_region_reachability", 0.1),
                ("mean_edge_length_inverted", 0.1),
                ("compactness_proxy", 0.1),
            ],
            "required_keys": ["outgoing_density_to_memory_head", "region_to_memory_head_reachability"],
            "reason": "memory_structure_features_missing",
        },
        "prediction": {
            "weights": [
                ("outgoing_density_to_prediction_head", 0.22),
                ("region_to_prediction_head_reachability", 0.18),
                ("input_to_region_reachability", 0.12),
                ("internal_density", 0.1),
                ("mean_edge_length_inverted", 0.1),
                ("long_edge_fraction_inverted", 0.08),
                ("surface_fraction", 0.06),
                ("harmonic_closeness", 0.07),
                ("compactness_proxy", 0.07),
            ],
            "required_keys": ["outgoing_density_to_prediction_head", "region_to_prediction_head_reachability", "input_to_region_reachability"],
            "reason": "prediction_structure_features_missing",
        },
    },
    "activity_only": {
        "control": {
            "weights": [
                ("danger_flag_response", 0.26),
                ("control_mask_correlation", 0.28),
                ("danger_probe_auc", 0.18),
                ("input_response_correlation", 0.14),
                ("activation_autocorrelation", 0.14),
            ],
            "required_keys": ["control_mask_correlation", "danger_flag_response"],
            "reason": "control_activity_features_missing",
        },
        "memory": {
            "weights": [
                ("memory_query_response", 0.3),
                ("memory_query_correlation", 0.28),
                ("mem_mask_correlation", 0.2),
                ("activation_autocorrelation", 0.12),
                ("temporal_activation_variance", 0.1),
            ],
            "required_keys": ["memory_query_correlation", "mem_mask_correlation"],
            "reason": "memory_activity_features_missing",
        },
        "prediction": {
            "weights": [
                ("pred_mask_correlation", 0.22),
                ("prediction_target_response", 0.2),
                ("prediction_target_correlation", 0.18),
                ("prediction_error_correlation", 0.14),
                ("temporal_activation_variance", 0.12),
                ("activation_autocorrelation", 0.08),
                ("input_response_correlation", 0.06),
            ],
            "required_keys": ["pred_mask_correlation", "prediction_target_response", "prediction_target_correlation"],
            "reason": "prediction_activity_features_missing",
        },
    },
}

TASK_FEATURE_SPECS["hybrid_nonlesion"] = {
    task_name: {
        "weights": TASK_FEATURE_SPECS["structure_only"][task_name]["weights"] + TASK_FEATURE_SPECS["activity_only"][task_name]["weights"],
        "required_keys": TASK_FEATURE_SPECS["structure_only"][task_name]["required_keys"],
        "activity_required_keys": TASK_FEATURE_SPECS["activity_only"][task_name]["required_keys"],
        "reason": f"{task_name}_hybrid_features_missing",
    }
    for task_name in ["control", "memory", "prediction"]
}


def _safe_float(value: object) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def normalize_metric(values: dict[int, float]) -> dict[int, float]:
    if not values:
        return {}
    array = np.array(list(values.values()), dtype=np.float64)
    min_value = float(array.min())
    max_value = float(array.max())
    if max_value - min_value < 1e-12:
        return {key: 0.0 for key in values}
    return {key: float((value - min_value) / (max_value - min_value)) for key, value in values.items()}


def _region_feature_value(stats: dict, feature_name: str) -> float:
    if feature_name == "compactness_proxy":
        return 1.0 / (1.0 + max(0.0, _safe_float(stats.get("mean_radius", 0.0))))
    if feature_name == "mean_edge_length_inverted":
        return 1.0 / (1.0 + max(0.0, _safe_float(stats.get("mean_edge_length", 0.0))))
    if feature_name == "long_edge_fraction_inverted":
        return 1.0 - _safe_float(stats.get("long_edge_fraction", 0.0))
    return _safe_float(stats.get(feature_name, 0.0))


def _activity_signal_present(stats: dict, feature_names: list[str]) -> bool:
    return any(abs(_region_feature_value(stats, name)) > 1e-9 for name in feature_names)


def score_regions_for_task(region_stats: dict[int, dict] | dict[str, dict], task_name: str, feature_set: str) -> dict[int, float]:
    if feature_set not in TASK_FEATURE_SPECS:
        raise ValueError(f"Unsupported feature_set={feature_set}")
    region_stats_int = {int(region_id): stats for region_id, stats in region_stats.items()}
    spec = TASK_FEATURE_SPECS[feature_set][task_name]
    normalized_by_key: dict[str, dict[int, float]] = {}
    for key, _weight in spec["weights"]:
        raw_values = {region_id: _region_feature_value(stats, key) for region_id, stats in region_stats_int.items()}
        normalized_by_key[key] = normalize_metric(raw_values)
    score_map: dict[int, float] = {}
    for region_id in region_stats_int:
        score_map[region_id] = float(sum(weight * normalized_by_key[key].get(region_id, 0.0) for key, weight in spec["weights"]))
    return score_map


def rank_task_regions_without_lesion(region_stats: dict[int, dict] | dict[str, dict], task_name: str, feature_set: str) -> dict:
    region_stats_int = {int(region_id): stats for region_id, stats in region_stats.items()}
    spec = TASK_FEATURE_SPECS[feature_set][task_name]
    score_map = score_regions_for_task(region_stats_int, task_name, feature_set)
    has_required_signal = any(_activity_signal_present(region_stats_int[region_id], spec.get("required_keys", [])) for region_id in region_stats_int)
    if feature_set == "hybrid_nonlesion":
        has_required_signal = has_required_signal and any(_activity_signal_present(region_stats_int[region_id], spec.get("activity_required_keys", [])) for region_id in region_stats_int)
    if feature_set == "activity_only":
        has_required_signal = any(_activity_signal_present(region_stats_int[region_id], spec.get("required_keys", [])) for region_id in region_stats_int)
    if not score_map or not has_required_signal:
        return {"ranking": [], "available": False, "reason": spec["reason"]}
    ranking = sorted(
        [{"region_id": int(region_id), "score": float(score)} for region_id, score in score_map.items()],
        key=lambda item: item["score"],
        reverse=True,
    )
    return {"ranking": ranking, "available": True, "reason": None}


def predict_task_regions_without_lesion(region_stats: dict[int, dict] | dict[str, dict], feature_set: str = "structure_only", unique_regions: bool = True) -> dict[str, dict]:
    available_regions = set(int(region_id) for region_id in region_stats)
    predictions: dict[str, dict] = {}
    used_regions: set[int] = set()
    for task_name in ["control", "memory", "prediction"]:
        ranking_bundle = rank_task_regions_without_lesion(region_stats, task_name, feature_set)
        if not ranking_bundle["available"]:
            predictions[task_name] = {"region_id": "not_available", "score": 0.0, "available": False, "reason": ranking_bundle["reason"]}
            continue
        chosen_region = None
        for candidate in ranking_bundle["ranking"]:
            region_id = int(candidate["region_id"])
            if (not unique_regions) or region_id not in used_regions or len(used_regions) >= len(available_regions) - 1:
                chosen_region = candidate
                break
        if chosen_region is None:
            predictions[task_name] = {"region_id": "not_available", "score": 0.0, "available": False, "reason": ranking_bundle["reason"]}
            continue
        region_id = int(chosen_region["region_id"])
        used_regions.add(region_id)
        predictions[task_name] = {
            "region_id": region_id,
            "score": float(chosen_region["score"]),
            "available": True,
            "reason": None,
        }
    return predictions


def _task_specific_posthoc(lesion_row: dict, task_name: str) -> bool:
    delta_key = TASK_TO_DELTA_KEY[task_name]
    ls_key = TASK_TO_LS_KEY[task_name]
    other_delta_keys = [value for key, value in TASK_TO_DELTA_KEY.items() if key != task_name]
    return _safe_float(lesion_row.get(ls_key)) > 0.0 and all(_safe_float(lesion_row.get(delta_key)) > _safe_float(lesion_row.get(other_key)) for other_key in other_delta_keys)


def _lesion_lookup(lesion_matrix: dict[int, dict] | dict[str, dict]) -> dict[int, dict]:
    return {int(region_id): payload for region_id, payload in lesion_matrix.items()}


def best_posthoc_region(region_stats: dict[int, dict] | dict[str, dict], lesion_matrix: dict[int, dict] | dict[str, dict], task_name: str) -> tuple[int | None, dict | None]:
    lesion_lookup = _lesion_lookup(lesion_matrix)
    if not lesion_lookup:
        return None, None
    delta_key = TASK_TO_DELTA_KEY[task_name]
    ls_key = TASK_TO_LS_KEY[task_name]
    ranking = sorted(
        lesion_lookup.items(),
        key=lambda item: (_safe_float(item[1].get(delta_key)), _safe_float(item[1].get(ls_key))),
        reverse=True,
    )
    for region_id, row in ranking:
        if _task_specific_posthoc(row, task_name):
            return int(region_id), row
    if not ranking:
        return None, None
    region_id, row = ranking[0]
    return int(region_id), row


def build_prediction_candidate_rows(
    region_stats: dict[int, dict] | dict[str, dict],
    lesion_matrix: dict[int, dict] | dict[str, dict] | None = None,
    seed: int | None = None,
    model_family: str | None = None,
    condition: str | None = None,
) -> list[dict]:
    region_stats_int = {int(region_id): stats for region_id, stats in region_stats.items()}
    lesion_lookup = _lesion_lookup(lesion_matrix or {})
    rankings = {mode: rank_task_regions_without_lesion(region_stats_int, "prediction", mode) for mode in ["structure_only", "activity_only", "hybrid_nonlesion"]}
    rank_maps: dict[str, dict[int, int]] = {}
    score_maps: dict[str, dict[int, float]] = {}
    for mode, payload in rankings.items():
        rank_maps[mode] = {int(item["region_id"]): rank for rank, item in enumerate(payload["ranking"], start=1)}
        score_maps[mode] = {int(item["region_id"]): float(item["score"]) for item in payload["ranking"]}
    rows = []
    for region_id, stats in sorted(region_stats_int.items()):
        lesion_row = lesion_lookup.get(region_id, {})
        rows.append(
            {
                "seed": int(seed) if seed is not None else -1,
                "model_family": model_family or "unknown",
                "condition": condition or "unknown",
                "region_id": int(region_id),
                "rank_structure_only": int(rank_maps["structure_only"].get(region_id, 0)),
                "score_structure_only": float(score_maps["structure_only"].get(region_id, 0.0)),
                "rank_activity_only": int(rank_maps["activity_only"].get(region_id, 0)),
                "score_activity_only": float(score_maps["activity_only"].get(region_id, 0.0)),
                "rank_hybrid_nonlesion": int(rank_maps["hybrid_nonlesion"].get(region_id, 0)),
                "score_hybrid_nonlesion": float(score_maps["hybrid_nonlesion"].get(region_id, 0.0)),
                "region_size": int(_safe_float(stats.get("size", 0.0))),
                "pred_head_coupling": float(_safe_float(stats.get("outgoing_density_to_prediction_head", 0.0))),
                "pred_activity_score": float(
                    np.mean(
                        [
                            _safe_float(stats.get("pred_mask_correlation", 0.0)),
                            _safe_float(stats.get("prediction_target_response", stats.get("prediction_activity_response", 0.0))),
                            _safe_float(stats.get("prediction_target_correlation", 0.0)),
                            _safe_float(stats.get("prediction_error_correlation", 0.0)),
                        ]
                    )
                ),
                "pred_reachability": float(_safe_float(stats.get("region_to_prediction_head_reachability", 0.0))),
                "posthoc_pred_lesion_drop_for_evaluation_only": float(_safe_float(lesion_row.get("Delta Pred", 0.0))),
                "posthoc_LS_pred_for_evaluation_only": float(_safe_float(lesion_row.get("LS_pred", 0.0))),
            }
        )
    return rows


def evaluate_prediction_rankings(candidate_rows: list[dict], best_region_id: int | None = None) -> dict[str, dict[str, float | int | None]]:
    if not candidate_rows:
        return {
            mode: {
                "top1_accuracy": 0.0,
                "top2_accuracy": 0.0,
                "top3_accuracy": 0.0,
                "mean_rank_of_best_posthoc_prediction_region": None,
                "best_posthoc_prediction_region_id": best_region_id,
            }
            for mode in ["structure_only", "activity_only", "hybrid_nonlesion"]
        }
    if best_region_id is None:
        best_row = sorted(
            candidate_rows,
            key=lambda row: (float(row.get("posthoc_pred_lesion_drop_for_evaluation_only", 0.0)), float(row.get("posthoc_LS_pred_for_evaluation_only", 0.0))),
            reverse=True,
        )[0]
        best_region_id = int(best_row["region_id"])
    summary = {}
    for mode in ["structure_only", "activity_only", "hybrid_nonlesion"]:
        rank_key = f"rank_{mode}"
        matched = [row for row in candidate_rows if int(row["region_id"]) == int(best_region_id)]
        best_rank = int(matched[0][rank_key]) if matched else 0
        summary[mode] = {
            "top1_accuracy": 1.0 if best_rank == 1 else 0.0,
            "top2_accuracy": 1.0 if 0 < best_rank <= 2 else 0.0,
            "top3_accuracy": 1.0 if 0 < best_rank <= 3 else 0.0,
            "mean_rank_of_best_posthoc_prediction_region": float(best_rank) if best_rank > 0 else None,
            "best_posthoc_prediction_region_id": int(best_region_id),
        }
    return summary


def write_candidate_rankings_json(path: Path, rows: list[dict], evaluation: dict | None = None) -> None:
    payload = {
        "rows": rows,
        "evaluation": evaluation or {},
        "forbidden_features": sorted(FORBIDDEN_NONLESION_FEATURES),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
