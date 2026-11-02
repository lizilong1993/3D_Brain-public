#!/usr/bin/env python3
"""Leave-one-seed-out (LOSO) vs leave-one-run-out (LORO) predictor audit.

Self-contained: reads analysis/results.json directly (no torch import), uses the
21 structure-only features from src/blind_region_predictor.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT_RUNS = Path(os.environ.get("RUNS_MULTISEED", "runs_multiseed"))
FEATURES = [
    "size", "mean_radius", "surface_fraction", "internal_density",
    "incoming_density_from_inputs", "outgoing_density_to_control_head",
    "outgoing_density_to_prediction_head", "outgoing_density_to_memory_head",
    "mean_edge_length", "long_edge_fraction", "degree_mean", "degree_std",
    "ctrl_coupling", "ctrl_path_mean", "pred_path_mean", "mem_path_mean",
    "region_to_control_head_reachability", "region_to_prediction_head_reachability",
    "region_to_memory_head_reachability", "input_to_region_reachability",
    "harmonic_closeness",
]


def safe_float(value):
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def build_rows(root: Path):
    rows = []
    for path in sorted(root.glob("seed_*/formal_*/analysis/results.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("model_type") == "moe":
            continue
        control_region_id = int(payload.get("control_region_id", -1))
        run_id = str(path.parent.parent)
        seed_match = re.search(r"seed_(\d+)", run_id)
        if not seed_match:
            continue
        seed = int(seed_match.group(1))
        for region_id_raw, stats in payload.get("region_stats", {}).items():
            region_id = int(region_id_raw)
            row = {
                "run_id": run_id,
                "seed": seed,
                "region_id": region_id,
                "label": 1 if region_id == control_region_id and control_region_id >= 0 else 0,
            }
            for key in FEATURES:
                row[key] = safe_float(stats.get(key, 0.0))
            rows.append(row)
    return rows


def top1(rows, scores):
    ranked = sorted(zip(rows, scores), key=lambda item: float(item[1]), reverse=True)
    return float(any(int(row["label"]) == 1 for row, _ in ranked[:1]))


def fit_eval(train, test):
    x_train = np.array([[row[name] for name in FEATURES] for row in train])
    y_train = np.array([int(row["label"]) for row in train])
    x_test = np.array([[row[name] for name in FEATURES] for row in test])
    y_test = np.array([int(row["label"]) for row in test])
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    clf.fit(x_train, y_train)
    scores = clf.predict_proba(x_test)[:, 1]
    return float(roc_auc_score(y_test, scores)), top1(test, scores)


def summarize(items, key):
    vals = [item[key] for item in items]
    return {"n": len(vals), "mean": float(np.mean(vals)), "std": float(np.std(vals)),
            "min": float(np.min(vals)), "max": float(np.max(vals))}


def main():
    ap = argparse.ArgumentParser(description="Leave-one-seed-out vs leave-one-run-out predictor audit.")
    ap.add_argument("--root", default=str(ROOT_RUNS), help="directory containing seed_*/formal_*/analysis/results.json")
    ap.add_argument("--out", default="predictor_loso.json")
    ap.add_argument("--split-out", default=None, help="optional CSV documenting the LOSO/LORO splits")
    args = ap.parse_args()

    rows = build_rows(Path(args.root))
    print("rows", len(rows), "runs", len({r["run_id"] for r in rows}))

    loro = []
    for run_id in sorted({r["run_id"] for r in rows}):
        train = [r for r in rows if r["run_id"] != run_id]
        test = [r for r in rows if r["run_id"] == run_id]
        if len({int(r["label"]) for r in test}) < 2 or len({int(r["label"]) for r in train}) < 2:
            continue
        auc, acc = fit_eval(train, test)
        loro.append({"run_id": run_id, "auc": auc, "top1": acc})

    loso = []
    for seed in range(42, 52):
        train = [r for r in rows if r["seed"] != seed]
        for suffix in ("formal_main_B", "formal_sparse_B"):
            test = [r for r in rows if r["seed"] == seed and r["run_id"].endswith(suffix)]
            if len({int(r["label"]) for r in test}) < 2:
                continue
            auc, acc = fit_eval(train, test)
            loso.append({"seed": seed, "run": suffix, "auc": auc, "top1": acc})

    result = {
        "feature_set": "structure_only",
        "root": str(args.root),
        "loro": {"folds": loro, "auc": summarize(loro, "auc"), "top1": summarize(loro, "top1")},
        "loso": {"folds": loso, "auc": summarize(loso, "auc"), "top1": summarize(loso, "top1")},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    if args.split_out:
        split_path = Path(args.split_out)
        split_path.parent.mkdir(parents=True, exist_ok=True)
        with split_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["design", "seed", "run_id", "n_regions", "n_positive"])
            for run_id in sorted({r["run_id"] for r in rows}):
                run_rows = [r for r in rows if r["run_id"] == run_id]
                writer.writerow(["loro", run_rows[0]["seed"], run_id, len(run_rows),
                                 sum(int(r["label"]) for r in run_rows)])
            for seed in range(42, 52):
                for suffix in ("formal_main_B", "formal_sparse_B"):
                    run_rows = [r for r in rows if r["seed"] == seed and r["run_id"].endswith(suffix)]
                    if not run_rows:
                        continue
                    writer.writerow(["loso", seed, run_rows[0]["run_id"], len(run_rows),
                                     sum(int(r["label"]) for r in run_rows)])

    print(json.dumps({"loro_auc": result["loro"]["auc"], "loso_auc": result["loso"]["auc"],
                      "loro_top1": result["loro"]["top1"], "loso_top1": result["loso"]["top1"]}, indent=2))
    print("saved", out)


if __name__ == "__main__":
    main()
