#!/usr/bin/env python3
"""Summarize event-local RMT assay outputs.

For each run JSON, computes RRP (strictly-less fraction) and tie-inclusive
p_MC = (1 + n_ge) / (M + 1) against each null family, then all-four pass.

Usage:
  python3 summarize_event_local.py --root T4 --out T4/summary
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

NULL_TYPES = ("same_size", "degree_matched", "surface_fraction_matched", "distance_matched")


def summarize_run(payload):
    seed = payload["seed"]
    if payload.get("status") == "NE" or not payload.get("candidate"):
        return {"seed": seed, "status": "NE", "candidate": [], "rmt": None, "all_four": False}

    rows = {row["label"]: row for row in payload["rows"]}
    cand_rmt = rows["candidate"]["rmt"]
    out = {"seed": seed, "status": "ok", "candidate": payload["candidate"], "rmt": cand_rmt}
    all_four = True
    for null_type in NULL_TYPES:
        null_rmts = [row["rmt"] for label, row in rows.items() if label.startswith(null_type + "_")]
        m = len(null_rmts)
        n_lt = sum(1 for value in null_rmts if value < cand_rmt)
        n_ge = m - n_lt
        rrp = n_lt / m
        p = (1 + n_ge) / (m + 1)
        passed = rrp >= 0.90 and p <= 0.10
        all_four = all_four and passed
        out[null_type] = {"rrp": round(rrp, 4), "p": round(p, 6), "pass": passed}
    out["all_four"] = all_four
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="directory containing seed_<model>_d<data>/ subdirs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.root)
    records = []
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("seed_")):
        model_seed, data_seed = run_dir.name.split("_")[1], run_dir.name.split("_")[2][1:]
        files = sorted(run_dir.glob("event_local_seed*.json"))
        if not files:
            continue
        payload = json.loads(files[-1].read_text(encoding="utf-8"))
        record = summarize_run(payload)
        record["model_seed"] = int(model_seed)
        record["data_seed"] = int(data_seed)
        records.append(record)

    by_model = defaultdict(list)
    for record in records:
        by_model[record["model_seed"]].append(record)

    model_rows = []
    for model_seed in sorted(by_model):
        runs = by_model[model_seed]
        if all(r["status"] == "NE" for r in runs):
            model_rows.append({"model_seed": model_seed, "n_runs": len(runs), "passes": 0, "stable": False, "status": "NE"})
            continue
        passes = sum(1 for r in runs if r["all_four"])
        model_rows.append({
            "model_seed": model_seed,
            "n_runs": len(runs),
            "passes": passes,
            "stable": passes >= 3,
            "status": "ok",
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(
        json.dumps({"runs": records, "models": model_rows}, indent=2), encoding="utf-8"
    )
    with out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model_seed", "data_seed", "status", "candidate", "rmt", "all_four"] +
                        [f"{t}_{k}" for t in NULL_TYPES for k in ("rrp", "p", "pass")])
        for record in records:
            row = [record["model_seed"], record["data_seed"], record["status"],
                   ";".join(map(str, record["candidate"])), record["rmt"], record["all_four"]]
            for null_type in NULL_TYPES:
                stat = record.get(null_type, {})
                row += [stat.get("rrp", ""), stat.get("p", ""), stat.get("pass", "")]
            writer.writerow(row)

    print(json.dumps(model_rows, indent=2))


if __name__ == "__main__":
    main()
