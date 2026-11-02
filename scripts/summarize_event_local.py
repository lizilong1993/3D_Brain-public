#!/usr/bin/env python3
"""Summarize event-local RMT assay outputs.

For each run JSON, computes the tie-inclusive RRP = (n_less + n_equal) / M and
p_MC = (1 + n_ge) / (M + 1), with ties at tolerance 1e-8, against each null
family, then the all-four status. Model-level summaries report both the
conditional rate over evaluable seeds and the full-flow rate over
training-valid seeds, plus the 4-draw stability count.

Usage:
  python3 summarize_event_local.py --root T4/out --out T4/summary
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

NULL_TYPES = ("same_size", "degree_matched", "surface_fraction_matched", "distance_matched")
TOL = 1e-8
INTERFACE = set(range(0, 24))
TRAINING_VALID_SEEDS = tuple(range(42, 51))
SEED_51 = 51


def load_candidate_fields(payload):
    if "raw_candidate" in payload:
        raw = [int(x) for x in payload.get("raw_candidate", [])]
        corrected = [int(x) for x in payload.get("corrected_candidate", payload.get("candidate", []))]
        raw_iai = payload.get("raw_iai")
        return raw, corrected, raw_iai
    run_dir = Path(payload["run_dir"])
    analysis = json.loads((run_dir / "analysis" / "results.json").read_text(encoding="utf-8"))
    region_id = analysis["predicted_regions_nonlesion"]["control"]["region_id"]
    region_map = {
        int(k): int(v)
        for k, v in json.loads((run_dir / "analysis" / "regions.json").read_text(encoding="utf-8")).items()
    }
    raw = sorted(n for n, r in region_map.items() if r == int(region_id))
    corrected = [n for n in raw if n not in INTERFACE]
    raw_iai = len(set(raw) & INTERFACE) / len(raw) if raw else None
    return raw, corrected, raw_iai


def summarize_run(payload):
    seed = int(payload["seed"])
    raw, corrected, raw_iai = load_candidate_fields(payload)
    base = {
        "seed": seed,
        "raw_candidate": raw,
        "corrected_candidate": corrected,
        "raw_candidate_size": len(raw),
        "corrected_candidate_size": len(corrected),
        "raw_iai": raw_iai,
        "ne_type": payload.get("ne_type"),
    }
    if payload.get("status") == "NE" or not corrected:
        base.update({"status": "NE", "rmt": None, "all_four": False})
        return base

    rows = {row["label"]: row for row in payload["rows"]}
    cand_rmt = rows["candidate"]["rmt"]
    out = {"status": "ok", "rmt": cand_rmt}
    all_four = True
    for null_type in NULL_TYPES:
        null_rmts = [row["rmt"] for label, row in rows.items() if label.startswith(null_type + "_")]
        m = len(null_rmts)
        n_lt = sum(1 for value in null_rmts if value < cand_rmt - TOL)
        n_eq = sum(1 for value in null_rmts if abs(value - cand_rmt) <= TOL)
        n_gt = m - n_lt - n_eq
        rrp = (n_lt + n_eq) / m
        p = (1 + n_gt + n_eq) / (m + 1)
        passed = rrp >= 0.90 and p <= 0.10
        all_four = all_four and passed
        out[null_type] = {"rrp": round(rrp, 4), "p": round(p, 6), "pass": passed}
    out["all_four"] = all_four
    base.update(out)
    return base


def write_table9(records, models, out_path):
    by_model = defaultdict(list)
    for record in records:
        by_model[record["model_seed"]].append(record)
    evaluable = [m["model_seed"] for m in models
                 if m["status"] == "ok" and m["model_seed"] in TRAINING_VALID_SEEDS]
    training_valid = list(TRAINING_VALID_SEEDS)
    rows = []
    for spec in list(NULL_TYPES) + ["all_four"]:
        single_cond = stable_cond = 0
        for seed in evaluable:
            runs = by_model[seed]
            primary = next((r for r in runs if r["data_seed"] == seed), None)
            if primary is None:
                continue

            def passed(record):
                if spec == "all_four":
                    return bool(record["all_four"])
                return bool(record.get(spec, {}).get("pass", False))

            single_cond += int(passed(primary))
            stable_cond += int(sum(1 for r in runs if passed(r)) >= 3)
        rows.append([spec, single_cond, len(evaluable), single_cond, len(training_valid),
                     stable_cond, len(evaluable), stable_cond, len(training_valid)])
    rows.append(["NE_empty_corrected", 0, 0, 0, 0, 0, 0, 0, 0])
    rows.append(["seed_51", 0, 0, 0, 0, 0, 0, 0, 0])
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["specification", "single_conditional_pass", "single_conditional_n",
                         "single_fullflow_pass", "single_fullflow_n",
                         "stability_conditional_pass", "stability_conditional_n",
                         "stability_fullflow_pass", "stability_fullflow_n"])
        writer.writerows(rows)


def write_table10(records, out_path):
    fields = ["model_seed", "data_seed", "status", "ne_type", "raw_candidate",
              "corrected_candidate", "raw_candidate_size", "corrected_candidate_size",
              "raw_iai", "rmt", "all_four"]
    for null_type in NULL_TYPES:
        fields += [f"{null_type}_rrp", f"{null_type}_p", f"{null_type}_pass"]
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {key: record.get(key, "") for key in fields if key in record}
            row["raw_candidate"] = ";".join(map(str, record["raw_candidate"]))
            row["corrected_candidate"] = ";".join(map(str, record["corrected_candidate"]))
            for null_type in NULL_TYPES:
                stat = record.get(null_type, {})
                row[f"{null_type}_rrp"] = stat.get("rrp", "")
                row[f"{null_type}_p"] = stat.get("p", "")
                row[f"{null_type}_pass"] = stat.get("pass", "")
            writer.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="directory containing seed_<model>_d<data>/ subdirs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--table9-out", default=None)
    ap.add_argument("--table10-out", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    records = []
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("seed_")):
        parts = run_dir.name.split("_")
        model_seed, data_seed = int(parts[1]), int(parts[2][1:])
        files = sorted(run_dir.glob("event_local_seed*.json"))
        if not files:
            continue
        payload = json.loads(files[-1].read_text(encoding="utf-8"))
        record = summarize_run(payload)
        record["model_seed"] = model_seed
        record["data_seed"] = data_seed
        records.append(record)

    by_model = defaultdict(list)
    for record in records:
        by_model[record["model_seed"]].append(record)

    models = []
    for model_seed in sorted(by_model):
        runs = by_model[model_seed]
        if all(r["status"] == "NE" for r in runs):
            models.append({"model_seed": model_seed, "n_runs": len(runs), "passes": 0,
                           "stable": False, "status": "NE"})
            continue
        passes = sum(1 for r in runs if r["all_four"])
        models.append({"model_seed": model_seed, "n_runs": len(runs), "passes": passes,
                       "stable": passes >= 3, "status": "ok"})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(
        json.dumps({"runs": records, "models": models}, indent=2), encoding="utf-8"
    )
    with out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model_seed", "data_seed", "status", "ne_type", "raw_candidate",
                         "corrected_candidate", "raw_candidate_size", "corrected_candidate_size",
                         "raw_iai", "rmt", "all_four"] +
                        [f"{t}_{k}" for t in NULL_TYPES for k in ("rrp", "p", "pass")])
        for record in records:
            row = [record["model_seed"], record["data_seed"], record["status"],
                   record.get("ne_type", ""), ";".join(map(str, record["raw_candidate"])),
                   ";".join(map(str, record["corrected_candidate"])),
                   record["raw_candidate_size"], record["corrected_candidate_size"],
                   record.get("raw_iai", ""), record.get("rmt", ""), record["all_four"]]
            for null_type in NULL_TYPES:
                stat = record.get(null_type, {})
                row += [stat.get("rrp", ""), stat.get("p", ""), stat.get("pass", "")]
            writer.writerow(row)

    if args.table9_out:
        write_table9(records, models, args.table9_out)
    if args.table10_out:
        write_table10(records, args.table10_out)

    print(json.dumps(models, indent=2))


if __name__ == "__main__":
    main()
