#!/usr/bin/env python3
"""Report actual matched-null deviations from the candidate statistics.

Degree- and distance-matched samplers use best-of-N draws without a hard
tolerance; this script quantifies the residual mismatch for one assay output.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.null_model_samplers import _undirected_degree, compute_candidate_stats  # noqa: E402

FAMILIES = ("same_size", "degree_matched", "surface_fraction_matched", "distance_matched")


def region_stats(nodes, node_types, degree, coords):
    return compute_candidate_stats(nodes, node_types, degree, coords)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--assay-json", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    payload = json.loads(Path(args.assay_json).read_text(encoding="utf-8"))
    labels = {row["label"]: row["region_nodes"] for row in payload["rows"]}
    candidate = [int(x) for x in labels["candidate"].split(";") if x]

    snapshot = sorted((Path(args.run_dir) / "snapshots").glob("step_*"))[-1]
    edge = torch.load(snapshot / "edge_index.pt", map_location="cpu")
    coords = torch.load(snapshot / "coords.pt", map_location="cpu").numpy()
    node_types = torch.load(snapshot / "node_types.pt", map_location="cpu").numpy()
    degree = _undirected_degree(edge, len(node_types))
    target = region_stats(candidate, node_types, degree, coords)

    rows = []
    for family in FAMILIES:
        deviations = []
        for label, node_string in labels.items():
            if not label.startswith(family + "_"):
                continue
            nodes = [int(x) for x in node_string.split(";") if x]
            stats = region_stats(nodes, node_types, degree, coords)
            deviations.append([
                abs(stats["degree_mean"] - target["degree_mean"]),
                abs(stats["surface_fraction"] - target["surface_fraction"]),
                abs(stats["mean_pairwise_distance"] - target["mean_pairwise_distance"]),
            ])
        arr = np.array(deviations, dtype=float)
        rows.append([family, len(arr),
                     round(float(np.mean(arr[:, 0])), 6) if len(arr) else "",
                     round(float(np.mean(arr[:, 1])), 6) if len(arr) else "",
                     round(float(np.mean(arr[:, 2])), 6) if len(arr) else ""])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["family", "n", "mean_abs_degree_dev",
                         "mean_abs_surface_fraction_dev", "mean_abs_mean_pairwise_distance_dev"])
        writer.writerows(rows)
    print(out)


if __name__ == "__main__":
    main()
