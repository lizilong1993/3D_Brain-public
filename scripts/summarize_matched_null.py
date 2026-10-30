#!/usr/bin/env python3
"""Summarize per-null lesion effects into RRP and Monte Carlo tail probability.

Tie handling follows the manuscript's unified rule:

    rrp   = (n_< + n_=) / M
    p_MC  = (1 + n_> + n_=) / (M + 1)

where M is the number of null draws (1000) and tolerance for ties is 1e-8.
The input is the CSV written by run_matched_null_assay.py; it must contain
10 seeds x 4 null types x 1000 rows.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import statistics
from pathlib import Path

TOLERANCE = 1e-8
EXPECTED_GROUPS = 40
EXPECTED_DRAWS = 1000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = list(csv.DictReader(open(args.input, encoding="utf-8")))
    groups = collections.defaultdict(list)
    for row in rows:
        groups[(row["seed"], row["null_type"])].append(row)

    out = []
    for (seed, null_type), group in sorted(groups.items(), key=lambda item: (int(item[0][0]), item[0][1])):
        targets = [float(row["target_rmt"]) for row in group]
        target = statistics.median(targets)
        values = [float(row["rmt_value"]) for row in group]
        n_less = sum(value < target - TOLERANCE for value in values)
        n_equal = sum(abs(value - target) <= TOLERANCE for value in values)
        n_greater = len(values) - n_less - n_equal
        rrp = (n_less + n_equal) / len(values)
        p_mc = (1 + n_greater + n_equal) / (len(values) + 1)
        out.append(
            {
                "seed": int(seed),
                "null_type": null_type,
                "n_null": len(values),
                "target_rmt_median": target,
                "target_rmt_min": min(targets),
                "target_rmt_max": max(targets),
                "target_rmt_tolerance": TOLERANCE,
                "n_less": n_less,
                "n_equal": n_equal,
                "n_greater": n_greater,
                "rrp_empirical": rrp,
                "p_mc": p_mc,
                "level2_pass": rrp >= 0.90 and p_mc <= 0.10,
            }
        )

    if len(out) != EXPECTED_GROUPS or any(item["n_null"] != EXPECTED_DRAWS for item in out):
        raise SystemExit(
            f"expected {EXPECTED_GROUPS} groups of {EXPECTED_DRAWS}; "
            f"got {len(out)} groups"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
