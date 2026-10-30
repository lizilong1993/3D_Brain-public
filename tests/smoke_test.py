#!/usr/bin/env python3
"""CPU smoke test for the minimal 3D Brain release.

Exercises: config loading -> model/data build -> evaluation -> recurrent lesion
-> matched-null sampling -> summary script on synthetic data.
"""
from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluate import evaluate_model_on_batches  # noqa: E402
from src.lesion import run_node_lesion, summarize_lesion_effects  # noqa: E402
from src.null_model_samplers import (  # noqa: E402
    _undirected_degree,
    build_null_samplers,
    compute_candidate_stats,
)
from src.train import build_model_and_data, load_config  # noqa: E402
from src.utils import set_seed  # noqa: E402

INTERFACE = set(range(16, 24))


def main() -> None:
    config = load_config(str(ROOT / "configs" / "smoke_main_B.yaml"))
    config["device"] = "cpu"
    assert config["num_nodes"] == 128, config["num_nodes"]

    set_seed(0)
    model, _generator, val_generator, _device = build_model_and_data(config)
    cached = val_generator.sample_dataset(1, config["batch_size"])
    baseline = evaluate_model_on_batches(model, cached)

    lesioned = run_node_lesion(model, [24, 25], val_generator, 1, config, cached_batches=cached)
    effects = summarize_lesion_effects(baseline, lesioned)
    assert math.isfinite(effects["LS_ctrl"]), effects

    types = model.node_types.detach().cpu().numpy()
    coords = model.coords.detach().cpu().numpy()
    degree = _undirected_degree(model.edge_index.detach().cpu(), config["num_nodes"])
    stats = compute_candidate_stats([24, 25], types, degree, coords)

    pool = np.array([n for n in range(config["num_nodes"]) if n not in INTERFACE], dtype=int)
    rng = np.random.default_rng(0)
    samplers = build_null_samplers(
        len([24, 25]),
        stats["degree_mean"],
        stats["surface_fraction"],
        stats["mean_pairwise_distance"],
        types,
        degree,
        coords,
        rng,
        eligible_ids=pool,
    )
    for name, sampler in samplers.items():
        region = [int(x) for x in sampler()]
        assert len(region) == 2, (name, region)
        assert not (set(region) & INTERFACE), (name, region)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        rows_path = tmp_path / "nulls.csv"
        out_path = tmp_path / "summary.json"
        with rows_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "seed",
                    "null_type",
                    "region_index",
                    "region_nodes",
                    "target_rmt",
                    "rmt_value",
                    "candidate_label",
                    "assay_seed",
                ]
            )
            for seed in range(42, 52):
                for null_type in (
                    "same_size",
                    "degree_matched",
                    "surface_fraction_matched",
                    "distance_matched",
                ):
                    for index in range(1000):
                        writer.writerow(
                            [seed, null_type, index, "24;25", 0.1, index / 1000.0 - 0.5, "demo", 0]
                        )
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "summarize_matched_null.py"),
                "--input",
                str(rows_path),
                "--output",
                str(out_path),
            ],
            check=True,
        )
        summary = json.loads(out_path.read_text(encoding="utf-8"))
        assert len(summary) == 40, len(summary)
        assert all(row["n_null"] == 1000 for row in summary)
        row = summary[0]
        expected = (row["n_greater"] + row["n_equal"] + 1) / 1001
        assert abs(row["p_mc"] - expected) < 1e-12, (row["p_mc"], expected)

    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
