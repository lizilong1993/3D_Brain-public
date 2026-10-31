#!/usr/bin/env python3
"""Formal Route A assay: event-window state replacement vs temporal control.

Multi-batch version of pilot_event_local_rmt.py. For each validation batch:
- baseline hidden trajectory and control logits;
- one event window and one non-overlapping control window (from ctrl_mask);
- per region: replace hidden states inside the recurrence (event -> control
  states; control -> event states), recompute control logits, average CE over
  the window and samples; RMT = mean(event delta) - mean(control delta),
  averaged over batches.

Strict interface mode excludes 0-23 from candidate and null pool; if the
corrected candidate is empty, the run exits with status "NE".
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.null_model_samplers import (  # noqa: E402
    _undirected_degree,
    build_null_samplers,
    compute_candidate_stats,
)
from src.train import build_model_and_data, load_config  # noqa: E402
from src.utils import set_seed  # noqa: E402


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def forward_with_replacement(model, obs, mem_query, danger_flag, nodes, replace_map, source_hidden):
    batch_size, seq_len, _ = obs.shape
    device = obs.device
    model.reset_state(batch_size, device)
    hidden_t = model.last_hidden
    ctrl_logits = []
    region = torch.as_tensor(nodes, dtype=torch.long, device=device)
    for t in range(seq_len):
        input_t = torch.cat([obs[:, t], mem_query[:, t], danger_flag[:, t]], dim=-1)
        rec_message = model._message_pass(hidden_t)
        input_message = torch.zeros(batch_size, model.num_nodes, device=device)
        input_message[:, model.input_node_ids] = input_t @ model.W_in
        preact = rec_message + input_message + model.bias.unsqueeze(0)
        hidden_t = torch.tanh(preact)
        if model.active_node_mask is not None:
            hidden_t = hidden_t.masked_fill(~model.active_node_mask.view(1, -1), 0.0)
        if t in replace_map:
            src_t = replace_map[t]
            hidden_t = hidden_t.clone()
            hidden_t[:, region] = source_hidden[:, src_t, region]
        _, _, ctrl_pool = model.compute_readout_pools(hidden_t)
        ctrl_logits.append(model.ctrl_head(ctrl_pool))
    return torch.stack(ctrl_logits, dim=1)


def window_ce_batch(logits, target, window):
    idx = torch.as_tensor(window, dtype=torch.long, device=logits.device)
    logp = torch.log_softmax(logits[:, idx], dim=-1)
    tar = target[:, idx]
    picked = logp.gather(-1, tar.unsqueeze(-1)).squeeze(-1)
    return float(-picked.mean().item())


def choose_windows(ctrl_mask, seq_len, forced_center=None):
    if forced_center is not None:
        center = int(forced_center)
    else:
        active = [t for t in range(seq_len) if float(ctrl_mask[:, t].float().mean()) > 0.5]
        center = active[len(active) // 2] if len(active) >= 3 else seq_len // 2
    event = [center - 1, center, center + 1]
    event = [t for t in event if 0 <= t < seq_len]
    length = len(event)
    shift = -(length + 2)
    control = [t + shift for t in event]
    if min(control) < 0:
        shift = length + 2
        control = [t + shift for t in event]
    control = [t for t in control if 0 <= t < seq_len]
    if len(control) != length or set(control) & set(event):
        raise RuntimeError(f"invalid windows: event={event} control={control}")
    return event, control


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--validation-batches", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--iface-start", type=int, default=0)
    ap.add_argument("--iface-end", type=int, default=24)
    ap.add_argument("--candidate-nodes", default="")
    ap.add_argument("--event-center", type=int, default=None, help="override event window center (window specificity control)")
    args = ap.parse_args()
    interface = set(range(args.iface_start, args.iface_end))

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(str(run_dir / "resolved_config.yaml"))
    config["device"] = args.device if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)
    model, _, val_gen, device = build_model_and_data(config)
    ckpt = sorted((run_dir / "checkpoints").glob("step_*.pt"))[-1]
    payload = torch.load(ckpt, map_location=device)
    model.load_state_dict(payload["model_state"], strict=False)
    model.eval()

    analysis = load_json(run_dir / "analysis" / "results.json")
    region_id = analysis["predicted_regions_nonlesion"]["control"]["region_id"]
    region_map = {int(k): int(v) for k, v in load_json(run_dir / "analysis" / "regions.json").items()}
    candidate = sorted(n for n, r in region_map.items() if r == int(region_id) and n not in interface)
    if args.candidate_nodes:
        candidate = sorted(int(x) for x in args.candidate_nodes.split(";") if x)

    status = "ok"
    if not candidate:
        status = "NE"
        payload_out = {
            "run_dir": str(run_dir),
            "checkpoint": str(ckpt),
            "seed": args.seed,
            "status": "NE",
            "reason": "corrected candidate is empty under the declared interface",
            "interface": [min(interface), max(interface)],
            "candidate": [],
        }
        (out_dir / f"event_local_seed{args.seed}.json").write_text(json.dumps(payload_out, indent=2))
        print(json.dumps(payload_out, indent=2))
        return

    snap = sorted((run_dir / "snapshots").glob("step_*"))[-1]
    edge = torch.load(snap / "edge_index.pt", map_location="cpu")
    coords = torch.load(snap / "coords.pt", map_location="cpu").numpy()
    types = torch.load(snap / "node_types.pt", map_location="cpu").numpy()
    degree = _undirected_degree(edge, config["num_nodes"])
    stats = compute_candidate_stats(candidate, types, degree, coords)
    pool = np.array([n for n in range(config["num_nodes"]) if n not in interface], dtype=int)
    samplers = build_null_samplers(
        len(candidate),
        stats["degree_mean"],
        stats["surface_fraction"],
        stats["mean_pairwise_distance"],
        types,
        degree,
        coords,
        np.random.default_rng(args.seed),
        eligible_ids=pool,
    )

    regions = [("candidate", candidate)]
    for null_type, sampler in samplers.items():
        for i in range(args.draws):
            region = sorted(int(x) for x in sampler())
            if set(region) & interface:
                raise AssertionError((null_type, region))
            regions.append((f"{null_type}_{i}", region))

    batches = val_gen.sample_dataset(args.validation_batches, config["batch_size"])
    batch_data = []
    for batch in batches:
        obs = batch["obs"].to(device)
        mem_query = batch["mem_query"].to(device)
        danger = batch["danger_flag"].to(device)
        target = batch["ctrl_target"].to(device)
        mask = batch["ctrl_mask"].to(device)
        with torch.no_grad():
            base = model(obs, mem_query, danger)
        event, control = choose_windows(mask, obs.shape[1], args.event_center)
        batch_data.append(
            {
                "obs": obs,
                "mem_query": mem_query,
                "danger": danger,
                "target": target,
                "base_hidden": base["hidden_traj"],
                "base_ctrl": base["ctrl_logits"],
                "event": event,
                "control": control,
                "base_event_ce": window_ce_batch(base["ctrl_logits"], target, event),
                "base_control_ce": window_ce_batch(base["ctrl_logits"], target, control),
            }
        )

    rows = []
    t_start = time.perf_counter()
    for label, region in regions:
        ev_deltas = []
        ct_deltas = []
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for bd in batch_data:
            event_map = {t: bd["control"][k] for k, t in enumerate(bd["event"])}
            control_map = {t: bd["event"][k] for k, t in enumerate(bd["control"])}
            with torch.no_grad():
                ev_logits = forward_with_replacement(
                    model, bd["obs"], bd["mem_query"], bd["danger"], region, event_map, bd["base_hidden"]
                )
                ct_logits = forward_with_replacement(
                    model, bd["obs"], bd["mem_query"], bd["danger"], region, control_map, bd["base_hidden"]
                )
            ev_deltas.append(window_ce_batch(ev_logits, bd["target"], bd["event"]) - bd["base_event_ce"])
            ct_deltas.append(window_ce_batch(ct_logits, bd["target"], bd["control"]) - bd["base_control_ce"])
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        event_delta = float(np.mean(ev_deltas))
        control_delta = float(np.mean(ct_deltas))
        rows.append(
            {
                "label": label,
                "region_nodes": ";".join(map(str, region)),
                "region_size": len(region),
                "event_delta": event_delta,
                "control_delta": control_delta,
                "rmt": event_delta - control_delta,
                "seconds": elapsed,
            }
        )
    total_s = time.perf_counter() - t_start

    payload_out = {
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt),
        "seed": args.seed,
        "status": status,
        "device": str(device),
        "interface": [min(interface), max(interface)],
        "candidate": candidate,
        "candidate_stats": stats,
        "validation_batches": args.validation_batches,
        "batch_size": config["batch_size"],
        "windows": [{"event": bd["event"], "control": bd["control"]} for bd in batch_data],
        "baseline_event_ce": [bd["base_event_ce"] for bd in batch_data],
        "baseline_control_ce": [bd["base_control_ce"] for bd in batch_data],
        "n_regions": len(rows),
        "draws_per_null": args.draws,
        "total_seconds": total_s,
        "seconds_per_region": total_s / len(rows),
        "rows": rows,
    }
    (out_dir / f"event_local_seed{args.seed}.json").write_text(json.dumps(payload_out, indent=2))
    with (out_dir / f"event_local_seed{args.seed}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({k: payload_out[k] for k in (
        "status", "interface", "candidate", "validation_batches", "n_regions",
        "total_seconds", "seconds_per_region")}, indent=2))


if __name__ == "__main__":
    main()
