#!/usr/bin/env python3
"""Run recurrent lesion assays on blind candidates and matched null regions.

Output schema is directly consumable by compute_routeA_causal_dispersion_metrics.py.
No RRP is inferred when a checkpoint, candidate, or assay row is missing.
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train import build_model_and_data, load_config
from src.utils import set_seed
from src.evaluate import evaluate_model_on_batches
from src.lesion import run_node_lesion, summarize_lesion_effects
from src.null_model_samplers import build_null_samplers, compute_candidate_stats, _undirected_degree

INTERFACE = set(range(16, 24))

def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields=["seed","null_type","region_index","region_nodes","target_rmt","rmt_value","candidate_label","assay_seed"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

def load_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))

def run_one(project_root, run_dir, seed, n_per_type, batches, assay_seed, null_types=None, start_index=0):
    run_dir=Path(run_dir); config_path=run_dir/"resolved_config.yaml"
    config=load_config(str(config_path)); config["device"]="cuda" if torch.cuda.is_available() else "cpu"
    set_seed(assay_seed + seed)
    model, generator, val_generator, device=build_model_and_data(config)
    ckpts=sorted((run_dir/"checkpoints").glob("step_*.pt"))
    if not ckpts: raise FileNotFoundError(f"no checkpoint: {run_dir}")
    payload=torch.load(ckpts[-1], map_location=device); model.load_state_dict(payload["model_state"], strict=False); model.eval()
    analysis=load_json(run_dir/"analysis"/"results.json")
    candidate_region=analysis["predicted_regions_nonlesion"]["control"]["region_id"]
    region_map={int(k):int(v) for k,v in load_json(run_dir/"analysis"/"regions.json").items()}
    candidate=sorted(n for n,r in region_map.items() if r==candidate_region and n not in INTERFACE)
    if not candidate: raise ValueError(f"empty blind candidate seed={seed}")
    cached=val_generator.sample_dataset(batches, config["batch_size"])
    # Every chunk for a seed must reuse the same paired validation batches.
    set_seed(assay_seed + seed)
    baseline=evaluate_model_on_batches(model, cached)
    target=summarize_lesion_effects(baseline, run_node_lesion(model,candidate,val_generator,batches,config,cached_batches=cached))["LS_ctrl"]
    snap=sorted((run_dir/"snapshots").glob("step_*"))[-1]
    edge=torch.load(snap/"edge_index.pt",map_location="cpu"); coords=torch.load(snap/"coords.pt",map_location="cpu").numpy(); types=torch.load(snap/"node_types.pt",map_location="cpu").numpy()
    degree=_undirected_degree(edge, config["num_nodes"])
    pool=np.array([n for n in range(config["num_nodes"]) if n not in INTERFACE],dtype=int)
    stats=compute_candidate_stats(candidate,types,degree,coords)
    rng=np.random.default_rng(assay_seed)
    # Keep sampler statistics in the original node-ID coordinate system.
    samplers=build_null_samplers(len(candidate),stats["degree_mean"],stats["surface_fraction"],stats["mean_pairwise_distance"],types,degree,coords,rng,eligible_ids=pool)
    rows=[]
    selected_types = null_types or ["same_size","degree_matched","surface_fraction_matched","distance_matched"]
    for nt in selected_types:
        for i in range(n_per_type):
            local=[int(x) for x in samplers[nt]()]
            if len(local) != len(candidate) or set(local) & INTERFACE:
                raise AssertionError(f"invalid null region seed={seed} type={nt}: {local}")
            value=summarize_lesion_effects(baseline,run_node_lesion(model,local,val_generator,batches,config,cached_batches=cached))["LS_ctrl"]
            rows.append({"seed":seed,"null_type":nt,"region_index":start_index+i,"region_nodes":";".join(map(str,local)),"target_rmt":float(target),"rmt_value":float(value),"candidate_label":"blind_nonlesion_control","assay_seed":assay_seed})
    return rows

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--run-dir",required=True); ap.add_argument("--seed",type=int,required=True); ap.add_argument("--output",required=True); ap.add_argument("--n-per-type",type=int,default=1000); ap.add_argument("--validation-batches",type=int,default=2); ap.add_argument("--assay-seed",type=int,default=20260723); ap.add_argument("--null-types",default="",help="comma-separated null types"); ap.add_argument("--start-index",type=int,default=0); a=ap.parse_args(); types=[x for x in a.null_types.split(",") if x] or None; rows=run_one(Path.cwd(),a.run_dir,a.seed,a.n_per_type,a.validation_batches,a.assay_seed,types,a.start_index); write_rows(Path(a.output),rows); print(a.output,len(rows))
if __name__ == "__main__": main()
