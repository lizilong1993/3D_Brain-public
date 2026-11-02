# srnn-cra

Code accompanying *When Localized Causal Effects Do Not Imply Functional Regions in Spatial Recurrent Neural Networks*.

## 1. Why this repository exists

The paper asks when a localized causal effect in a spatial recurrent neural network is evidence for a functional region. It separates three claims: an intervention changes behavior at a particular place and time (event-local sensitivity), the affected node set is stronger than matched random regions (region-specific support), and the same region carries the function across seeds and configurations (stable regionalization). The audit distinguishes these claims with interface exclusion, temporal controls, matched nulls, and replication.

This repository is the code layer of that audit. It contains the implementation used to train the 128-node models, detect candidate regions, run the two intervention operators (a recurrent lesion and an event-local state replacement), sample matched null regions, and compute the reported RRP and Monte Carlo tail probabilities. It is a code release: pretrained checkpoints and archived run outputs are not included.

## 2. What is in the repository

```
.
├── configs/                     model and task configurations
│   ├── default.yaml             base defaults (128 nodes, 64-step sequences, 30k steps)
│   ├── formal_main_B.yaml       main model on world B with the control loss enabled
│   ├── smoke_main_B.yaml        40-step CPU configuration used by the smoke test
│   └── world_B.yaml             world-B task and loss overrides
├── src/                         model, training, analysis, and assay code
├── scripts/                     command-line entry points for the audits
├── tests/
│   └── smoke_test.py            CPU end-to-end check of the code path
├── requirements.txt             Python dependencies
└── .gitignore                   excludes caches, checkpoints, run directories, logs
```

### `src/`

| File | Purpose |
|---|---|
| `model_main.py` | Boundary/interior sparse recurrent network and the localized sparse-adapter variant. |
| `model_registry.py` | Builds a model from the configured `model_type`. |
| `dense_rnn.py` | Dense recurrent sequence baseline. |
| `baseline_3d_sparse.py` | Sparse 3D baseline sharing the main recurrent architecture. |
| `baseline_moe.py` | Mixture-of-experts sequence baseline. |
| `data_generator.py` | Procedural world-A/world-B sequence tasks (no external dataset). |
| `graph_init.py` | Initial spatial graph construction. |
| `graph_metrics.py` | Directed graph metrics used by region analysis. |
| `rewiring.py` | Graph rewiring during training. |
| `losses.py` | Task losses and regularizers. |
| `train.py` | Training loop and CLI; writes checkpoints, snapshots, logs, resolved config, and manifest. |
| `evaluate.py` | Evaluation of prediction, memory, and control targets. |
| `analysis_regions.py` | Graph thresholding, community detection, region statistics, and region-map comparison. |
| `analyze_run.py` | CLI that analyzes a completed run and writes `analysis/results.json`, `analysis/regions.json`, and plots. |
| `blind_region_predictor.py` | Blind structure-only region prediction and leave-one-run-out evaluation. |
| `predict_task_regions.py` | Fixed structure-only weighted scoring used to select control, memory, and prediction regions. |
| `interface_exclusion.py` | Registry of input, readout, and adapter interfaces excluded from candidates and null pools. |
| `lesion.py` | Recurrent node-lesion assays. |
| `null_model_samplers.py` | Same-size, degree-matched, surface-fraction-matched, and distance-matched null-region samplers. |
| `statistics.py` | RRP, Monte Carlo tail probability, and confidence-interval helpers. |
| `task_ecologies.py` | Configuration resolution and world/task defaults. |
| `task_space_ecologies.py` | Task-space world definitions. |
| `utils.py` | Config, I/O, seed, and device utilities. |
| `visualize.py` | Graph and lesion visualizations. |

### `scripts/`

| File | Purpose |
|---|---|
| `event_local_rmt_batch.py` | Event-window state-replacement assay. Replaces hidden states inside the recurrence, computes RMT against a non-overlapping temporal control, excludes interface nodes 0–23, and marks empty corrected candidates as `NE`. |
| `summarize_event_local.py` | Computes tie-inclusive RRP and `p_MC` for the four null families from the event-local JSON outputs and reports each seed's all-four status. |
| `run_matched_null_assay.py` | Recurrent-lesion assay on a candidate and its matched null regions. Excludes control-readout nodes 16–23 and writes one CSV row per null region. |
| `summarize_matched_null.py` | Computes RRP and `p_MC` from the lesion-assay CSV. Expects 10 seeds × 4 null types × 1000 draws. |
| `predictor_loso.py` | Leave-one-seed-out versus leave-one-run-out audit of the 21-feature structure-only classifier; writes fold metrics and an optional split table. |
| `report_match_quality.py` | Quantifies the residual degree, surface-fraction, and mean-distance mismatch for the approximately matched null families. |
| `build_confirmatory_master.py` | Combines per-seed analysis results and the matched-null summary into the confirmatory table. Expects `<analysis-root>/seed_<42..51>/main_B/analysis/`. |

## 3. How to run

### Requirements

- Python 3.10
- PyTorch 2.x (CPU is sufficient for the smoke test)
- The remaining dependencies are listed in `requirements.txt`.

### Install

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Smoke test (CPU, about one minute)

```bash
python tests/smoke_test.py
```

The test loads a config, builds the model and data, evaluates one batch, runs a two-node lesion, samples matched null regions, and exercises the summary script. It prints `SMOKE TEST PASSED`.

### Train and analyze a model

```bash
python -m src.train \
  --config configs/formal_main_B.yaml \
  --run-dir runs/seed_42 \
  --seed 42

python -m src.analyze_run \
  --config configs/formal_main_B.yaml \
  --run-dir runs/seed_42
```

Training writes `resolved_config.yaml`, `run_manifest.json`, `train_log.csv`, `checkpoints/`, and `snapshots/`. Analysis adds `analysis/results.json` and `analysis/regions.json`.

### Event-local RMT assay

A run directory must contain:

```
<run_dir>/
  resolved_config.yaml
  checkpoints/step_*.pt
  snapshots/step_*/{edge_index,coords,node_types}.pt
  analysis/results.json
  analysis/regions.json
```

Run the assay and summarize it:

```bash
python scripts/event_local_rmt_batch.py \
  --run-dir runs/seed_42 --seed 42 \
  --draws 1000 --validation-batches 8 \
  --iface-start 0 --iface-end 24 \
  --out out/seed_42_d42

python scripts/summarize_event_local.py --root out --out out/summary
```

The assay uses CUDA by default; add `--device cpu` for a CPU run. The default is 8 validation batches. Each JSON records the raw pre-interface candidate, the corrected candidate, raw IAI, candidate sizes, per-batch baseline cross-entropy, and the event/control windows. Each output directory must follow `out/seed_<model>_d<data>/` for the summary script.

### Recurrent-lesion matched-null assay

```bash
python scripts/run_matched_null_assay.py \
  --run-dir runs/seed_42 --seed 42 \
  --output nulls.csv --n-per-type 1000

python scripts/summarize_matched_null.py \
  --input nulls.csv --output summary.json
```

`summarize_matched_null.py` expects a single CSV containing 10 seeds × 4 null types × 1000 draws. To audit fewer seeds, concatenate their rows before summarizing.

### Build the confirmatory table

```bash
python scripts/build_confirmatory_master.py \
  --analysis-root <analysis-root> \
  --null-summary summary.json \
  --output confirmatory_table.csv
```

`<analysis-root>` must contain `seed_42/main_B/analysis/results.json` through `seed_51/main_B/analysis/results.json`.

## 4. How to obtain and cite this repository

The repository is available at:

```
https://github.com/lizilong1993/srnn-cra
```

### Zenodo archive

This release is archived on Zenodo: [10.5281/zenodo.23876014](https://doi.org/10.5281/zenodo.23876014). The reproducibility package (assay outputs, configs, split table) is archived as [10.5281/zenodo.23879010](https://doi.org/10.5281/zenodo.23879010). Deposition uses a personal access token supplied through the `ZENODO_TOKEN` environment variable. The token is not stored in this repository or in any file tracked by it.

### Citation

Paper:

```bibtex
@article{wu2026localized,
  author  = {Wu, Yang and Chen, Huanhuan and Song, Wenping and Wang, Xinsheng and Li, Zilong},
  title   = {When Localized Causal Effects Do Not Imply Functional Regions in Spatial Recurrent Neural Networks},
  journal = {Scientific Reports},
  year    = {2026},
  note    = {Manuscript submitted for publication}
}
```

Code:

```bibtex
@misc{srnncra2026,
  author       = {Wu, Yang and Chen, Huanhuan and Song, Wenping and Wang, Xinsheng and Li, Zilong},
  title        = {Code for "When Localized Causal Effects Do Not Imply Functional Regions in Spatial Recurrent Neural Networks"},
  year         = {2026},
  howpublished = {\url{https://github.com/lizilong1993/srnn-cra}}
}
```

### License

This release is distributed under the Creative Commons Attribution 4.0 International license (CC BY 4.0), as recorded in the Zenodo metadata.
