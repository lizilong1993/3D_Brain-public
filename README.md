# 3D Brain: lesion-based causal audit for spatial recurrent networks

Minimal, self-contained code release for training and auditing 128-node spatial
recurrent networks. The package supports:

- training a boundary/interior 3D sparse recurrent network on procedurally
  generated sequence tasks (world A/B);
- region detection and structure-only region analysis;
- recurrent node-lesion assays;
- matched-null random-region audits with same-size, degree-matched,
  surface-fraction-matched and distance-matched nulls;
- empirical RRP and Monte Carlo tail probability (`p_MC`) with the
  tie-inclusive rule `p_MC = (1 + n_>=) / (M + 1)`.

> Scope note: this minimal release implements two operators: the **event-local
> state-replacement operator** (primary; hidden states inside the event window
> are replaced with states from a non-overlapping temporal-control window) and
> the **recurrent lesion operator** (`LS_ctrl`; permanent node mask, auxiliary).
> It does not include pretrained checkpoints or manuscript results. It is a code
> release, not a data release.

## Requirements

- Python 3.10
- PyTorch 2.x (CPU is sufficient for the smoke test)
- See `requirements.txt` for the other dependencies.

Tested with Python 3.10, PyTorch 2.2.2, NumPy 1.26, SciPy 1.15,
scikit-learn 1.7, networkx 3.4.

## Install

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Minimal run (CPU, about a minute)

```bash
python tests/smoke_test.py
```

The test builds a 128-node model, evaluates one validation batch, runs a
two-node lesion, samples matched null regions, and exercises the summary
script on synthetic matched-null data. It prints `SMOKE TEST PASSED`.

## Running a matched-null audit

A run directory must contain:

```
<run_dir>/
  resolved_config.yaml
  checkpoints/step_*.pt
  snapshots/step_*/{edge_index,coords,node_types}.pt
  analysis/results.json
  analysis/regions.json
```

Then:

```bash
python scripts/run_matched_null_assay.py \
  --run-dir <run_dir> --seed 42 --output nulls.csv --n-per-type 1000

python scripts/summarize_matched_null.py \
  --input nulls.csv --output summary.json
```

`run_matched_null_assay.py` filters nodes 16-23 (the registered control-readout
interface) from the candidate and the null pool, uses the latest checkpoint in
`checkpoints/`, and writes one row per null region.

`summarize_matched_null.py` expects 10 seeds x 4 null types x 1000 draws and
writes per-seed/per-null counts (`n_less`, `n_equal`, `n_greater`), `RRP`, and
the tie-inclusive `p_MC`.

## Event-local RMT assay

`scripts/event_local_rmt_batch.py` implements the event-window state-replacement
operator. For each validation batch it chooses a three-timestep event window at
the centre of the control-mask segment and a non-overlapping temporal-control
window, replaces hidden states inside the recurrence, and computes

```
RMT = (event CE - baseline CE on event window)
    - (control CE - baseline CE on control window)
```

Example (strict interface 0-23):

```bash
python scripts/event_local_rmt_batch.py \
  --run-dir <run_dir> --seed 42 --draws 1000 --validation-batches 2 \
  --iface-start 0 --iface-end 24 --out out/seed42
```

If the corrected candidate is empty, the script writes a JSON record with
`status: "NE"`.

## Data

All observations are generated procedurally by `src/data_generator.py`; no
external dataset is required.

## Repository layout

```
configs/        default, world B, formal and smoke configs
src/            model, training, analysis, lesion, null samplers (24 modules)
scripts/        matched-null assay, summary, candidate-table builder
tests/          CPU smoke test
```

## License

No license file is included in this release. Contact the corresponding author
before reuse.
