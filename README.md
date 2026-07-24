# Zero-shot probabilistic storm-surge forecasting: global observational benchmark

Code, frozen splits, model checkpoints, and evaluation artifacts for the paper
*"Zero-shot probabilistic storm-surge forecasting from gauged to ungauged coasts"* (Chen, 2026).

Every headline number in the paper is asserted directly from the committed evaluation
artifacts by the audit gate:

```bash
python scripts/verify_numbers.py        # 69 checks; exit 0 = every number reproduces
```

See **NUMBERS.md** for the claim-by-claim provenance table (paper number → artifact → producing script).

## Repository map

```
src/               model (encoder–decoder LSTM, 6.35M) + data loading + surge extraction
scripts/
  pipeline/        GESLA-3 download → ERA5 point forcing → segmented detide → QC
  train.py         training (validation-fold checkpoint selection; --forcing_only for gauge-free)
  eval_full.py     canonical evaluation ([A] per-lead, [B] event-conditioned, [C] peak buckets, [D] quantiles)
  experiments/     gauge-free ladder (σ̂ regression, EOT20 statics), GTSM head-to-head,
                   GEFS real-forecast forcing, trailing-window causality check, pinball scores, ENSO stratification
  figures/         all paper figures
  verify_numbers.py  audit gate
catalog/           FROZEN pre-registered splits (deployment + 4 rotations + 7-rung ladder + ablations),
                   station attributes, QC table, M7+ earthquake catalog (tsunami masking)
models/            checkpoints of record (deployment, gauge-free, 4 rotation models)
outputs/           evaluation artifacts of record (per-gauge CSVs + full eval logs) + paper figures
docs/SPLITS.md     how the frozen splits were constructed (buffers, dedup, stratification)
```

## Reproducing the pipeline (from raw data)

1. `python scripts/pipeline/download_gesla.py` — GESLA-3 water levels (public S3).
2. `python scripts/pipeline/fetch_era5.py` — ERA5 point forcing via Open-Meteo (open, no registration).
3. `python scripts/pipeline/batch_detide.py` — segmented harmonic detide + 30-day high-pass + QC
   (validated against NOAA verified water levels; see paper §2.1).
4. `python scripts/train.py --split catalog/exp_split_final.csv` — trains the deployment model
   (12 epochs; checkpoint selected by zero-shot skill on the 40-gauge validation fold, never the test fold).
5. `python scripts/eval_full.py --ckpt models/deploy_760_best.pt --tag myrun` — full evaluation;
   output format matches `outputs/eval_full_lstmq_v2final.{csv,log}`.

Steps 1–3 rebuild the corpus from public sources (hours of download + CPU). Steps 4–5 reproduce
the deployment result from the frozen splits. All experiment scripts under `scripts/experiments/`
are runnable the same way; each writes artifacts in the exact format committed under `outputs/`.

Three helper scripts were certified by regeneration: `extract_eot20_statics.py` and
`fit_sigma_hat.py` (with and without `--eot`) regenerate their committed artifacts **byte-identically**
(deterministic; GBR seeded with `random_state=0`).

## Data sources (all open)

| Source | Use | Access |
|---|---|---|
| GESLA-3 tide gauges | water levels / surge targets | public |
| ERA5 via Open-Meteo | point atmospheric forcing | open, no registration |
| EOT20 (SEANOE) | gauge-free tidal statics | open, no registration |
| USGS M7+ catalog | tsunami/seiche masking | open |
| GEFSv12 reforecast (AWS) | real-forecast forcing spot check | open |
| GTSM v3 surge reanalysis (CDS) | physics baseline, comparison only | registration (the only registered product; not part of the forecasting pipeline) |

## License

Code and artifacts: MIT. Underlying third-party datasets retain their own licenses/terms.
