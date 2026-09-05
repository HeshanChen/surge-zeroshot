# Learning transferable storm-surge forecasts from a global tide-gauge network

Code, frozen splits, model checkpoints, and evaluation artifacts for the manuscript
*"Learning transferable storm-surge forecasts from a global tide-gauge network"* (Chen and co-author, 2026;
earlier preprint title *"Zero-shot probabilistic storm-surge forecasting from gauged to ungauged coasts"*).

Every headline number in the manuscript is asserted directly from the committed evaluation
artifacts by two audit gates:

```bash
python scripts/verify_numbers_cont.py   # September 2026 revision: continuous-window numbers (primary)
python scripts/verify_numbers.py        # July 2026 preprint: all-window numbers (69 checks)
```

See **NUMBERS.md** for the claim-by-claim provenance table (paper number → artifact → producing script).

## September 2026 revision

The manuscript now reports every result on evaluation windows whose 256 rows are consecutive hours
(`--continuous_only` in `scripts/eval_full.py`, `eval_gtsm_symmetric.py`, `eval_forcing_only.py`,
`eval_pinball.py`, `baseline_chronos.py`); all-window results remain as the sensitivity set
(`compare_cont.py`). New experiments and artifacts since the preprint, all under the same clean protocol
(validation-fold checkpoint selection, seismic-masked records, 12 epochs):

- symmetric hydrodynamic head-to-head (`scripts/experiments/eval_gtsm_symmetric.py`: GTSM high-passed like the
  target, 25-h mean-error bias correction, two information tiers) and its map (`plot_gis_gtsm_sym.py`);
- spatial-block bootstrap intervals (`scripts/bootstrap_cis.py`) replacing per-gauge p-values;
- Europe corpus intervention (299 vs 593 Europe-free gauges, `catalog/exp_split_eu299v.csv`), fixed-count
  composition ablation (`exp_split_c298strat.csv`, `exp_split_c298usjp.csv`), matched-update control
  (64 and 256 gauges at the 760-gauge step budget);
- clean architecture factor study: recurrent model vs attention model with hour and channel forcing embeddings
  (`src/models/surge_jepa_v7e.py`, `scripts/experiments/train_v7e.py`) vs Chronos-bolt;
- GEFSv12 reforecast check re-issued so that every catalogued peak lies inside the 48-h horizon
  (`fetch_gefs_cycles.py`, `eval_gefs.py --inits outputs/gfs_inits_inwindow.csv`);
- event-level peak coverage, hourly coverage by lead, and window-continuity counters in `eval_full.py`.

Two additions on the same day: a season-scale real-forecast check (`scripts/experiments/fetch_gefs_season.py`,
`eval_gefs_season.py`; GEFSv12 reforecast control forcing at every 00 UTC cycle of 2012 and 2017, sampled at the 84 test
gauges and committed under `outputs/gefs_season/`, one npz per cycle, so the check reruns without touching AWS) and a
coverage indicator (`scripts/coverage_gap.py`; `catalog/surge_forecast_systems.csv` with one sourced evidence line per
country in `docs/surge_forecast_systems_sources.md`; World Bank inputs under `outputs/coverage_gap/`). Both are gated by
`scripts/verify_numbers_cont.py`.

`docs/experiment_ledger.md` lists every experiment on disk and where the manuscript uses it;
`docs/protocol_symmetry_audit.md`, `docs/gtsm_symmetry_audit.md`, and `docs/audit2_results.md` record the
audits that led to these changes. Per-window predictions of the GTSM comparison (160 MB) are regenerable with
`eval_gtsm_symmetric.py` and are not committed.

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
