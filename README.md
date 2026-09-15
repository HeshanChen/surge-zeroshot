# Storm surge forecasts from a global tide-gauge network

Code, frozen splits, model checkpoints and evaluation artifacts for the manuscript *Storm surge forecasts from a global
tide-gauge network* (Heshan Chen and Andrew Kruczkiewicz, 2026). The July 2026 preprint of the same work carried the title
*Zero-shot probabilistic storm-surge forecasting from gauged to ungauged coasts*; its artifacts are kept here as the record
of that version.

Every number in the manuscript is asserted from the committed artifacts by two audit gates:

```bash
python scripts/verify_numbers_cont.py   # manuscript: continuous evaluation windows (the primary set), 168 checks
python scripts/verify_numbers.py        # July 2026 preprint: all evaluation windows, 69 checks
```

`NUMBERS.md` maps every manuscript claim to its artifact and to the script that produced it.

## What the manuscript uses and where it lives

| Manuscript element | Artifacts and models |
|---|---|
| Deployment: 84 pre-registered marine test gauges, 760-gauge model (Table 1, Fig. 2, Extended Data Table 5) | `outputs/eval_full_lstmq_v2final_cont.{csv,log}` (all windows: `eval_full_lstmq_v2final.*`), `outputs/perlead_full_cont.csv`, `models/deploy_760_best.pt` |
| Four leave-region-out rotations (Fig. 3, Extended Data Table 2) | `outputs/eval_full_lstmq_v2rot{jp,eu,na,ocr2}_cont.*`, `models/rot_*_best.pt`; composition of the test sets (lake, river and inner-estuary gauges) in `catalog/rotation_test_domain.csv` |
| Europe intervention, fixed-count composition ablation, matched-update control (Results 2 and 3, Methods) | `outputs/eval_full_lstmq_v2eu299_cont.*`, `v2c298strat_cont.*`, `v2c298usjp_cont.*`, `v2g64ms_cont.*`, `v2g256ms_cont.*`; checkpoints under `models/`; training logs `outputs/train_lstmq_v2{eu299,c298strat,c298usjp,g64ms,g256ms}.log` |
| Scaling ladder, 64 to 760 gauges (Fig. 4, Extended Data Table 3) | `outputs/eval_full_lstmq_v2g{64,128,256,384,512,512r2,640}_cont.*` |
| Gauge-free ladder and hydrodynamic head-to-head (Table 2, Fig. 5) | `outputs/eval_forcing_only_*_cont.csv`, `outputs/eval_gtsm_symmetric_cont.{csv,log}`, `outputs/gtsm_mapping.csv`, `models/gaugefree_760_best.pt` |
| Envelopes and pinball scores (Results 6) | `outputs/eval_full_lstmq_v2final_cont.log`, `outputs/eval_pinball_cont.{csv,log}` |
| Real-forecast checks (Extended Data Table 1 and Fig. 3, Supplementary Table 2) | `outputs/eval_gefs_season*.{csv,log}`, `outputs/gefs_season/*.npz` (sampled GEFSv12 reforecast forcing, one file per 00 UTC cycle of 2012 and 2017), `outputs/eval_gefs_v3.{csv,log}` |
| Architecture factor study (Extended Data Table 4) | `outputs/eval_full_lstmq_v2n298_cont.*`, `outputs/eval_full_v7e_n298v_cont.*`, `outputs/eval_chronos_bolt-small_masked_cont.*`, `outputs/train_v7e_*.log` |
| Block-bootstrap intervals (Methods) | `outputs/bootstrap_cis_cont.txt` |
| Dataset chain, effective fold sizes, join share, seismic mask (Methods, Supplementary Table 3 and Fig. 3) | `outputs/audit_universe_loadability.csv`, `outputs/join_share_audit.csv`, `outputs/audit_window_count.csv`, `outputs/seismic_exceedance_audit.csv`, `outputs/tsunami_mask_annex.{csv,tex}`, `outputs/figS3_tsunami_mask.pdf` |

Training logs exist for the seven runs made after run logging was added (September 2026); the earlier runs are
represented by their checkpoints and evaluation logs. Per-window predictions of the hydrodynamic comparison (160 MB) are
regenerable with `scripts/experiments/eval_gtsm_symmetric.py` and are not committed.

## Repository map

```
src/               model (encoder-decoder LSTM, 6.35M parameters), attention model of the factor study, data loading, surge extraction
scripts/
  pipeline/        GESLA-3 download, point forcing via Open-Meteo, segmented detide and QC, static attributes, split construction
  train.py         training (validation-fold checkpoint selection; --forcing_only for the gauge-free variant)
  eval_full.py     canonical evaluation ([A] per-lead, [B] event-conditioned, [C] peak buckets, [D] quantiles, [W] window continuity)
  experiments/     gauge-free ladder, hydrodynamic head-to-head, GEFSv12 forcing checks, trailing-window causality check,
                   pinball scores, ENSO stratification, Chronos baseline, attention model training
  figures/         every figure of the manuscript, including the seismic-mask annex (tsunami_mask_annex.py)
  verify_numbers_cont.py, verify_numbers.py   audit gates
  bootstrap_cis.py, compare_cont.py, audit2_numbers.py, summarize_eval_logs.py   derived tables and intervals
catalog/           frozen splits (deployment, four rotations, ladder, ablations), station attributes, QC table,
                   cleaned station list, M7+ earthquake catalogue, rotation test-set domain classification
models/            fifteen checkpoints of record
outputs/           evaluation artifacts of record (per-gauge tables and full logs), provenance tables, figures
docs/              how the splits were built (SPLITS.md), experiment ledger, protocol and evaluation audits
```

## Running the code

Every script resolves the repository root from its own location (or from `SURGE_ROOT` if set), so the scripts run from
any working directory; checkpoints are referenced by their names under `models/`. Raw data (`data/`) is not included;
steps 1 to 3 rebuild it from public sources (hours of download plus CPU time).

1. `python scripts/pipeline/download_gesla.py` (GESLA-3 water levels, public S3 bucket)
2. `python scripts/pipeline/fetch_era5.py` (point forcing via the Open-Meteo historical archive, open, no registration)
3. `python scripts/pipeline/batch_detide.py` (segmented harmonic detide, 30-day high-pass, QC)
4. `python scripts/train.py --split catalog/exp_split_final.csv` (12 epochs; checkpoint selected by zero-shot skill on the 40-gauge validation fold, never the test fold)
5. `python scripts/eval_full.py --model lstmq --ckpt models/deploy_760_best.pt --split exp_split_final.csv --tag myrun --continuous_only`
   (output format matches `outputs/eval_full_lstmq_v2final_cont.{csv,log}`)

Steps 4 and 5 reproduce the deployment result from the frozen splits. `scripts/compare_cont.py` and
`scripts/audit2_numbers.py` regenerate `outputs/compare_cont.txt` and `outputs/audit2_numbers.txt` from the committed
evaluation tables; `extract_eot20_statics.py` and `fit_sigma_hat.py` regenerate their artifacts byte-identically.

## Data sources (all open)

| Source | Use | Access |
|---|---|---|
| GESLA-3 tide gauges | water levels, surge targets | public |
| Open-Meteo historical archive (ERA5, ERA5-Land, IFS) | point atmospheric forcing | open, no registration |
| EOT20 (SEANOE) | gauge-free tidal statics | open, no registration |
| USGS M7+ catalogue | seismic (tsunami and seiche) masking | open |
| GEFSv12 reforecast (AWS) | real-forecast forcing checks | open |
| GTSM v3 surge reanalysis (Copernicus CDS) | hydrodynamic baseline, comparison only | registration (the only registered product; not part of the forecasting pipeline) |

## License

Code and artifacts: MIT. The underlying third-party datasets retain their own licences and terms.
