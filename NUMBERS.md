# Provenance: every paper number → artifact → producing script

Two gates. `scripts/verify_numbers_cont.py` checks the September 2026 manuscript (continuous evaluation windows,
the primary set); `scripts/verify_numbers.py` checks the July 2026 preprint (all windows; 69 assertions). Run both
before trusting anything else. "Artifact" paths are committed in this repository. Producing script = what
generated the artifact.

## September 2026 manuscript (continuous windows; `*_cont` artifacts)

| Manuscript claim | Value | Artifact | Producing script |
|---|---|---|---|
| Deployment pooled / @8h / NNSE / wins (84 marine) | +37% / +30% / 0.782 / 81/84 (all windows: +39 / +31 / 0.786 / 82/84) | `outputs/eval_full_lstmq_v2final_cont.csv` | `scripts/eval_full.py --continuous_only` + `models/deploy_760_best.pt` |
| Storm per-lead 4–12 h / high-water / capture / tail / event-level peak coverage | +32..40% / +45..48% / 0.59 / 0.75 / 41% | `outputs/eval_full_lstmq_v2final_cont.log` | same |
| Per-lead 1–48 h (+11%@1h → +40%@48h, all positive) | 48/48 | `outputs/perlead_full_cont.csv` | `scripts/figures/plot_perlead_full.py --continuous_only` |
| Rotations Japan / Europe / N. America / Oceania (median run) | +30 / +11 / +10 / +8% pooled; union 633 evals, 477 wins | `outputs/eval_full_lstmq_v2rot{jp,eu,na,ocr2}_cont.csv` | `scripts/eval_full.py --continuous_only` + rotation ckpts |
| Oceania three runs | +5 / +8 / +10% (all windows +8.4 / +11.0 / +13.6) | `outputs/eval_full_lstmq_v2rotoc{,r2,r3}{,_cont}.csv` | same |
| Europe intervention 299 → 593 Europe-free gauges (231 shared) | 0 → +11% pooled; 593 lower at 186/231 | `outputs/eval_full_lstmq_v2eu299_cont.csv`, `v2roteu_cont.csv` | same + `models/europe_intervention_299_best.pt` |
| Composition ablation, 298 gauges, 49 countries vs USA+Japan | +24% vs −2% at 8 h; 67/84 | `outputs/eval_full_lstmq_v2c298{strat,usjp}_cont.csv` | same + `models/composition_298_*` |
| Ladder 64…760 pooled | +19 / 24 / 28 / 31 / 35 / 36 / 37%; 5.2 points per doubling | `outputs/eval_full_lstmq_v2g*_cont.csv`, `v2final_cont.csv` | same |
| Matched-update control (77,028 steps) | 64: +19→+20; 256: +28→+27 | `outputs/eval_full_lstmq_v2g{64,256}ms_cont.csv` | same + `models/ladder_*_matched_steps_best.pt` |
| Replicate difference (512) | 0.8 pooled / 0.8 @8h / 0.004 NNSE / 0.01 capture | `outputs/eval_full_lstmq_v2g512{,r2}_cont.csv` | same |
| GTSM head-to-head, two tiers (80 gauges; 61 with ≥5 storm windows) | tier A 6.74 vs 6.21 cm (n.s.), storms 11.79 vs 8.75; tier B 4.17 vs 5.47 (66/80), storms 7.88 vs 7.70 (n.s.), peak 0.62 vs 0.66 (n.s.) | `outputs/eval_gtsm_symmetric_cont.{csv,log}` | `scripts/experiments/eval_gtsm_symmetric.py --continuous_only` |
| Block-bootstrap intervals (all Table 2b cells, rotations, interventions) | see file | `outputs/bootstrap_cis_cont.txt` | `scripts/bootstrap_cis.py _cont` |
| Gauge-free ladder | 0.593 → 0.598 → 0.618 → 0.782; zero-water-level ablation 0.522 | `outputs/eval_forcing_only_v2fonly{_eot2,_predscale,}_cont.csv`, `eval_forcing_only_v2final_cont.csv` | `scripts/experiments/eval_forcing_only.py --continuous_only` |
| Factor study (298 gauges, Japan held out) | LSTM +15/+14/+31/0.58; attention v7e +7/0/+11/0.38; Chronos −1/+5/+3/0.32; LSTM lower @8h at 71/77 | `outputs/eval_full_lstmq_v2n298_cont.*`, `eval_full_v7e_n298v_cont.*`, `eval_chronos_bolt-small_masked_cont.*` | `eval_full.py`, `baseline_chronos.py --continuous_only` |
| Pinball proper score | q90 +44.7% (82/84), q99 +47.2% (77/84) | `outputs/eval_pinball_cont.{csv,log}` | `scripts/experiments/eval_pinball.py --continuous_only` |
| GEFS real-forecast check, nine cases with peaks inside the window | 37.1 → 48.2 cm (pers 73.8); 8/9 beat persistence; capture 0.36; q99 above peak 2/9 | `outputs/eval_gefs_v2.{csv,log}`, `gfs_inits_inwindow.csv` | `scripts/experiments/eval_gefs.py --inits outputs/gfs_inits_inwindow.csv` |
| Table 1 geographic grouping, rotation union, scaling fit | see file | `outputs/audit2_numbers.txt` | `scripts/audit2_numbers.py` |
| Every log quantity, all vs continuous | see file | `outputs/eval_log_summary.txt`, `outputs/compare_cont.txt` | `scripts/summarize_eval_logs.py`, `scripts/compare_cont.py` |

## July 2026 preprint (all windows)

| Paper claim | Value | Artifact | Producing script |
|---|---|---|---|
| Deployment pooled / @8h / NNSE / wins (84 marine) | +39% / +31% / 0.786 / 82/84 | `outputs/eval_full_lstmq_v2final.csv` | `scripts/eval_full.py` + `models/deploy_760_best.pt` |
| Deployment storm per-lead / high-water / capture / tail | +29..36% / +43..48% / 0.60 / 0.78 | `outputs/eval_full_lstmq_v2final.log` ([B],[C],[D]) | same |
| Per-lead 1–48 h all positive (+11%@1h → ~+42%@48h) | 48/48 | `outputs/perlead_full.csv` | `scripts/figures/plot_perlead_full.py` |
| Ladder 64/128/256/384/512/640/760 | +21/26/30/34/37/38/39% | `outputs/eval_full_lstmq_v2g*.csv` | `scripts/eval_full.py` per rung |
| Seed-replicate bound (512) | ±0.6 pooled / ±0.2 @8h | `outputs/eval_full_lstmq_v2g512{,r2}.csv` | same |
| Rotation Japan / Europe / N.America | +35/+26, +13/+12, +12/+10 | `outputs/eval_full_lstmq_v2rot{jp,eu,na}.csv` | `scripts/eval_full.py` + rotation ckpts |
| Rotation Oceania = median of 3 runs | +11/+10 (runs +8.4/+11.0/+13.6) | `outputs/eval_full_lstmq_v2rotoc{,r2,r3}.csv` | same |
| Rotation union | 635 evals, 510 wins, +14/+12% | union of the four rotation CSVs | `scripts/verify_numbers.py` §4 |
| Storm ranges per rotation | jp +31..37, eu +8..13, na +16..19, oc −1..+11 | rotation `.log` files, block [B] | `scripts/eval_full.py` |
| Gauge-free ladder | 0.606 → 0.611 → 0.631 (→ 0.786) | `outputs/eval_forcing_only_v2fonly{_eot2,_predscale,}.csv` | `scripts/experiments/eval_forcing_only.py` + `models/gaugefree_760_best.pt` |
| Gauge-free storm: capture 0.50, q99@peak 0.98, cov99 0.983 | | `outputs/eval_forcing_only_v2fonly_eot2.csv` | same |
| σ̂ regression (log-corr 0.85, median err 14%) | | `outputs/sigma_hat_test.csv` | `scripts/experiments/fit_sigma_hat.py` (regen-certified) |
| EOT20 statics (range log-corr 0.933) | | `outputs/eot20_statics_test.csv` | `scripts/experiments/extract_eot20_statics.py` (regen-certified) |
| GTSM head-to-head @8h (4.4 vs 8.3 vs 6.4 cm; 79/80) | | `outputs/eval_gtsm.{csv,log}` + `outputs/gtsm_mapping.csv` | `scripts/experiments/eval_gtsm.py` |
| GTSM honest exception (p99.9@48h: 13.2 vs 12.5; pers 27.6) | | `outputs/eval_gtsm.log` | same |
| GEFS real-forcing (21.5→38.2 cm, cap 0.71→0.34, 5/6) | | `outputs/eval_gefs.csv` + `outputs/gfs_events.csv` | `scripts/experiments/eval_gefs.py` |
| (correction, September 2026) the preprint's GEFS run de-normalized the already-raw anchor value twice, inflating the degradation; the artifact above is kept unchanged as the record of the preprint, and the corrected check with every peak inside the horizon is `outputs/eval_gefs_v2.{csv,log}` (see the September table) | | | `scripts/experiments/eval_gefs.py --inits outputs/gfs_inits_inwindow.csv` |
| Trailing causality (pooled +36, NNSE 0.818, storm@8h +40, 80/84) | | `outputs/eval_trailing_{centered,trailing}.csv`, `.log` | `scripts/experiments/{exp_trailing_baseline,eval_trailing}.py` |
| Pinball proper score (q90 +45.4% 82/84; q99 +46.2% 77/84) | | `outputs/eval_pinball.csv` | `scripts/experiments/eval_pinball.py` |
| ENSO/season stratification (phase-flat; SI) | | `outputs/eval_enso_{ocrot,deploy}.csv`, `.log` | `scripts/experiments/eval_enso.py` |
| Lake ablations nil (706-marine; matched-640) | ≤0.7 pt | `outputs/eval_full_lstmq_v2marine.csv`, `v2g640{,marine}.csv` | `scripts/eval_full.py` |
| Composition ablation (298 less-diverse pool) | @8h +10% vs +18% | `outputs/eval_full_lstmq_v2n298.csv` | same |
| Out-of-domain set | 15 evaluated, 14 beat persistence | `outputs/eval_full_lstmq_v2final_xdom.csv` | same |
| Tsunami-mask impact (up to 45% of p99.9 hours) | | `outputs/seismic_exceedance_audit.csv` | pipeline QC |

Notes.
- The **factor study** (Table: recurrence vs attention vs self-supervision vs Chronos) predates the final
  protocol era; its rows come from the masked re-evaluation logs of the 298-gauge era and are reported in the
  paper with that framing (caption discloses the era and the clean-protocol re-anchor +20/+18). Those legacy
  artifacts are retained in the working-history archive rather than this release.
- p-values are capped at 1e-15 in the paper (spatial correlation between gauges; see paper §4).
