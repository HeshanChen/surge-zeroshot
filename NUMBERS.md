# Provenance: every manuscript number, its artifact and the script that produced it

Two gates. `scripts/verify_numbers_cont.py` checks the manuscript (continuous evaluation windows, the primary set; 168
checks). `scripts/verify_numbers.py` checks the July 2026 preprint (all windows; 69 checks). Artifact paths are
committed in this repository. Uncertainty in the manuscript is given as spatial-block bootstrap intervals
(`outputs/bootstrap_cis_cont.txt`), never as p-values.

## Manuscript (continuous windows; `*_cont` artifacts)

| Manuscript claim | Value | Artifact | Producing script |
|---|---|---|---|
| Deployment pooled / 8-h / NNSE@8h / wins (84 marine gauges) | +37% / +30% / 0.782 / 81 of 84 (all windows +39 / +31 / 0.786 / 82 of 84) | `outputs/eval_full_lstmq_v2final_cont.csv` | `scripts/eval_full.py --continuous_only` + `models/deploy_760_best.pt` |
| Storm per-lead 4 to 12 h / high-water / peak capture / tail coverage / share of peaks under q99 | +32 to +40% / +45 to +48% / 0.59 / 0.75 / 41% | `outputs/eval_full_lstmq_v2final_cont.log` | same |
| Per-lead skill 1 to 48 h (+11% at 1 h to about +40% at 48 h, all positive) | 48 of 48 | `outputs/perlead_full_cont.csv` | `scripts/figures/plot_perlead_full.py --continuous_only` |
| Rotations Japan / Europe / North America / Oceania (median run) | +30 / +11 / +10 / +8% pooled; union 477 of 633 | `outputs/eval_full_lstmq_v2rot{jp,eu,na,ocr2}_cont.csv` | `scripts/eval_full.py --continuous_only` + rotation checkpoints |
| Oceania three runs | +5 / +8 / +10% pooled | `outputs/eval_full_lstmq_v2rotoc{,r2,r3}_cont.csv` | same |
| Rotation test sets: non-marine gauges; marine-only skill; union | North America 80 of 214 (Europe 7 of 231, Oceania 1 of 111); marine-only North America +16% pooled, +14% at 8 h, NNSE 0.713, 110 of 134; union 414 of 545 | `catalog/rotation_test_domain.csv` + rotation CSVs | `scripts/verify_numbers_cont.py` section 9 |
| Duplicate pairs inside rotation test sets (< 2 km) | 16 (Europe 7, North America 2, Oceania 7); union after dropping the higher-gap member 465 of 617 | rotation CSVs + `catalog/processing_qc.csv` | same |
| Europe intervention, 299 to 593 Europe-free gauges (231 shared test gauges) | 0 to +11% pooled; 593 lower at 186 of 231; high-water +8 to +21%; capture 0.29 to 0.38 | `outputs/eval_full_lstmq_v2eu299_cont.csv`, `v2roteu_cont.csv` | same + `models/europe_intervention_299_best.pt` |
| Composition ablation, 298 gauges, 49 countries vs USA and Japan | +29 vs +7% pooled; +24 vs -2% at 8 h; 67 of 84 | `outputs/eval_full_lstmq_v2c298{strat,usjp}_cont.csv` | same + `models/composition_298_*` |
| Ladder 64 to 760 pooled | +19 / 24 / 28 / 31 / 35 / 36 / 37%; 5.2 points per doubling | `outputs/eval_full_lstmq_v2g*_cont.csv`, `v2final_cont.csv` | same |
| Matched-update control | 64 gauges +19 to +20 pooled (+17 to +19 at 8 h); 256 gauges +28 to +27; budgets 77,077 and 77,922 steps | `outputs/eval_full_lstmq_v2g{64,256}ms_cont.csv`, `outputs/train_lstmq_v2g{64,256}ms.log` | same + `models/ladder_*_matched_steps_best.pt` |
| Replicate difference (512 gauges) | 0.8 pooled / 0.8 at 8 h / 0.004 NNSE / 0.01 capture | `outputs/eval_full_lstmq_v2g512{,r2}_cont.csv` | same |
| Hydrodynamic head-to-head, two tiers (80 gauges; 61 with at least five storm windows) | tier A 6.7 vs 6.2 cm (38 of 80), storms 11.8 vs 8.8; tier B 4.2 vs 5.5 (66 of 80), storms 7.9 vs 7.7, peak 0.62 vs 0.66; NNSE 0.60 / 0.62 and 0.80 / 0.67 / 0.67 | `outputs/eval_gtsm_symmetric_cont.{csv,log}` | `scripts/experiments/eval_gtsm_symmetric.py --continuous_only` |
| Block-bootstrap intervals (deployment, rotations, Table 2b, interventions) | see file | `outputs/bootstrap_cis_cont.txt` | `scripts/bootstrap_cis.py _cont` |
| Gauge-free ladder | 0.593 to 0.598 to 0.618 to 0.782; water level zeroed at inference 0.522 | `outputs/eval_forcing_only_v2fonly{_eot2,_predscale,}_cont.csv`, `eval_forcing_only_v2final_cont.csv` | `scripts/experiments/eval_forcing_only.py --continuous_only` |
| Factor study (298 gauges, Japan withheld) | recurrent +15 / +14 / +31 / 0.58; attention +7 / 0 / +11 / 0.38; Chronos -1 / +5 / +3 / 0.32; recurrent lower at 8 h at 71 of 77 | `outputs/eval_full_lstmq_v2n298_cont.*`, `eval_full_v7e_n298v_cont.*`, `eval_chronos_bolt-small_masked_cont.*` | `eval_full.py`, `baseline_chronos.py --continuous_only` |
| Pinball score against climatological quantiles | q90 +45% (82 of 84), q99 +47% (77 of 84) | `outputs/eval_pinball_cont.{csv,log}` | `scripts/experiments/eval_pinball.py --continuous_only` |
| Season-scale real-forecast check (2012 and 2017, 79 gauges, 35,228 windows) | pooled +37.3% reanalysis / +30.1% GEFS; 8 h +28.2 / +24.2; 48 h +41.4 / +32.9; 72 of 79 beat persistence; storm RMSE 13.4 / 15.4 / 24.7 cm; capture 0.65 / 0.69 | `outputs/eval_gefs_season{,_2017,_2012}.{csv,log}`, `_perlead.csv`, `outputs/gefs_season/*.npz` | `scripts/experiments/eval_gefs_season.py 2017 2012` |
| Nine storm cases, real forcing | window RMSE 37.1 to 49.3 cm (persistence 73.8); capture 0.47 to 0.36; 8 of 9 beat persistence; q99 above the peak in 2 of 9 | `outputs/eval_gefs_v3.{csv,log}`, `gfs_inits_inwindow.csv` (an earlier run with a precipitation-bucket fault is kept as `eval_gefs_v2.*`) | `scripts/experiments/eval_gefs.py --inits outputs/gfs_inits_inwindow.csv --tag _v3` |
| Dataset chain | 1,054 joined gauges = 100 test + 127 inside 0.5-degree buffers + 23 dropped members of 0.02-degree pairs + 4 catalogue exclusions + 800 pool; 26 residual pairs among the 1,054 | `outputs/audit_universe_loadability.csv`, `catalog/exp_split_final.csv`, `catalog/static_attributes.csv`, `catalog/clean_stations.csv` | `scripts/verify_numbers_cont.py` section 9 |
| Effective fold sizes (at least 3,000 joined hours) | 1,047 of 1,054; training 756 of 760, validation 39 of 40, out-of-domain test 15 of 16; ladder 63 / 126 / 254 / 380 / 508 / 636 / 756; rotations 592 / 587 / 606 / 785 | `outputs/audit_universe_loadability.csv` + split files | same |
| Join share and forcing losses | median gauge joins 44% of its surge record (quartiles 27 to 73%, minimum 1%); 251 of 1,047 gauges lose 23 to 78% (median 54%) of their post-2000 hours to whole null years of forcing; the others lose at most 20% | `outputs/join_share_audit.csv` | same |
| Seismic mask share of hours | median 0.8% (Japan 2.2%, Germany 0.1%); 16 gauges lose more than 10% of their p99.9 exceedance hours (Vung Tau 45%) | `outputs/audit_window_count.csv`, `outputs/seismic_exceedance_audit.csv` | same |
| Seismic mask annex (Supplementary Table 3, Fig. 3) | Tohoku 2011: 105 h removed at Onahama, 354 gauges inside the radius, tsunami removed; Sumatra 2004: 138 h, 276 gauges, removed; Sandy 2012: 0 h, 76 gauges, 151 cm peak kept; Haiyan 2013: 0 h, 0 gauges, kept | `outputs/tsunami_mask_annex.{csv,tex}`, `outputs/figS3_tsunami_mask.pdf` | `scripts/figures/tsunami_mask_annex.py` |
| Table 1 grouping, rotation union, scaling fit | see file | `outputs/audit2_numbers.txt` | `scripts/audit2_numbers.py` |
| Every log quantity, all vs continuous windows | see file | `outputs/eval_log_summary.txt`, `outputs/compare_cont.txt` | `scripts/summarize_eval_logs.py`, `scripts/compare_cont.py` |

## July 2026 preprint (all windows)

| Preprint claim | Value | Artifact | Producing script |
|---|---|---|---|
| Deployment pooled / 8-h / NNSE / wins (84 marine gauges) | +39% / +31% / 0.786 / 82 of 84 | `outputs/eval_full_lstmq_v2final.csv` | `scripts/eval_full.py` + `models/deploy_760_best.pt` |
| Deployment storm per-lead / high-water / capture / tail | +29 to 36% / +43 to 48% / 0.60 / 0.78 | `outputs/eval_full_lstmq_v2final.log` ([B], [C], [D]) | same |
| Per-lead 1 to 48 h all positive (+11% at 1 h to about +42% at 48 h) | 48 of 48 | `outputs/perlead_full.csv` | `scripts/figures/plot_perlead_full.py` |
| Ladder 64 / 128 / 256 / 384 / 512 / 640 / 760 | +21 / 26 / 30 / 34 / 37 / 38 / 39% | `outputs/eval_full_lstmq_v2g*.csv` | `scripts/eval_full.py` per size |
| Replicate bound (512) | 0.6 pooled / 0.2 at 8 h | `outputs/eval_full_lstmq_v2g512{,r2}.csv` | same |
| Rotation Japan / Europe / North America | +35 / +26, +13 / +12, +12 / +10 | `outputs/eval_full_lstmq_v2rot{jp,eu,na}.csv` | `scripts/eval_full.py` + rotation checkpoints |
| Rotation Oceania, median of three runs | +11 / +10 (runs +8.4 / +11.0 / +13.6) | `outputs/eval_full_lstmq_v2rotoc{,r2,r3}.csv` | same |
| Rotation union | 635 evaluations, 510 wins, +14 / +12% | union of the four rotation CSVs | `scripts/verify_numbers.py` section 4 |
| Storm ranges per rotation | Japan +31 to 37, Europe +8 to 13, North America +16 to 19, Oceania -1 to +11 | rotation `.log` files, block [B] | `scripts/eval_full.py` |
| Gauge-free ladder | 0.606 to 0.611 to 0.631 (to 0.786) | `outputs/eval_forcing_only_v2fonly{_eot2,_predscale,}.csv` | `scripts/experiments/eval_forcing_only.py` + `models/gaugefree_760_best.pt` |
| Gauge-free storms: capture 0.50, q99 at the peak 0.98, coverage 0.983 | | `outputs/eval_forcing_only_v2fonly_eot2.csv` | same |
| Surge-scale regression (log-correlation 0.85, median error 14%) | | `outputs/sigma_hat_test.csv` | `scripts/experiments/fit_sigma_hat.py` (regenerates byte-identically) |
| EOT20 statics (tidal range log-correlation 0.933) | | `outputs/eot20_statics_test.csv` | `scripts/experiments/extract_eot20_statics.py` (regenerates byte-identically) |
| Hydrodynamic head-to-head at 8 h (4.4 vs 8.3 vs 6.4 cm; 79 of 80) | | `outputs/eval_gtsm.{csv,log}`, `outputs/gtsm_mapping.csv` | `scripts/experiments/eval_gtsm.py` |
| Hydrodynamic exception (p99.9 at 48 h: 13.2 vs 12.5; persistence 27.6) | | `outputs/eval_gtsm.log` | same |
| GEFS real forcing, six cases (21.5 to 38.2 cm, capture 0.71 to 0.34, 5 of 6) | | `outputs/eval_gefs.csv`, `outputs/gfs_events.csv` | `scripts/experiments/eval_gefs.py` |
| Correction (September 2026): that run de-normalized the already-raw anchor value twice and inflated the degradation; the artifact is kept unchanged as the record of the preprint, and the corrected nine-case check with every peak inside the horizon is `outputs/eval_gefs_v3.*` (manuscript table above) | | | |
| Trailing causality (pooled +36, NNSE 0.818, storm 8 h +40, 80 of 84) | | `outputs/eval_trailing_{centered,trailing}.csv`, `.log` | `scripts/experiments/{exp_trailing_baseline,eval_trailing}.py` |
| Pinball score (q90 +45.4% 82 of 84; q99 +46.2% 77 of 84) | | `outputs/eval_pinball.csv` | `scripts/experiments/eval_pinball.py` |
| ENSO and season stratification | | `outputs/eval_enso_{ocrot,deploy}.csv`, `.log` | `scripts/experiments/eval_enso.py` |
| Lake ablations (706 marine; matched 640) | at most 0.7 points | `outputs/eval_full_lstmq_v2marine.csv`, `v2g640{,marine}.csv` | `scripts/eval_full.py` |
| Composition ablation of the preprint (298 less-diverse pool) | 8 h +10% vs +18% | `outputs/eval_full_lstmq_v2n298.csv` | same |
| Out-of-domain set | 15 evaluated, 14 beat persistence | `outputs/eval_full_lstmq_v2final_xdom.csv` | same |
| Seismic-mask impact (up to 45% of p99.9 hours) | | `outputs/seismic_exceedance_audit.csv` | pipeline QC |

Notes.
- The factor study of the manuscript is the clean-protocol comparison (recurrent model, attention model with hour and channel
  forcing embeddings, Chronos-Bolt; all with validation-fold checkpoint selection on seismic-masked records). The preprint's
  legacy factor rows were retired with the September 2026 revision.
- The coverage indicator that appeared in a September 2026 draft was withdrawn from the manuscript on 2026-09-10 and its
  artifacts were removed from this repository on 2026-09-14.
