# Provenance: every paper number → artifact → producing script

Machine-checked by `scripts/verify_numbers.py` (69 assertions; run it before trusting anything else).
"Artifact" paths are committed in this repository. Producing script = what generated the artifact.

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
