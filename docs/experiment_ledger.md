# Experiment ledger (2026-09-05)

One row per experiment that exists on disk, sorted into three bins: what the paper cites, what is superseded,
what was audit-only. `outputs/eval_full_<tag>.{csv,log}` unless another path is given. "Clean protocol" means:
validation-fold checkpoint selection (never test), seismic-masked records, 12 epochs, `eval_full.py` on the
full test set.

## A. In the paper (all clean protocol)

| experiment | what it tests | tags / artifacts | where in the paper |
|---|---|---|---|
| Deployment, 760-gauge model | zero-shot skill at 84 pre-registered marine gauges | `lstmq_v2final` (+ `_xdom` for the 16 non-marine) | Results 1, Table tab:deploy, Figs skillmap, perlead |
| Gauge-free ladder | value of each kind of local information | `eval_forcing_only_v2fonly{,_eot2,_predscale}.csv`, `fit_sigma_hat`, `eot20_statics_test` | Results 2, Table tab:ladder |
| Physics head-to-head, matched information | gauge-free vs GTSM (tier A); flagship vs GTSM + bias correction (tier B) | `eval_gtsm_symmetric.{csv,log}`, `gtsm_sym_preds/`, `gis_gtsm_delta_sym` | Results 3 (was Discussion), Table tab:gtsm, Fig gtsmmap |
| Four rotations | transfer to whole held-out regions | `lstmq_v2rotjp`, `v2roteu`, `v2rotna`, `v2rotoc` (+ `r2`, `r3`) | Results 4, Table tab:rot, Fig rotmap, Fig ppbay |
| Europe corpus intervention | does more non-European data help Europe | small arm `lstmq_v2eu299` (299 gauges) vs `lstmq_v2roteu` (593), 233 gauges | Results 4, abstract, Discussion |
| Scaling ladder | skill vs training-gauge count | `lstmq_v2g64`, `v2g128`, `v2g256`, `v2g384`, `v2g512`, `v2g640`, `v2final` | Results 5, Table tab:scaling, Fig scaling |
| Seed replicate | run-to-run variance bound | `lstmq_v2g512` vs `lstmq_v2g512r2` | Results 5, Limitations |
| Lake/river ablations | do non-marine training gauges hurt marine skill | `lstmq_v2marine` (706), `lstmq_v2g640marine` vs `v2g640` | Methods (splits) |
| Composition ablation, equal era | diversity at fixed count | `lstmq_v2c298strat` (49 countries) vs `lstmq_v2c298usjp` (USA+JPN) | Results 5 (pending, running today) |
| Matched-update control | is the ladder data or compute | `lstmq_v2g64ms`, `lstmq_v2g256ms` at 77,028 steps | Results 5 (pending, tonight) |
| Factor study | recurrence vs attention vs generic TSFM | `lstmq_v2n298` (LSTM), `v7e_n298v` (attention with hour+channel forcing embeddings), `eval_chronos_bolt-small_masked`; `v5_masked` kept as a footnoted legacy row | Methods + Extended Data table |
| Real-forecast forcing | perfect prognosis vs GEFSv12 reforecast | `eval_gefs.{csv,log}` (anchor bug fixed 2026-09-04; 7 storms) | Results 1, Table tab:gefs |
| Causal-target re-evaluation | retrospective vs trailing surge definition | `eval_trailing` (`data/processed_trailing`) | Methods (extraction) |
| Quantile scoring | envelopes vs per-gauge climatological quantiles | `eval_pinball` | Results 6 |
| Continuity / window integrity | gap-crossing windows | my re-derivation in `docs/protocol_symmetry_audit.md` (agent script in `Codex/`) | Limitations sentence |

| Continuous-window re-evaluation (primary from 2026-09-05) | every headline checkpoint scored on windows whose 256 rows are consecutive hours | `eval_full_<tag>_cont.*` for v2final, rotjp/eu/na, rotoc/r2/r3, eu299, g64..g640, g64ms, g256ms, g512r2, marine, g640marine, c298strat/usjp, n298, v7e_n298v; `eval_gtsm_symmetric_cont.*`; `eval_forcing_only_{v2fonly,_eot2,_predscale,v2final}_cont.csv`; `eval_chronos_bolt-small_masked_cont.*`; `eval_pinball_cont.*`; summaries `compare_cont.txt`, `audit2_numbers.txt`, `eval_log_summary.txt`, `bootstrap_cis_cont.txt` | every table and figure; all-window values in ED Table 5 |

## B. Superseded (kept for the record; the paper does not cite them)

| experiment | why superseded | tags |
|---|---|---|
| Legacy factor rows | test-fold selection; v7 forcing tokens position-blind; v5 ran 9 epochs | `lstmq_best`, `lstmq298_masked`, `v7_best`, `v7_masked`, `v5_best` (v5_masked survives as a footnoted row) |
| JEPA / transformer development line | replaced by the recurrent model | `v1_best`, `v2`, `v4_best`; `lstm_best` (point-only LSTM) |
| Legacy Europe intervention | 75 gauges, test-fold selection, pre-mask | `lstmq_eu`, `lstmq_eu_masked`, `lstmq_euscale_full`, `lstmq_euscale_on_eu75` |
| Legacy rotations | pre-mask / test-fold selection | `lstmq_na`, `lstmq_na_masked`, `lstmq_nascale_full`, `lstmq_nascale_on_na85`, `lstmq_oc`, `lstmq_oc_masked`, `lstmq_jp_masked` |
| Japan single-region ladder | test-fold selection; sentence removed from the paper | `lstmq_n64`, `lstmq_n137`, `lstmq_n298`, `lstmq_scale`, `lstmq_scale4`, `lstmq_scale_r2` |
| Pre-clean global ladder | replaced by the v2 ladder | `lstmq_final`, `lstmq_gfull`, `lstmq_g64` to `lstmq_g640` |
| Asymmetric GTSM comparison | observation and target-filter asymmetry | `eval_gtsm.{csv,log}`, `gis_gtsm_delta` |
| GEFS spot check, first run | anchor de-normalised twice | `eval_gefs_v1_anchorbug.*` |

## C. Audit-only (never in the paper)

| check | result | artifacts |
|---|---|---|
| Delivered checkpoint = last epoch? | yes, identical numbers | `lstmq_v2final_LASTEPOCH` |
| GTSM symmetry, first pass (instantaneous anchor) | superseded by the 25-h bias-corrected version | `eval_gtsm_symmetric_v1.log` |
| v7 permutation probe | output invariant to forcing order and channel (1e-6) | in `docs/protocol_symmetry_audit.md` |
| Agent helper scripts | window integrity, attention probe, GEFS anchor diagnostic | `Codex/` (untracked; numbers there are agent-produced) |
| Dropped runs | lr 1e-3 for v7e; no-embedding v7 clean pair | never run (`scripts/stop_after_matched_steps.sh`) |

## Reading the Europe intervention across protocols (the question of 2026-09-05)

| arm and protocol | gauges | pooled | @4h | @8h | @12h |
|---|---|---|---|---|---|
| legacy 299 (test-fold selection, pre-mask) | 75 | +9.3% | +8.9% | +7.8% | -0.7% |
| clean 299, same 75 gauges | 75 | +0.9% | +9.9% | +5.2% | -8.1% |
| clean 299, all 233 | 233 | +3.0% | +10.3% | +7.2% | -2.2% |
| clean 593, same 75 gauges | 75 | +10.3% | +12.2% | +9.3% | +0.9% |
| clean 593, all 233 | 233 | +12.7% | +12.8% | +11.5% | +6.4% |

The 12-h weakness of the small arm was already there in the legacy run (-0.7%, which the paper described as
"bulk skill remains positive" because pooled was +9%). The clean protocol removes the flattery of test-fold
selection (pooled +9.3% to +0.9% on the same 75 gauges) and the 233-gauge set adds Mediterranean and Iberian
gauges where the small arm does fine. By country, the small arm's 12-h skill is negative exactly on the
shallow-shelf North Sea and Baltic gauges (Germany -15%, Sweden -12%, Denmark -8%, Netherlands -4%) and
positive in Britain and France (+8%, +9%): the missing regime is the large-amplitude shelf surge, as the paper
says.
