# Protocol symmetry audit, every baseline and table (2026-09-04)

Scope: back-application of the 2026-07-11 scoreboard checklist to every comparison in `paper/main.tex`.
Method: four read-only code audits (one per baseline family) followed by my own verification of every
load-bearing claim at the cited lines. Marks: **[V]** = I verified it in code or by re-running;
**[A]** = agent-reported, mechanism plausible, numbers not independently re-derived. Agent helper scripts
and their outputs sit in the untracked `Codex/` directory (created today by the audit agents, not a prior
external audit); treat numbers there as [A] unless listed here as [V].

Companion documents: `docs/gtsm_symmetry_audit.md` (GTSM, done first), `docs/local_knowledge_insertions.md`.

## Summary by contribution

| paper contribution / comparison | verdict | what is wrong | fix |
|---|---|---|---|
| vs persistence (deployment, rotations, per-lead) | **clean** | none; persistence is the model's own zero-delta output, same t0 sample [V] | disclose the in-sample scalars (below) |
| vs GTSM (Table tab:gtsm) | **inverted on extremes** | observation and target-filter asymmetry [V] | replace table, figure, paragraph (done in `gtsm_symmetry_audit.md`) |
| vs Ebel 2024 (abstract, 0.606 vs 0.556) | **not comparable** | pooled vs per-station NNSE, storm-biased single-hour targets, different detiding, spatial+temporal vs spatial holdout [V]; our own `docs/baseline_ebel2024.md:104` calls the direct comparison invalid | reword abstract to "same order of magnitude under a different protocol"; head-to-head only if their model is run on our gauges |
| factor study (Table tab:factor, contribution 4) | **not supported as stated** | attention baseline is position-blind and channel-blind on forcing [V]; v5 ran 9 epochs and is scored at epoch 3 against a stated 12-epoch matched budget [V]; LR 1e-3 vs 3e-4 [V]; test-fold checkpoint selection for all rows [V]; mixed pre-mask/masked cells and unbacked Chronos extreme cells [A] | retrain v7 with hour and channel embeddings under the clean protocol; re-run every row through one eval path; reword contribution 4 until then |
| scaling curve (Table tab:scaling) | **confounded, shape probably survives** | optimizer steps grow about 12x with corpus [V]; composition ablation not equal-recipe and cites a nonexistent SI [V]; Japan ladder is legacy protocol [A]; Europe intervention arms legacy and pre-mask [A] | matched-update control at 2 rungs; rebuild ablation from clean artifacts or drop; add SI or delete the citations |
| quantile calibration / pinball vs climatology | clean | climatology quantiles are in-sample per gauge, which favours the baseline [A] | none |
| real-forecast spot check (GEFS) | **numbers were wrong against us** | anchor de-normalized twice [V]; fixed and re-run [V] | update Section 4.1 and Limitations with the corrected numbers |
| corpus recipe / evaluation windows | disclosure gap | 28.7% of evaluation windows span a data gap, 9.0% of targets, 2.2% of "8 h" leads are not 8 real hours [V, my own count]; symmetric across methods | add a continuity-restricted sensitivity and a sentence in Data |

## 1. Persistence and the core protocol (`scripts/eval_full.py`)

Clean [V]: persistence is `last*100` and the model anchor is `last/tstd`, the same array element
(`eval_full.py:67,95,107`); the head is zero-initialised so the model starts at persistence
(`baseline_lstm.py:23,33`); thresholds, storm windows, high-water hours and peak hours come from truth only
and index model and persistence errors in the same statement (`eval_full.py:60,120,125,140`); the delivered
checkpoint equals the last epoch (`eval_full_lstmq_v2final_LASTEPOCH.log` is identical to the published
log in every reported number), so which fold selected it is moot [V].

Disclosure items (not asymmetries, model-side in-sample statistics), all [V] at the cited lines:

* `tstd` (output scale) is the std of the gauge's full record, which includes every scored hour (`eval_full.py:59`).
* Forcing channels are standardised with the gauge's full-record mean and std (`eval_full.py:61`); train and eval agree.
* Local statics (tidal range, form factor) are fitted on the gauge's last 10 years of sea level (`build_static_attributes.py:16`).
* There is no temporal holdout anywhere: splits are spatial only (26 split files carry `name,fold` only); test-gauge years overlap training years at other gauges. Stated only in `docs/technical_report_cn.md:268`, not in the paper.
* The centered 30-day baseline and centered despiking use future hours; disclosed, and the causal re-evaluation uses the same checkpoint with persistence recomputed symmetrically (`eval_trailing.py:21,46,55`) [V].

Reporting inconsistency [V]: high-water-hour skill (abstract "+43 to +48%") is pooled over windows across
gauges (`eval_full.py:120-127`), while `main.tex:338` says statistics are gauge-averaged unless noted.
Audit gate gap [V]: `verify_numbers.py:66` regexes `high-water-hour RMSE` but the log prints
`timestep-level (high-water hours, ...)`, so the abstract's high-water numbers are checked by nothing.

Gap-crossing windows [V, re-derived independently]: `load_station` drops rows (|s| >= 4 m, missing ERA5,
seismic mask) and concatenates positionally with no time check. On the 84 test gauges: 239,757 windows,
68,802 (28.7%) span at least one gap, 21,609 (9.0%) have a broken target span, 5,284 (2.2%) have an "8 h"
lead that is not 8 real hours. Worst gauges: Rumoi 93%, Varennes 89%, Niigata 86%, Andenes 85%. Symmetric
across methods; the agent's continuity-restricted re-score (ours 4.44 to 4.17 cm, persistence 6.43 to
5.95 cm at 8 h, `Codex/continuity_sensitivity_summary.csv`) is [A] and should be reproduced as an
`eval_full.py --continuous_only` flag before it is quoted.

## 2. GTSM

See `docs/gtsm_symmetry_audit.md`. Bulk margin 47% to 22% at 8 h; physics wins peak magnitude and 48 h
storm windows; gauge-free equals GTSM on bulk and loses 24% on storms; the Baltic/North Sea is where
physics wins (8 of 9 gauges).

## 3. Ebel et al. 2024 (StormSurgeCastNet)

All [V] in `external/StormSurgeCastNet/`:

* Their NNSE is one pooled value over all concatenated test targets (`test.py:223-228`, `util/losses.py:30-37`), with the cross-gauge variance as denominator. Ours is per gauge at the 8 h lead, then averaged (`eval_forcing_only.py:71,75,82`).
* Their test targets are one hour per gauge, drawn at IBTrACS landfall times when available (`util/dataLoader.py:462-482`), else 50% extreme / 50% random. Ours: every stride-48 window, all hours.
* Their target: full-record linear detrend, single full-record utide fit, 5-hour centred smoothing, no low-frequency high-pass (`util/decomposeSeaHeight.py:119-174`). Ours: segmented 8-year fit, 30-day high-pass, despiking.
* Their holdout is spatial and temporal (test from 2014-04-03); ours is spatial only.
* Their densification keeps neighbouring gauges' observations in the input; our gauge-free variant uses none, but does use the target gauge's own ERA5 record statistics for normalisation.
* Two NNSE prints exist in their code (standardised and metre-units, the latter mixing GESLA and GTSM stds); nothing records which produced 0.556.
* Our own `docs/baseline_ebel2024.md:104`: "Direct number comparison is invalid, four axes differ", recommending relative-to-persistence comparison instead.

Consequence: the abstract sentence "exceeding the published simulation-based densification benchmark
(0.556, on the respective test corpora) while using strictly less information at inference" must become a
statement of comparable magnitude under a different metric and sample, or be dropped. The "strictly less
information" clause is true and can stay.

## 4. Factor study (Table tab:factor)

Load-bearing finding [V, my own probe on `outputs/surge_v7_best.pt`]: the hourly-forcing transformer's
192 forcing tokens are produced by one `Linear(1, D)` per scalar and concatenated into the key/value set
with no hour embedding and no channel embedding (`src/models/surge_jepa_v7.py:24,30-31`); only the queries
carry an hour embedding. Reversing the 48 forcing hours changes the output by 9.5e-7, permuting the four
channels by 1.2e-6, permuting all 192 tokens by 9.5e-7; zeroing the forcing changes it by 2.7 (the forcing
is used, but as an unordered, unlabelled set). The LSTM changes by 2.2 and 1.8 under the same permutations.
So "identical hourly forcing access" (`main.tex:22,252`) and "resolution is not the story" (`main.tex:252`)
compare a recurrent model against an attention model that structurally cannot integrate forcing in time.
The result "recurrence dominates attention" is not established by this table.

Further [V]:

* v5 (self-supervised row) trained 9 of 12 epochs and is scored at epoch 3 (`docs/v5_report.md:3,30-38`); `main.tex:241,308` state a 12-epoch matched budget.
* Learning rate 1e-3 for the LSTM vs 3e-4 for both transformers (`train_baseline_lstmq.py:62`, `train_v7.py:50`, `train_foundation_v5.py:45`); no sweep on either side.
* All three trained rows select checkpoints on the 77-gauge test fold (`exp_split.csv` has no val fold; `train_v7.py:24`, `train_foundation_v5.py:21`, and the pre-49be7c8 LSTM trainer). `main.tex:81` describes val-fold selection; that is true of the delivered model only.

Further [A], mechanism checked, cells not re-derived:

* Evaluation of the three trained rows used `exp_split_n298*.csv` while training used `exp_split.csv`; static-attribute standardisation constants at eval therefore differ from training (the two 298-gauge train folds share 144 gauges).
* v5 row mixes pre-mask (pooled +7%) and masked (extremes) artifacts; persistence peak capture 0.25 is pre-mask, masked logs say 0.27.
* Chronos row is `bolt-small` (script default is `bolt-base`), pre-mask, evaluated by its own summary code with no log; its two extreme cells (+5%, 0.29) are backed by no committed artifact, only `docs/v5_report.md:91`.
* `verify_numbers.py` checks nothing in this table.

Fix: retrain v7 with hour and channel embeddings on the forcing tokens (and v5 likewise if it is kept),
under the clean protocol (val fold, masked, full 12 epochs, one learning-rate choice applied to both
families or a two-point sweep for each), evaluate every row with one `eval_full.py` invocation on one
split file, regenerate the Chronos extremes from a logged run. Until then contribution 4 should be reworded
to what the data support: a recurrent model with per-hour forcing beat every attention variant we built,
and the attention variants we built did not encode forcing time or channel identity.

## 5. Scaling (Table tab:scaling, Section 4.3)

* Compute confound [V]: `EP=12; bs=64; spe=N//bs; tot=EP*spe` (`train_baseline_lstmq.py:98`), so optimizer steps scale with training windows, about 12x from 64 to 760 gauges, and the cosine schedule stretches with it (`:104`). The Chinese report names this as an open confound (`technical_report_cn.md:168`); the paper does not. Fix: matched-update control (train g64 and g256 for the 760-rung step count) or state "data and compute scale together".
* Ladder hygiene, clean [V]: val fold (40) and test fold (84) identical across the seven rungs, no overlaps; identical hyperparameters; 760 rung is the delivered model from the same loop (`run_clean_ladder.sh:6-11`). Subsets not nested for 6 of 7 pairs [A]; split generator absent from the repo [A]; no seed set [A]; single unseeded replicate pair at 512 is the variance bound [V at `verify_numbers.py:35-37`].
* Composition ablation (298 vs 298, +10% vs +18%) [V]: the two arms come from different corpus eras sharing 144 of 298 gauges, both test-fold selected, on opposite sides of the seismic-mask cutoff; the manuscript once said "an earlier, less diverse pool" (`git 42382a2`) and now says "a less diverse pool" (`main.tex:235`); both this and the Japan ladder are cited to an SI that does not exist. Clean artifacts exist for the full-pool arm (`eval_full_lstmq_v2n298.csv`: +18% at 8 h) and a masked re-eval for the old-pool arm (`eval_full_lstmq298_masked.csv`); the old-pool arm has no clean-protocol retrain.
* Europe corpus intervention (299 to 633) [A]: both arms legacy test-fold selection, both pre-mask, scored on 75 gauges (rotation row uses 233), no audit-gate check. The paper already says count and composition move together. Fix: retrain the 299 arm under the clean protocol (one run) and score both on the same gauges.
* Japan single-region ladder ("+12 to +35% over 64 to 763") [A]: five legacy-protocol points, test-fold selected, nested; its own commit called the top a knee (`git bfe0659`).

## 6. Rotations

[V, my own computation from `catalog/`]: minimum train-to-test Chebyshev separation 2.33 deg (Japan),
7.50 (Europe), 2.18 (North America), 6.40 (Oceania); deployment split 0.51 deg; no train gauge within 0.5 deg
of a test gauge in any split. Exclusion is by region membership, so buffers differ 3.4x across rotations;
no script generating the rotation splits exists in the repo [A]. Val folds differ per rotation [A].
Disclose the per-rotation separation; nothing to re-run.

## 7. Real-forecast spot check (GEFS)

Bug [V]: `eval_gefs.py:84-88` built the t0 anchors of the interpolated GEFS series as `a[...]*fsd+fmu`
where `a` was already raw, so the first hours of every GEFS forcing series started from nonsense (MSLP of
order 11,000 hPa). Fixed 2026-09-04 (old script, csv and log kept as `*_v1_anchorbug.*`). Re-run [V]:

| | published (bug) | fixed, same 6 storms | fixed, 7 storms (Esbjerg now complete) |
|---|---|---|---|
| window RMSE era5 / gefs / persistence (cm) | 21.5 / 38.2 / 58.8 | 21.5 / 27.5 / 58.8 | 25.9 / 32.0 / 65.0 |
| peak capture era5 / gefs | 0.71 / 0.34 | 0.71 / 0.56 | 0.68 / 0.53 |
| q99 at peak era5 / gefs | 1.15 / 0.82 | 1.15 / 1.02 | 1.09 / 0.96 |
| median degradation era5 to gefs | +70% | +30% | +19% |
| beats persistence | 5/6 | 6/6 | 7/7 |

Section 4.1 and Limitations ("storm RMSE +70%, envelopes at 0.82, 5 of 6") must be updated. Remaining
caveats [A]: issue time is floored to 00Z so intended peak leads are 40 to 55 h and for 3 of 6 storms the
catalogued peak lies outside the 48 h window (the table's "peak" is the in-window maximum); ERA5 analysis
leaks into hours 1 to 2 through the interpolation anchor; the storm list (`outputs/gfs_events.csv`) and
the GEFS download have no generating script.

## 8. Reproducibility gaps found along the way

* `src/data/dataset_v0.py`, which implements the seismic mask, is not tracked by git [A, `git ls-files src/`].
* No training stdout is persisted for any run; best-epoch identity is unrecoverable except for v5.
* `eval_full.py` prints the checkpoint path before installing its log tee (`:31` vs `:44`); no eval log records which checkpoint produced it.
* Generators missing: ladder splits, rotation splits, `gfs_events.csv`, GEFS download.
* `verify_numbers.py` covers 1 of 7 columns of the ladder, none of the factor table, none of the Europe intervention, and its high-water check is a silent no-op.

## Status 2026-09-04 23:00

Working manuscript is `paper/restyle/main_restyled.tex` (the two-column variant is regenerated from it with
`scripts/make_twocol.py paper/restyle/main_restyled.tex paper/restyle/main_restyled_twocol.tex`; `paper/main.tex`
was left at git HEAD, its trial edits saved as `paper/restyle/audit_edits_from_main_2026-09-04.patch`).

Applied to the working manuscript (result-independent, humanitarian framing where it fits):
abstract Ebel clause and tier-A/tier-B sentence; contribution 2 clause; gauge-free paragraph local-knowledge
ranking; GEFS table, paragraph and Limitations numbers (7 storms, +19%, 7/7); scaling paragraph compute
disclosure and local-knowledge closing; earlier-era wording for the 298 ablation and removal of the two
nonexistent-SI citations (Japan ladder sentence dropped); Discussion "gauge's value is twofold" and
"gauge or model" sentences; Discussion Ebel clause; GTSM table, figure (`gis_gtsm_delta_sym`) and paragraph
replaced by the matched-information version with `\label{sec:gtsm}`; Limitations disclosures (Baltic,
no temporal holdout and in-sample scalars, gap-crossing windows, per-rotation separation); appendix
pooled-statistics note. `latexmk` compiles clean (29 pages, no undefined references).

Pending the re-run queue (`scripts/run_audit_queue.sh`, log `outputs/audit_queue.log`, marked in the .tex as
`% AUDIT 2026-09-04` comments): contribution 4 and Table tab:factor (v7e vs v7 clean pair, lr 3e-4 and 1e-3,
Chronos masked); Europe intervention numbers (eu299v vs roteu on 234 gauges); 298 composition ablation
(c298strat vs c298usjp); matched-update control sentence (g64ms, g256ms).

### Re-run results as they land

**v7e (2026-09-05 01:15, clean protocol, lr 3e-4, 12 epochs, best = epoch 12, Japan 77-gauge test):**
pooled +7% (legacy v7: +2%; LSTM clean: +20%); @4h -3%, @8h +2%, @12h +4% (LSTM +14/+18/+17); NNSE@8h 0.711
(LSTM 0.779); p99.9 window skill +10%, high-water +10%, storm @8h +4% (LSTM +28/+30/+28); peak capture 0.34
(LSTM 0.61, persistence 0.27); q99 peak envelope 0.69 (LSTM 1.01); tail coverage 0.55 (LSTM 0.71). LSTM has the
lower 8-h RMSE at 75 of 77 gauges (sign p = 4e-20), lower pooled RMSE at 70 of 77. Reading: restoring hour and
channel identity to the forcing tokens helps attention (pooled +2% to +7%) but the recurrent model still leads on
every axis, and attention variants cluster at peak capture 0.30 to 0.35 against 0.61. Contribution 4 survives in
weakened, honest form; the lr 1e-3 row and the no-embedding clean pair are still queued and belong in the same table.
Val-proxy during training (40 global gauges) read pooled +18% / @8h +6%; the Japan test is harder for this model
than the val fold, another reason the paper must never quote proxies.

**Chronos-bolt-small, masked record, same 77 gauges (2026-09-05 01:22, `outputs/eval_chronos_bolt-small_masked.{csv,log}`):**
pooled +1%, @4h +7%, @6h +8%, @8h +5%, @10h +3%, @12h -0%; high-water p99 +2%, p99.5 +2%, p99.9 +3%; peak capture 0.31
(persistence 0.27). Replaces the paper's unbacked cells (+5% high-water, 0.29 peak capture) with logged values; the
qualitative reading is unchanged: a generic time-series foundation model with no forcing matches the domain model only
inside the autocorrelation-dominated first hours and has no extreme-event skill.

**Queue trimmed (2026-09-05):** the lr 1e-3 and no-embedding v7 runs were dropped as padding (`scripts/stop_after_matched_steps.sh`);
the factor table will carry LSTM clean, v7e clean, Chronos masked, and the legacy v5 SSL row footnoted.

**Europe 299 arm (clean protocol) trained 2026-09-05 01:42:** best = epoch 12 on the 40 extra-Europe-free val gauges
(val proxy pooled +24%, @8h +22%); test evaluation on the 234-gauge Europe set running.

**Europe corpus intervention, clean protocol (2026-09-05 01:46, `eval_full_lstmq_v2eu299` vs `eval_full_lstmq_v2roteu`, 233 gauges):**
299-gauge Europe-free arm: pooled +3%, @8h +7%, @12h -2%, storm per-lead +5 to +8%, high-water p99/p99.5/p99.9 +8/+7/+5%,
peak capture 0.31, q99 capture 0.60, tail coverage 0.44. 593-gauge arm (the clean rotation row): pooled +13%, @8h +12%,
storm +8 to +13%, high-water +20/+19/+16%, peak capture 0.39, q99 0.70, tail 0.54. The larger arm has the lower 8-h RMSE
at 194 of 233 gauges (sign p = 7e-26) and lower pooled RMSE at 218 of 233. The legacy claim "storm skill zero (+0 to +2%)
turns positive (+3 to +11%)" does not survive: on the full 233-gauge set with validation-fold selection the small arm
already has +5 to +8% storm skill. The clean statement, now in the manuscript (abstract, Section 4.2, Discussion):
enlarging the Europe-free corpus from 299 to 593 gauges lifts European pooled skill from +3 to +13% and high-water skill
from +8 to +20%.

**Composition ablation at fixed count, clean protocol (2026-09-05 02:39, `eval_full_lstmq_v2c298strat` vs `v2c298usjp`, 84 gauges, same val/test as the ladder):**
298 stratified across 49 countries: pooled +31%, @4h +22%, @8h +25%, @12h +21%, NNSE@8h 0.759, p99.9 high-water +38%, peak capture 0.55
(sits between the 256 and 384 ladder rungs; not distinguishable from 256, 46/84). 298 USA+JPN: pooled +12%, @4h -5%, @8h 0%,
@12h -5%, NNSE 0.664, high-water +3%, peak capture 0.39. Diverse corpus lower 8-h RMSE at 67/84 (Wilcoxon p = 1e-10), pooled 68/84.
By region (@8h, stratified vs two-country): Japan +34 vs +33, N America +26 vs +15, Europe +25 vs -12, Oceania +22 vs -61,
Africa/Indian Ocean +32 vs +14, Pacific +25 vs +16. Replaces the legacy +10 vs +18 sentence (main_restyled.tex 4.3 and Discussion,
and the restructured draft). Remaining slot: matched-update control (g64ms running since 02:37, g256ms next).

**Matched-update control (2026-09-05 03:59 and 05:08, `eval_full_lstmq_v2g64ms`, `v2g256ms`, 84 gauges):** 64 gauges at 77,028 steps
(143 epochs, best at epoch 18): pooled +22% (ladder +21%), @8h +19% (+17%), NNSE 0.738 (0.731), high-water +26% (+28%), peak capture 0.48
(0.50); matched-steps lower 8-h RMSE at 52/84 (Wilcoxon p = 0.01), pooled 37/84 (n.s.). 256 gauges at 77,028 steps (37 epochs, best at
epoch 21): pooled +30% (+30%), @8h +24% (+23%), NNSE 0.755 (0.756), high-water +37% (+34%), peak capture 0.56 (0.54); 8-h 33/84 (n.s.),
pooled 26/84 (ladder lower, p = 6e-4). Twelve times the updates buy one to two points at 64 gauges and nothing at 256; doubling the
gauges buys five points pooled. The compute confound is closed; sentence added to Section 4.3 and the Discussion. **All audit re-runs
are complete (queue trimmed at 05:06; the no-embedding v7 and lr 1e-3 runs were dropped).** Factor study rewritten 2026-09-05 05:30 per Heshan's
decision: v5 SSL row deleted, self-supervision dropped from the abstract and contribution 4; Table tab:factor now LSTM clean /
attention with hour+channel forcing embeddings / Chronos masked / persistence; "Two factor conclusions" text; Discussion
inductive-bias paragraph rewritten. No `% AUDIT` comments remain in the working manuscript.

## Required actions, ordered

1. Text now, no compute: GTSM section (drafted); Ebel sentence in abstract; contribution 4 rewording; SI citations; "gauge-averaged" statement in the appendix; per-rotation separation; in-sample scalars and no-temporal-holdout sentence; gap-window sentence; GEFS numbers.
2. Cheap compute (hours): `eval_full.py --continuous_only` sensitivity; Chronos extremes from a logged run; regenerate the GTSM figure (done, `gis_gtsm_delta_sym`).
3. Retrains (one to two days on MPS): v7 with hour and channel embeddings, clean protocol; Europe 299 arm, clean protocol; matched-update control at g64 and g256.
4. Gate: extend `verify_numbers.py` to every table cell and add a protocol-symmetry section (information at t0, target filter, forcing form, selection fold, mask status, budget) that must match across the rows of each table.
5. Repo: track `dataset_v0.py`, persist training logs, log the checkpoint path, add the missing generators.
