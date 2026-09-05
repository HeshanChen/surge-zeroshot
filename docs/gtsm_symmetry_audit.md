# GTSM head-to-head: symmetry audit (2026-09-04)

Script: `scripts/eval_gtsm_symmetric.py` -> `outputs/eval_gtsm_symmetric.{log,csv}`, per-window
predictions cached in `outputs/gtsm_sym_preds/*.npz` (new baseline variants need no model re-run).
Original: `scripts/eval_gtsm.py` -> `outputs/eval_gtsm.log` (reproduced exactly: 4.44 | 8.33 | 6.43 cm).

## What Table `tab:gtsm` in main.tex actually compares

| side | information at issue time t0 | target it is scored against |
|---|---|---|
| "ours" (`baseline_lstmq_v2final_best.pt`) | 208 h of observed surge (context), observed surge at t0 as additive anchor (`baseline_lstm.py:33`), ERA5 forcing over context and the 48 h horizon, 5 statics | our surge: obs minus utide tide minus centered 30-day mean (`clean_surge.py:43`) |
| GTSM (CDS v2 `storm_surge_residual`, nearest point, median 1.5 km, max 238 km) | ERA5 forcing only; assimilates nothing | same target, but GTSM had only its 2010-2018 mean removed (`eval_gtsm.py:54`) |
| persistence | observed surge at t0 | same |

Three asymmetries, all in our favour:

1. **Information.** Persistence alone (zero physics) beats GTSM at 60/80 gauges at 8 h (p = 9e-6),
   44/80 at 24 h, 19/80 at 48 h. Half of the 8 h gap between "ours" and GTSM is covered by persistence.
   The paper's "79/80" mostly measures who holds the observation.
2. **Target definition.** Our truth has a centered 720-h mean removed; GTSM did not. High-passing GTSM
   with the same filter lowers its 8 h RMSE from 8.33 to 6.49 cm and lifts NNSE@8h from 0.52 to 0.63.
   22% of GTSM's reported error was low-frequency signal we had filtered out of the truth.
3. **Representation.** Nearest output point, no local calibration; 3 gauges have the nearest point
   > 25 km away (Varennes 238 km, St Lawrence; Cherry Point, Tacoma in Puget Sound). Dropping them
   changes nothing material (Section 4 of the log), so this one is minor.

The paper's two "fairness notes" do not address (1): the note that GTSM is a lead-independent hindcast
is neutral, because our forecast also consumes concurrent (perfect-prognosis) ERA5 at every lead.

## Symmetric protocol

Same 80 gauges, same 48 h windows, same seismic mask, 2010-2018.

* Tier A, no local observation on either side: gauge-free model (`v2fonly`, EOT20 statics, sigma_hat,
  the paper's 0.606 configuration) vs GTSM_hp (high-passed like the target).
* Tier B, observation at t0 on both sides: flagship vs GTSM_hp + bias correction with the mean
  error over the last 25 h (standard operational correction; the 25-h mean removes a semidiurnal
  component present in y - GTSM that made instantaneous anchoring oscillate with lead).
  Instantaneous anchoring (tau = inf, 24 h, 12 h) is also reported; the 25-h version is the
  strongest and is the one to cite.

## Results (mean over gauges; storm windows = p99.9, 74 gauges with >= 5 windows)

| comparison | full @8h | full @24h | full @48h | wins @8h | storm @8h | storm @48h | peak capture |
|---|---|---|---|---|---|---|---|
| paper: ours vs GTSM (raw) | 4.44 vs 8.33 | 5.41 vs 8.24 | 6.35 vs 8.24 | 79/80 | 9.41 vs 12.23 | 13.18 vs 12.34 | 0.64 vs 0.69 |
| **Tier B: ours vs GTSM_hp + 25 h bias corr.** | **4.44 vs 5.69** | 5.41 vs 6.12 | 6.35 vs 6.73 | 66/80 (p = 3e-9) | 9.41 vs 8.72 (42/74, n.s.) | 13.18 vs 10.79 (33/74, Wilcoxon p = 0.04) | 0.64 vs 0.68 (p = 0.007) |
| **Tier A: gauge-free vs GTSM_hp** | 7.13 vs 6.49 | 7.01 vs 6.40 | 7.06 vs 6.39 | 38/80 (n.s.) | 13.76 vs 10.50 (25/74, p = 3e-4) | 13.86 vs 10.74 (p = 0.004) | 0.57 vs 0.64 |
| GTSM_hp + bias corr. vs persistence | 5.69 vs 6.43 | 6.12 vs 8.87 | 6.73 vs 11.26 | 44/80 (n.s.) | 8.72 vs 15.40 | 10.79 vs 27.73 | 0.68 vs 0.32 |

Reading:

* **Bulk accuracy with a gauge feed:** the learned model still wins at every lead, but the margin over
  the physics model is 22% at 8 h and 6% at 48 h, not 47%. NNSE@8h 0.80 vs 0.69, not 0.80 vs 0.52.
* **Extremes with a gauge feed:** parity at 8 h (median favours us, mean favours physics, neither
  significant); physics wins at 48 h; physics captures peak magnitude better at every lead
  (0.68 vs 0.64, significant). This inverts the paper's sentence "everywhere else, including storm
  windows at operational leads, the learned model leads".
* **Without a gauge:** open-data ML and the global physics reanalysis are statistically
  indistinguishable on the bulk, and physics is 24% better on storm windows.
* **Physics adds nothing to the observation on the bulk at 8 h** (44/80 vs persistence) but is the
  entire story on extremes (8.72 vs 15.40 cm). The observation carries the bulk; the physics carries
  the peaks; the learned model beats the physics only on the part the observation already carries,
  plus a modest, shrinking margin beyond.

## Where the physics wins: a regime, not noise

Regional breakdown of Tier B (flagship vs GTSM_hp + 25 h bias correction), figure
`outputs/gis_gtsm_delta_sym.{pdf,png}` (`scripts/plot_gis_gtsm_sym.py`):

| region | n | bulk wins (ours) | median bulk delta | peak wins (ours) | median peak delta |
|---|---|---|---|---|---|
| Baltic / North Sea | 9 | 1 | -67% | 0 / 8 | -0.14 |
| US East / Canada | 3 | 2 | +2% | 0 / 2 | -0.03 |
| S America | 5 | 5 | +13% | 2 / 5 | -0.11 |
| Pacific islands / other | 15 | 11 | +19% | 7 / 14 | 0.00 |
| W Europe / Iberia Atlantic | 8 | 7 | +21% | 1 / 8 | -0.07 |
| US / Canada West | 6 | 6 | +27% | 4 / 6 | +0.09 |
| Japan / E Asia | 14 | 14 | +38% | 5 / 13 | -0.03 |
| Australia / NZ | 11 | 11 | +43% | 4 / 10 | -0.01 |
| Africa / Indian Ocean | 9 | 9 | +51% | 2 / 8 | -0.02 |

Outside the Baltic/North Sea the forecast wins bulk accuracy at 65 of 71 gauges. Inside it the physics
model wins 8 of 9 on bulk (by 30 to 92%) and 8 of 8 on peaks. The Baltic surge is a basin-filling and
seiche response to wind over the whole basin: the information is in the spatial wind field, which the
hydrodynamic model consumes and a point-forced model cannot see. This is the same regime the rotation
section already identifies as the hardest (Europe), now with a physical baseline that shows what the
missing information is worth. The paper's earlier "one exception, Ballen (Denmark)" was the visible tip
of this: with the baseline symmetrized, it is 8 of 9 Baltic gauges. Spatial forcing context (already the
paper's "identified next lever") is the fix, and the Baltic is where to test it.

## What changes in the paper

Table `tab:gtsm`, Figure `gis_gtsm_delta`, and the paragraph at main.tex:305 must be replaced. Numbers
in the abstract are unaffected (the abstract does not cite GTSM), but the Discussion sentence "beats
persistence by ... together with q99 envelopes" stands while any sentence implying the learned model
supersedes the physics model does not. Proposed replacement paragraph (numbers from the log):

> We also ran the head-to-head against the hydrodynamic baseline that a physical oceanographer would
> ask for, with the information on both sides matched. Against the ERA5-forced GTSM v3 surge
> reanalysis (nearest output point, high-passed with the same 30-day filter as the target, and given
> the same observation through a 25-hour mean-error bias correction), the gauge-fed forecast has lower
> full-distribution error at every lead (4.4 versus 5.7 cm at 8 h, 66 of 80 gauges; 6.4 versus 6.7 cm
> at 48 h, 53 of 80), while the physics model captures storm peak magnitude better (0.68 versus 0.64 of
> true peak height at the true peak hour) and has lower storm-window error at 48 h (10.8 versus
> 13.2 cm), with parity at 8 h. Without any local observation, the gauge-free variant and the
> reanalysis are statistically indistinguishable on the full distribution (NNSE 0.61 versus 0.63) and
> the reanalysis is 24% better inside storm windows. The two products are complementary: the
> observation carries short-lead bulk skill, the physics carries peak magnitude, and the learned model
> improves on the physics model only where the observation is available. A learned post-processor
> that ingests the reanalysis is the obvious hybrid and is left to future work because it would
> reintroduce the registered product the rest of the pipeline avoids.

Also delete "the exception is Ballen" from the figure caption (the per-gauge map must be regenerated
from `rmse8_ours` vs `rmse8_anch_hp_lp`).

## For Andrew (framing only; do not touch numbers)

The symmetric result is a cleaner Nature Water story than "beats physics", provided it is told as a
decision for an under-resourced coastal service:

1. Intro: name the incumbent. GTSM/GLOSSIS is the global physics option; its outputs reach national
   services through registered portals and are not locally runnable. The question a service faces is
   not "ML or physics" but "what does a single tide gauge plus an open-data model buy me, relative to
   the global physics product I may or may not be able to access".
2. Discussion (new paragraph, "A gauge or a model?"): with a gauge feed, the open-data model gives
   the best everyday accuracy at operational leads and calibrated envelopes; the physics model gives
   better peak magnitude; without a gauge, the two are equivalent on the bulk and physics is better on
   storms. The marginal value of one gauge (Tier A to Tier B: NNSE 0.61 to 0.80) exceeds the marginal
   value of the whole physics model at 8 h (persistence to physics+obs: 0.67 to 0.69). This is the
   observing-system argument the scaling section already makes, now stated in currency a ministry
   understands.
3. Keep the humanitarian verbs evidence-bounded as before ("is deployable at", "lowers the barrier").
   Do not write "outperforms physics-based models" anywhere. "Matches the global physics reanalysis
   without a gauge and exceeds it on everyday accuracy with one" is the defensible sentence.
4. The number Nature Water will want and Andrew is best placed to source: population and coastline in
   countries with no operational surge forecast (WMO Coastal Inundation Forecasting Initiative country
   lists, EW4All capacity assessments, WMO Hydromet Gap Report 2021), intersected with the Tier A and
   Tier B skill maps.
