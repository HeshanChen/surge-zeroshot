# Audit 2 (external review response), results and manuscript changes, 2026-09-05

Trigger: the external review of the Nature Water draft (eight points plus five inconsistencies). Nothing was
retrained. Everything below is re-evaluation of existing checkpoints, new metrics on existing outputs, or new
baseline/forcing data. Artifacts: `outputs/compare_cont.txt`, `outputs/bootstrap_cis{,_cont}.txt`,
`outputs/eval_gtsm_symmetric{,_cont}.log`, `outputs/eval_gefs_v2.{csv,log}`, `outputs/eval_full_*_cont.{csv,log}`.
Scripts: `eval_full.py --continuous_only` (plus new metrics), `bootstrap_cis.py`, `compare_cont.py`,
`fetch_gefs_cycles.py`, `eval_gefs.py --inits`, `eval_gtsm_symmetric.py --continuous_only` (+208-h baseline).

## 1. Gap-crossing windows (review point 1): measured, conclusions hold, numbers shift 1 to 5 points

`load_station` concatenates rows after dropping gaps; 71% of deployment windows are 256 consecutive hours
(rotations 55 to 80%). A persistence forecast across a gap spans more than the nominal lead, which favours the
model on those windows. Re-evaluating every headline checkpoint on continuous windows only:

| | pooled all / cont | @8h all / cont | NNSE all / cont | high-water all / cont | peak capture | peak covered % |
|---|---|---|---|---|---|---|
| deployment 84 | +39 / +37 | +31 / +30 | 0.786 / 0.782 | +43 / +45 | 0.60 / 0.59 | 46 / 41 |
| Japan rotation | +35 / +30 | +26 / +23 | 0.808 / 0.803 | +41 / +34 | 0.65 / 0.59 | 54 / 46 |
| Europe rotation | +13 / +11 | +12 / +11 | 0.719 / 0.719 | +16 / +16 | 0.39 / 0.38 | 26 / 23 |
| North America rotation | +12 / +10 | +10 / +9 | 0.687 / 0.680 | +20 / +20 | 0.45 / 0.41 | 30 / 26 |
| Oceania rotation (median run) | +8 / +5 | +10 / +9 | 0.732 / 0.728 | +7 / +5 | 0.43 / 0.41 | 26 / 25 |
| Europe 299 arm | +3 / 0 | +7 / +7 | 0.698 / 0.698 | +5 / +4 | 0.31 / 0.29 | 19 / 17 |
| ladder 64 to 640 | each 2 points lower pooled, 0 to 1 at 8 h | | | | | |
| 298 stratified vs USA+JPN | +31 / +29 vs +12 / +7 | +25 / +24 vs 0 / -2 | | | | |

GTSM head-to-head on continuous windows: tier A gauge-free 6.74 vs GTSM 6.21 cm at 8 h (38/80, n.s.), storms
11.79 vs 8.75; tier B forecast 4.17 vs 5.47 cm (66/80), 48 h 5.95 vs 6.34, storms 7.88 vs 7.70 at 8 h and 11.18
vs 9.64 at 48 h, peak capture 0.62 vs 0.66. Same picture as all windows.

Decision: keep all-window numbers as primary (they are what the figures show), add Extended Data Table 5 with both
sets, add one sentence in Results 1 and a Methods paragraph. Retraining on continuous windows is not required by
these numbers.

## 2. Event-level peak coverage (review point 4): the "bound peaks" claim fails

The q99 envelope lies above the true peak in 46% of p99.9 storm windows in deployment (64% for peaks within 6 h,
48% at 7 to 12 h, 41% at 13 to 24 h, 43% at 25 to 48 h); 25 to 54% across rotations; 41% on continuous windows.
Hourly q99 coverage is 0.98 at every lead and q90 0.88 to 0.90. The capture ratio (mean q99 at peak / mean peak)
of 0.92 to 1.01 is a ratio of means and averaged over under- and over-prediction. All "bound peaks" language
replaced by the coverage numbers (abstract, Here-we, Results 1, Results 6, Discussion).

## 3. Real-forecast check redone with peaks inside the window (review point 3)

Three of the seven storms had catalogued peaks 51 to 55 h after issue, outside the 48-h horizon; the table had
scored in-window maxima (Ike 145 vs 173 cm, Anchorage 35 vs 142, Saint-Nazaire 62 vs 127). New GEFS cycles were
fetched by byte range from the public archive (`fetch_gefs_cycles.py`, reproducible) so that every catalogued
peak lies inside the window (true leads 27 to 45 h), and the two storms with missing variables (Viken, Ballen)
were completed: nine storms. Result: window RMSE reanalysis 37.1 / GEFS 48.2 / persistence 73.8 cm; median
degradation +19%; GEFS beats persistence 8 of 9 (Ballen fails); peak capture 0.36 with GEFS (0.47 with reanalysis
forcing); q99 above the peak in 2 of 9 storms (Ike, Sandy); Anchorage and Ballen peaks essentially missed.
"Pessimistic bound" wording removed. Extended Data Table 4 now lists issue time, catalogued peak lead, and peak.

## 4. Spatial-block bootstrap (review point 6)

Regions (11) and countries as blocks, 4,000 resamples. Deployment pooled +39% [+35, +43], 8-h +31% [+27, +35];
storm 8-h +36% [+27, +44]. Rotations: Japan +35 [+17, +50], Europe +13 [+4, +23], North America +12 [+4, +29],
Oceania +8 [+8, +25] (3 to 4 regions each, intervals wide). Europe intervention 8-h RMSE difference 593 minus 299:
-0.47 cm [-0.7, -0.3]. Composition ablation 8-h difference: -1.7 cm [-3.0, -0.6]. GTSM tier A 8-h difference
+0.5 cm [-0.5, +1.9] (no detectable difference), 48-h storm difference +3.5 cm [+0.1, +9.5] (physics better).
Tier B 8-h difference -1.3 cm [-2.1, -0.2] (forecast better), 48-h -0.4 cm [-1.2, +0.8] (no detectable
difference), peak capture difference -0.03 [-0.09, +0.01] (no detectable difference). The p-value cap sentence
is replaced by this description; "equivalent" is replaced by "no detectable difference" with the interval.

## 5. GTSM with the learned model's history length (review point 5)

A mean-error correction over the full 208-h context is weaker than the 25-h correction (6.63 vs 5.69 cm at 8 h
on all windows; 6.29 vs 5.47 continuous), so the 25-h baseline stands as the stronger one; stated in Results 3
and Methods.

## 6. Forcing product (review point 8, Open-Meteo)

`fetch_era5.py` used the archive API default (`best_match`: ERA5 0.25 deg, ERA5-Land 0.1 deg over land, ECMWF
IFS 9 km from 2017). Methods now say so and call it reanalysis forcing; "ERA5" removed from the forcing
description and the GEFS table. Not yet done: an ERA5-only refetch for the 84 test gauges as a sensitivity
(would take about two hours; the 2017 to 2018 part of the GTSM comparison is where it could matter).

## 7. Text corrections applied in `Nature_water_restructured_draft.tex`

Abstract (149 words); Here-we paragraph; Results 1 (storm skill by lead, coverage sentence, continuity sentence,
nine-storm real-forecast paragraph); Results 2 (envelope "averages"); Results 3 (intervals, 208-h sentence,
no "equivalent"); Results 4 ("every region" wording; interval instead of 1e-25); Results 5 ("The curve measures
data" removed; interval instead of 1e-9); Results 6 (coverage numbers); Discussion (level, coverage, "help
transfer most", nine-storm limits); Ebel description; Methods (forcing product, bootstrap paragraph, GTSM
208-h, real-forecast method, continuity paragraph, dress-rehearsal vs delivered relabeled gauges, 5-cm audit
wording, factor p-value); Extended Data Table 4 replaced, Table 5 added, domain-zoom figure moved to SI (ED
stays at 10 items).

## 8. Still open

Temporal holdout and an end-to-end causal-inference test (statistics from pre-issue data): not run; the paper's
claims are scoped to perfect-prognosis spatial transfer plus the nine-storm check. Extra seeds on the ladder:
not run. ERA5-only forcing sensitivity: not run. Gauge-free ladder on continuous windows: only the local-statics
configuration finished (0.606 all windows; continuous value in `eval_forcing_only_v2fonly_cont.csv`).

## 9. Continuous windows made primary (review 2, 2026-09-05 afternoon)

Every number in `paper/restyle/Nature_water_restructured_draft.tex` now comes from the continuous-window
evaluations (`outputs/eval_full_<tag>_cont.*`, `eval_gtsm_symmetric_cont.*`, `eval_forcing_only_*_cont.csv`,
`eval_chronos_bolt-small_masked_cont.*`, `eval_pinball_cont.*`); all-window values stay in Extended Data Table 5
as the sensitivity. No retraining. Derived numbers that no single log prints are in `outputs/audit2_numbers.txt`
(`scripts/audit2_numbers.py`); every quoted log quantity is tabulated in `outputs/eval_log_summary.txt`
(`scripts/summarize_eval_logs.py`); intervals in `outputs/bootstrap_cis_cont.txt`.

What changed against the all-window draft:

- Deployment: 81 of 84 gauges beat persistence (Veracruz moves from +0.5 to -18% pooled skill on continuous
  windows; its continuous subset is 263 of 965 windows). Pooled +37% [+33, +41], 8 h +30% [+26, +34],
  NNSE 0.782, storm per-lead +32 to +40% at 4 to 12 h, peak capture 0.59, event-level peak coverage 41% [31, 51].
- Oceania rotation: the median of the three clean runs is `lstmq_v2rotocr2` on both window sets
  (all: +8.4 / +11.0 / +13.6; continuous: +5 / +8 / +10); the paper uses r2 everywhere, +8% continuous
  (+11% all windows). ED Table 5 had used `v2rotoc` (+8 all / +5 cont); fixed.
- Rotation union 633 gauge evaluations (Europe drops to 231 gauges with at least 20 continuous windows):
  pooled +11%, 8 h +11%, 477 of 633 beat persistence.
- Europe intervention on the 231 shared gauges: 299 arm pooled -0.4% (written as zero), 593 arm +11%;
  593 lower at 186 of 231; 8-h difference -0.45 cm [-0.51, -0.16]; high water +8 -> +21%; peak 0.29 -> 0.38.
- GTSM head-to-head (continuous; 61 gauges with >= 5 storm windows): tier A full 8 h gf 6.74 vs GTSM 6.21
  (n.s.), storms 11.79 vs 8.75 and 13.32 vs 9.85 (GTSM better, intervals exclude zero), peak 0.57 vs 0.61 (n.s.);
  tier B 8 h 4.17 vs 5.47 (forecast better, 66/80), 48 h 5.95 vs 6.34 (n.s.), storms 7.88 vs 7.70 and
  11.18 vs 9.64 (both n.s.), peak 0.62 vs 0.66 (n.s.). Bold in Table 2b now follows the intervals exactly:
  only tier A storm cells (GTSM) and tier B full 8 h (forecast). Text calls the comparison "equal local
  information" and states that the two systems differ in meteorological input (ERA5 fields vs the Open-Meteo
  point series).
- Gauge-free ladder: 0.593 (all open) -> 0.598 -> 0.618 -> 0.782; zero-water-level ablation 0.522.
- Scaling: pooled 19.1, 23.7, 27.9, 31.5, 34.7, 36.4, 37.3; log-linear slope 5.2 points per doubling
  (8 h: 3.8). The old "5, 4, 3.5, 3.5 decaying" sentence is gone (not supported on either window set).
  Replicate bound on continuous windows: 0.8 pooled, 0.8 at 8 h, 0.004 NNSE, 0.01 peak capture; the 640 -> 760
  step at 8 h (0.6) is inside it, stated. Matched steps: g64 19.1 -> 19.7 pooled, 17.3 -> 18.8 at 8 h;
  g256 27.9 -> 27.0, 22.7 -> 23.0.
- Composition: 49-country +29 / +24; two-country +7 / -2; regional breakdown of the two-country model with a
  stated grouping (Japan +30 at 8 h, USA+Canada +20, Europe -14, Australia -70).
- Factor study: LSTM +15 / +14 / +31 / 0.58, v7e +7 / 0 / +11 / 0.38, Chronos -1 / +5 / +3 / 0.32,
  persistence peak 0.30; LSTM lower 8-h RMSE at 71 of 77.
- Table 1 regrouped with an explicit geographic rule (`scripts/audit2_numbers.py`): Asia 18, Europe 18,
  North America 15 (incl. Mexico, US Virgin Islands), Oceania 10 (Australia), Pacific islands 7 (all tropical
  Pacific island gauges), South America 8 (incl. Falklands, Antarctic Peninsula), Africa 8 (incl. Canary
  Islands, Madeira). The July grouping (18/18/14/12/8/8/4/2) could not be reproduced from any documented rule
  (best fit put Pohnpei in "other"), so it was replaced; Methods sentence updated to match.
- Title: "Learning transferable storm-surge forecasts from a global tide-gauge network" (Heshan, 2026-09-05 evening; earlier candidate "Open-data storm-surge forecasting for coasts without a local model"). Results 6 heading no longer
  says the envelopes "recover" peak information; probabilistic claims are coverage and pinball only.
- GEFS: "nine storm cases (eight storms; Xaver at two gauges)", issue rule stated as implemented.
- Figures regenerated on continuous windows: `gis_skillmap_cont`, `perlead_full_cont`, `gis_gtsm_delta_sym_cont`,
  `gis_rotations_cont`, `scaling_figure_cont`, `gis_ppbay_cont` (scripts take `_cont` / `--continuous_only`).
- Stated as all-window where no continuous version exists: causal-target check, dress-rehearsal relabeling
  effect, out-of-domain 16 gauges, the +17% test-side-selected Oceania number.

## 10. Third external read (2026-09-05, later): wording only, no new experiments

Applied by `paper/restyle/review3_edits.py` (each replacement asserted once):
- Abstract re-centred on the title: four held-out regions +8 to +30% pooled and the fixed-count composition
  contrast (+24% vs -2% at 8 h) now in the abstract; the envelope caveat is one clause; closing sentence
  "argue for a gauge network covering more regimes" replaces "the next gauge is worth most where its regime is
  unrepresented". 149 words.
- Results order: deployment -> regions -> scaling and composition -> local information -> GTSM -> envelopes
  (figures renumber automatically: rotations Fig. 3, scaling Fig. 4, GTSM map Fig. 5). Discussion opens with
  the network paragraph ("Transfer is a property of the training network"); the service paragraph is framed as
  what the local observation buys vs what the network buys.
- GEFS issue time stated as implemented: the 00 UTC cycle 24 to 48 h before the catalogued peak (code default
  is peak-36 h floored to 00Z, i.e. 36 to 60 h, moved one day later for the three storms where that left the
  peak outside the horizon: Ike, Anchorage, Saint-Nazaire; `outputs/gfs_inits_inwindow.csv`). "1.5 to 2 days"
  -> "27 to 45 h" everywhere; "last 00 UTC cycle" removed.
- "level" / "matches" -> "no detectable difference", with an explicit sentence that the intervals do not
  establish equivalence.
- The 512-gauge replicate is "one observation of run-to-run variation", no longer a "bound".
