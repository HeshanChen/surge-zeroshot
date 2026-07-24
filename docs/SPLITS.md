# Frozen splits: construction rules and status

The split CSVs under `catalog/` are the **pre-registration artifacts of this study**. They are frozen:
they are not meant to be regenerated, and downstream code treats them as ground truth. This document
records how they were constructed.

## Deployment split (`exp_split_final.csv`: 760 train / 40 val / 84 test + 16 out-of-domain)
1. Universe: 1,029 QC-passing gauges (of 1,054 with joined observations+forcing).
2. 100 test gauges pre-registered, stratified across continents.
3. Exclusion buffer: no training gauge within 0.5 deg of any test gauge.
4. Cross-provider dedup: 0.02-deg box pass on top of the catalog-level 2-km dedup; ties keep the lower-gap record.
5. Validation fold: 40 gauges carved from the training pool (continent-stratified), excluded from training,
   used ONLY for checkpoint selection.
6. Two post-registration water-body audits relabeled 16 gauges out of domain (5 Great Lakes, 1 Antarctic
   ice shelf, 10 estuary/river/weir/lake); effects quantified in both directions in the paper (Section 2.4).
   The relabeling is the only change ever made to the registered test list.

## Rotation splits (`exp_split_{jp,eu,na,oc}_v2.csv`)
Leave-region-out with a 2-deg buffer around the held-out region; training pool = world minus region minus
buffer; validation fold re-carved from each rotation's training pool. Oceania was trained three times with
the same pipeline (seeds); the paper reports the median run (checkpoint shipped here).

## Ladder splits (`exp_split_g{64..640}.csv`, `g640marine`, `marine_abl`, `n298`)
Nested subsets of the deployment training pool, country-stratified so regime composition stays
approximately proportional at every size; `g640marine` removes lake/river gauges at matched count;
`n298` is the reduced-diversity composition ablation.

## Verifying freeze integrity
The working-history archive (separate repository) contains the timestamped commit at which each split file
was frozen, before the corresponding training runs. The audit gate (`scripts/verify_numbers.py`) asserts
fold sizes (760/40/84, rotation n = 77/233/214/111) on every run.
