# Frozen splits: construction and status

The split CSVs under `catalog/` are the pre-registration artifacts of this study. They are frozen: they are not meant to be
regenerated, and every script treats them as ground truth. This document records how they were built and what the
manuscript says about them; `scripts/verify_numbers_cont.py` (section 9) recomputes every count below from the files in
this repository.

## Dataset chain (`exp_split_final.csv`: 760 train, 40 validation, 84 marine test, 16 out-of-domain test)

1,054 gauges have both a processed surge record and a forcing record (`outputs/audit_universe_loadability.csv`); all pass
the QC screen. Removing the 100 pre-registered test gauges (84 marine primary, 16 out of domain), the 127 gauges inside a
0.5-degree box of a test gauge, one member of each of 23 remaining co-located pairs (0.02-degree box), and four gauges that
the catalogue cleaning of `catalog/clean_stations.csv` had already excluded (three lake, lagoon or river sites and one
co-located twin) leaves the 800-gauge split pool (760 training, 40 validation). A 0.02-degree pass finds 26 residual
co-located pairs among the 1,054 gauges; the release keeps one member of each. The 40-gauge validation fold is carved from
the pool (continent-stratified), excluded from training and used only for checkpoint selection.

Fold sizes are nominal. Seven gauges have fewer than 3,000 joined hours after seismic masking (their records end at or
before the start of the forcing archive in 2000, or their wind forcing was lost to a download failure) and contribute no
windows: four in the training fold (756 effective training gauges), one in the validation fold (39), one among the
out-of-domain test gauges (15) and one outside the split. Joined hours per gauge are in `outputs/audit_universe_loadability.csv`.

Two post-registration water-body audits relabelled 16 of the 100 test gauges out of domain (five Great Lakes, one
Antarctic ice-shelf station, ten estuary, river, weir or lake sites); the manuscript quantifies the effect in both
directions. The relabelling is the only change ever made to the registered test list.

## Rotation splits (`exp_split_{jp,eu,na,oc}_v2.csv`)

Leave-region-out by membership: the training pool is the world minus the region, and a 40-gauge validation fold is
re-carved from each rotation's pool. Because regions are defined by membership, the minimum train-to-test separation
differs by rotation (2.2 degrees North America, 2.3 Japan, 6.4 Oceania, 7.5 Europe, measured as the larger of the
latitude and longitude differences); no training gauge lies within 0.5 degrees of a test gauge in any split. Effective
training sizes after the joined-hours minimum are 592 (Japan), 587 (Europe), 606 (North America) and 785 (Oceania).

Rotation test sets contain every gauge of the region that passes the joined-hours minimum, including lake, river and
inner-estuary gauges that the marine deployment set excludes: 80 of the 214 evaluated North American gauges, 7 of 231
European and 1 of 111 Oceanian. The classification and its rule are in `catalog/rotation_test_domain.csv` (Great Lakes
and connecting channels, St Lawrence above the head of tide, Columbia River above the estuary, Sacramento-San Joaquin
Delta, Elbe above Cuxhaven, Loire above Saint-Nazaire, Rhine-Meuse tidal rivers, Gippsland Lakes). The rotation splits
predate the final 0.02-degree dedupe pass, so 16 cross-provider pairs within 2 km remain inside the test sets (Europe 7,
North America 2, Oceania 7); the manuscript reports the effect of both facts on the rotation skills. Oceania was trained
three times with the same pipeline; the manuscript reports the median run, whose checkpoint ships here with the other two.

## Ladder and ablation splits (`exp_split_g{64..640}.csv`, `g640marine`, `marine_abl`, `n298v`, `eu299v`, `c298strat`, `c298usjp`)

Nested subsets of the deployment training pool, country-stratified so that regime composition stays approximately
proportional at every size (effective sizes 63, 126, 254, 380, 508 and 636 for 64 to 640 nominal gauges); `g640marine`
removes lake and river gauges at matched count; `n298v` is the 298-gauge training set of the factor study (Japan
withheld); `eu299v` the 299-gauge Europe-free training set of the Europe intervention; `c298strat` and `c298usjp` the
equal-count composition pair (49 countries against the United States and Japan).

## Freeze integrity

The split files ship here unchanged. The commit history of the working repository from which this release is built
records when each file was frozen, before the corresponding training runs; this release's history begins on 2026-07-24
with the frozen files. Both gates assert the fold sizes on every run.
