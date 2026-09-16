# v1.0.0: submission release (September 2026)

Release accompanying the manuscript *Storm surge forecasts from a global tide-gauge network* (Heshan Chen and Andrew
Kruczkiewicz, 2026), submitted to Nature Water. This is the version the manuscript's Data and code availability statement
points to; Zenodo archives it and assigns the DOI quoted there.

## Contents

- `src/`, `scripts/`: dataset recipe, training and evaluation code, figure scripts. Paths resolve from the repository root or
  from `SURGE_ROOT`.
- `catalog/`: quality-screened gauge catalogue, the pre-registered deployment split (760 training, 40 validation, 100 test
  gauges: 84 marine primary, 16 out of domain), the four leave-region-out rotation splits, the scaling-ladder splits, the
  Europe intervention and fixed-count composition splits, the rotation test-set domain classification, and the M7 earthquake
  list used by the seismic mask.
- `models/`: 15 trained checkpoints (deployment model, gauge-free variant, rotations, ladder, composition and matched-update
  runs).
- `outputs/`: per-gauge evaluation tables and logs on continuous evaluation windows (`*_cont`) and on all windows (July 2026
  preprint), block-bootstrap intervals, the sampled GEFSv12 reforecast forcing at the test gauges, training logs of the seven
  runs made after run logging was added, the data-provenance audits (dataset chain, effective fold sizes, join share, seismic
  mask annex), and the figures of record.
- `NUMBERS.md`: every manuscript number, its artifact and the script that produced it.

## Audit gates

```bash
python scripts/verify_numbers_cont.py   # manuscript, continuous windows: 168 checks
python scripts/verify_numbers.py        # July 2026 preprint, all windows: 69 checks
```

Both exit 0 at this tag.

## Changes since v1.0-manuscript-2026-09-14

- Figures of record regenerated with the manuscript's final wording (withheld regions, written-out gauge names).
- Supplementary Fig. 1 regenerated with the physical non-marine rule (head of tide at Portneuf on the St Lawrence, named
  estuary and inland-waterway gauges); the figure script carries the rule.
- `.zenodo.json` and `CITATION.cff` added; README gains a citation section.

## Data

GESLA-3 water levels and the Open-Meteo historical archive are open and are not redistributed here; the recipe rebuilds the
cleaned dataset from them. The hydrodynamic baseline (Copernicus Climate Data Store) was used for comparison only.
