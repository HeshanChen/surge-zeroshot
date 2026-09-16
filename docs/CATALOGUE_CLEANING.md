# Catalogue cleaning: the counts quoted in Methods and where they come from

Methods ("Cleaning beyond the standard pipeline") states that the documented catalogue cleaning removed 74 lake and
river stations, 23 gross-range failures and 321 cross-provider duplicates (377 sub-2-km pairs, 42% of the candidate
stations), and that a final 0.02-degree-box dedupe in the split builder catches 26 residual pairs.

## Chain

| Step | Stations | Record |
|---|---|---|
| GESLA-3 files scanned | 5,119 | `catalog/station_index.csv` in the working repository (not needed here) |
| Selected for processing (at least 15 years, dense record) | 1,733 | `catalog/processing_qc.csv` (1,610 ok, 122 rejected, 1 error) |
| Candidate stations after processing | 1,585 | working-repository data-quality log, 2026-06-26 |
| Co-location scan on the 1,585 candidates | 377 pairs closer than 2 km, involving 671 stations (42%) | same log |
| Removed: lake and river domain | 74 | same log |
| Removed: gross-range failures (max abs surge above 5 m) | 23 | same log |
| Dedupe input (candidates minus the two removals) | 1,490 | `catalog/station_clusters.csv` (1,490 rows) |
| Co-located clusters within 2 km, cleanest twin kept per cluster | 1,169 clusters, 321 stations removed | `catalog/station_clusters.csv`: 1,490 rows, 1,169 distinct `cluster` values |
| Clean independent stations | 1,169 | `catalog/clean_stations.csv` |
| Residual 0.02-degree pairs among the 1,054 joined gauges (split builder) | 26 pairs, 23 dropped members with a partner in the pool | `scripts/verify_numbers_cont.py`, section 9 |

## Recompute

```python
import pandas as pd
c = pd.read_csv('catalog/station_clusters.csv')
print(len(c), c.cluster.nunique(), len(c) - c.cluster.nunique())   # 1490 1169 321
```

The 377-pair and 42% figures were measured on the 1,585 candidates before the domain and range removals; on the 1,490
dedupe inputs the same 2-km scan gives 347 pairs involving 615 stations (41%). The 321 removed duplicates are the
difference between rows and clusters in the cluster map. Sources were prioritized so that the cleanest twin was kept
(CMEMS deprioritized where a national provider serves the same gauge).
