"""Figure for the season-scale real-forecast check: skill against persistence by lead, full distribution and p99.9 storm
windows, for the reanalysis-forced (perfect-prognosis) and GEFSv12-forced runs on identical windows.
Reads outputs/eval_gefs_season{tag}_perlead.csv -> outputs/gefs_season_perlead{tag}.{pdf,png}
Usage: python3 scripts/plot_gefs_season.py [tag]"""
import sys, pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
sys.path.insert(0, '/Users/heshan/Desktop/surge_fm/scripts')
from pubstyle import apply, save_pub, PAL, W1
import matplotlib.pyplot as plt
ROOT = '/Users/heshan/Desktop/surge_fm'; TAG = sys.argv[1] if len(sys.argv) > 1 else ''
apply()
d = pd.read_csv(f'{ROOT}/outputs/eval_gefs_season{TAG}_perlead.csv')
fig, axes = plt.subplots(1, 2, figsize=(W1, 2.2), sharey=True)
for ax, (a, b, ttl) in zip(axes, (('skill_era5', 'skill_gefs', 'full distribution'), ('storm_skill_era5', 'storm_skill_gefs', 'p99.9 storm windows'))):
    ax.plot(d.lead, d[a], '-', color=PAL['blue'], lw=1.3, label='reanalysis forcing (perfect prognosis)')
    ax.plot(d.lead, d[b], '-', color='#D55E00', lw=1.3, label='GEFSv12 reforecast forcing')
    ax.axhline(0, color='#999999', lw=0.7, ls=(0, (4, 3)))
    ax.set_xticks([1, 6, 12, 24, 36, 48]); ax.set_xlabel('lead time (h)'); ax.set_title(ttl, fontsize=7.5); ax.grid(alpha=0.3, lw=0.4)
axes[0].set_ylabel('skill vs persistence (%)'); axes[1].legend(loc='lower right', fontsize=5.8, handlelength=1.4, framealpha=0.9)
axes[0].text(0.0, 1.06, 'a', transform=axes[0].transAxes, fontsize=10, fontweight='bold'); axes[1].text(0.0, 1.06, 'b', transform=axes[1].transAxes, fontsize=10, fontweight='bold')
fig.tight_layout(); save_pub(fig, f'{ROOT}/outputs/gefs_season_perlead{TAG}')
print(f'wrote outputs/gefs_season_perlead{TAG}: full L8 rean {d.skill_era5[7]:+.0f}% gefs {d.skill_gefs[7]:+.0f}% | L48 rean {d.skill_era5[47]:+.0f}% gefs {d.skill_gefs[47]:+.0f}%')
