"""Extended Data figure for the season-scale real-forecast check: skill against persistence by lead, full distribution
and p99.9 storm windows, for the reanalysis-forced (perfect-prognosis) and GEFSv12-forced runs on identical windows.
Reads outputs/eval_gefs_season{tag}_perlead.csv -> outputs/gefs_season_perlead{tag}.{pdf,png}
Usage: python3 scripts/plot_gefs_season.py [tag]"""
import os as _os
_ROOT = _os.environ.get('SURGE_ROOT') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
import sys, pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
sys.path.insert(0, f'{_ROOT}/scripts')
from pubstyle import apply, save_pub, PAL, W2, panel_label
import matplotlib.pyplot as plt
ROOT = _ROOT; TAG = sys.argv[1] if len(sys.argv) > 1 else ''
apply()
d = pd.read_csv(f'{ROOT}/outputs/eval_gefs_season{TAG}_perlead.csv')
g = pd.read_csv(f'{ROOT}/outputs/eval_gefs_season{TAG}.csv')
pooled = {k: 100*(1 - g[f'rmse_p_{k}'].mean()/g.rmse_p_pers.mean()) for k in ('era5', 'gefs')}
fig, axes = plt.subplots(1, 2, figsize=(W2, 2.6), sharey=True)
panels = [('skill_era5', 'skill_gefs', f'full distribution: pooled skill {pooled["era5"]:+.0f}% and {pooled["gefs"]:+.0f}%'),
          ('storm_skill_era5', 'storm_skill_gefs', f'p99.9 storm windows: {int(g.n_storm.sum())} windows at {len(g)} gauges')]
for ax, letter, (a, b, ttl) in zip(axes, 'ab', panels):
    ax.fill_between(d.lead, d[b], d[a], where=d[a] >= d[b], color=PAL['verm'], alpha=0.18, lw=0, label='cost of real forcing')
    ax.plot(d.lead, d[a], '-', color=PAL['blue'], lw=1.2, label='reanalysis forcing (perfect prognosis)')
    ax.plot(d.lead, d[b], '-', color=PAL['verm'], lw=1.2, label='GEFSv12 reforecast forcing')
    ax.axhline(0, color=PAL['grey'], lw=0.7, ls='--')
    ax.set_xlim(1, 48); ax.set_xticks([1, 6, 12, 24, 36, 48]); ax.set_ylim(-3, 60); ax.set_yticks([0, 20, 40, 60])
    ax.set_xlabel('lead time (h)'); ax.grid(alpha=0.25, lw=0.4)
    ax.set_title(ttl, fontsize=6.5)
    panel_label(ax, letter, dx=-0.1)
axes[0].set_ylabel('skill vs persistence (%)')
axes[0].text(47.5, 1.5, 'persistence parity', fontsize=5.4, color='#777777', ha='right', va='bottom')
handles, labels = axes[0].get_legend_handles_labels()
order = [1, 2, 0]
fig.legend([handles[i] for i in order], [labels[i] for i in order], loc='upper center', bbox_to_anchor=(0.5, 1.03), ncol=3)
fig.tight_layout(rect=[0, 0, 1, 0.93])
save_pub(fig, f'{ROOT}/outputs/gefs_season_perlead{TAG}')
print(f'wrote outputs/gefs_season_perlead{TAG}: full L8 rean {d.skill_era5[7]:+.0f}% gefs {d.skill_gefs[7]:+.0f}% | L48 rean {d.skill_era5[47]:+.0f}% gefs {d.skill_gefs[47]:+.0f}%')
