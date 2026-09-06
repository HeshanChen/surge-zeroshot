"""Extended Data figure for the season-scale real-forecast check: skill against persistence by lead, full distribution
and p99.9 storm windows, for the reanalysis-forced (perfect-prognosis) and GEFSv12-forced runs on identical windows.
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
g = pd.read_csv(f'{ROOT}/outputs/eval_gefs_season{TAG}.csv')
pooled = {k: 100*(1 - g[f'rmse_p_{k}'].mean()/g.rmse_p_pers.mean()) for k in ('era5', 'gefs')}
C_REAN, C_GEFS, C_GAP = PAL['blue'], '#D55E00', '#f2c9b0'

fig, axes = plt.subplots(1, 2, figsize=(W1, 2.55), sharey=True)
panels = [('skill_era5', 'skill_gefs', 'Full distribution', f'pooled skill {pooled["era5"]:+.0f}% and {pooled["gefs"]:+.0f}%'),
          ('storm_skill_era5', 'storm_skill_gefs', 'p99.9 storm windows', f'{int(g.n_storm.sum())} windows at {len(g)} gauges')]
for ax, letter, (a, b, ttl, sub) in zip(axes, 'ab', panels):
    ax.fill_between(d.lead, d[b], d[a], where=d[a] >= d[b], color=C_GAP, alpha=0.7, lw=0, label='cost of real forcing')
    ax.plot(d.lead, d[a], '-', color=C_REAN, lw=1.15, label='reanalysis forcing (perfect prognosis)')
    ax.plot(d.lead, d[b], '-', color=C_GEFS, lw=1.15, label='GEFSv12 reforecast forcing')
    ax.axhline(0, color='#8a8a8a', lw=0.7, ls=(0, (4, 3)))
    ax.set_xlim(1, 48); ax.set_xticks([1, 6, 12, 24, 36, 48]); ax.set_ylim(-3, 60); ax.set_yticks([0, 20, 40, 60])
    ax.set_xlabel('lead time (h)'); ax.grid(alpha=0.3, lw=0.4)
    ax.set_title(ttl, fontsize=7.6, pad=9); ax.text(0.5, 1.005, sub, transform=ax.transAxes, ha='center', va='bottom', fontsize=6.0, color='#555555')
    ax.text(-0.02, 1.16, letter, transform=ax.transAxes, fontsize=10, fontweight='bold', va='top', ha='right')
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
axes[0].set_ylabel('skill vs persistence (%)')
axes[0].text(47.5, 1.5, 'persistence parity', fontsize=5.4, color='#777777', ha='right', va='bottom')
handles, labels = axes[0].get_legend_handles_labels()
order = [1, 2, 0]
fig.legend([handles[i] for i in order], [labels[i] for i in order], loc='lower center', ncol=3, fontsize=6.2, frameon=False,
           handlelength=1.6, columnspacing=1.6, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.09, 1, 1))
save_pub(fig, f'{ROOT}/outputs/gefs_season_perlead{TAG}')
print(f'wrote outputs/gefs_season_perlead{TAG}: full L8 rean {d.skill_era5[7]:+.0f}% gefs {d.skill_gefs[7]:+.0f}% | L48 rean {d.skill_era5[47]:+.0f}% gefs {d.skill_gefs[47]:+.0f}%')
