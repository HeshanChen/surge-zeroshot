"""Replacement for the GTSM delta map (gis_gtsm_delta) using the information-symmetric head-to-head
(scripts/eval_gtsm_symmetric.py). Two panels on the same 80 gauges, identical windows, 2010-2018:
(a) bulk: RMSE@8h reduction of the gauge-fed forecast vs GTSM high-passed like the target and given the same
    observation through a 25-h mean-error bias correction;
(b) extremes: peak-capture difference (ours minus physics) inside p99.9 storm windows.
-> outputs/gis_gtsm_delta_sym.{pdf,png}. Does not overwrite the old figure."""
import warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd, numpy as np

ROOT = '/Users/heshan/Desktop/surge_fm'
import sys as _sys
SUF = _sys.argv[1] if len(_sys.argv) > 1 else ''   # '_cont' = continuous windows only (audit 2)
plt.rcParams.update({'font.family': 'Helvetica', 'font.size': 7.5, 'axes.linewidth': 0.5})
OCEAN, LAND, COAST = '#eef4f9', '#f0efe9', '#b9b7b0'
sa = pd.read_csv(f'{ROOT}/catalog/static_attributes.csv').set_index('name')

def base(ax, scale='110m'):
    ax.add_feature(cfeature.OCEAN.with_scale(scale), facecolor=OCEAN, zorder=0)
    ax.add_feature(cfeature.LAND.with_scale(scale), facecolor=LAND, zorder=1)
    ax.add_feature(cfeature.COASTLINE.with_scale(scale), edgecolor=COAST, linewidth=0.35, zorder=2)
    ax.gridlines(draw_labels=False, linewidth=0.25, color='#c9cdd4', linestyle=(0, (2, 3)), zorder=3)
    ax.spines['geo'].set_edgecolor('#888888'); ax.spines['geo'].set_linewidth(0.5)

g = pd.read_csv(f'{ROOT}/outputs/eval_gtsm_symmetric{SUF}.csv')
g['lat'] = [sa.loc[n, 'lat'] for n in g.stn]; g['lon'] = [sa.loc[n, 'lon'] for n in g.stn]
g['delta_bulk'] = 100*(1 - g.rmse8_ours/g.rmse8_anch_hp_lp)
g['delta_peak'] = g.pkcap_ours - g.pkcap_anch_hp_lp
nb = int((g.delta_bulk > 0).sum()); n = len(g)
gp = g.dropna(subset=['delta_peak']); npk = int((gp.delta_peak > 0).sum()); npn = len(gp)

fig = plt.figure(figsize=(7.0, 6.6))
for k, (col, cmap, norm, lab, title, key) in enumerate([
        ('delta_bulk', 'RdYlBu_r', mcolors.TwoSlopeNorm(vmin=-30, vcenter=0, vmax=60),
         'RMSE@8h reduction vs GTSM + observation (%)',
         f'a  bulk accuracy, both sides given the t0 observation: forecast better at {nb}/{n} gauges', g),
        ('delta_peak', 'PuOr_r', mcolors.TwoSlopeNorm(vmin=-0.3, vcenter=0, vmax=0.3),
         'peak capture difference, forecast minus GTSM + observation (fraction of true peak)',
         f'b  storm peak magnitude (p99.9 windows): forecast better at {npk}/{npn} gauges', gp)]):
    ax = fig.add_subplot(2, 1, k+1, projection=ccrs.Robinson(central_longitude=11))
    base(ax); ax.set_global()
    o = key.sort_values(col)
    sc = ax.scatter(o.lon, o.lat, s=13, c=o[col], cmap=cmap, norm=norm, edgecolor='none',
                    transform=ccrs.PlateCarree(), zorder=6)
    cax = fig.add_axes([0.33, 0.555 - 0.492*k, 0.34, 0.016])
    cb = fig.colorbar(sc, cax=cax, orientation='horizontal')
    cb.set_label(lab, fontsize=6.8)
    cb.ax.tick_params(labelsize=6.3, length=2); cb.outline.set_linewidth(0.4)
    ax.set_title(title, fontsize=7.4, pad=3, loc='left')
fig.suptitle('Zero-shot forecast vs ERA5-forced GTSM reanalysis, information matched (80 gauges, identical windows, 2010-2018)',
             fontsize=7.8, y=0.985)
fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.075, hspace=0.32)
fig.savefig(f'{ROOT}/outputs/gis_gtsm_delta_sym{SUF}.pdf'); fig.savefig(f'{ROOT}/outputs/gis_gtsm_delta_sym{SUF}.png', dpi=170)
print(f'bulk: forecast better at {nb}/{n}; peak capture: forecast better at {npk}/{npn}; '
      f'median bulk delta {g.delta_bulk.median():+.1f}%, median peak delta {gp.delta_peak.median():+.3f}')
