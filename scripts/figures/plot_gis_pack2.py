"""GIS pack, part 2: (C) training-network density/record-length map,
(D) Port Phillip Bay regime-limit zoom (10m coastline), (E) GTSM head-to-head delta map."""
import warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd, numpy as np

ROOT = '/Users/heshan/Desktop/surge_fm'
plt.rcParams.update({'font.family': 'Helvetica', 'font.size': 7.5, 'axes.linewidth': 0.5})
OCEAN, LAND, COAST = '#eef4f9', '#f0efe9', '#b9b7b0'
import sys as _sys
SUF = _sys.argv[1] if len(_sys.argv) > 1 else ''   # '_cont' = continuous windows only (audit 2), applies to the Oceania panel
sa = pd.read_csv(f'{ROOT}/catalog/static_attributes.csv').set_index('name')

def base(ax, scale='110m'):
    ax.add_feature(cfeature.OCEAN.with_scale(scale), facecolor=OCEAN, zorder=0)
    ax.add_feature(cfeature.LAND.with_scale(scale), facecolor=LAND, zorder=1)
    ax.add_feature(cfeature.COASTLINE.with_scale(scale), edgecolor=COAST, linewidth=0.35, zorder=2)
    ax.gridlines(draw_labels=False, linewidth=0.25, color='#c9cdd4', linestyle=(0, (2, 3)), zorder=3)
    ax.spines['geo'].set_edgecolor('#888888'); ax.spines['geo'].set_linewidth(0.5)

# ---------- C: training density ----------
tr = pd.read_csv(f'{ROOT}/outputs/train_density.csv')
tr['yr'] = tr.hours/8760
fig = plt.figure(figsize=(7.0, 3.9))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson(central_longitude=11))
base(ax); ax.set_global()
o = tr.sort_values('yr')
sc = ax.scatter(o.lon, o.lat, s=10, c=o.yr, cmap='viridis',
                norm=mcolors.LogNorm(vmin=2, vmax=60), edgecolor='none',
                transform=ccrs.PlateCarree(), zorder=5, alpha=0.9)
cax = fig.add_axes([0.33, 0.10, 0.34, 0.028])
cb = fig.colorbar(sc, cax=cax, orientation='horizontal', ticks=[2, 5, 10, 20, 40, 60])
cb.ax.set_xticklabels(['2', '5', '10', '20', '40', '60'])
cb.set_label('quality-screened surge record (years)', fontsize=7)
cb.ax.tick_params(labelsize=6.5, length=2); cb.outline.set_linewidth(0.4)
ax.set_title('training network: 760 gauges, median 36 yr of surge record (forcing joins from 2000)', fontsize=7.6, pad=4)
fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.14)
fig.savefig(f'{ROOT}/outputs/gis_density.pdf'); fig.savefig(f'{ROOT}/outputs/gis_density.png', dpi=170)
plt.close(fig); print('C density done')

# ---------- D: Port Phillip zoom (two panels) ----------
oc = pd.read_csv(f'{ROOT}/outputs/eval_full_lstmq_v2rotocr2{SUF}.csv')
oc['sk'] = 100*(1 - oc.rmse_p/oc.prmse_p)
oc['lat'] = [sa.loc[n, 'lat'] for n in oc.stn]; oc['lon'] = [sa.loc[n, 'lon'] for n in oc.stn]
sub = oc[(oc.lon > 143.5) & (oc.lon < 148.4) & (oc.lat > -39.2) & (oc.lat < -37.3)].copy()
sub = sub.sort_values('lon').reset_index(drop=True)
cmap = plt.get_cmap('RdYlBu_r'); norm = mcolors.TwoSlopeNorm(vmin=-25, vcenter=0, vmax=50)
fig = plt.figure(figsize=(7.0, 3.0))
EXTS = [([143.8, 145.75, -38.95, -37.65], 'Port Phillip Bay and Western Port', 0.055, 0.60),
        ([147.45, 148.30, -38.15, -37.70], 'Gippsland Lakes', 0.665, 0.155)]
for (ext, ttl, x0, w) in EXTS:
    ax = fig.add_axes([x0, 0.13, w, 0.74], projection=ccrs.PlateCarree())
    ax.set_extent(ext, crs=ccrs.PlateCarree())
    base(ax, '10m')
    ss = sub[(sub.lon > ext[0]) & (sub.lon < ext[1]) & (sub.lat > ext[2]) & (sub.lat < ext[3])]
    ax.scatter(ss.lon, ss.lat, s=30, c=ss.sk, cmap=cmap, norm=norm, edgecolor='none',
               transform=ccrs.PlateCarree(), zorder=6)
    for k, r in ss.iterrows():
        ax.annotate(str(k+1), xy=(r.lon, r.lat), xytext=(4, 4), textcoords='offset points',
                    fontsize=6.0, color='#222222', zorder=8)
    gl = ax.gridlines(draw_labels=True, linewidth=0.25, color='#c9cdd4', linestyle=(0, (2, 3)))
    gl.top_labels = False; gl.right_labels = False
    gl.xlabel_style = {'size': 5.6}; gl.ylabel_style = {'size': 5.6}
    ax.set_title(ttl, fontsize=7.0, pad=3)
    if ttl.startswith('Port'):
        ax.text(144.93, -38.02, 'Port Phillip Bay', fontsize=6.6, style='italic', color='#728598', ha='center', zorder=5)
        ax.annotate('"The Rip" (~3 km)', xy=(144.64, -38.30), xytext=(144.05, -38.62),
                    fontsize=6.2, color='#31538f', arrowprops=dict(arrowstyle='-', color='#31538f', lw=0.6), zorder=7)
        ax.text(145.35, -38.45, 'Western\nPort', fontsize=6.0, style='italic', color='#728598', ha='center', zorder=5)
sc = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
lines = [f"{k+1}  {r.stn.split('-')[0].replace('_',' ').replace(' corio bay','').replace(' inner bullock island','').replace('melbourne ','').replace(' stony point','')}  {r.sk:+.0f}%" for k, r in sub.iterrows()]
fig.text(0.845, 0.87, '\n'.join(lines), fontsize=6.3, va='top', ha='left', linespacing=1.6)
cax = fig.add_axes([0.848, 0.13, 0.012, 0.30])
cb = fig.colorbar(sc, cax=cax); cb.set_label('pooled skill (%)', fontsize=6.2)
cb.ax.tick_params(labelsize=5.8, length=2); cb.outline.set_linewidth(0.4)
fig.suptitle('enclosed lagoons: the reproducible regime limit (Oceania rotation, median run)', fontsize=7.8, y=0.985)
fig.savefig(f'{ROOT}/outputs/gis_ppbay{SUF}.pdf'); fig.savefig(f'{ROOT}/outputs/gis_ppbay{SUF}.png', dpi=170)
plt.close(fig); print('D ppbay done')

# ---------- E: GTSM delta ----------
g = pd.read_csv(f'{ROOT}/outputs/eval_gtsm.csv')
g['delta'] = 100*(1 - g.rmse8_ours/g.rmse8_gtsm)
g['lat'] = [sa.loc[n, 'lat'] for n in g.stn]; g['lon'] = [sa.loc[n, 'lon'] for n in g.stn]
fig = plt.figure(figsize=(7.0, 3.9))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson(central_longitude=11))
base(ax); ax.set_global()
o = g.sort_values('delta')
sc = ax.scatter(o.lon, o.lat, s=13, c=o.delta, cmap='RdYlBu_r',
                norm=mcolors.TwoSlopeNorm(vmin=-30, vcenter=0, vmax=80),
                edgecolor='none', transform=ccrs.PlateCarree(), zorder=6)
cax = fig.add_axes([0.33, 0.10, 0.34, 0.028])
cb = fig.colorbar(sc, cax=cax, orientation='horizontal')
cb.set_label('RMSE@8h reduction vs GTSM hindcast (%)', fontsize=7)
cb.ax.tick_params(labelsize=6.5, length=2); cb.outline.set_linewidth(0.4)
ax.set_title('zero-shot forecast vs ERA5-forced GTSM reanalysis: 79/80 gauges improve (identical windows, 2010-2018)', fontsize=7.6, pad=4)
fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.14)
fig.savefig(f'{ROOT}/outputs/gis_gtsm_delta.pdf'); fig.savefig(f'{ROOT}/outputs/gis_gtsm_delta.png', dpi=170)
plt.close(fig); print('E gtsm delta done')
