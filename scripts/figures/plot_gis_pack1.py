"""GIS pack, part 1: (A) global deployment skill map, (B) 2x2 rotation transfer maps.
Robinson projection, Natural Earth base, diverging skill colormap centered at 0."""
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
sa = pd.read_csv(f'{ROOT}/catalog/static_attributes.csv').set_index('name')

def base(ax):
    ax.add_feature(cfeature.OCEAN.with_scale('110m'), facecolor=OCEAN, zorder=0)
    ax.add_feature(cfeature.LAND.with_scale('110m'), facecolor=LAND, zorder=1)
    ax.add_feature(cfeature.COASTLINE.with_scale('110m'), edgecolor=COAST, linewidth=0.35, zorder=2)
    gl = ax.gridlines(draw_labels=False, linewidth=0.25, color='#c9cdd4', linestyle=(0, (2, 3)), zorder=3)
    ax.set_global(); ax.spines['geo'].set_edgecolor('#888888'); ax.spines['geo'].set_linewidth(0.5)

def skill(df):
    return 100*(1 - df.rmse_p/df.prmse_p)

cmap = plt.get_cmap('RdYlBu_r')
norm = mcolors.TwoSlopeNorm(vmin=-25, vcenter=0, vmax=50)

# ---------- A: deployment skill map ----------
d = pd.read_csv(f'{ROOT}/outputs/eval_full_lstmq_v2final.csv')
d['sk'] = skill(d)
d['lat'] = [sa.loc[n, 'lat'] for n in d.stn]; d['lon'] = [sa.loc[n, 'lon'] for n in d.stn]
fig = plt.figure(figsize=(7.0, 3.9))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson(central_longitude=11))
base(ax)
tr = pd.read_csv(f'{ROOT}/outputs/train_density.csv')
ax.scatter(tr.lon, tr.lat, s=4.5, c='#8093a6', alpha=0.6, lw=0, transform=ccrs.PlateCarree(), zorder=4, label='training (760)')
o = d.sort_values('sk')
sc = ax.scatter(o.lon, o.lat, s=13, c=o.sk, cmap=cmap, norm=norm, edgecolor='none',
                transform=ccrs.PlateCarree(), zorder=6)
cax = fig.add_axes([0.33, 0.10, 0.34, 0.028])
cb = fig.colorbar(sc, cax=cax, orientation='horizontal')
cb.set_label('pooled skill vs persistence (%)', fontsize=7)
cb.ax.tick_params(labelsize=6.5, length=2); cb.outline.set_linewidth(0.4)
ax.set_title('zero-shot deployment: 84 pre-registered marine gauges, 82/84 beat persistence (grey: 760 training gauges)', fontsize=7.6, pad=4)
fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.14)
fig.savefig(f'{ROOT}/outputs/gis_skillmap.pdf'); fig.savefig(f'{ROOT}/outputs/gis_skillmap.png', dpi=170)
plt.close(fig); print('A skillmap done')

# ---------- B: rotation 2x2 regional zooms ----------
ROTS = [('v2rotjp', 'Japan + Pacific', '+35% pooled', True, 'a'),
        ('v2roteu', 'Europe', '+13% pooled', False, 'b'),
        ('v2rotna', 'North America', '+12% pooled', False, 'c'),
        ('v2rotocr2', 'Oceania', '+11% pooled (median of 3)', True, 'd')]
tr = pd.read_csv(f'{ROOT}/outputs/train_density.csv')
fig = plt.figure(figsize=(7.0, 5.8))
for i, (tag, name, sub, pacific, letter) in enumerate(ROTS):
    r0 = pd.read_csv(f'{ROOT}/outputs/eval_full_lstmq_{tag}.csv')
    la = np.array([sa.loc[n, 'lat'] for n in r0.stn]); lo = np.array([sa.loc[n, 'lon'] for n in r0.stn])
    cl = 180 if pacific else 0
    lo_p = ((lo - cl + 180) % 360) - 180
    pad_x = max(3, (lo_p.max()-lo_p.min())*0.06); pad_y = max(3, (la.max()-la.min())*0.08)
    ax = fig.add_subplot(2, 2, i+1, projection=ccrs.PlateCarree(central_longitude=cl))
    ax.set_extent([lo_p.min()-pad_x, lo_p.max()+pad_x, la.min()-pad_y, la.max()+pad_y], crs=ccrs.PlateCarree(central_longitude=cl))
    ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor=OCEAN, zorder=0)
    ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor=LAND, zorder=1)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor=COAST, linewidth=0.3, zorder=2)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), edgecolor='#d8d6cf', linewidth=0.2, zorder=2)
    gl = ax.gridlines(draw_labels=False, linewidth=0.25, color='#c9cdd4', linestyle=(0, (2, 3)), zorder=3)
    r = pd.read_csv(f'{ROOT}/outputs/eval_full_lstmq_{tag}.csv')
    r['sk'] = skill(r)
    r['lat'] = [sa.loc[n, 'lat'] for n in r.stn]; r['lon'] = [sa.loc[n, 'lon'] for n in r.stn]
    r = r.sort_values('sk')
    sc = ax.scatter(r.lon, r.lat, s=9, c=r.sk, cmap=cmap, norm=norm, edgecolor='none',
                    transform=ccrs.PlateCarree(), zorder=6)
    nneg = int((r.sk < 0).sum())
    ax.set_title(f'held out: {name}  ({sub}, $n$={len(r)}, {nneg} negative)', fontsize=7.4, pad=3)
    ax.text(0.0, 1.05, letter, transform=ax.transAxes, fontsize=10, fontweight='bold', va='bottom')
    ax.spines['geo'].set_edgecolor('#888888'); ax.spines['geo'].set_linewidth(0.5)
    if tag == 'v2rotocr2':
        ax.annotate('Port Phillip Bay /\nGippsland Lakes', xy=(145.2, -38.4), xytext=(120, -44),
                    transform=ccrs.PlateCarree(), fontsize=6.4, color='#31538f',
                    arrowprops=dict(arrowstyle='-', color='#31538f', lw=0.6))
cax = fig.add_axes([0.33, 0.062, 0.34, 0.02])
cb = fig.colorbar(sc, cax=cax, orientation='horizontal')
cb.set_label('per-gauge pooled skill vs persistence (%)', fontsize=7)
cb.ax.tick_params(labelsize=6.5, length=2); cb.outline.set_linewidth(0.4)
fig.subplots_adjust(left=0.015, right=0.985, top=0.955, bottom=0.125, wspace=0.06, hspace=0.16)
fig.savefig(f'{ROOT}/outputs/gis_rotations.pdf'); fig.savefig(f'{ROOT}/outputs/gis_rotations.png', dpi=170)
plt.close(fig); print('B rotations done')
