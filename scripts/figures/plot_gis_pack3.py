"""GIS pack, part 3: (F) case-study storm maps — GEFS reforecast 10 m wind speed field +
MSLP contours at the forecast hour nearest each observed surge peak, gauge starred.
Panel order mirrors the case-study figure."""
import warnings, glob; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd, numpy as np, xarray as xr

ROOT = '/Users/heshan/Desktop/surge_fm'
plt.rcParams.update({'font.family': 'Helvetica', 'font.size': 7.5, 'axes.linewidth': 0.5})
sa = pd.read_csv(f'{ROOT}/catalog/static_attributes.csv').set_index('name')

CASES = [  # (fragment prefix, gauge catalog name substr, storm title, init tag, f-hour, extent half-widths)
    ('esbjerg', 'esbjerg', 'Gudrun (Jan 2005)', '2005010700', 39, (16, 10)),
    ('dunkerque', 'dunkerque', 'Xaver (Dec 2013)', '2013120400', 45, (16, 10)),
    ('anchorage2019', 'anchorage', 'Anchorage storm (Dec 2019)', '2019122400', 48, (18, 10)),
    ('morgans_point', 'morgans_point', 'Hurricane Ike (Sep 2008)', '2008091100', 48, (14, 9)),
    ('viken', 'viken', 'Xaver in the Kattegat (Dec 2013)', '2013120500', 39, (16, 10)),
    ('atlantic_city', 'atlantic_city', 'Sandy (Oct 2012)', '2012102800', 45, (14, 9)),
]

def read_field(prefix, tag, var, hh):
    f = f'{ROOT}/data/raw/gefs/{prefix}_{tag}_{var}_f{hh:03d}.grib2'
    ds = xr.open_dataset(f, engine='cfgrib', backend_kwargs={'indexpath': ''})
    v = list(ds.data_vars)[0]
    return ds[v]

fig = plt.figure(figsize=(7.0, 8.6))
for i, (prefix, gname, title, tag, hh, (hw, hh_lat)) in enumerate(CASES):
    stn = [n for n in sa.index if n.startswith(gname)][0]
    la, lo = float(sa.loc[stn, 'lat']), float(sa.loc[stn, 'lon'])
    u = read_field(prefix, tag, 'ugrd_hgt', hh); v = read_field(prefix, tag, 'vgrd_hgt', hh)
    p = read_field(prefix, tag, 'pres_msl', hh)/100.0
    spd = np.sqrt(u**2 + v**2)
    cl = 180 if abs(lo) > 150 else 0
    ax = fig.add_subplot(3, 2, i+1, projection=ccrs.PlateCarree(central_longitude=cl))
    lo_p = ((lo - cl + 180) % 360) - 180
    ax.set_extent([lo_p-hw, lo_p+hw, la-hh_lat, la+hh_lat], crs=ccrs.PlateCarree(central_longitude=cl))
    lons = spd.longitude.values; lats = spd.latitude.values
    pm = ax.pcolormesh(lons, lats, spd.values, cmap='YlGnBu', vmin=0, vmax=32,
                       transform=ccrs.PlateCarree(), zorder=1, rasterized=True, shading='auto')
    cs = ax.contour(lons, lats, p.values, levels=np.arange(940, 1052, 4), colors='white',
                    linewidths=0.45, transform=ccrs.PlateCarree(), zorder=2, alpha=0.85)
    ax.clabel(cs, cs.levels[::2], fontsize=4.6, fmt='%d', inline_spacing=1)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='#3b3b3b', linewidth=0.45, zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), edgecolor='#3b3b3b', linewidth=0.2, alpha=0.4, zorder=3)
    ax.plot(lo, la, marker='*', markersize=11, markerfacecolor='#e41a1c', markeredgecolor='none',
             transform=ccrs.PlateCarree(), zorder=6)
    valid = pd.Timestamp(f'{tag[:4]}-{tag[4:6]}-{tag[6:8]}') + pd.Timedelta(hours=hh)
    ax.set_title(f'{title} — {stn.split("-")[0].replace("_", " ")}\nGEFS +{hh} h, valid {valid:%Y-%m-%d %H} UTC', fontsize=7.0, pad=3)
    ax.text(0.0, 1.07, chr(97+i), transform=ax.transAxes, fontsize=10, fontweight='bold', va='bottom')
    ax.spines['geo'].set_edgecolor('#888888'); ax.spines['geo'].set_linewidth(0.5)
cax = fig.add_axes([0.33, 0.045, 0.34, 0.014])
cb = fig.colorbar(pm, cax=cax, orientation='horizontal')
cb.set_label('GEFS reforecast 10 m wind speed (m s$^{-1}$)', fontsize=7)
cb.ax.tick_params(labelsize=6.5, length=2); cb.outline.set_linewidth(0.4)
fig.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.085, wspace=0.08, hspace=0.28)
fig.savefig(f'{ROOT}/outputs/gis_case_maps.pdf'); fig.savefig(f'{ROOT}/outputs/gis_case_maps.png', dpi=150)
plt.close(fig); print('F case maps done')
