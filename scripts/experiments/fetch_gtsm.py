"""GTSM/CoDEC physics-baseline fetch: ERA5-driven GTSM surge reanalysis (CDS, v2, 10-min),
streamed year by year; extracts the nearest GTSM output point for each of the 84 marine test
gauges, hourly-means it, appends to per-gauge parquet, deletes the bulk files.
Usage: python3 fetch_gtsm.py [y0] [y1]   (default 2000 2018)"""
import os as _os
_ROOT = _os.environ.get('SURGE_ROOT') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
import sys, os, glob, zipfile, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import cdsapi
from netCDF4 import Dataset

ROOT = _ROOT
GD = f'{ROOT}/data/raw/gtsm'
os.makedirs(f'{GD}/series', exist_ok=True)
y0, y1 = (int(sys.argv[1]) if len(sys.argv) > 1 else 2000), (int(sys.argv[2]) if len(sys.argv) > 2 else 2018)

sa = pd.read_csv(f'{ROOT}/catalog/static_attributes.csv').set_index('name')
sp = pd.read_csv(f'{ROOT}/catalog/exp_split_final.csv')
te = [n for n in sp[sp.fold == 'test'].name if n in sa.index]

# nearest GTSM point per gauge (from the probe file's coordinates)
probe = glob.glob(f'{GD}/reanalysis_surge_10min_*_v2.nc')[0]
d = Dataset(probe)
gx = np.array(d.variables['station_x_coordinate'][:]); gy = np.array(d.variables['station_y_coordinate'][:])
idx = {}
for n in te:
    la, lo = float(sa.loc[n, 'lat']), float(sa.loc[n, 'lon']); lo = lo if lo <= 180 else lo-360
    dd = (gy-la)**2 + (np.minimum(np.abs(gx-lo), 360-np.abs(gx-lo))*np.cos(np.radians(la)))**2
    idx[n] = int(dd.argmin())
km = {n: float(np.sqrt((gy[i]-sa.loc[n,'lat'])**2 + (np.minimum(abs(gx[i]-(sa.loc[n,'lon'] if sa.loc[n,'lon']<=180 else sa.loc[n,'lon']-360)),360-abs(gx[i]-(sa.loc[n,'lon'] if sa.loc[n,'lon']<=180 else sa.loc[n,'lon']-360)))*np.cos(np.radians(sa.loc[n,'lat'])))**2))*111 for n,i in idx.items()}
pd.DataFrame({'stn': list(idx), 'gtsm_idx': list(idx.values()), 'dist_km': [km[n] for n in idx]}).to_csv(f'{GD}/gtsm_mapping.csv', index=False)
print(f'mapping: {len(idx)} gauges, median dist {np.median(list(km.values())):.1f} km, max {max(km.values()):.1f} km', flush=True)

c = cdsapi.Client(quiet=True)
cols = sorted(set(idx.values()))
for year in range(y0, y1+1):
    if os.path.exists(f'{GD}/series/done_{year}'):
        print(f'{year} done, skip', flush=True); continue
    z = f'{GD}/y{year}.zip'
    try:
        c.retrieve('sis-water-level-change-timeseries-cmip6',
                   dict(variable=['storm_surge_residual'], experiment=['reanalysis'],
                        temporal_aggregation=['10_min'], year=[str(year)],
                        month=[f'{m:02d}' for m in range(1, 13)], version=['v2']), z)
    except Exception as e:
        print(f'{year} FETCH FAIL: {str(e)[:100]}', flush=True); continue
    frames = []
    with zipfile.ZipFile(z) as zf:
        for nm in sorted(zf.namelist()):
            zf.extract(nm, GD)
            nc = Dataset(f'{GD}/{nm}')
            t = pd.to_datetime(np.array(nc.variables['time'][:]), unit='s', origin='1900-01-01', utc=True)
            sub = np.array(nc.variables['surge'][:, cols])
            df = pd.DataFrame(sub, index=t, columns=cols).resample('1h').mean()
            frames.append(df); nc.close(); os.remove(f'{GD}/{nm}')
    yr = pd.concat(frames).sort_index()
    yr.to_parquet(f'{GD}/series/gtsm_{year}.parquet')
    os.remove(z); open(f'{GD}/series/done_{year}', 'w').close()
    print(f'{year}: {len(yr)} hourly rows x {len(cols)} pts saved', flush=True)
print('GTSM FETCH DONE', flush=True)
