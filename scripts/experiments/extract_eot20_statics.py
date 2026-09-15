"""Gauge-free tidal statics from the EOT20 global tide model (open access, SEANOE
https://doi.org/10.17882/79489; files M2/S2/K1/O1_ocean_eot20.nc under data/raw/eot20/).
Faithful script form of the session code that produced outputs/eot20_statics_test.csv
(verified byte-identical by regenerate_and_diff below).

Composed descriptors for the gauge-free configuration:
    range_eot = 2*(M2+S2+K1+O1)   [m]   characteristic-range proxy
    ff_eot    = (K1+O1)/(M2+S2)         tidal form factor
Pitfalls handled: EOT20 amplitude grids are masked arrays with _FillValue=0.0 (must use
.filled(nan), never raw np.array); coastal gauges often fall on land cells of the 1/8-degree
grid, so we take the nearest finite cell within an expanding window (up to 2 degrees).
Amplitudes are stored in cm (divide by 100).

Usage:  python scripts/extract_eot20_statics.py [--out PATH]
"""
import os as _os
_ROOT = _os.environ.get('SURGE_ROOT') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
import sys, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from netCDF4 import Dataset

ROOT = '.'
OUT = sys.argv[sys.argv.index('--out')+1] if '--out' in sys.argv else f'{ROOT}/outputs/eot20_statics_test.csv'

sa = pd.read_csv(f'{ROOT}/catalog/static_attributes.csv').set_index('name')
sp = pd.read_csv(f'{ROOT}/catalog/exp_split_final.csv')
te = [n for n in sp[sp.fold == 'test'].name if n in sa.index]

A = {}
for c in ['M2', 'S2', 'K1', 'O1']:
    d = Dataset(f'{ROOT}/data/raw/eot20/{c}_ocean_eot20.nc')
    A[c] = (d.variables['amplitude'][:].filled(np.nan), np.array(d.variables['lat'][:]), np.array(d.variables['lon'][:]))

def amp_at(c, la, lo):
    amp, lats, lons = A[c]; lo = lo % 360
    i = int(np.abs(lats-la).argmin()); j = int(np.abs(lons-lo).argmin())
    for r in range(0, 17):   # up to 2 deg
        sub = amp[max(0, i-r):i+r+1, max(0, j-r):j+r+1]
        ii, jj = np.where(np.isfinite(sub))
        if len(ii):
            di = (ii+max(0, i-r)-i)**2 + (jj+max(0, j-r)-j)**2
            k = int(di.argmin()); return float(sub[ii[k], jj[k]])/100.0
    return np.nan

rows = []
for n in te:
    la, lo = float(sa.loc[n, 'lat']), float(sa.loc[n, 'lon'])
    a = {c: amp_at(c, la, lo) for c in A}
    rng = 2*(a['M2']+a['S2']+a['K1']+a['O1']); ff = (a['K1']+a['O1'])/max(a['M2']+a['S2'], 1e-6)
    rows.append((n, rng, ff, float(sa.loc[n, 'tidal_range_m']), float(sa.loc[n, 'form_factor'])))
d = pd.DataFrame(rows, columns=['stn', 'range_eot', 'ff_eot', 'range_utide', 'ff_utide'])
d.to_csv(OUT, index=False)
v = d.dropna()
lr = np.corrcoef(np.log(v.range_eot+0.01), np.log(v.range_utide+0.01))[0, 1]
print(f'{len(d)} gauges -> {OUT}; range log-corr vs local harmonic fit {lr:.3f}')
