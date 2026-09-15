"""Gauge-free local surge scale: predict per-gauge sigma_s (std of the surge record) from
covariates available without any local water-level instrument. Faithful script form of the
session code that produced outputs/sigma_hat_test.csv and outputs/sigma_hat_eot_test.csv
(GradientBoostingRegressor with random_state=0 -> deterministic; regression-checked).

Eight features: spherical position (3), tidal range, form factor, ERA5 wind-speed std,
wind-speed q99, MSLP std. Target: log sigma_s. Trained on the 760 training gauges
(train side always uses local harmonic statics); at test time either
  (a) default: local utide statics                -> sigma_hat_test.csv        (ladder rung 0.611)
  (b) --eot:   EOT20 open-product statics + NaN fallback -> sigma_hat_eot_test.csv (ladder rung 0.606)

Usage:  python scripts/fit_sigma_hat.py [--eot] [--out PATH]
"""
import os as _os
_ROOT = _os.environ.get('SURGE_ROOT') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
import sys, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

ROOT = '.'
USE_EOT = '--eot' in sys.argv
OUT = sys.argv[sys.argv.index('--out')+1] if '--out' in sys.argv else \
      (f'{ROOT}/outputs/sigma_hat_eot_test.csv' if USE_EOT else f'{ROOT}/outputs/sigma_hat_test.csv')

sa = pd.read_csv(f'{ROOT}/catalog/static_attributes.csv').set_index('name')
sp = pd.read_csv(f'{ROOT}/catalog/exp_split_final.csv')
tr = [n for n in sp[sp.fold == 'train'].name if n in sa.index]
te = [n for n in sp[sp.fold == 'test'].name if n in sa.index]
eot = pd.read_csv(f'{ROOT}/outputs/eot20_statics_test.csv').set_index('stn') if USE_EOT else None

def feats(n, rng=None, ff=None):
    r = sa.loc[n]; la, lo = np.radians(float(r.lat)), np.radians(float(r.lon))
    e = pd.read_parquet(f'{ROOT}/data/raw/era5/{n}.parquet')
    ws = pd.to_numeric(e['wind_speed_10m'], errors='coerce'); mp = pd.to_numeric(e['pressure_msl'], errors='coerce')
    RNG = float(r.tidal_range_m) if rng is None else rng
    FF = float(r.form_factor) if ff is None else ff
    return [np.cos(la)*np.cos(lo), np.cos(la)*np.sin(lo), np.sin(la), RNG, FF,
            float(ws.std()), float(ws.quantile(0.99)), float(mp.std())]

def sigma(n):
    s = pd.read_parquet(f'{ROOT}/data/processed/{n}.parquet')['surge']
    s = s.where(s.abs() < 4).dropna(); return float(s.std())

X, y = [], []
for n in tr:
    try:
        f = feats(n); sg = sigma(n)
        if np.all(np.isfinite(f)) and sg > 0: X.append(f); y.append(np.log(sg))
    except Exception: pass
g = GradientBoostingRegressor(random_state=0).fit(np.array(X), np.array(y))

rows = []
for n in te:
    if USE_EOT:
        rng = float(eot.loc[n, 'range_eot']) if n in eot.index and np.isfinite(eot.loc[n, 'range_eot']) else float(sa.loc[n, 'tidal_range_m'])
        ff = float(eot.loc[n, 'ff_eot']) if n in eot.index and np.isfinite(eot.loc[n, 'ff_eot']) else float(sa.loc[n, 'form_factor'])
        rows.append((n, float(np.exp(g.predict(np.array([feats(n, rng, ff)]))[0])), rng, ff))
    else:
        rows.append((n, float(np.exp(g.predict(np.array([feats(n)]))[0])), sigma(n)))
cols = ['stn', 'sigma_hat', 'range_eot', 'ff_eot'] if USE_EOT else ['stn', 'sigma_hat', 'sigma_true']
d = pd.DataFrame(rows, columns=cols)
d.to_csv(OUT, index=False)
if not USE_EOT:
    lc = np.corrcoef(np.log(d.sigma_hat), np.log(d.sigma_true))[0, 1]
    print(f'{len(d)} gauges -> {OUT}; log-corr {lc:.3f}, median |err| {float((abs(d.sigma_hat/d.sigma_true-1)).median())*100:.0f}%')
else:
    print(f'{len(d)} gauges -> {OUT}')
