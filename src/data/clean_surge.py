"""Single-station storm-surge extraction. Validated vs NOAA (docs/data_quality_log.md).
SEGMENTED harmonic de-tide (8-yr blocks) so decadal drift in tidal amplitude/phase does NOT
bias local extreme peaks (single-fit on 101-yr Battery underestimated Sandy 2.58->1.92).
Pipeline: native gross clip -> hourly -> segmented utide de-tide (Sa/Ssa + linear trend)
          -> 30-d high-pass -> surge-level spike removal -> QC + record/gap gates."""
import os as _os
_ROOT = _os.environ.get('SURGE_ROOT') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
import numpy as np, pandas as pd
from utide import solve, reconstruct

def to_hourly(s, gross_m=10.0):
    s = s[~s.index.duplicated(keep='first')].sort_index()
    s.index = s.index.tz_localize('UTC') if s.index.tz is None else s.index.tz_convert('UTC')
    bad = ((s - float(s.median())).abs() > gross_m).fillna(False)
    return s.mask(bad).resample('1h').mean(), int(bad.sum())

def _td64(idx): return idx.tz_convert('UTC').tz_localize(None).to_numpy()
def _amp(c, nm):
    n = list(c['name']); return float(c['A'][n.index(nm)]) if nm in n else None

def _segmented_tide(wl, lat, seg=8):
    y0, y1 = int(wl.index.year.min()), int(wl.index.year.max())
    tide = pd.Series(index=wl.index, dtype=float); n = 0; m2 = None; ys = y0
    while ys <= y1:
        ye = min(ys+seg-1, y1); blk = wl.loc[str(ys):str(ye)].dropna(); ys = ye+1
        if len(blk) < 180*24: continue                         # need >= ~0.5yr to fit a block
        c = solve(_td64(blk.index), blk.values, lat=lat, method='ols', conf_int='none', trend=True, verbose=False)
        tide.loc[blk.index] = reconstruct(_td64(blk.index), c, verbose=False).h; n += 1
        if m2 is None: m2 = _amp(c, 'M2')
    return tide, n, m2

def compute_storm_surge(water_level, lat, min_years=1.0, max_gap_frac=0.5,
                        highpass_days=30, gross_m=10.0, spike_mad=8.0, seg_years=8, causal=False):
    qc = {'status': 'ok'}
    wl, n_gross = to_hourly(water_level, gross_m=gross_m); qc['n_gross_removed'] = n_gross
    valid = wl.notna(); qc.update(span_years=round((wl.index[-1]-wl.index[0]).days/365.25,2),
                                  gap_frac=round(float(1-valid.mean()),3), n_hourly=int(len(wl)))
    if valid.sum() < min_years*365.25*24:
        qc.update(status='rejected', reason=f'<{min_years}yr valid'); return None, qc
    if (1-valid.mean()) > max_gap_frac:
        qc.update(status='rejected', reason=f"gap {qc['gap_frac']}>{max_gap_frac}"); return None, qc
    tide, n_seg, m2 = _segmented_tide(wl, lat, seg=seg_years)
    qc['n_tide_segments'] = n_seg; qc['M2_cm'] = round(m2*100,1) if m2 else None
    resid = wl - tide
    base = resid.rolling(highpass_days*24, center=(not causal), min_periods=highpass_days*12).mean()
    surge = resid - base
    dev = surge - surge.rolling(25, center=(not causal), min_periods=5).median()
    mad = dev.abs().rolling(25, center=(not causal), min_periods=5).median()
    spk = (dev.abs() > spike_mad*1.4826*mad).fillna(False)
    qc['n_spike_removed'] = int(spk.sum()); surge[spk] = np.nan
    s = surge.dropna(); qc.update(mean_cm=round(float(s.mean())*100,2), std_cm=round(float(s.std())*100,2))
    try:
        chk = s[s.index >= s.index.max()-pd.Timedelta(days=1825)]    # M2-leak check on recent <=5yr (speed)
        c2 = solve(_td64(chk.index), chk.values, lat=lat, method='ols', conf_int='none', trend=False, verbose=False)
        qc['M2_leak_cm'] = round(_amp(c2,'M2')*100,2) if _amp(c2,'M2') is not None else None
    except Exception: qc['M2_leak_cm'] = None
    lf = s.rolling(720, center=True, min_periods=360).mean(); qc['lowfreq_frac_pct'] = round(float(lf.var()/s.var())*100,1)
    return surge, qc
