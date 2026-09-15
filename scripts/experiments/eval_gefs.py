"""Real-forecast forcing spot check (reviewer fire-point #2): replace ERA5 perfect-prognosis forcing
with GEFSv12 reforecast (control member) for 12 named storms at test gauges; identical model, identical
windows; compare ERA5-forcing vs GEFS-forcing vs persistence. GEFS 3-hourly -> hourly interpolation,
anchored at ERA5 analysis at issue time (raw units; the 2026-07-03 run de-normalized these already-raw
anchors a second time, see outputs/eval_gefs_v1_anchorbug.*; fixed 2026-09-04). Conventions: our wu=spd*sin(dir)=-u10, wv=-v10; msl Pa->hPa;
apcp 3h-accum -> hourly/3. -> outputs/eval_gefs.csv + .log"""
import os as _os
_ROOT = _os.environ.get('SURGE_ROOT') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
import sys, glob, warnings; warnings.filterwarnings('ignore')
import torch, pandas as pd, numpy as np, xarray as xr
sys.path.insert(0, f'{_ROOT}/src')
from models.baseline_lstm import GlobalLSTM
ROOT = _ROOT
Tctx, W, H = 208, 256, 48

sa = pd.read_csv(f'{ROOT}/catalog/static_attributes.csv').set_index('name')
sp = pd.read_csv(f'{ROOT}/catalog/exp_split_final.csv')
tr = [n for n in sp[sp.fold == 'train'].name if n in sa.index]
import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument('--inits', default='', help='optional CSV stn,init (YYYYMMDD00) overriding the default issue time (peak-36h floored to 00Z) per storm; audit 2026-09-05')
_ap.add_argument('--tag', default='', help='output suffix: outputs/eval_gefs{tag}.csv/.log')
_args = _ap.parse_args()
ev = pd.read_csv(f'{ROOT}/outputs/gfs_events.csv', parse_dates=['peak_time'])
INITS = dict(pd.read_csv(f'{ROOT}/{_args.inits}', dtype=str).values) if _args.inits else {}
def sfeat(n):
    r = sa.loc[n]; la, lo = np.radians(float(r.lat)), np.radians(float(r.lon))
    return np.array([np.cos(la)*np.cos(lo), np.cos(la)*np.sin(lo), np.sin(la), float(r.tidal_range_m), float(r.form_factor)], dtype='float32')
arr = np.stack([sfeat(n) for n in tr]); smu = arr.mean(0); ssd = arr.std(0)+1e-6
m = GlobalLSTM(n_out=3); m.load_state_dict(torch.load(f'{ROOT}/models/deploy_760_best.pt', map_location='cpu')); m.eval()

from data.dataset_v0 import seismic_mask
def load_with_time(name):
    s = pd.read_parquet(f'{ROOT}/data/processed/{name}.parquet')['surge']; s = s.where(s.abs() < 4)
    e = pd.read_parquet(f'{ROOT}/data/raw/era5/{name}.parquet')
    if e.index.tz is None: e.index = e.index.tz_localize('UTC')
    for c in ['wind_speed_10m','wind_direction_10m','pressure_msl','precipitation']:
        e[c] = pd.to_numeric(e[c], errors='coerce')
    spd = e['wind_speed_10m'].values; dd = np.deg2rad(e['wind_direction_10m'].values)
    e2 = pd.DataFrame({'wu':spd*np.sin(dd),'wv':spd*np.cos(dd),'mslp':e['pressure_msl'].values,'precip':e['precipitation'].values}, index=e.index)
    df = pd.DataFrame({'surge':s}).join(e2, how='inner').dropna()
    r = sa.loc[name]; return df[~seismic_mask(float(r.lat), float(r.lon), df.index)]

def gefs_point(stn_short, tag, la, lo):
    """hourly forcing frame t0+1..t0+48 from GEFS fragments + ERA5 anchor at t0"""
    lo360 = lo % 360
    out = {}
    for var, col in [('ugrd_hgt','u'),('vgrd_hgt','v'),('pres_msl','msl'),('apcp_sfc','tp')]:
        vals = {}
        for f in sorted(glob.glob(f'{ROOT}/data/raw/gefs/{stn_short}_{tag}_{var}_f*.grib2')):
            hh = int(f.split('_f')[-1][:3])
            ds = xr.open_dataset(f, engine='cfgrib', backend_kwargs={'indexpath':''})
            v = list(ds.data_vars)[0]
            vals[hh] = float(ds[v].sel(latitude=la, longitude=lo360, method='nearest').values)
            ds.close()
        out[var] = vals
    return out

rows = []
for _, r in ev.iterrows():
    name = r.stn; short = name.split('-')[0]
    init = (r.peak_time - pd.Timedelta(hours=36)).floor('D')
    if short in INITS: init = pd.Timestamp(pd.to_datetime(INITS[short], format='%Y%m%d%H')).tz_localize('UTC')
    tag = init.strftime('%Y%m%d00'); peak_lead_true = (r.peak_time - init).total_seconds()/3600
    df = load_with_time(name)
    t0 = init
    if t0 not in df.index:
        cand = df.index[df.index <= t0]
        if not len(cand): print(f'{short}: no context'); continue
        t0 = cand[-1]
    rng = pd.date_range(t0-pd.Timedelta(hours=Tctx-1), t0+pd.Timedelta(hours=H), freq='1h')
    seg = df.reindex(rng)
    cov = float(seg.notna().all(1).mean())
    if cov < 0.9: print(f'{short}: coverage {cov:.2f} too low'); continue
    seg = seg.interpolate(limit=6).ffill().bfill()
    a = seg.values.astype('float32'); tstd = float(df.surge.std())+1e-6
    fmu = df.iloc[:, 1:].mean(0).values; fsd = df.iloc[:, 1:].std(0).values+1e-6
    ctxa = a.copy(); ctxa[:, 1:] = (ctxa[:, 1:]-fmu)/fsd
    mu = ctxa[:Tctx, 0].mean(); sd = ctxa[:Tctx, 0].std()+1e-6
    ctx = np.concatenate([((ctxa[:Tctx, 0]-mu)/sd)[:, None], ctxa[:Tctx, 1:]], 1).T[None]
    sfv = (sfeat(name)-smu)/ssd
    anc = a[Tctx-1, 0]/tstd
    tru = a[Tctx:, 0]*100
    # ERA5-forcing run
    ffe = ctxa[Tctx:, 1:].T[None]
    # GEFS-forcing run
    try:
        g = gefs_point(short, tag, float(sa.loc[name,'lat']), float(sa.loc[name,'lon']))
        hh3 = sorted(g['pres_msl'])
        t3 = [t0 + pd.Timedelta(hours=h) for h in hh3]
        hourly = pd.date_range(t0+pd.Timedelta(hours=1), t0+pd.Timedelta(hours=H), freq='1h')
        def interp(vals, anchor0):
            s3 = pd.Series([anchor0]+[vals[h] for h in hh3], index=[t0]+t3)
            return s3.reindex(s3.index.union(hourly)).interpolate('time').reindex(hourly).values
        wu = interp({h: -g['ugrd_hgt'][h] for h in hh3}, a[Tctx-1, 1])
        wv = interp({h: -g['vgrd_hgt'][h] for h in hh3}, a[Tctx-1, 2])
        ms = interp({h: g['pres_msl'][h]/100.0 for h in hh3}, a[Tctx-1, 3])
        # reforecast APCP records alternate 0-3, 0-6, 6-9, 6-12, ...: the 6-hourly records are 6-h accumulations that
        # include the preceding 3-h record, so rebuild 3-h buckets before converting to an hourly rate (fix 2026-09-05)
        acc = g['apcp_sfc']; tp3 = {h: max((acc[h] - acc[h-3]) if (h % 6 == 0 and (h-3) in acc) else acc[h], 0)/3.0 for h in hh3}
        tp = interp(tp3, a[Tctx-1, 4])
        Fg = np.stack([wu, wv, ms, tp], 1)
        ffg = ((Fg-fmu)/fsd).T[None].astype('float32')
    except Exception as e:
        print(f'{short}: GEFS build fail {str(e)[:60]}'); continue
    with torch.no_grad():
        _, oe = m.predict_window(torch.tensor(ctx).float(), torch.tensor(ffe).float(), H, torch.tensor(sfv)[None].float(), torch.tensor([anc]).float())
        _, og = m.predict_window(torch.tensor(ctx).float(), torch.tensor(ffg).float(), H, torch.tensor(sfv)[None].float(), torch.tensor([anc]).float())
    pe = oe[0,:,0].numpy()*tstd*100; pg = og[0,:,0].numpy()*tstd*100
    qe = oe[0,:,2].numpy()*tstd*100; qg = og[0,:,2].numpy()*tstd*100
    per = a[Tctx-1, 0]*100
    ip = int(tru.argmax())
    rows.append(dict(stn=short, init=tag, peak_cm=float(tru.max()), peak_lead=ip+1, catalog_peak_cm=float(r.peak_cm), catalog_peak_lead_h=float(peak_lead_true), peak_in_window=bool(peak_lead_true <= 48),
        rmse_era5=float(np.sqrt(((pe-tru)**2).mean())), rmse_gefs=float(np.sqrt(((pg-tru)**2).mean())),
        rmse_per=float(np.sqrt(((per-tru)**2).mean())),
        cap_era5=float(pe[ip]/tru[ip]), cap_gefs=float(pg[ip]/tru[ip]),
        q99cap_era5=float(qe[ip]/tru[ip]), q99cap_gefs=float(qg[ip]/tru[ip])))
    print(f"{short:16s} peak {tru.max():5.0f}cm@+{ip+1:２d}h | RMSE era5 {rows[-1]['rmse_era5']:.1f} gefs {rows[-1]['rmse_gefs']:.1f} per {rows[-1]['rmse_per']:.1f} | cap {rows[-1]['cap_era5']:.2f}/{rows[-1]['cap_gefs']:.2f} | q99 {rows[-1]['q99cap_era5']:.2f}/{rows[-1]['q99cap_gefs']:.2f}", flush=True)

d = pd.DataFrame(rows)
if not len(d): print('NO EVENTS EVALUATED'); raise SystemExit
d.to_csv(f'{ROOT}/outputs/eval_gefs{_args.tag}.csv', index=False)
out = [f'GEFS REAL-FORECAST FORCING SPOT CHECK (n={len(d)} storms)',
       f'window RMSE: era5-forcing {d.rmse_era5.mean():.1f} | gefs-forcing {d.rmse_gefs.mean():.1f} | persistence {d.rmse_per.mean():.1f} cm',
       f'peak capture: era5 {d.cap_era5.mean():.2f} | gefs {d.cap_gefs.mean():.2f}',
       f'q99 envelope at peak: era5 {d.q99cap_era5.mean():.2f} | gefs {d.q99cap_gefs.mean():.2f}',
       f'median RMSE degradation era5->gefs: {((d.rmse_gefs-d.rmse_era5)/d.rmse_era5).median()*100:+.0f}%',
       f'storms where gefs-forcing still beats persistence: {int((d.rmse_gefs<d.rmse_per).sum())}/{len(d)}']
out.append(f'storms whose catalogued peak lies inside the 48-h window: {int(d.peak_in_window.sum())}/{len(d)}')
txt = '\n'.join(out); open(f'{ROOT}/outputs/eval_gefs{_args.tag}.log','w').write(txt+'\n'); print(txt)
