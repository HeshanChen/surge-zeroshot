"""Season-scale real-forecast evaluation: the delivered model driven by GEFSv12 reforecast control forcing at every
00 UTC cycle of the requested years, at all 84 marine test gauges, against the same model with reanalysis
(perfect-prognosis) forcing and against persistence, on identical windows. Windows are the model's deployment windows
(208 h of observed context ending at the cycle time, 48-h horizon) restricted to fully continuous records; GEFS 3-hourly
values are interpolated to hourly and anchored on the reanalysis value at issue time (same construction as
eval_gefs.py), with 3-h precipitation buckets rebuilt from the reforecast's 0-3 / 0-6 / 6-9 / 6-12 ... accumulations.
Inputs: data/raw/gefs_season/{YYYYMMDD00}.npz from scripts/fetch_gefs_season.py.
Usage: python3 scripts/eval_gefs_season.py 2017 2012 [--device mps] [--tag NAME]
-> outputs/eval_gefs_season{tag}.csv (per gauge) + .log (aggregate) + _perlead.csv (skill by lead)"""
import sys, os, glob, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, torch
sys.path.insert(0, '/Users/heshan/Desktop/surge_fm/src')
from models.baseline_lstm import GlobalLSTM
from data.dataset_v0 import load_station
ROOT = '/Users/heshan/Desktop/surge_fm'
Tctx, H, W = 208, 48, 256
YEARS = [int(a) for a in sys.argv[1:] if a.isdigit() and len(a) == 4]
DEV = sys.argv[sys.argv.index('--device') + 1] if '--device' in sys.argv else 'cpu'
TAG = sys.argv[sys.argv.index('--tag') + 1] if '--tag' in sys.argv else ''
PCT = 99.9

sa = pd.read_csv(f'{ROOT}/catalog/static_attributes.csv').set_index('name')
sp = pd.read_csv(f'{ROOT}/catalog/exp_split_final.csv')
tr = [n for n in sp[sp.fold == 'train'].name if n in sa.index]
def sfeat(n):
    r = sa.loc[n]; la, lo = np.radians(float(r.lat)), np.radians(float(r.lon))
    return np.array([np.cos(la)*np.cos(lo), np.cos(la)*np.sin(lo), np.sin(la), float(r.tidal_range_m), float(r.form_factor)], dtype='float32')
arr = np.stack([sfeat(n) for n in tr]); smu = arr.mean(0); ssd = arr.std(0) + 1e-6
dev = torch.device(DEV if (DEV != 'mps' or torch.backends.mps.is_available()) else 'cpu')
m = GlobalLSTM(n_out=3); m.load_state_dict(torch.load(f'{ROOT}/outputs/baseline_lstmq_v2final_best.pt', map_location='cpu')); m.eval().to(dev)

# ---- GEFS cycles: stack all npz into arrays indexed by cycle ----
files = sorted(f for y in YEARS for f in glob.glob(f'{ROOT}/data/raw/gefs_season/{y}*.npz'))
assert files, 'no GEFS season files'
z0 = np.load(files[0], allow_pickle=True); gauges = list(z0['gauges']); hours = list(z0['hours'])
G = {k: np.stack([np.load(f)[k] for f in files]) for k in ('u', 'v', 'msl', 'apcp')}   # (ncyc, 16, 84)
inits = pd.to_datetime([os.path.basename(f)[:10] for f in files], format='%Y%m%d%H', utc=True)
init_h = np.asarray(inits.asi8) // 3_600_000_000_000            # hours since the epoch, matching load_station's time axis
# 3-h precipitation buckets: records at 6,12,...,48 are 6-h accumulations that include the preceding 3-h record
apcp3 = G['apcp'].copy()
for k, h in enumerate(hours):
    if h % 6 == 0: apcp3[:, k] = G['apcp'][:, k] - G['apcp'][:, k-1]
apcp3 = np.clip(apcp3, 0, None)
print(f'{len(files)} GEFS cycles ({inits[0].date()} .. {inits[-1].date()}), {len(gauges)} gauges', flush=True)

def hourly_from_3h(anchor, vals3):
    """anchor at lead 0 plus 16 values at leads 3..48 -> 48 hourly values by linear interpolation (per window, vectorised)"""
    x = np.concatenate([[0], hours]).astype(float); y = np.concatenate([anchor[:, None], vals3], 1)    # (n, 17)
    xq = np.arange(1, H + 1, dtype=float)
    out = np.empty((len(anchor), H), 'float32')
    for i in range(len(anchor)): out[i] = np.interp(xq, x, y[i])
    return out

LEADS = np.arange(1, H + 1)
rows = []; SE = {k: np.zeros(H) for k in ('era5', 'gefs', 'pers')}; SEs = {k: np.zeros(H) for k in SE}; N = np.zeros(H); Ns = np.zeros(H)
peaks = {'true': [], 'era5': [], 'gefs': [], 'pers': [], 'q99e': [], 'q99g': []}
cov = {'era5': [0, 0], 'gefs': [0, 0]}
for gi, name in enumerate(gauges):
    a, t = load_station(name, return_time=True)
    if a is None: continue
    tstd = float(a[:, 0].std()) + 1e-6; fmu = a[:, 1:].mean(0); fsd = a[:, 1:].std(0) + 1e-6
    thr = float(np.percentile(a[:, 0], PCT))
    pos = {int(h): i for i, h in enumerate(t)}
    # windows: context ends at the cycle time t0 (row index i0 with t[i0] == init hour); continuity of all 256 rows required
    i0s, cyc = [], []
    for c, h0 in enumerate(init_h):
        i0 = pos.get(int(h0))
        if i0 is None or i0 < Tctx - 1 or i0 + H >= len(a): continue
        if t[i0 + H] - t[i0 - Tctx + 1] != W - 1: continue
        i0s.append(i0); cyc.append(c)
    if len(i0s) < 5: continue
    i0s = np.array(i0s); cyc = np.array(cyc)
    ctx_raw = np.stack([a[i - Tctx + 1:i + 1] for i in i0s])            # (n, 208, 5)
    fut_raw = np.stack([a[i + 1:i + 1 + H] for i in i0s])               # (n, 48, 5)
    ctx = ctx_raw.copy(); ctx[:, :, 1:] = (ctx[:, :, 1:] - fmu) / fsd
    mu = ctx[:, :, 0].mean(1, keepdims=True); sd = ctx[:, :, 0].std(1, keepdims=True) + 1e-6
    ctx[:, :, 0] = (ctx[:, :, 0] - mu) / sd
    ffe = ((fut_raw[:, :, 1:] - fmu) / fsd).astype('float32')
    anc_raw = ctx_raw[:, -1, :]                                          # raw values at t0 (surge, wu, wv, mslp, precip)
    # GEFS forcing: our wu = -u10, wv = -v10 (wind-direction convention), msl Pa -> hPa, precip 3-h bucket -> hourly rate
    wu = hourly_from_3h(anc_raw[:, 1], -G['u'][cyc, :, gi]); wv = hourly_from_3h(anc_raw[:, 2], -G['v'][cyc, :, gi])
    ms = hourly_from_3h(anc_raw[:, 3], G['msl'][cyc, :, gi] / 100.0); tp = hourly_from_3h(anc_raw[:, 4], apcp3[cyc, :, gi] / 3.0)
    Fg = np.stack([wu, wv, ms, tp], 2); ffg = ((Fg - fmu) / fsd).astype('float32')
    ok = np.isfinite(ffg).all((1, 2))
    if ok.sum() < 5: continue
    ctx, ffe, ffg, fut_raw, anc_raw = ctx[ok], ffe[ok], ffg[ok], fut_raw[ok], anc_raw[ok]
    sfv = torch.tensor((sfeat(name) - smu) / ssd)[None].expand(len(ctx), -1).float().to(dev)
    with torch.no_grad():
        outs = {}
        for key, ff in (('era5', ffe), ('gefs', ffg)):
            o = []
            for j in range(0, len(ctx), 512):
                _, q = m.predict_window(torch.tensor(ctx[j:j+512]).float().transpose(1, 2).to(dev), torch.tensor(ff[j:j+512]).float().transpose(1, 2).to(dev), H,
                                        sfv[j:j+512], torch.tensor(anc_raw[j:j+512, 0] / tstd).float().to(dev))
                o.append(q.cpu().numpy())
            outs[key] = np.concatenate(o) * tstd * 100                    # (n, 48, 3) in cm
    tru = fut_raw[:, :, 0] * 100; pers = np.repeat(anc_raw[:, 0:1] * 100, H, 1)
    pe, pg = outs['era5'][:, :, 0], outs['gefs'][:, :, 0]; qe, qg = outs['era5'][:, :, 2], outs['gefs'][:, :, 2]
    storm = (fut_raw[:, :, 0] >= thr).any(1)
    for key, p in (('era5', pe), ('gefs', pg), ('pers', pers)):
        SE[key] += ((p - tru) ** 2).sum(0); SEs[key] += ((p[storm] - tru[storm]) ** 2).sum(0)
    N += len(tru); Ns += storm.sum()
    cov['era5'][0] += (tru <= qe).sum(); cov['gefs'][0] += (tru <= qg).sum(); cov['era5'][1] += tru.size; cov['gefs'][1] += tru.size
    if storm.any():
        ip = tru[storm].argmax(1); rr = np.arange(storm.sum())
        peaks['true'] += list(tru[storm][rr, ip]); peaks['era5'] += list(pe[storm][rr, ip]); peaks['gefs'] += list(pg[storm][rr, ip])
        peaks['pers'] += list(pers[storm][rr, ip]); peaks['q99e'] += list(qe[storm][rr, ip]); peaks['q99g'] += list(qg[storm][rr, ip])
    r8 = lambda p: float(np.sqrt(((p[:, 7] - tru[:, 7]) ** 2).mean()))
    rp = lambda p, y=None: float(np.sqrt(((p - (tru if y is None else y)) ** 2).mean()))
    rows.append(dict(stn=name, n_win=len(tru), n_storm=int(storm.sum()), rmse_p_era5=rp(pe), rmse_p_gefs=rp(pg), rmse_p_pers=rp(pers),
                     r8_era5=r8(pe), r8_gefs=r8(pg), r8_pers=r8(pers), r48_era5=float(np.sqrt(((pe[:, 47]-tru[:, 47])**2).mean())),
                     r48_gefs=float(np.sqrt(((pg[:, 47]-tru[:, 47])**2).mean())), r48_pers=float(np.sqrt(((pers[:, 47]-tru[:, 47])**2).mean())),
                     storm_rmse_era5=rp(pe[storm], tru[storm]) if storm.any() else np.nan, storm_rmse_gefs=rp(pg[storm], tru[storm]) if storm.any() else np.nan,
                     storm_rmse_pers=rp(pers[storm], tru[storm]) if storm.any() else np.nan))
    if (gi + 1) % 10 == 0: print(f'  ...{gi+1}/{len(gauges)} gauges', flush=True)

d = pd.DataFrame(rows); d.to_csv(f'{ROOT}/outputs/eval_gefs_season{TAG}.csv', index=False)
rm = {k: np.sqrt(SE[k] / N) for k in SE}; rs = {k: np.sqrt(SEs[k] / Ns) for k in SEs}
pl = pd.DataFrame({'lead': LEADS, 'rmse_era5': rm['era5'], 'rmse_gefs': rm['gefs'], 'rmse_pers': rm['pers'],
                   'skill_era5': 100*(1-rm['era5']/rm['pers']), 'skill_gefs': 100*(1-rm['gefs']/rm['pers']),
                   'storm_rmse_era5': rs['era5'], 'storm_rmse_gefs': rs['gefs'], 'storm_rmse_pers': rs['pers'],
                   'storm_skill_era5': 100*(1-rs['era5']/rs['pers']), 'storm_skill_gefs': 100*(1-rs['gefs']/rs['pers'])})
pl.to_csv(f'{ROOT}/outputs/eval_gefs_season{TAG}_perlead.csv', index=False)
P = {k: np.array(v) for k, v in peaks.items()}
def gsk(col_m, col_p): return 100*(1 - d[col_m].mean()/d[col_p].mean())
out = [f'SEASON-SCALE REAL-FORECAST FORCING: GEFSv12 reforecast control vs reanalysis forcing vs persistence, years {YEARS}',
       f'gauges evaluated {len(d)}; windows {int(N[0])} (continuous 256-h records at 00 UTC cycles); storm windows (p{PCT}) {int(Ns[0])}',
       f'pooled RMSE skill vs persistence (gauge-mean): reanalysis {gsk("rmse_p_era5","rmse_p_pers"):+.1f}% | GEFS {gsk("rmse_p_gefs","rmse_p_pers"):+.1f}%',
       f'8-h skill: reanalysis {gsk("r8_era5","r8_pers"):+.1f}% | GEFS {gsk("r8_gefs","r8_pers"):+.1f}%   48-h skill: reanalysis {gsk("r48_era5","r48_pers"):+.1f}% | GEFS {gsk("r48_gefs","r48_pers"):+.1f}%',
       f'gauges where GEFS-forced model beats persistence: pooled {int((d.rmse_p_gefs<d.rmse_p_pers).sum())}/{len(d)}, 8 h {int((d.r8_gefs<d.r8_pers).sum())}/{len(d)}   (reanalysis: {int((d.rmse_p_era5<d.rmse_p_pers).sum())}/{len(d)}, {int((d.r8_era5<d.r8_pers).sum())}/{len(d)})',
       f'median per-gauge pooled RMSE degradation reanalysis->GEFS: {((d.rmse_p_gefs-d.rmse_p_era5)/d.rmse_p_era5).median()*100:+.0f}%',
       'per-lead skill (hour-pooled): ' + ' | '.join(f'L{L}: rean {pl.skill_era5[L-1]:+.0f}% gefs {pl.skill_gefs[L-1]:+.0f}%' for L in (1, 4, 8, 12, 24, 36, 48)),
       'storm-window per-lead skill: ' + ' | '.join(f'L{L}: rean {pl.storm_skill_era5[L-1]:+.0f}% gefs {pl.storm_skill_gefs[L-1]:+.0f}%' for L in (4, 8, 12, 24, 48)),
       f'storm-window RMSE (all leads): reanalysis {np.sqrt((SEs["era5"]).sum()/Ns.sum()):.1f} | GEFS {np.sqrt((SEs["gefs"]).sum()/Ns.sum()):.1f} | persistence {np.sqrt((SEs["pers"]).sum()/Ns.sum()):.1f} cm',
       f'peak capture at the true peak hour (n={len(P["true"])} storm windows): reanalysis {P["era5"].sum()/P["true"].sum():.2f} | GEFS {P["gefs"].sum()/P["true"].sum():.2f} | persistence {P["pers"].sum()/P["true"].sum():.2f}',
       f'q99 above the true peak: reanalysis {100*(P["q99e"]>=P["true"]).mean():.0f}% | GEFS {100*(P["q99g"]>=P["true"]).mean():.0f}% of storm windows; hourly q99 coverage reanalysis {cov["era5"][0]/cov["era5"][1]:.3f} | GEFS {cov["gefs"][0]/cov["gefs"][1]:.3f}']
txt = '\n'.join(out); open(f'{ROOT}/outputs/eval_gefs_season{TAG}.log', 'w').write(txt + '\n'); print(txt); print('SEASON EVAL DONE', flush=True)
