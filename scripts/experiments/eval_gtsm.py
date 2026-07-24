"""Physics-baseline head-to-head: our zero-shot forecast vs GTSM-ERA5 reanalysis (CDS v2, hourly)
vs persistence, on the 84 marine test gauges over the 2010-2018 overlap. Same 48h windows.
GTSM is a lead-independent hindcast (its value at t+l is the same regardless of issue time).
Per-gauge mean difference removed from GTSM (datum/model-mean alignment; disclosed).
-> outputs/eval_gtsm.csv + .log"""
import sys, warnings; warnings.filterwarnings('ignore')
import torch, pandas as pd, numpy as np
sys.path.insert(0, '/Users/heshan/Desktop/surge_fm/src')
from models.baseline_lstm import GlobalLSTM
from data.dataset_v0 import load_station
ROOT = '/Users/heshan/Desktop/surge_fm'
Tctx, W, H = 208, 256, 48

sa = pd.read_csv(f'{ROOT}/catalog/static_attributes.csv').set_index('name')
sp = pd.read_csv(f'{ROOT}/catalog/exp_split_final.csv')
tr = [n for n in sp[sp.fold == 'train'].name if n in sa.index]
te = [n for n in sp[sp.fold == 'test'].name if n in sa.index]
mp = pd.read_csv(f'{ROOT}/data/raw/gtsm/gtsm_mapping.csv').set_index('stn')
G = pd.read_parquet(f'{ROOT}/data/raw/gtsm/gtsm_hourly_2010_2018.parquet')
G.columns = [int(c) for c in G.columns]

def sfeat(n):
    r = sa.loc[n]; la, lo = np.radians(float(r.lat)), np.radians(float(r.lon))
    return np.array([np.cos(la)*np.cos(lo), np.cos(la)*np.sin(lo), np.sin(la), float(r.tidal_range_m), float(r.form_factor)], dtype='float32')
SF = {n: sfeat(n) for n in tr+te}; arr = np.stack([SF[n] for n in tr]); smu = arr.mean(0); ssd = arr.std(0)+1e-6
for n in SF: SF[n] = (SF[n]-smu)/ssd
m = GlobalLSTM(n_out=3); m.load_state_dict(torch.load(f'{ROOT}/outputs/baseline_lstmq_v2final_best.pt', map_location='cpu')); m.eval()

# need timestamps: rebuild the loader join with time index (mirror load_station, keeping index)
from data.dataset_v0 import seismic_mask
def load_with_time(name):
    s = pd.read_parquet(f'{ROOT}/data/processed/{name}.parquet')['surge']; s = s.where(s.abs() < 4)
    e = pd.read_parquet(f'{ROOT}/data/raw/era5/{name}.parquet')
    if e.index.tz is None: e.index = e.index.tz_localize('UTC')
    for c in ['wind_speed_10m','wind_direction_10m','pressure_msl','precipitation']:
        e[c] = pd.to_numeric(e[c], errors='coerce')
    spd = e['wind_speed_10m'].values; d = np.deg2rad(e['wind_direction_10m'].values)
    e2 = pd.DataFrame({'wu':spd*np.sin(d),'wv':spd*np.cos(d),'mslp':e['pressure_msl'].values,'precip':e['precipitation'].values}, index=e.index)
    df = pd.DataFrame({'surge':s}).join(e2, how='inner').dropna()
    if len(df):
        r = sa.loc[name]; df = df[~seismic_mask(float(r.lat), float(r.lon), df.index)]
    return df

rows = []
RM=[]; RP=[]; RG=[]; SM=[]; SP=[]; SG=[]
for k, name in enumerate(te):
    df = load_with_time(name)
    df = df[(df.index >= pd.Timestamp('2010-01-01', tz='UTC')) & (df.index <= pd.Timestamp('2018-12-31 23:00', tz='UTC'))]
    if len(df) < 3000: continue
    gi = int(mp.loc[name, 'gtsm_idx'])
    g = G[gi].reindex(df.index)
    ok = g.notna(); df = df[ok]; g = g[ok]
    if len(df) < 3000: continue
    g = g - (g.mean() - df.surge.mean())              # per-gauge mean alignment
    a = df.values.astype('float32'); gid = g.values.astype('float32')
    tstd = float(a[:, 0].std())+1e-6
    af = a.copy(); af[:, 1:] = (af[:, 1:]-af[:, 1:].mean(0))/(af[:, 1:].std(0)+1e-6)
    starts = list(range(0, len(af)-W, H))
    ws = np.stack([af[st:st+W] for st in starts])
    raw = np.stack([a[st:st+W, 0] for st in starts])
    gw = np.stack([gid[st+Tctx:st+W] for st in starts])
    mu = ws[:, :Tctx, 0].mean(1, keepdims=True); sd = ws[:, :Tctx, 0].std(1, keepdims=True)+1e-6
    ctx = np.concatenate([((ws[:, :Tctx, 0]-mu)/sd)[:, :, None], ws[:, :Tctx, 1:]], 2).transpose(0, 2, 1)
    ff = ws[:, Tctx:, 1:].transpose(0, 2, 1)
    last = raw[:, Tctx-1]
    ps = []
    with torch.no_grad():
        for j in range(0, len(ws), 512):
            _, o = m.predict_window(torch.tensor(ctx[j:j+512]).float(), torch.tensor(ff[j:j+512]).float(), H,
                                    torch.tensor(SF[name])[None].expand(min(512, len(ws)-j), -1).float(),
                                    torch.tensor(last[j:j+512]/tstd).float())
            ps.append(o[..., 0].cpu())
    pred = torch.cat(ps).numpy()*tstd*100
    tru = raw[:, Tctx:]*100; per = (last*100)[:, None]; gts = gw*100
    em = np.sqrt(((pred-tru)**2).mean(0)); ep = np.sqrt(((per-tru)**2).mean(0)); eg = np.sqrt(((gts-tru)**2).mean(0))
    RM.append(em); RP.append(ep); RG.append(eg)
    thr = np.quantile(tru, 0.999); wm = tru.max(1) >= thr
    if wm.sum() >= 5:
        SM.append(np.sqrt(((pred[wm]-tru[wm])**2).mean(0))); SP.append(np.sqrt(((per[wm]-tru[wm])**2).mean(0))); SG.append(np.sqrt(((gts[wm]-tru[wm])**2).mean(0)))
    def nnse(e, y):
        ns = 1-(e**2).mean()/max(((y-y.mean())**2).mean(), 1e-9); return 1/(2-ns)
    rows.append(dict(stn=name, n_win=len(ws), dist_km=float(mp.loc[name, 'dist_km']),
                rmse8_ours=float(em[7]), rmse8_gtsm=float(eg[7]), rmse8_per=float(ep[7]),
                nnse8_ours=float(nnse(pred[:, 7]-tru[:, 7], tru[:, 7])), nnse8_gtsm=float(nnse(gts[:, 7]-tru[:, 7], tru[:, 7]))))
    if (k+1) % 20 == 0: print(f'  {k+1}/{len(te)}', flush=True)

d = pd.DataFrame(rows); d.to_csv(f'{ROOT}/outputs/eval_gtsm.csv', index=False)
lead = np.arange(1, H+1)
am = np.mean(RM, 0); ap = np.mean(RP, 0); ag = np.mean(RG, 0)
sm = np.mean(SM, 0); spp = np.mean(SP, 0); sg = np.mean(SG, 0)
out = []
out.append(f'GTSM HEAD-TO-HEAD (n={len(d)} gauges, 2010-2018 overlap, same 48h windows)')
out.append(f'full-dist @8h RMSE: ours {am[7]:.2f} | GTSM {ag[7]:.2f} | pers {ap[7]:.2f} cm')
out.append(f'full-dist @24h    : ours {am[23]:.2f} | GTSM {ag[23]:.2f} | pers {ap[23]:.2f}')
out.append(f'full-dist @48h    : ours {am[47]:.2f} | GTSM {ag[47]:.2f} | pers {ap[47]:.2f}')
out.append(f'NNSE@8h mean: ours {d.nnse8_ours.mean():.3f} | GTSM {d.nnse8_gtsm.mean():.3f}')
out.append(f'gauges where ours beats GTSM @8h RMSE: {int((d.rmse8_ours<d.rmse8_gtsm).sum())}/{len(d)}')
out.append(f'p99.9 storm windows @8h: ours {sm[7]:.2f} | GTSM {sg[7]:.2f} | pers {spp[7]:.2f} cm')
out.append(f'p99.9 storm windows @48h: ours {sm[47]:.2f} | GTSM {sg[47]:.2f} | pers {spp[47]:.2f}')
txt = '\n'.join(out)
open(f'{ROOT}/outputs/eval_gtsm.log', 'w').write(txt+'\n')
print(txt)
