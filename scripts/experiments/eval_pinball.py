"""Proper probabilistic scoring for the delivered model's quantile heads: per-gauge pinball loss
at tau=0.90 and 0.99 for the q90/q99 trajectories on the 84 marine test gauges, against a
climatological constant-quantile reference (per-gauge q90/q99 of the surge record).
Skill = 1 - pinball_model/pinball_clim (positive = beats climatological quantiles under a proper score).
-> outputs/eval_pinball.csv + .log"""
import sys, warnings; warnings.filterwarnings('ignore')
import torch, pandas as pd, numpy as np
sys.path.insert(0, '/Users/heshan/Desktop/surge_fm/src')
from models.baseline_lstm import GlobalLSTM
from data.dataset_v0 import seismic_mask
ROOT = '/Users/heshan/Desktop/surge_fm'
Tctx, W, H = 208, 256, 48
CONT = '--continuous_only' in sys.argv; SUF = '_cont' if CONT else ''

sa = pd.read_csv(f'{ROOT}/catalog/static_attributes.csv').set_index('name')
sp = pd.read_csv(f'{ROOT}/catalog/exp_split_final.csv')
tr = [n for n in sp[sp.fold == 'train'].name if n in sa.index]
te = [n for n in sp[sp.fold == 'test'].name if n in sa.index]

def sfeat(n):
    r = sa.loc[n]; la, lo = np.radians(float(r.lat)), np.radians(float(r.lon))
    return np.array([np.cos(la)*np.cos(lo), np.cos(la)*np.sin(lo), np.sin(la), float(r.tidal_range_m), float(r.form_factor)], dtype='float32')
arr = np.stack([sfeat(n) for n in tr]); smu = arr.mean(0); ssd = arr.std(0)+1e-6
m = GlobalLSTM(n_out=3); m.load_state_dict(torch.load(f'{ROOT}/outputs/baseline_lstmq_v2final_best.pt', map_location='cpu')); m.eval()

def pinball(y, q, tau):
    u = y - q
    return float(np.mean(np.maximum(tau*u, (tau-1)*u)))

def load(name):
    s = pd.read_parquet(f'{ROOT}/data/processed/{name}.parquet')['surge']; s = s.where(s.abs() < 4)
    e = pd.read_parquet(f'{ROOT}/data/raw/era5/{name}.parquet')
    if e.index.tz is None: e.index = e.index.tz_localize('UTC')
    for c in ['wind_speed_10m','wind_direction_10m','pressure_msl','precipitation']:
        e[c] = pd.to_numeric(e[c], errors='coerce')
    spd = e['wind_speed_10m'].values; dd = np.deg2rad(e['wind_direction_10m'].values)
    e2 = pd.DataFrame({'wu':spd*np.sin(dd),'wv':spd*np.cos(dd),'mslp':e['pressure_msl'].values,'precip':e['precipitation'].values}, index=e.index)
    df = pd.DataFrame({'surge':s}).join(e2, how='inner').dropna()
    r = sa.loc[name]; return df[~seismic_mask(float(r.lat), float(r.lon), df.index)]

rows = []
for k, name in enumerate(te):
    try: df = load(name)
    except Exception: continue
    if len(df) < 3000: continue
    a = df.values.astype('float32'); tstd = float(a[:, 0].std())+1e-6
    af = a.copy(); af[:, 1:] = (af[:, 1:]-af[:, 1:].mean(0))/(af[:, 1:].std(0)+1e-6)
    starts = np.arange(0, len(af)-W, H)
    if CONT:   # audit 2: keep only windows whose 256 rows are consecutive hours
        th = (df.index.values.astype('datetime64[h]').astype('int64')); starts = starts[(th[starts+W-1]-th[starts]) == (W-1)]
    if len(starts) < 20: continue
    ws = np.stack([af[st:st+W] for st in starts]); raw = np.stack([a[st:st+W, 0] for st in starts])
    mu = ws[:, :Tctx, 0].mean(1, keepdims=True); sd = ws[:, :Tctx, 0].std(1, keepdims=True)+1e-6
    ctx = np.concatenate([((ws[:, :Tctx, 0]-mu)/sd)[:, :, None], ws[:, :Tctx, 1:]], 2).transpose(0, 2, 1)
    ff = ws[:, Tctx:, 1:].transpose(0, 2, 1); last = raw[:, Tctx-1]
    qs = []
    with torch.no_grad():
        for j in range(0, len(ws), 512):
            _, o = m.predict_window(torch.tensor(ctx[j:j+512]).float(), torch.tensor(ff[j:j+512]).float(), H,
                                    torch.tensor((sfeat(name)-smu)/ssd)[None].expand(min(512, len(ws)-j), -1).float(),
                                    torch.tensor(last[j:j+512]/tstd).float())
            qs.append(o.cpu())
    q = torch.cat(qs).numpy()*tstd*100          # (n, 48, 3): pt, q90, q99
    tru = raw[:, Tctx:]*100
    c90, c99 = np.quantile(a[:, 0]*100, 0.90), np.quantile(a[:, 0]*100, 0.99)
    rows.append(dict(stn=name,
        pb90=pinball(tru, q[:, :, 1], 0.90), pb90_clim=pinball(tru, np.full_like(tru, c90), 0.90),
        pb99=pinball(tru, q[:, :, 2], 0.99), pb99_clim=pinball(tru, np.full_like(tru, c99), 0.99)))
    if (k+1) % 20 == 0: print(f'  {k+1}/{len(te)}', flush=True)

d = pd.DataFrame(rows)
d['sk90'] = 100*(1-d.pb90/d.pb90_clim); d['sk99'] = 100*(1-d.pb99/d.pb99_clim)
d.to_csv(f'{ROOT}/outputs/eval_pinball{SUF}.csv', index=False)
out = [f'PINBALL (proper score) vs climatological quantiles, n={len(d)} gauges, all 48 leads pooled:',
       f'q90: model {d.pb90.mean():.3f} vs clim {d.pb90_clim.mean():.3f} cm -> skill {100*(1-d.pb90.mean()/d.pb90_clim.mean()):+.1f}%  (win {int((d.pb90<d.pb90_clim).sum())}/{len(d)})',
       f'q99: model {d.pb99.mean():.3f} vs clim {d.pb99_clim.mean():.3f} cm -> skill {100*(1-d.pb99.mean()/d.pb99_clim.mean()):+.1f}%  (win {int((d.pb99<d.pb99_clim).sum())}/{len(d)})',
       f'per-gauge median skill: q90 {d.sk90.median():+.1f}%  q99 {d.sk99.median():+.1f}%']
txt='\n'.join(out); open(f'{ROOT}/outputs/eval_pinball{SUF}.log','w').write(txt+'\n'); print(txt)
print('PINBALL EVAL DONE', flush=True)
