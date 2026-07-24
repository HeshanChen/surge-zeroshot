"""Centered vs trailing (causal) surge definition: evaluate the delivered checkpoint zero-shot
on both definitions of the 84 marine test gauges with the identical window protocol.
-> outputs/eval_trailing.log + per-gauge CSV"""
import sys, warnings; warnings.filterwarnings('ignore')
import torch, pandas as pd, numpy as np
sys.path.insert(0, '/Users/heshan/Desktop/surge_fm/src')
from models.baseline_lstm import GlobalLSTM
from data.dataset_v0 import seismic_mask
ROOT = '/Users/heshan/Desktop/surge_fm'
Tctx, W, H = 208, 256, 48

sa = pd.read_csv(f'{ROOT}/catalog/static_attributes.csv').set_index('name')
sp = pd.read_csv(f'{ROOT}/catalog/exp_split_final.csv')
tr = [n for n in sp[sp.fold == 'train'].name if n in sa.index]
te = [n for n in sp[sp.fold == 'test'].name if n in sa.index]

def sfeat(n):
    r = sa.loc[n]; la, lo = np.radians(float(r.lat)), np.radians(float(r.lon))
    return np.array([np.cos(la)*np.cos(lo), np.cos(la)*np.sin(lo), np.sin(la), float(r.tidal_range_m), float(r.form_factor)], dtype='float32')
arr = np.stack([sfeat(n) for n in tr]); smu = arr.mean(0); ssd = arr.std(0)+1e-6
m = GlobalLSTM(n_out=3); m.load_state_dict(torch.load(f'{ROOT}/outputs/baseline_lstmq_v2final_best.pt', map_location='cpu')); m.eval()

def load(name, pdir):
    s = pd.read_parquet(f'{ROOT}/{pdir}/{name}.parquet')['surge']; s = s.where(s.abs() < 4)
    e = pd.read_parquet(f'{ROOT}/data/raw/era5/{name}.parquet')
    if e.index.tz is None: e.index = e.index.tz_localize('UTC')
    for c in ['wind_speed_10m','wind_direction_10m','pressure_msl','precipitation']:
        e[c] = pd.to_numeric(e[c], errors='coerce')
    spd = e['wind_speed_10m'].values; dd = np.deg2rad(e['wind_direction_10m'].values)
    e2 = pd.DataFrame({'wu':spd*np.sin(dd),'wv':spd*np.cos(dd),'mslp':e['pressure_msl'].values,'precip':e['precipitation'].values}, index=e.index)
    df = pd.DataFrame({'surge':s}).join(e2, how='inner').dropna()
    r = sa.loc[name]; return df[~seismic_mask(float(r.lat), float(r.lon), df.index)]

def eval_def(pdir, tag):
    rows = []; SM=[]; SP=[]
    for name in te:
        try: df = load(name, pdir)
        except Exception: continue
        if len(df) < 3000: continue
        a = df.values.astype('float32'); tstd = float(a[:, 0].std())+1e-6
        af = a.copy(); af[:, 1:] = (af[:, 1:]-af[:, 1:].mean(0))/(af[:, 1:].std(0)+1e-6)
        starts = list(range(0, len(af)-W, H))
        ws = np.stack([af[st:st+W] for st in starts]); raw = np.stack([a[st:st+W, 0] for st in starts])
        mu = ws[:, :Tctx, 0].mean(1, keepdims=True); sd = ws[:, :Tctx, 0].std(1, keepdims=True)+1e-6
        ctx = np.concatenate([((ws[:, :Tctx, 0]-mu)/sd)[:, :, None], ws[:, :Tctx, 1:]], 2).transpose(0, 2, 1)
        ff = ws[:, Tctx:, 1:].transpose(0, 2, 1); last = raw[:, Tctx-1]
        ps = []
        with torch.no_grad():
            for j in range(0, len(ws), 512):
                _, o = m.predict_window(torch.tensor(ctx[j:j+512]).float(), torch.tensor(ff[j:j+512]).float(), H,
                                        torch.tensor((sfeat(name)-smu)/ssd)[None].expand(min(512, len(ws)-j), -1).float(),
                                        torch.tensor(last[j:j+512]/tstd).float())
                ps.append(o[..., 0].cpu())
        pred = torch.cat(ps).numpy()*tstd*100
        tru = raw[:, Tctx:]*100; per = (last*100)[:, None]
        em = np.sqrt(((pred-tru)**2).mean()); ep = np.sqrt(((per-tru)**2).mean())
        e8 = float(np.sqrt(((pred[:, 7]-tru[:, 7])**2).mean())); p8 = float(np.sqrt(((per[:, 0]-tru[:, 7])**2).mean()))
        nse = 1-((pred[:, 7]-tru[:, 7])**2).mean()/max(((tru[:, 7]-tru[:, 7].mean())**2).mean(), 1e-9)
        thr = np.quantile(tru, 0.999); wm = tru.max(1) >= thr
        cap = np.nan
        if wm.sum() >= 3:
            tw = tru[wm]; ipk = tw.argmax(1); ar = np.arange(len(ipk))
            cap = float(pred[wm][ar, ipk].sum()/(tw[ar, ipk].sum()+1e-6))
            SM.append(np.sqrt(((pred[wm]-tw)**2).mean(0))); SP.append(np.sqrt(((per[wm]-tw)**2).mean(0)))
        rows.append(dict(stn=name, rmse_p=float(em), prmse_p=float(ep), r8=e8, p8=p8, n8=float(1/(2-nse)), cap=cap))
    d = pd.DataFrame(rows)
    sm = np.mean(SM, 0); spp = np.mean(SP, 0)
    out = dict(tag=tag, n=len(d),
               pooled=100*(1-d.rmse_p.mean()/d.prmse_p.mean()), at8=100*(1-d.r8.mean()/d.p8.mean()),
               nnse=d.n8.mean(), cap=d.cap.mean(),
               storm8=100*(1-sm[7]/spp[7]), storm_rng=(100*(1-sm[[3,5,7,9,11]]/spp[[3,5,7,9,11]])).round(0).tolist(),
               wins=int((d.rmse_p < d.prmse_p).sum()))
    d.to_csv(f'{ROOT}/outputs/eval_trailing_{tag}.csv', index=False)
    return out

lines = []
for pdir, tag in [('data/processed', 'centered'), ('data/processed_trailing', 'trailing')]:
    r = eval_def(pdir, tag)
    lines.append(f"{r['tag']:9s} n={r['n']} pooled {r['pooled']:+.1f}% @8h {r['at8']:+.1f}% NNSE {r['nnse']:.3f} "
                 f"cap {r['cap']:.2f} storm@8h {r['storm8']:+.1f}% storm4-12 {r['storm_rng']} wins {r['wins']}/{r['n']}")
    print(lines[-1], flush=True)
open(f'{ROOT}/outputs/eval_trailing.log', 'w').write('\n'.join(lines)+'\n')
print('TRAILING EVAL DONE', flush=True)
