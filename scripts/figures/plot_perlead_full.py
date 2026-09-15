"""Per-lead skill for ALL 48 leads (the full decomposition we preach): v2final on the 84 marine
test gauges, skill(l) = 1 - <RMSE_mdl(l)>/<RMSE_per(l)>, plus the p99.9 storm-window curve.
CPU. -> outputs/perlead_full.{pdf,png} + outputs/perlead_full.csv"""
import os as _os
_ROOT = _os.environ.get('SURGE_ROOT') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
import sys, warnings; warnings.filterwarnings('ignore')
import torch, pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
sys.path.insert(0, f'{_ROOT}/scripts')
from pubstyle import apply, save_pub, PAL, W1
apply()
import matplotlib.pyplot as plt
sys.path.insert(0, f'{_ROOT}/src')
from models.baseline_lstm import GlobalLSTM
from data.dataset_v0 import load_station
ROOT = _ROOT
Tctx, W, H = 208, 256, 48
CONT = '--continuous_only' in sys.argv; SUF = '_cont' if CONT else ''   # audit 2: continuous windows only
REPLOT = '--replot' in sys.argv   # redraw from the saved CSV without model inference

sa = pd.read_csv(f'{ROOT}/catalog/static_attributes.csv').set_index('name')
sp = pd.read_csv(f'{ROOT}/catalog/exp_split_final.csv')
tr = [n for n in sp[sp.fold == 'train'].name if n in sa.index]
te = [n for n in sp[sp.fold == 'test'].name if n in sa.index]
def sfeat(n):
    r = sa.loc[n]; la, lo = np.radians(float(r.lat)), np.radians(float(r.lon))
    return np.array([np.cos(la)*np.cos(lo), np.cos(la)*np.sin(lo), np.sin(la), float(r.tidal_range_m), float(r.form_factor)], dtype='float32')
SF = {n: sfeat(n) for n in tr+te}; arr = np.stack([SF[n] for n in tr]); smu = arr.mean(0); ssd = arr.std(0)+1e-6
for n in SF: SF[n] = (SF[n]-smu)/ssd
m = GlobalLSTM(n_out=3); m.load_state_dict(torch.load(f'{ROOT}/models/deploy_760_best.pt', map_location='cpu')); m.eval()

THR = [0.90, 0.95, 0.99, 0.999]
RM = np.zeros((0, H)); RP = np.zeros((0, H)); SM = {t: [] for t in THR}; SP = {t: [] for t in THR}
for k, name in enumerate([] if REPLOT else te):
    a, tt = load_station(name, return_time=True)
    if a is None or len(a) < 3000: continue
    tstd = float(a[:, 0].std())+1e-6
    af = a.copy(); af[:, 1:] = (af[:, 1:]-af[:, 1:].mean(0))/(af[:, 1:].std(0)+1e-6)
    starts = np.arange(0, len(af)-W, H)
    if CONT: starts = starts[(tt[starts+W-1]-tt[starts]) == (W-1)]
    if len(starts) < 20: continue
    ws = np.stack([af[st:st+W] for st in starts])
    raw = np.stack([a[st:st+W, 0] for st in starts])
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
    tru = raw[:, Tctx:]*100; per = (last*100)[:, None]
    em = np.sqrt(((pred-tru)**2).mean(0)); ep = np.sqrt(((per-tru)**2).mean(0))
    RM = np.vstack([RM, em[None]]); RP = np.vstack([RP, ep[None]])
    for t in THR:
        thr = np.quantile(tru, t); wm = tru.max(1) >= thr
        if wm.sum() >= 5:
            SM[t].append(np.sqrt(((pred[wm]-tru[wm])**2).mean(0))); SP[t].append(np.sqrt(((per[wm]-tru[wm])**2).mean(0)))
    if (k+1) % 20 == 0: print(f'  {k+1}/{len(te)}', flush=True)

if REPLOT:
    _d = pd.read_csv(f'{ROOT}/outputs/perlead_full{SUF}.csv'); lead = _d.lead.values; skill = _d.skill_full.values
    sk = {t: _d[f'skill_p{str(t)[2:]}'].values for t in THR}
else:
    lead = np.arange(1, H+1)
    skill = 100*(RP.mean(0)-RM.mean(0))/RP.mean(0)
    out = {'lead': lead, 'skill_full': skill}
    sk = {}
    for t in THR:
        sk[t] = 100*(np.mean(SP[t], 0)-np.mean(SM[t], 0))/np.mean(SP[t], 0)
        out[f'skill_p{str(t)[2:]}'] = sk[t]
    pd.DataFrame(out).to_csv(f'{ROOT}/outputs/perlead_full{SUF}.csv', index=False)

fig, ax = plt.subplots(figsize=(W1, 2.3))
ax.plot(lead, skill, '-', color=PAL['blue'], lw=1.3, label='full distribution')
shades = ['#F3C09E', '#E8945B', '#D55E00', '#8C3D00']
labs = ['p90 windows', 'p95', 'p99', 'p99.9']
for t, c, lb in zip(THR, shades, labs):
    ax.plot(lead, sk[t], '-', color=c, lw=1.0, label=lb)
ax.axhline(0, color='#999999', lw=0.7, ls=(0,(4,3)))
ax.text(1.5, 1.2, 'persistence parity', fontsize=5.2, color='#777777', ha='left')
ax.set_xlabel('lead time (h)'); ax.set_ylabel('skill vs persistence (%)')
ax.set_xticks([1, 6, 12, 24, 36, 48]); ax.grid(alpha=0.3, lw=0.4)
ax.legend(loc='lower right', handlelength=1.4, fontsize=5.8, labelspacing=0.25)
fig.tight_layout()
save_pub(fig, f'{ROOT}/outputs/perlead_full{SUF}')
print(f'wrote outputs/perlead_full{SUF}: full@1h {skill[0]:+.0f}% @48h {skill[47]:+.0f}% | p999@8h {sk[0.999][7]:+.0f}%')
