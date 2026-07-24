"""Appendix full-page exhibit: 12 UNSELECTED forecast windows — random gauges (continent-stratified,
seed 0), random times (seed per gauge), no peak criterion. Anti-cherry-picking companion to the
extreme-event case studies. Model: v2final deliverable, CPU. -> outputs/figS2_random.{pdf,png}"""
import sys, warnings; warnings.filterwarnings('ignore')
import torch, pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
sys.path.insert(0, '/Users/heshan/Desktop/surge_fm/scripts')
from pubstyle import apply, save_pub, PAL, W2, panel_label
apply()
import matplotlib.pyplot as plt
sys.path.insert(0, '/Users/heshan/Desktop/surge_fm/src')
from models.baseline_lstm import GlobalLSTM
from data.dataset_v0 import load_station
ROOT = '/Users/heshan/Desktop/surge_fm'
Tctx, W, H = 208, 256, 48

sa = pd.read_csv(f'{ROOT}/catalog/static_attributes.csv').set_index('name')
sp = pd.read_csv(f'{ROOT}/catalog/exp_split_final.csv')
tr = [n for n in sp[sp.fold == 'train'].name if n in sa.index]
te = [n for n in sp[sp.fold == 'test'].name if n in sa.index]
def sfeat(n):
    r = sa.loc[n]; la, lo = np.radians(float(r.lat)), np.radians(float(r.lon))
    return np.array([np.cos(la)*np.cos(lo), np.cos(la)*np.sin(lo), np.sin(la), float(r.tidal_range_m), float(r.form_factor)], dtype='float32')
SF = {n: sfeat(n) for n in tr+te}; arr = np.stack([SF[n] for n in tr]); smu = arr.mean(0); ssd = arr.std(0)+1e-6
for n in SF: SF[n] = (SF[n]-smu)/ssd

def cont(n):
    la, lo = float(sa.loc[n,'lat']), float(sa.loc[n,'lon']); lo = lo if lo <= 180 else lo-360
    if la > 15 and -130 <= lo <= -50: return 'NA'
    if la <= 15 and -90 <= lo <= -30: return 'SA'
    if la > 35 and -15 <= lo <= 45: return 'EU'
    if -37 <= la <= 35 and -20 <= lo <= 52: return 'AF'
    if la > 0 and 45 < lo <= 180: return 'AS'
    if la <= 0 and 90 <= lo <= 185: return 'OC'
    return 'other'
rng = np.random.RandomState(0)
by = {}
for n in rng.permutation(sorted(te)): by.setdefault(cont(n), []).append(n)
picks = []
for c in ['AS','EU','NA','OC','SA','AF']: picks += by.get(c, [])[:2]
picks = picks[:12]

m = GlobalLSTM(n_out=3); m.load_state_dict(torch.load(f'{ROOT}/outputs/baseline_lstmq_v2final_best.pt', map_location='cpu')); m.eval()
fig, axes = plt.subplots(4, 3, figsize=(W2, 7.6))
for k, (name, ax) in enumerate(zip(picks, axes.flat)):
    a = load_station(name)
    st = np.random.RandomState(k).randint(0, len(a) - W)          # random time, per-panel seed
    tstd = float(a[:, 0].std())+1e-6
    af = a.copy(); af[:, 1:] = (af[:, 1:]-af[:, 1:].mean(0))/(af[:, 1:].std(0)+1e-6)
    w = af[st:st+W]
    mu = w[:Tctx, 0].mean(); sd = w[:Tctx, 0].std()+1e-6
    ctx = np.concatenate([((w[:Tctx, 0]-mu)/sd)[:, None], w[:Tctx, 1:]], 1).T[None]
    ff = w[Tctx:, 1:].T[None]
    with torch.no_grad():
        _, o = m.predict_window(torch.tensor(ctx).float(), torch.tensor(ff).float(), H,
                                torch.tensor(SF[name])[None].float(), torch.tensor([a[st+Tctx-1, 0]/tstd]).float())
    o = o[0].numpy()*tstd*100
    hx = np.arange(1, 49); tru = a[st+Tctx:st+W, 0]*100
    cx = np.arange(-24, 0); trc = a[st+Tctx-24:st+Tctx, 0]*100
    ax.plot(cx, trc, c='k', lw=0.6, alpha=0.5)
    ax.fill_between(hx, o[:, 1], o[:, 2], color=PAL['blue'], alpha=0.18, lw=0, label='q90 to q99')
    ax.plot(hx, tru, c='k', lw=0.9, label='observed')
    ax.plot(hx, o[:, 0], c=PAL['blue'], lw=0.9, label='point')
    ax.axhline(a[st+Tctx-1, 0]*100, c=PAL['grey'], ls='--', lw=0.6, label='persistence')
    ax.axvline(0, c=PAL['grey'], lw=0.4, alpha=0.6)
    ax.set_title(f"{name.split('-')[0]}", fontsize=6.5)
    ax.grid(alpha=0.25, lw=0.4)
for ax in axes[-1]: ax.set_xlabel('lead time (h)')
for ax in axes[:, 0]: ax.set_ylabel('surge (cm)')
h_, l_ = axes.flat[0].get_legend_handles_labels()
fig.legend(h_, l_, loc='upper center', bbox_to_anchor=(0.5, 1.005), ncol=4)
fig.tight_layout(rect=[0, 0, 1, 0.975], h_pad=1.2)
save_pub(fig, f'{ROOT}/outputs/figS2_random')
print('wrote outputs/figS2_random.{pdf,png}:', ', '.join(p.split('-')[0] for p in picks))
