"""Case-study figure: biggest extreme events across the 84-gauge marine test set — v2final zero-shot forecasts.

Selection: across all test stations, take forecast windows whose 48h horizon contains that station's largest
surge peaks (>= p99.9); rank by true peak height, keep one event per station, plot top N.
Each panel: 24h of observed context, then the 48h horizon with ground truth, point forecast, q90/q99 envelope,
and persistence. Model: v2final deliverable (760-station, clean protocol), CPU (ablations own MPS).
Output: outputs/case_studies_extremes.png
"""
import sys, warnings; warnings.filterwarnings('ignore')
import torch, pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
sys.path.insert(0,'/Users/heshan/Desktop/surge_fm/scripts')
from pubstyle import apply, save_pub, PAL, W2, panel_label
apply()
import matplotlib.pyplot as plt
sys.path.insert(0, '/Users/heshan/Desktop/surge_fm/src')
from models.baseline_lstm import GlobalLSTM
from data.dataset_v0 import seismic_mask
ROOT = '/Users/heshan/Desktop/surge_fm'
CKPT = f'{ROOT}/outputs/baseline_lstmq_v2final_best.pt'
SPLIT = 'exp_split_final.csv'; N_PANELS = 6
Tctx, Ttgt, W, H = 208, 48, 256, 48

sa = pd.read_csv(f'{ROOT}/catalog/static_attributes.csv').set_index('name')
sp = pd.read_csv(f'{ROOT}/catalog/{SPLIT}')
tr = [n for n in sp[sp.fold == 'train'].name if n in sa.index]
te = [n for n in sp[sp.fold == 'test'].name if n in sa.index]
def sfeat(n):
    r = sa.loc[n]; la, lo = np.radians(float(r.lat)), np.radians(float(r.lon))
    return np.array([np.cos(la)*np.cos(lo), np.cos(la)*np.sin(lo), np.sin(la), float(r.tidal_range_m), float(r.form_factor)], dtype='float32')
SF = {n: sfeat(n) for n in tr+te}; arr = np.stack([SF[n] for n in tr]); smu = arr.mean(0); ssd = arr.std(0)+1e-6
for n in SF: SF[n] = (SF[n]-smu)/ssd

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
        r = sa.loc[name]
        df = df[~seismic_mask(float(r.lat), float(r.lon), df.index)]
    return (df.values.astype('float32'), df.index) if len(df) >= 3000 else (None, None)

# pass 1: find the biggest p99.9 events (ground truth only, no model)
events = []
for n in te:
    a, idx = load_with_time(n)
    if a is None: continue
    thr = float(np.percentile(a[:, 0], 99.9))*100
    starts = list(range(0, len(a)-W, Ttgt))
    tru = np.stack([a[st+Tctx:st+W, 0] for st in starts])*100
    wm = tru.max(1)
    for wi in np.where(wm >= thr)[0]:
        pk_lead = int(tru[wi].argmax())+1
        events.append(dict(stn=n, wi=int(wi), start=starts[wi], peak=float(wm[wi]), lead=pk_lead,
                           t0=idx[starts[wi]+Tctx-1]))
ev = pd.DataFrame(events).sort_values('peak', ascending=False).drop_duplicates('stn').head(N_PANELS)
print(ev[['stn','peak','lead','t0']].to_string(index=False), flush=True)

model = GlobalLSTM(n_out=3)
model.load_state_dict(torch.load(CKPT, map_location='cpu')); model.eval()

fig, axes = plt.subplots(2, 3, figsize=(W2, 3.6), sharex=False)
for ax, (_, r) in zip(axes.flat, ev.iterrows()):
    a, idx = load_with_time(r.stn)
    tstd = float(a[:, 0].std())+1e-6
    af = a.copy(); af[:, 1:] = (af[:, 1:]-af[:, 1:].mean(0))/(af[:, 1:].std(0)+1e-6)
    st = int(r.start); w = af[st:st+W]
    mu = w[:Tctx, 0].mean(); sd = w[:Tctx, 0].std()+1e-6
    ctx = np.concatenate([((w[:Tctx, 0]-mu)/sd)[:, None], w[:Tctx, 1:]], 1).T[None]
    ff = w[Tctx:, 1:].T[None]
    anc = torch.tensor([a[st+Tctx-1, 0]/tstd]).float()
    with torch.no_grad():
        _, out = model.predict_window(torch.tensor(ctx).float(), torch.tensor(ff).float(), H,
                                      torch.tensor(SF[r.stn])[None].float(), anc)
    out = out[0].numpy()*tstd*100                                   # (48,3) cm
    tru_c = a[st+Tctx-24:st+Tctx, 0]*100                            # 24h context (observed)
    tru_h = a[st+Tctx:st+W, 0]*100                                  # 48h horizon truth
    pers = a[st+Tctx-1, 0]*100
    hx = np.arange(1, H+1); cx = np.arange(-23, 1)
    ax.plot(cx, tru_c, color='k', lw=0.7, alpha=0.5)
    ax.plot(hx, tru_h, color='k', lw=1.0, label='observed')
    ax.plot(hx, out[:, 0], color=PAL['blue'], lw=1.0, label='model point')
    ax.fill_between(hx, out[:, 1], out[:, 2], color=PAL['blue'], alpha=0.18, lw=0, label='q90–q99 envelope')
    ax.axhline(pers, color=PAL['grey'], ls='--', lw=0.7, label='persistence')
    ax.axvline(0, color=PAL['grey'], lw=0.5, alpha=0.6)
    ax.set_title(f"{r.stn.split('-')[0]}  {str(r.t0)[:10]}\npeak {r.peak:.0f} cm @ +{int(r.lead)} h", fontsize=6.5)
    ax.grid(alpha=0.25, lw=0.4)
for ax in axes[1]: ax.set_xlabel('lead time (h)')
for ax in axes[:, 0]: ax.set_ylabel('surge (cm)')
h_, l_ = axes.flat[0].get_legend_handles_labels()
for ax,letter in zip(axes.flat,'abcdef'): panel_label(ax,letter,dx=-0.16)
h_, l_ = axes.flat[0].get_legend_handles_labels()
fig.legend(h_, l_, loc='upper center', bbox_to_anchor=(0.5, 1.03), ncol=4)
fig.tight_layout(rect=[0, 0, 1, 0.95], h_pad=1.4)
save_pub(fig, f'{ROOT}/outputs/case_studies_extremes')
print('wrote outputs/case_studies_extremes.pdf+png')
