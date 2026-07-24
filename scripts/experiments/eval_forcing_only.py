"""Forcing-only LOWER BOUND: v2final with the surge-context channel zeroed and anchor=0
(inference-time ablation; a model trained with surge context sees OOD zeros, so a trained
forcing-only variant can only be better). NNSE@8h + RMSE@8h on the 84 marine test gauges.
CPU (MPS busy). Reference: Ebel densification NNSE 0.556 (their 708-gauge set)."""
import sys, warnings; warnings.filterwarnings('ignore')
import torch, pandas as pd, numpy as np, argparse
_ap=argparse.ArgumentParser(); _ap.add_argument('--ckpt', default='outputs/baseline_lstmq_v2final_best.pt'); _ap.add_argument('--tag', default='v2final')
_ap.add_argument('--sigma_file', default='', help='CSV stn,sigma_hat: use predicted scale instead of the record std (fully gauge-free)')
_ap.add_argument('--statics_file', default='', help='CSV stn,range_eot,ff_eot: override test-gauge tidal statics with open-tide-model values')
_args=_ap.parse_args()
sys.path.insert(0, '/Users/heshan/Desktop/surge_fm/src')
sys.path.insert(0, '/Users/heshan/Desktop/surge_fm/scripts')
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
SF = {n: sfeat(n) for n in tr+te}
if _args.statics_file:
    _ov=pd.read_csv(f'{ROOT}/{_args.statics_file}').set_index('stn')
    for n in te:
        if n in _ov.index and np.isfinite(_ov.loc[n,'range_eot']):
            SF[n] = SF[n].copy(); SF[n][3]=float(_ov.loc[n,'range_eot']); SF[n][4]=float(_ov.loc[n,'ff_eot'])
    print(f'statics override applied to {sum(1 for n in te if n in _ov.index)} test gauges', flush=True)
arr = np.stack([SF[n] for n in tr]); smu = arr.mean(0); ssd = arr.std(0)+1e-6
for n in SF: SF[n] = (SF[n]-smu)/ssd

SIG={}
if _args.sigma_file:
    _sd=pd.read_csv(f'{ROOT}/{_args.sigma_file}')
    SIG=dict(zip(_sd.stn,_sd.sigma_hat))
m = GlobalLSTM(n_out=3); m.load_state_dict(torch.load(f'{ROOT}/{_args.ckpt}', map_location='cpu')); m.eval()
rows = []
for i, name in enumerate(te):
    try: a = load_station(name)
    except Exception: continue
    if a is None or len(a) < 3000: continue
    a = a.copy(); tstd = float(a[:, 0].std())+1e-6
    a[:, 1:] = (a[:, 1:]-a[:, 1:].mean(0))/(a[:, 1:].std(0)+1e-6)
    ws = np.stack([a[st:st+W] for st in range(0, len(a)-W, H)])
    ctx = np.concatenate([np.zeros_like(ws[:, :Tctx, :1]), ws[:, :Tctx, 1:]], 2).transpose(0, 2, 1)  # surge ch = 0
    ff = ws[:, Tctx:, 1:].transpose(0, 2, 1)
    tru = ws[:, Tctx:, 0]*100
    s = torch.tensor(SF[name]).unsqueeze(0).expand(len(ws), -1)
    anc = torch.zeros(len(ws))                                                              # anchor = 0
    ps = []; ps3 = []
    with torch.no_grad():
        for j in range(0, len(ws), 512):
            _, o = m.predict_window(torch.tensor(ctx[j:j+512]).float(), torch.tensor(ff[j:j+512]).float(), H,
                                    s[j:j+512].float(), anc[j:j+512].float())
            ps.append(o[..., 0].cpu()); ps3.append(o.cpu())
    out3 = torch.cat(ps3).numpy()*(SIG.get(name, tstd) if SIG else tstd)*100 if ps3 else None
    scale = SIG.get(name, tstd) if SIG else tstd
    pred = torch.cat(ps).numpy()*scale*100
    e8 = pred[:, 7]-tru[:, 7]
    thrq = np.quantile(tru, 0.999); wmq = tru.max(1) >= thrq
    if wmq.sum() >= 3:
        tw = tru[wmq]; ipk = tw.argmax(1); ar = np.arange(len(ipk))
        pk_cap = float(pred[wmq][ar, ipk].sum()/(tw[ar, ipk].sum()+1e-6))
        q99v = out3[..., 2] if out3 is not None else None
        q99_cap = float(q99v[wmq][ar, ipk].sum()/(tw[ar, ipk].sum()+1e-6)) if q99v is not None else np.nan
        sw_rmse = float(np.sqrt(((pred[wmq]-tw)**2).mean()))
    else: pk_cap=q99_cap=sw_rmse=np.nan
    nse = 1 - (e8**2).mean()/max(((tru[:, 7]-tru[:, 7].mean())**2).mean(), 1e-9)
    q99 = out3[..., 2]; q90 = out3[..., 1]
    thr = np.quantile(tru, 0.99)
    hw = tru >= thr
    rows.append(dict(stn=name, rmse8=float(np.sqrt((e8**2).mean())), nnse8=float(1/(2-nse)), pk_cap=pk_cap, q99_cap=q99_cap, sw_rmse=sw_rmse,
                cov99=float((tru <= q99).mean()), cov90=float((tru <= q90).mean()),
                tail99=float((tru[hw] <= q99[hw]).mean()) if hw.any() else np.nan,
                cross=float((q90 > q99).mean())))
    if (i+1) % 20 == 0: print(f'  {i+1}/{len(te)}', flush=True)
d = pd.DataFrame(rows)
d.to_csv(f'{ROOT}/outputs/eval_forcing_only_{_args.tag}.csv', index=False)
print(f'\nFORCING-ONLY (n={len(d)}): NNSE@8h {d.nnse8.mean():.3f} | RMSE@8h {d.rmse8.mean():.2f} cm')
print(f'quantiles: cov99 {d.cov99.mean():.3f} cov90 {d.cov90.mean():.3f} | tail99 {d.tail99.mean():.3f} | crossing {d.cross.mean():.4f}')
print(f'storm p99.9: peak capture {d.pk_cap.mean():.2f} | q99 envelope at peak {d.q99_cap.mean():.2f} | window RMSE {d.sw_rmse.mean():.1f} cm')
