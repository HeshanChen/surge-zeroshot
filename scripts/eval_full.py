"""CANONICAL eval — one run, one report, full test set, ALL decisive metrics. Use this for EVERY model
comparison. NEVER conclude from proxies (in-training peakcap), subsamples, or pooled RMSE alone.

Single forward pass per station -> derives every metric (held-out gauges, cm):
  A. Full-distribution per-lead: RMSE / MAE / NNSE + skill vs persistence (report L=4/6/8/10/12h + pooled 1-48h)
  B. Extreme (p99/99.5/99.9): window-level, timestep-level (high-water hours), peak amplitude, lead-time — vs persistence
  C. Peak-capture-by-lead: is the peak height captured better at short lead than long? (information-limit check)

Rule: proxies/subsamples are ONLY for "is training moving". Model conclusions come from this, on full data,
head-to-head. Prints progress; writes outputs/eval_full_<tag>.csv (per-station) + summary block.

Usage: python scripts/eval_full.py --ckpt <path> [--residual] [--device mps] [--nstations N] [--tag NAME]
"""
import sys, warnings, argparse, collections; warnings.filterwarnings('ignore')
import torch, pandas as pd, numpy as np
sys.path.insert(0, '/Users/heshan/Desktop/surge_fm/src')
from models.surge_jepa_v1 import SurgeJEPA_v1
from data.dataset_v0 import load_station, ROOT

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', default=f'{ROOT}/outputs/surge_jepa_foundation_residual_best.pt')
ap.add_argument('--device', default='mps')
ap.add_argument('--residual', action='store_true')
ap.add_argument('--nstations', type=int, default=0, help='>0 = subsample (SMOKE TEST ONLY, not for conclusions)')
ap.add_argument('--tag', default='model')
ap.add_argument('--split', default='exp_split.csv', help='split file in catalog/')
ap.add_argument('--model', default='v1', choices=['v1', 'v5', 'lstm', 'lstmq', 'v7'], help='v5/lstmq = multi-quantile (adds [D]); lstm = GlobalLSTM point baseline')
args = ap.parse_args()
RESIDUAL = args.residual; V5 = args.model in ('v5', 'lstmq', 'v7')
dev = args.device if (args.device != 'mps' or torch.backends.mps.is_available()) else 'cpu'
print(f'device {dev} | ckpt {args.ckpt.split("/")[-1]} | residual {RESIDUAL} | tag {args.tag}', flush=True)

Tctx, Ttgt, W, H = 208, 48, 256, 48
CFG = dict(C=5, D=384, H=8, tower_depth=8, fusion_depth=3, pred_dim=192, pred_depth=6, n_static=5)
PCTS = [99, 99.5, 99.9]; PEAKP = 99.9; LEADS = [4, 6, 8, 10, 12]
PEAK_LEAD_BUCKETS = [(1, 6), (7, 12), (13, 24), (25, 48)]

class _Tee:
    def __init__(self, path):
        import sys; self.f=open(path,'w'); self.stdout=sys.stdout
    def write(self,x): self.f.write(x); self.stdout.write(x)
    def flush(self): self.f.flush(); self.stdout.flush()
import sys as _sys
_sys.stdout = _Tee(f'{ROOT}/outputs/eval_full_{args.tag}.log')   # every printed [A-D] number persists

sa = pd.read_csv(f'{ROOT}/catalog/static_attributes.csv').set_index('name')
sp = pd.read_csv(f'{ROOT}/catalog/{args.split}')
tr = [n for n in sp[sp.fold == 'train'].name if n in sa.index]
te = [n for n in sp[sp.fold == 'test'].name if n in sa.index]
if args.nstations: te = te[:args.nstations]
def sfeat(n):
    r = sa.loc[n]; la, lo = np.radians(float(r.lat)), np.radians(float(r.lon))
    return np.array([np.cos(la)*np.cos(lo), np.cos(la)*np.sin(lo), np.sin(la), float(r.tidal_range_m), float(r.form_factor)], dtype='float32')
SF = {n: sfeat(n) for n in tr+te}; arr = np.stack([SF[n] for n in tr]); smu = arr.mean(0); ssd = arr.std(0)+1e-6
for n in SF: SF[n] = (SF[n]-smu)/ssd
def prep(name):
    a = load_station(name)
    if a is None: return None
    a = a.copy(); tstd = float(a[:, 0].std())+1e-6
    thr = {p: float(np.percentile(a[:, 0], p)) for p in PCTS}
    a[:, 1:] = (a[:, 1:]-a[:, 1:].mean(0))/(a[:, 1:].std(0)+1e-6)
    ws = np.stack([a[st:st+W] for st in range(0, len(a)-W, Ttgt)])
    mu = ws[:, :Tctx, 0].mean(1, keepdims=True); sd = ws[:, :Tctx, 0].std(1, keepdims=True)+1e-6
    ctx = np.concatenate([((ws[:, :Tctx, 0]-mu)/sd)[:, :, None], ws[:, :Tctx, 1:]], 2).transpose(0, 2, 1)
    s = torch.tensor(SF[name]).unsqueeze(0).expand(len(ws), -1)
    return (torch.tensor(ctx).float(), torch.tensor(ws[:, Tctx:, 1:].transpose(0, 2, 1)).float(),
            torch.tensor(ws[:, Tctx:, 0]).float(), tstd, torch.tensor(ws[:, Tctx-1, 0]).float(), s, thr)

if args.model == 'v5':
    from models.surge_jepa_v5 import SurgeJEPA_v5
    model = SurgeJEPA_v5(**CFG).to(dev)
elif args.model == 'lstmq':
    from models.baseline_lstm import GlobalLSTM
    model = GlobalLSTM(n_out=3).to(dev)
elif args.model == 'v7':
    from models.surge_jepa_v7 import SurgeJEPA_v7
    model = SurgeJEPA_v7(**CFG).to(dev)
elif args.model == 'lstm':
    from models.baseline_lstm import GlobalLSTM
    model = GlobalLSTM().to(dev)
else:
    model = SurgeJEPA_v1(**CFG).to(dev)
model.load_state_dict(torch.load(args.ckpt, map_location=dev)); model.eval()

rows = []                                           # per-station full-dist arrays
Wacc = {p: collections.Counter() for p in PCTS}; Tacc = {p: collections.Counter() for p in PCTS}
Lacc = {L: collections.Counter() for L in LEADS}
peak = {'lead': [], 'true': [], 'pred': [], 'pers': [], 'q99': [], 'under': 0, 'n': 0}
Q = collections.Counter()          # v5 quantile stats: coverage / tail coverage / crossing / sharpness
def rmse(c, k): return (c[k]/c['n'])**0.5 if c['n'] else float('nan')

for idx, n in enumerate(te):
    r = prep(n)
    if r is None: continue
    ctx, ff, tru, tstd, last, s, thr = r
    anc = last/tstd
    preds = []; qps = []
    with torch.no_grad():
        for i in range(0, len(ctx), 512):
            ap_ = anc[i:i+512].to(dev) if RESIDUAL else None
            _, sh = model.predict_window(ctx[i:i+512].to(dev), ff[i:i+512].to(dev), H//model.L, s[i:i+512].to(dev), ap_)
            if V5:
                o = sh.reshape(sh.shape[0], -1, model.n_out).cpu()*tstd
                preds.append(o[..., 0]); qps.append(o[..., 1:])
            else:
                preds.append(sh.reshape(sh.shape[0], -1).cpu()*tstd)
    pred = (torch.cat(preds)[:, :H]*100).numpy(); tru = (tru[:, :H]*100).numpy(); pers = (last*100).numpy()[:, None].repeat(H, 1)
    qarr = (torch.cat(qps)[:, :H, :]*100).numpy() if V5 else None        # (Nw,H,2): q90,q99 (cm)
    e = pred-tru; ep = pers-tru
    # A. full-dist per-lead (per-station)
    rmse_h = np.sqrt((e**2).mean(0)); mae_h = np.abs(e).mean(0); prmse_h = np.sqrt((ep**2).mean(0))
    sstot_h = ((tru-tru.mean(0, keepdims=True))**2).sum(0); nse_h = 1-(e**2).sum(0)/(sstot_h+1e-9); nnse_h = 1/(2-nse_h)
    rmse_p = float(np.sqrt((e**2).mean())); prmse_p = float(np.sqrt((ep**2).mean()))
    sstot = ((tru-tru.mean())**2).sum(); nnse_p = 1/(2-(1-(e**2).sum()/(sstot+1e-9)))
    rows.append(dict(stn=n, rmse_p=rmse_p, prmse_p=prmse_p, nnse_p=nnse_p,
                     **{f'r{h}': rmse_h[h-1] for h in LEADS}, **{f'p{h}': prmse_h[h-1] for h in LEADS},
                     **{f'm{h}': mae_h[h-1] for h in LEADS}, **{f'n{h}': nnse_h[h-1] for h in LEADS}))
    # B. extreme
    for p in PCTS:
        t = thr[p]*100; wm = tru.max(1) >= t; c = Wacc[p]; c['ntot'] += len(tru); c['nwin'] += int(wm.sum())
        if wm.any():
            ew, epw = e[wm], ep[wm]
            c['se_m'] += float((ew**2).sum()); c['se_p'] += float((epw**2).sum())
            c['ae_m'] += float(np.abs(ew).sum()); c['ae_p'] += float(np.abs(epw).sum()); c['n'] += ew.size
        mt = tru >= t; ct = Tacc[p]
        if mt.any():
            ct['se_m'] += float((e[mt]**2).sum()); ct['se_p'] += float((ep[mt]**2).sum()); ct['n'] += int(mt.sum())
    if V5:
        q90 = qarr[..., 0]; q99 = qarr[..., 1]; t99 = thr[99]*100
        Q['n'] += tru.size; Q['c90'] += int((tru <= q90).sum()); Q['c99'] += int((tru <= q99).sum())
        mt = tru >= t99
        if mt.any(): Q['tn'] += int(mt.sum()); Q['tc'] += int((tru[mt] <= q99[mt]).sum())
        Q['xc'] += int((q90 > q99).sum())
        storm = tru.max(1) >= thr[PEAKP]*100; calm = tru.max(1) < t99
        if storm.any(): Q['ss'] += float((q99[storm]-pred[storm]).sum()); Q['sn'] += int(q99[storm].size)
        if calm.any(): Q['cs'] += float((q99[calm]-pred[calm]).sum()); Q['cn'] += int(q99[calm].size)
    tp = thr[PEAKP]*100; wm = tru.max(1) >= tp
    if wm.any():
        ew, epw, tw, pw, prw = e[wm], ep[wm], tru[wm], pred[wm], pers[wm]
        it = tw.argmax(1); a_ = np.arange(len(it))
        peak['lead'] += (it+1).tolist(); peak['true'] += tw[a_, it].tolist(); peak['pred'] += pw[a_, it].tolist(); peak['pers'] += prw[a_, it].tolist()
        if V5: peak['q99'] += qarr[wm][a_, it, 1].tolist()
        peak['under'] += int((pw[a_, it] < tw[a_, it]).sum()); peak['n'] += len(it)
        for L in LEADS:
            i = L-1; cL = Lacc[L]; cL['se_m'] += float((ew[:, i]**2).sum()); cL['se_p'] += float((epw[:, i]**2).sum()); cL['n'] += len(it)
    if (idx+1) % 20 == 0: print(f'  ...{idx+1}/{len(te)} stations', flush=True)

df = pd.DataFrame(rows); m = df.mean(numeric_only=True)
print(f'\n==================== eval_full [{args.tag}]  n={len(df)} gauges ====================')
print('\n[A] Full-distribution per-lead (mean across gauges):')
print(f'  {"lead":<8}{"mdlRMSE":>9}{"skill%":>8}{"NNSE":>7}{"mdlMAE":>8}')
for L in LEADS:
    sk = 100*(m[f'p{L}']-m[f'r{L}'])/m[f'p{L}']
    print(f'  L={L:<5}{m[f"r{L}"]:>9.2f}{sk:>7.0f}%{m[f"n{L}"]:>7.3f}{m[f"m{L}"]:>8.2f}')
skp = 100*(m.prmse_p-m.rmse_p)/m.prmse_p
print(f'  {"pooled":<8}{m.rmse_p:>9.2f}{skp:>7.0f}%{m.nnse_p:>7.3f}')
print(f'  >> Ebel @L=8h (m): RMSE {m.r8/100:.3f}  MAE {m.m8/100:.3f}  NNSE {m.n8:.3f}')

print('\n[B] Extreme — window-level (48h window contains surge>=thr):')
print(f'  {"thr":<6}{"#win":>7}{"mdlRMSE":>9}{"perRMSE":>9}{"skill":>7}')
for p in PCTS:
    c = Wacc[p]; print(f'  p{p:<5}{c["nwin"]:>7}{rmse(c,"se_m"):>9.2f}{rmse(c,"se_p"):>9.2f}{100*(rmse(c,"se_p")-rmse(c,"se_m"))/rmse(c,"se_p"):>6.0f}%')
print('    timestep-level (high-water hours, surge>=thr):')
for p in PCTS:
    c = Tacc[p]; print(f'  p{p:<5}{c["n"]:>7}{rmse(c,"se_m"):>9.2f}{rmse(c,"se_p"):>9.2f}{100*(rmse(c,"se_p")-rmse(c,"se_m"))/rmse(c,"se_p"):>6.0f}%')

tr_, pd_, pp_, ld_ = map(np.array, (peak['true'], peak['pred'], peak['pers'], peak['lead']))
print(f'\n[B] Peak amplitude (true peak hour of p{PEAKP} windows, n={peak["n"]}): true {tr_.mean():.1f}cm')
print(f'  model  : bias {(pd_-tr_).mean():+.2f}  under {100*peak["under"]/max(peak["n"],1):.0f}%  MAE {np.abs(pd_-tr_).mean():.2f}  capture {pd_.mean()/tr_.mean():.2f}')
print(f'  persist: bias {(pp_-tr_).mean():+.2f}  MAE {np.abs(pp_-tr_).mean():.2f}  capture {pp_.mean()/tr_.mean():.2f}')
print(f'\n[B] Lead-time skill on p{PEAKP} windows:')
for L in LEADS:
    c = Lacc[L]; print(f'  L={L:>2}h: mdl {rmse(c,"se_m"):5.2f}  pers {rmse(c,"se_p"):5.2f}  skill {100*(rmse(c,"se_p")-rmse(c,"se_m"))/rmse(c,"se_p"):+3.0f}%')

print(f'\n[C] Peak-capture by peak-lead-position (n per bucket):')
for lo, hi in PEAK_LEAD_BUCKETS:
    mk = (ld_ >= lo) & (ld_ <= hi)
    if mk.sum(): print(f'  peak at {lo:>2}-{hi:<2}h: n={int(mk.sum()):>4}  model capture {pd_[mk].mean()/tr_[mk].mean():.2f}  persist {pp_[mk].mean()/tr_[mk].mean():.2f}')

if V5:
    q9_ = np.array(peak['q99'])
    sh_storm = Q['ss']/max(Q['sn'], 1); sh_calm = Q['cs']/max(Q['cn'], 1)
    print(f'\n[D] Quantile section (v5). Pre-registered gates: tail-cov>=0.90, q99 peak capture>=0.6, sharpness ratio>=2, point non-inferior to v1.')
    print(f'  coverage overall : q90 {Q["c90"]/max(Q["n"],1):.3f} (nominal .90) | q99 {Q["c99"]/max(Q["n"],1):.3f} (nominal .99)')
    print(f'  TAIL coverage (hours y>=p99): q99 {Q["tc"]/max(Q["tn"],1):.3f}   (tail collapse if << .90)')
    print(f'  sharpness (q99-point): storm {sh_storm:.2f} cm | calm {sh_calm:.2f} cm | ratio {sh_storm/max(sh_calm,1e-6):.2f}  (constant envelope if ~1)')
    print(f'  crossing rate (q90>q99): {Q["xc"]/max(Q["n"],1):.4f}')
    print(f'  q99 peak-hour envelope capture (p{PEAKP} windows, n={len(q9_)}): {q9_.mean()/tr_.mean():.2f}   (point {pd_.mean()/tr_.mean():.2f}, persist {pp_.mean()/tr_.mean():.2f})')
    for lo, hi in PEAK_LEAD_BUCKETS:
        mk = (ld_ >= lo) & (ld_ <= hi)
        if mk.sum(): print(f'    peak at {lo:>2}-{hi:<2}h: q99 capture {q9_[mk].mean()/tr_[mk].mean():.2f}  (point {pd_[mk].mean()/tr_[mk].mean():.2f})')

df.to_csv(f'{ROOT}/outputs/eval_full_{args.tag}.csv', index=False)
print(f'\nwrote outputs/eval_full_{args.tag}.csv\ndone', flush=True)
