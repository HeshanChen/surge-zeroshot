"""LSTM + multi-quantile head — compose the falsifier's winning backbone (per-hour forcing decoder, pooled
+25% / extremes +29-32%) with v5's proven quantile envelope (q99 peak capture 0.92, calibrated).

Under EVERY branch of the paper decision this model is needed: it is either the new flagship or the strongest
baseline. Same recipe as train_baseline_lstm.py; head n_out=3 (point, q90, q99), loss = v1's weighted MSE on
point + unweighted pinball(.90,.99) on quantiles (v5 pattern). BEST by point RMSE@8h.
-> outputs/baseline_lstmq_best.pt ; verdict: eval_full --model lstmq
"""
import sys, math, time, warnings, argparse; warnings.filterwarnings('ignore')
import torch, pandas as pd, numpy as np
sys.path.insert(0, '/Users/heshan/Desktop/surge_fm/src')
from models.baseline_lstm import GlobalLSTM
from data.dataset_v0 import load_station, ROOT
ap = argparse.ArgumentParser()
ap.add_argument('--split', default='exp_split.csv', help='split file in catalog/ (exp_split.csv=Japan holdout, exp_split_eu.csv, exp_split_na.csv)')
ap.add_argument('--tag', default='', help='suffix for output ckpts, e.g. _eu')
ap.add_argument('--forcing_only', action='store_true', help='zero the surge context channel and the persistence anchor (true ungauged mode)')
args = ap.parse_args()
dev = 'mps' if torch.backends.mps.is_available() else 'cpu'; print(f'device {dev} | split {args.split} | tag {args.tag!r}', flush=True)
Tctx, Ttgt, W = 208, 48, 256
TAUS = (0.90, 0.99); LAM_Q = 1.0
sa = pd.read_csv(f'{ROOT}/catalog/static_attributes.csv').set_index('name')
sp = pd.read_csv(f'{ROOT}/catalog/{args.split}')
tr = [n for n in sp[sp.fold == 'train'].name if n in sa.index]
te = [n for n in sp[sp.fold == 'val'].name if n in sa.index]   # SELECTION FOLD = val (trainer never touches test)
assert te, f'{args.split} has no val fold; refusing to select checkpoints on test'
def sfeat(n):
    r = sa.loc[n]; la, lo = np.radians(float(r.lat)), np.radians(float(r.lon))
    return np.array([np.cos(la)*np.cos(lo), np.cos(la)*np.sin(lo), np.sin(la), float(r.tidal_range_m), float(r.form_factor)], dtype='float32')
SF = {n: sfeat(n) for n in tr+te}; arr = np.stack([SF[n] for n in tr]); smu = arr.mean(0); ssd = arr.std(0)+1e-6
for n in SF: SF[n] = (SF[n]-smu)/ssd
def prep(name, stride):
    a = load_station(name)
    if a is None: return None
    a = a.copy(); tstd = float(a[:, 0].std())+1e-6; a[:, 1:] = (a[:, 1:]-a[:, 1:].mean(0))/(a[:, 1:].std(0)+1e-6)
    ws = np.stack([a[st:st+W] for st in range(0, len(a)-W, stride)])
    mu = ws[:, :Tctx, 0].mean(1, keepdims=True); sd = ws[:, :Tctx, 0].std(1, keepdims=True)+1e-6
    ctx = np.concatenate([((ws[:, :Tctx, 0]-mu)/sd)[:, :, None], ws[:, :Tctx, 1:]], 2).transpose(0, 2, 1)
    s = torch.tensor(SF[name]).unsqueeze(0).expand(len(ws), -1)
    return (torch.tensor(ctx).float(), torch.tensor(ws[:, Tctx:, 1:].transpose(0, 2, 1)).float(),
            torch.tensor(ws[:, Tctx:, 0]/tstd).float(), tstd, torch.tensor(ws[:, Tctx-1, 0]).float(), s)
C=[]; Fr=[]; T=[]; St=[]; A=[]
skipped=0
for n in tr:
    try: r = prep(n, W)
    except Exception: skipped+=1; continue
    if r: C.append(r[0]); Fr.append(r[1]); T.append(r[2]); St.append(r[5]); A.append(r[4]/r[3])
if skipped: print(f'skipped {skipped} unloadable stations', flush=True)
C=torch.cat(C); Fr=torch.cat(Fr); T=torch.cat(T); St=torch.cat(St); A=torch.cat(A); N=len(C)
if args.forcing_only:
    C[:, 0, :] = 0.; A[:] = 0.
    print('FORCING-ONLY mode: surge context channel and anchors zeroed', flush=True)
print(f'train windows {N} | LAM_Q {LAM_Q}', flush=True)
TE=[]
for n in te:
    try:
        r=prep(n, Ttgt)
        if r is not None: TE.append((n,)+tuple(r))
    except Exception: pass; print(f'val stations cached {len(TE)}', flush=True)
model = GlobalLSTM(n_out=3).to(dev)
print(f'params {sum(p.numel() for p in model.parameters())/1e6:.2f}M', flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
def loss_q(ctx, ff, tgt, static, anchor):
    _, out = model.predict_window(ctx, ff, tgt.shape[1], static, anchor)   # (B,48,3)
    ph = out[..., 0]; wt = 1.0 + tgt.abs()
    Lg = (wt*(ph-tgt)**2).mean()/wt.mean()
    Lq = 0.0
    for k, tau in enumerate(TAUS):
        d = tgt - out[..., 1+k]
        Lq = Lq + torch.maximum(tau*d, (tau-1.0)*d).mean()
    return Lg + LAM_Q*Lq, Lg.detach(), (Lq.detach() if torch.is_tensor(Lq) else Lq)
def zeroshot():   # zero-shot on VAL fold (selection only); verdict = eval_full on test
    model.eval(); zsp=[]; pep=[]; zs8=[]; pe8=[]; pcap=[]; qcap=[]; cov=[]
    with torch.no_grad():
        for n, ctx, ff, tg, tstd, last, s in TE:
            if args.forcing_only:
                ctx = ctx.clone(); ctx[:, 0, :] = 0.; anc = torch.zeros_like(last)
            else:
                anc = last/tstd
            ps=[]
            for i in range(0, len(ctx), 512):
                _, o = model.predict_window(ctx[i:i+512].to(dev), ff[i:i+512].to(dev), 48, s[i:i+512].to(dev), anc[i:i+512].to(dev))
                ps.append(o.cpu())
            out = torch.cat(ps)[:, :48, :]*tstd*100
            pred = out[..., 0]; q99 = out[..., 2]
            tru = tg[:, :48]*tstd*100; per = (last*100).unsqueeze(1)
            e = pred-tru; ep = per-tru
            zsp.append(float(torch.sqrt((e**2).mean()))); pep.append(float(torch.sqrt((ep**2).mean())))
            zs8.append(float(torch.sqrt((e[:, 7]**2).mean()))); pe8.append(float(torch.sqrt((ep[:, 7]**2).mean())))
            cov.append(float((tru <= q99).float().mean()))
            thr = torch.quantile(tru.flatten(), 0.999); wm = tru.max(1).values >= thr
            if wm.any():
                tw = tru[wm]; it = tw.argmax(1); ar = torch.arange(len(it))
                pcap.append(float(pred[wm][ar, it].sum()/(tw[ar, it].sum()+1e-6)))
                qcap.append(float(q99[wm][ar, it].sum()/(tw[ar, it].sum()+1e-6)))
    model.train()
    return np.mean(zsp), np.mean(pep), np.mean(zs8), np.mean(pe8), float(np.mean(pcap)) if pcap else 0., float(np.mean(qcap)) if qcap else 0., float(np.mean(cov))
EP=12; bs=64; spe=N//bs; tot=EP*spe; warm=int(0.1*tot); step=0; t0=time.time(); best=999
z = zeroshot(); print(f'epoch 0 (fresh=persistence): pooled {z[0]:.2f}/{z[1]:.2f} | @8h {z[2]:.2f}/{z[3]:.2f} | ptcap {z[4]:.2f} q99cap {z[5]:.2f} cov99 {z[6]:.3f}', flush=True)
for ep in range(EP):
    perm = torch.randperm(N)
    for i in range(0, N-bs, bs):
        ix = perm[i:i+bs]
        lr = 1e-3*(step/warm if step < warm else 0.5*(1+math.cos(math.pi*(step-warm)/(tot-warm))))
        for g in opt.param_groups: g['lr'] = lr
        loss, lg, lq = loss_q(C[ix].to(dev), Fr[ix].to(dev), T[ix].to(dev), St[ix].to(dev), A[ix].to(dev))
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step(); opt.zero_grad(); step += 1
    torch.save(model.state_dict(), f'{ROOT}/outputs/baseline_lstmq{args.tag}.pt')
    if (ep+1) % 3 == 0 or ep == EP-1:
        zsp, pep, zs8, pe8, pc, qc, cv = zeroshot(); tag = ''
        if zs8 < best: best = zs8; torch.save(model.state_dict(), f'{ROOT}/outputs/baseline_lstmq{args.tag}_best.pt'); tag = ' <- BEST@8h(val)'
        print(f'ep {ep+1}/{EP} loss {loss.item():.3f}(Lg{lg:.3f} Lq{lq:.3f}) | pooled {zsp:.2f}/{pep:.2f} | @8h {zs8:.2f}/{pe8:.2f} skill {100*(pe8-zs8)/pe8:+.0f}% | ptcap {pc:.2f} q99cap {qc:.2f} cov99 {cv:.3f}{tag} {time.time()-t0:.0f}s', flush=True)
    else:
        print(f'ep {ep+1}/{EP} loss {loss.item():.3f}(Lg{lg:.3f} Lq{lq:.3f}) {time.time()-t0:.0f}s', flush=True)
print(f'\nBEST val RMSE@8h {best:.2f}cm -> outputs/baseline_lstmq{args.tag}_best.pt (last-epoch weights: baseline_lstmq{args.tag}.pt)', flush=True)
print('Verdict: python scripts/eval_full.py --model lstmq --ckpt outputs/baseline_lstmq_best.pt --residual --tag lstmq_best', flush=True)
