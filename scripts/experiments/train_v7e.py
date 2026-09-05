"""Clean-protocol trainer for the v7e transformer (hourly forcing tokens WITH hour+channel embeddings).
Same recipe as scripts/train_v7.py except: checkpoint selection on the split's VAL fold (never test),
seismic mask active (dataset_v0 default), full 12 epochs unless --epochs, --lr exposed so the learning-rate
asymmetry of the factor study (LSTM 1e-3 vs transformers 3e-4) can be tested both ways, and the training
log is persisted to outputs/train_v7e{tag}.log.
Usage: python3 scripts/train_v7e.py --split exp_split_n298v.csv --tag _n298v [--lr 3e-4] [--epochs 12]
Verdict: python3 scripts/eval_full.py --model v7e --ckpt outputs/surge_v7e{tag}_best.pt --residual --split <same split> --tag v7e{tag}"""
import sys, math, time, warnings, argparse; warnings.filterwarnings('ignore')
import torch, pandas as pd, numpy as np
sys.path.insert(0, '/Users/heshan/Desktop/surge_fm/src')
from models.surge_jepa_v7e import SurgeJEPA_v7e, loss_v7e
from data.dataset_v0 import load_station, ROOT
ap = argparse.ArgumentParser()
ap.add_argument('--split', default='exp_split_n298v.csv')
ap.add_argument('--tag', default='_n298v')
ap.add_argument('--lr', type=float, default=3e-4)
ap.add_argument('--epochs', type=int, default=12)
ap.add_argument('--max_stations', type=int, default=0, help='smoke test only')
ap.add_argument('--arch', default='v7e', choices=['v7e', 'v7'], help='v7 = the original position-blind forcing tokens, trained under this same clean protocol (controlled pair)')
args = ap.parse_args()

class _Tee:
    def __init__(self, path): self.f = open(path, 'w'); self.stdout = sys.stdout
    def write(self, x): self.f.write(x); self.stdout.write(x)
    def flush(self): self.f.flush(); self.stdout.flush()
sys.stdout = _Tee(f'{ROOT}/outputs/train_v7e{args.tag}.log')

dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'device {dev} | split {args.split} | tag {args.tag!r} | lr {args.lr} | epochs {args.epochs}', flush=True)
Tctx, Ttgt, W = 208, 48, 256
TAUS = (0.90, 0.99); LAM_Q = 1.0
sa = pd.read_csv(f'{ROOT}/catalog/static_attributes.csv').set_index('name')
sp = pd.read_csv(f'{ROOT}/catalog/{args.split}')
tr = [n for n in sp[sp.fold == 'train'].name if n in sa.index]
te = [n for n in sp[sp.fold == 'val'].name if n in sa.index]   # SELECTION FOLD = val (trainer never touches test)
assert te, f'{args.split} has no val fold; refusing to select checkpoints on test'
if args.max_stations: tr = tr[:args.max_stations]; te = te[:max(2, args.max_stations//2)]
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
C=[]; Fr=[]; T=[]; St=[]; A=[]; skipped=0
for n in tr:
    try: r = prep(n, W)
    except Exception: skipped += 1; continue
    if r: C.append(r[0]); Fr.append(r[1]); T.append(r[2]); St.append(r[5]); A.append(r[4]/r[3])
if skipped: print(f'skipped {skipped} unloadable stations', flush=True)
C=torch.cat(C); Fr=torch.cat(Fr); T=torch.cat(T); St=torch.cat(St); A=torch.cat(A); N=len(C)
print(f'train stations {len(tr)} | train windows {N} | LAM_Q {LAM_Q}', flush=True)
TE=[]
for n in te:
    try:
        r = prep(n, Ttgt)
        if r is not None: TE.append((n,)+tuple(r))
    except Exception: pass
print(f'val stations cached {len(TE)}', flush=True)
CFG = dict(C=5, D=384, H=8, tower_depth=8, fusion_depth=3, pred_dim=192, pred_depth=6, n_static=5)
if args.arch == 'v7e':
    model = SurgeJEPA_v7e(**CFG).to(dev)
else:
    from models.surge_jepa_v7 import SurgeJEPA_v7
    model = SurgeJEPA_v7(**CFG).to(dev)
print(f'arch {args.arch}', flush=True)
print(f'params {sum(p.numel() for p in model.parameters())/1e6:.2f}M', flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
def loss_q(ctx, ff, tgt, static, anchor):
    return loss_v7e(model, ctx, ff, tgt, static, anchor, lam_q=LAM_Q)
def zeroshot():   # PROXY ONLY (val fold); verdict = eval_full --model v7e
    model.eval(); zsp=[]; pep=[]; zs8=[]; pe8=[]; pcap=[]; qcap=[]; cov=[]
    with torch.no_grad():
        for n, ctx, ff, tg, tstd, last, s in TE:
            anc = last/tstd; ps=[]
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
EP=args.epochs; bs=64; spe=max(1, N//bs); tot=EP*spe; warm=max(1, int(0.1*tot)); step=0; t0=time.time(); best=999
z = zeroshot(); print(f'epoch 0 (fresh=persistence): pooled {z[0]:.2f}/{z[1]:.2f} | @8h {z[2]:.2f}/{z[3]:.2f} | ptcap {z[4]:.2f} q99cap {z[5]:.2f} cov99 {z[6]:.3f}', flush=True)
for ep in range(EP):
    perm = torch.randperm(N)
    for i in range(0, N-bs, bs):
        ix = perm[i:i+bs]
        lr = args.lr*(step/warm if step < warm else 0.5*(1+math.cos(math.pi*(step-warm)/max(1, tot-warm))))
        for g in opt.param_groups: g['lr'] = lr
        loss, lg, lq = loss_q(C[ix].to(dev), Fr[ix].to(dev), T[ix].to(dev), St[ix].to(dev), A[ix].to(dev))
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step(); opt.zero_grad(); step += 1
    torch.save(model.state_dict(), f'{ROOT}/outputs/surge_v7e{args.tag}.pt')
    if (ep+1) % 3 == 0 or ep == EP-1:
        zsp, pep, zs8, pe8, pc, qc, cv = zeroshot(); tag = ''
        if zs8 < best: best = zs8; torch.save(model.state_dict(), f'{ROOT}/outputs/surge_v7e{args.tag}_best.pt'); tag = ' <- BEST@8h(val)'
        print(f'ep {ep+1}/{EP} loss {loss.item():.3f}(Lg{lg:.3f} Lq{lq:.3f}) | pooled {zsp:.2f}/{pep:.2f} | @8h {zs8:.2f}/{pe8:.2f} skill {100*(pe8-zs8)/pe8:+.0f}% | ptcap {pc:.2f} q99cap {qc:.2f} cov99 {cv:.3f}{tag} {time.time()-t0:.0f}s', flush=True)
    else:
        print(f'ep {ep+1}/{EP} loss {loss.item():.3f}(Lg{lg:.3f} Lq{lq:.3f}) {time.time()-t0:.0f}s', flush=True)
print(f'\nBEST val RMSE@8h {best:.2f}cm -> outputs/surge_v7e{args.tag}_best.pt (last-epoch weights: surge_v7e{args.tag}.pt) | steps {step}', flush=True)
