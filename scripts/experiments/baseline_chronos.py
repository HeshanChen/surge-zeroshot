"""Generic-TSFM zero-shot baseline (docs/next_steps_sota.md #1c): Chronos-Bolt on the 77 held-out gauges.

The point of this comparison (Sun&Sun 2026): does an off-the-shelf time-series foundation model — which sees
ONLY the surge history, no wind/pressure forcing (inherent to generic TSFMs) — match a domain FM? Prediction:
it cannot anticipate forced storm onsets. Median of Chronos quantiles = point forecast.

Same windows/split as eval_full ([A] per-lead + pooled, [B-lite] extreme timestep + peak capture), persistence
compared on identical samples. torch threads capped to 4 to avoid starving the concurrent MPS training (P3).
Usage: python scripts/baseline_chronos.py [--modelsize bolt-base|bolt-small] [--nstations N]
"""
import sys, warnings, argparse, collections; warnings.filterwarnings('ignore')
import torch, pandas as pd, numpy as np
torch.set_num_threads(4)
sys.path.insert(0, '/Users/heshan/Desktop/surge_fm/src')
from data.dataset_v0 import load_station, ROOT
# boto3/botocore on this machine are version-mismatched and only probed by accelerate's SageMaker branch; return empty modules (audit 2026-09-04)
import importlib.abc, importlib.util, types
class _NilFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path, target=None):
        if name.split('.')[0] in ('boto3', 'botocore'): return importlib.util.spec_from_loader(name, self)
    def create_module(self, spec): return types.ModuleType(spec.name)
    def exec_module(self, m): pass
sys.meta_path.insert(0, _NilFinder())
from chronos import BaseChronosPipeline

ap = argparse.ArgumentParser()
ap.add_argument('--modelsize', default='bolt-base')
ap.add_argument('--nstations', type=int, default=0)
ap.add_argument('--split', default='exp_split.csv', help='split file in catalog/; use exp_split_n298v.csv to match the clean factor rows')
ap.add_argument('--tag', default='', help='output suffix, e.g. _masked')
ap.add_argument('--continuous_only', action='store_true', help='keep only windows whose 256 rows are consecutive hours (audit 2026-09-05); adds _cont to the outputs')
args = ap.parse_args()
if args.continuous_only: args.tag += '_cont'
class _Tee:
    def __init__(self, path): self.f = open(path, 'w'); self.stdout = sys.stdout
    def write(self, x): self.f.write(x); self.stdout.write(x)
    def flush(self): self.f.flush(); self.stdout.flush()
sys.stdout = _Tee(f'{ROOT}/outputs/eval_chronos_{args.modelsize}{args.tag}.log')

Tctx, Ttgt, W, H = 208, 48, 256, 48
LEADS = [4, 6, 8, 10, 12]; PCTS = [99, 99.5, 99.9]; PEAKP = 99.9
sp = pd.read_csv(f'{ROOT}/catalog/{args.split}')
te = list(sp[sp.fold == 'test'].name)
if args.nstations: te = te[:args.nstations]

pipe = BaseChronosPipeline.from_pretrained(f'amazon/chronos-{args.modelsize}', device_map='cpu', torch_dtype=torch.float32)
print(f'chronos-{args.modelsize} loaded | {len(te)} stations', flush=True)

rows = []; Tacc = {p: collections.Counter() for p in PCTS}; peak = {'true': [], 'pred': [], 'pers': [], 'n': 0}
def rmse(c, k): return (c[k]/c['n'])**0.5 if c['n'] else float('nan')

for idx, n in enumerate(te):
    a, tt = load_station(n, return_time=True)
    if a is None: continue
    thr = {p: float(np.percentile(a[:, 0], p))*100 for p in PCTS}
    starts = np.arange(0, len(a)-W, Ttgt)
    if args.continuous_only: starts = starts[(tt[starts+W-1]-tt[starts]) == (W-1)]
    if len(starts) == 0: continue
    ws = np.stack([a[st:st+W, 0] for st in starts])*100      # surge only, cm
    ctx = torch.tensor(ws[:, :Tctx]).float(); tru = ws[:, Tctx:Tctx+H]; last = ws[:, Tctx-1]
    preds = []
    with torch.no_grad():
        for i in range(0, len(ctx), 256):
            q, _ = pipe.predict_quantiles(ctx[i:i+256], prediction_length=H, quantile_levels=[0.5])
            preds.append(q[:, :, 0].numpy())
    pred = np.concatenate(preds)[:, :H]; pers = last[:, None].repeat(H, 1)
    e = pred - tru; ep = pers - tru
    rmse_h = np.sqrt((e**2).mean(0)); prmse_h = np.sqrt((ep**2).mean(0))
    rows.append(dict(stn=n, rmse_p=float(np.sqrt((e**2).mean())), prmse_p=float(np.sqrt((ep**2).mean())),
                     **{f'r{h}': rmse_h[h-1] for h in LEADS}, **{f'p{h}': prmse_h[h-1] for h in LEADS}))
    for p in PCTS:
        mt = tru >= thr[p]; c = Tacc[p]
        if mt.any(): c['se_m'] += float((e[mt]**2).sum()); c['se_p'] += float((ep[mt]**2).sum()); c['n'] += int(mt.sum())
    wm = tru.max(1) >= thr[PEAKP]
    if wm.any():
        tw = tru[wm]; it = tw.argmax(1); ar = np.arange(len(it))
        peak['true'] += tw[ar, it].tolist(); peak['pred'] += pred[wm][ar, it].tolist(); peak['pers'] += pers[wm][ar, it].tolist(); peak['n'] += len(it)
    if (idx+1) % 10 == 0: print(f'  ...{idx+1}/{len(te)}', flush=True)

df = pd.DataFrame(rows); m = df.mean(numeric_only=True)
print(f'\n==================== Chronos-{args.modelsize} zero-shot (history-only, NO forcing)  n={len(df)} ====================')
for L in LEADS:
    print(f'  L={L:>2}h : RMSE {m[f"r{L}"]:5.2f} | pers {m[f"p{L}"]:5.2f} | skill {100*(m[f"p{L}"]-m[f"r{L}"])/m[f"p{L}"]:+3.0f}%')
print(f'  pooled: RMSE {m.rmse_p:5.2f} | pers {m.prmse_p:5.2f} | skill {100*(m.prmse_p-m.rmse_p)/m.prmse_p:+3.0f}%')
for p in PCTS:
    c = Tacc[p]; print(f'  extreme timestep p{p}: RMSE {rmse(c,"se_m"):.2f} | pers {rmse(c,"se_p"):.2f} | skill {100*(rmse(c,"se_p")-rmse(c,"se_m"))/rmse(c,"se_p"):+.0f}%')
tr_, pd_, pp_ = map(np.array, (peak['true'], peak['pred'], peak['pers']))
print(f'  peak capture (p{PEAKP}, n={peak["n"]}): model {pd_.mean()/tr_.mean():.2f} | pers {pp_.mean()/tr_.mean():.2f}')
df.to_csv(f'{ROOT}/outputs/eval_chronos_{args.modelsize}{args.tag}.csv', index=False)
print(f'wrote outputs/eval_chronos_{args.modelsize}{args.tag}.csv\ndone', flush=True)
