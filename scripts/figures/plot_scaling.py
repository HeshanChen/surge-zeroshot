"""Scaling figure v2 (clean protocol): 7 rungs on the fixed 84-gauge marine test set.
All numbers parsed from the persisted eval logs (outputs/eval_full_lstmq_v2*.log) — no hardcoding.
-> outputs/scaling_figure.{pdf,png}"""
import os as _os
_ROOT = _os.environ.get('SURGE_ROOT') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
import sys, re, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, f'{_ROOT}/scripts')
import matplotlib; matplotlib.use('Agg')
from pubstyle import apply, save_pub, PAL, W2, panel_label
apply()
import pandas as pd
import matplotlib.pyplot as plt
ROOT = _ROOT
import sys as _sys
SUF = _sys.argv[1] if len(_sys.argv) > 1 else ''   # '_cont' = continuous windows only (audit 2)

RUNGS = [(64,'v2g64'),(128,'v2g128'),(256,'v2g256'),(384,'v2g384'),(512,'v2g512'),(640,'v2g640'),(760,'v2final')]
N=[]; pooled=[]; at8=[]; hw=[]; cap=[]
for n_, tag in RUNGS:
    d = pd.read_csv(f'{ROOT}/outputs/eval_full_lstmq_{tag}{SUF}.csv')
    sk = lambda m,p: 100*(d[p].mean()-d[m].mean())/d[p].mean()
    log = open(f'{ROOT}/outputs/eval_full_lstmq_{tag}{SUF}.log').read()
    hwm = re.search(r'p99\.9\s+\d+\s+([\d.]+)\s+([\d.]+)', log.split('timestep-level')[1])
    N.append(n_); pooled.append(sk('rmse_p','prmse_p')); at8.append(sk('r8','p8'))
    hw.append(100*(float(hwm.group(2))-float(hwm.group(1)))/float(hwm.group(2)))
    cap.append(float(re.search(r'capture ([\d.]+)\n  persist', log).group(1)))

fig, axes = plt.subplots(2, 2, figsize=(W2, 3.3)); axes = list(axes.flat)
panels = [('Pooled skill vs persistence (%)', pooled), ('Skill @8 h (%)', at8),
          ('High-water p99.9 skill (%)', hw), ('Peak capture (fraction)', cap)]
for ax, (title, y) in zip(axes, panels):
    ax.plot(N, y, 'o-', color=PAL['blue'], lw=1.1, ms=3.5)
    ax.set_xscale('log'); ax.set_xticks(N); ax.set_xticklabels(N, rotation=45, fontsize=6.5)
    ax.minorticks_off(); ax.grid(alpha=0.3, lw=0.4); ax.set_title(title, fontsize=7)
    ax.set_xlabel('training gauges')
for ax, letter in zip(axes, 'abcd'): panel_label(ax, letter, dx=-0.075, dy=1.13)
fig.tight_layout(w_pad=2.0, h_pad=1.6)
save_pub(fig, f'{ROOT}/outputs/scaling_figure{SUF}')
print(f'wrote outputs/scaling_figure{SUF}.{{pdf,png}} from logs')
