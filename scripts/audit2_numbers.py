"""Numbers the Nature draft quotes that are not printed by any single eval log, recomputed from the per-station CSVs
for all windows and continuous windows (audit 2, 2026-09-05): rotation union, continent table (explicit geographic
grouping), scaling per-doubling fit, gauge-free ladder mapping, factor-study win counts.
Usage: python3 scripts/audit2_numbers.py  -> prints and writes outputs/audit2_numbers.txt"""
import numpy as np, pandas as pd
ROOT = '/Users/heshan/Desktop/surge_fm'
L = []
def P(*a): s = ' '.join(str(x) for x in a); print(s); L.append(s)
def load(tag, suf): return pd.read_csv(f'{ROOT}/outputs/eval_full_{tag}{suf}.csv')
def sk(x, m='rmse_p', p='prmse_p'): return 100*(1-x[m].mean()/x[p].mean())

# continent grouping: geographic, stated in the Table 1 caption
def continent(n):
    cc = n.split('-')[-2]; base = n.split('-')[0]
    if cc in ('jpn', 'mys', 'omn', 'mdv'): return 'Asia'
    if base in ('fuerteventura', 'lapalma', 'arrecife', 'funchal') or cc in ('sen', 'ken', 'zaf'): return 'Africa'
    if cc in ('swe', 'dnk', 'nor', 'irl', 'est', 'esp', 'prt') or base in ('nice', 'saint_nazaire', 'dunkerque', 'weymouth', 'holyhead'): return 'Europe'
    if cc in ('usa', 'can', 'mex'): return 'N. America'
    if cc == 'aus': return 'Oceania'
    if cc in ('arg', 'bra', 'per', 'chl') or base == 'stanley': return 'S. America'
    return 'Pacific islands'   # kanton, penrhyn, papeete, rikitea, pohnpei, lautoka, numbo_noumea
ORDER = ['Asia', 'Europe', 'N. America', 'Oceania', 'Pacific islands', 'S. America', 'Africa']

for suf in ('', '_cont'):
    P(f'\n===== {suf or "all windows"} =====')
    # rotation union
    rots = [load(t, suf) for t in ('lstmq_v2rotjp', 'lstmq_v2roteu', 'lstmq_v2rotna', 'lstmq_v2rotocr2')]
    u = pd.concat(rots); P(f'rotation union: n={len(u)} pooled {sk(u):+.1f} @8h {sk(u, "r8", "p8"):+.1f} beating persistence {(u.rmse_p < u.prmse_p).sum()}/{len(u)}')
    for name, r in zip(('Japan', 'Europe', 'N America', 'Oceania r2'), rots): P(f'   {name}: n={len(r)} negative {(r.rmse_p >= r.prmse_p).sum()}')
    # continent table
    d = load('lstmq_v2final', suf); d['cont'] = [continent(n) for n in d.stn]
    P('continent table (n, pooled, @8h, NNSE@8h, wins):')
    for c in ORDER:
        x = d[d.cont == c]; P(f'   {c:16s} {len(x):3d} {sk(x):+5.0f} {sk(x, "r8", "p8"):+5.0f} {x.n8.mean():.3f} {(x.rmse_p < x.prmse_p).sum()}/{len(x)}')
    P(f'   {"All":16s} {len(d):3d} {sk(d):+5.0f} {sk(d, "r8", "p8"):+5.0f} {d.n8.mean():.3f} {(d.rmse_p < d.prmse_p).sum()}/{len(d)}')
    P('   members: ' + '; '.join(f'{c}: ' + ', '.join(sorted(n.split("-")[0] for n in d[d.cont == c].stn)) for c in ('Pacific islands', 'Africa', 'S. America', 'N. America', 'Oceania')))
    P('   negative gauges: ' + ', '.join(d[d.rmse_p >= d.prmse_p].stn))
    # scaling fit
    sizes = [64, 128, 256, 384, 512, 640, 760]; tags = [f'lstmq_v2g{s}' for s in sizes[:-1]] + ['lstmq_v2final']
    pooled = [sk(load(t, suf)) for t in tags]; s8 = [sk(load(t, suf), 'r8', 'p8') for t in tags]
    lg = np.log2(sizes)
    for lab, y in (('pooled', pooled), ('8h', s8)):
        slope = np.polyfit(lg, y, 1)[0]; steps = [f'{(y[i+1]-y[i])/(lg[i+1]-lg[i]):.1f}' for i in range(len(y)-1)]
        P(f'scaling {lab}: values {[round(v, 1) for v in y]}  log-linear slope {slope:.2f}/doubling; rung-to-rung per-doubling rates {steps}')
    # gauge-free ladder mapping
    for tag, lab in (('v2fonly_eot2', 'EOT20 statics + predicted sigma (all open)'), ('v2fonly_predscale', 'local statics + predicted sigma'), ('v2fonly', 'local statics + measured sigma'), ('v2final', 'flagship with water level zeroed at inference')):
        try:
            f = pd.read_csv(f'{ROOT}/outputs/eval_forcing_only_{tag}{suf}.csv'); col = [c for c in f.columns if 'nnse' in c.lower()][0]
            P(f'gauge-free {lab:48s} NNSE@8h {f[col].mean():.3f}  above climatology {(f[col] > 0.5).sum()}/{len(f)}  (cols {list(f.columns)[:8]})')
        except FileNotFoundError: P(f'gauge-free {lab}: no file for {tag}{suf}')
    # factor study wins
    try:
        a, b = load('lstmq_v2n298', suf).set_index('stn'), load('v7e_n298v', suf).set_index('stn'); j = a.join(b, rsuffix='_v7e', how='inner')
        P(f'factor: LSTM lower 8-h RMSE than v7e at {(j.r8 < j.r8_v7e).sum()}/{len(j)}; LSTM pooled {sk(a):+.0f} @8h {sk(a, "r8", "p8"):+.0f}; v7e pooled {sk(b):+.0f} @8h {sk(b, "r8", "p8"):+.0f}')
    except FileNotFoundError as e: P('factor:', e)
    # Europe intervention pairs and composition pairs
    for lab, ta, tb in (('Europe 593 vs 299', 'lstmq_v2roteu', 'lstmq_v2eu299'), ('composition 49-country vs 2-country', 'lstmq_v2c298strat', 'lstmq_v2c298usjp')):
        a, b = load(ta, suf).set_index('stn'), load(tb, suf).set_index('stn'); j = a.join(b, rsuffix='_b', how='inner')
        P(f'{lab}: n={len(j)} first lower 8-h RMSE at {(j.r8 < j.r8_b).sum()}/{len(j)}; pooled {sk(a):+.0f} vs {sk(b):+.0f}; @8h {sk(a, "r8", "p8"):+.0f} vs {sk(b, "r8", "p8"):+.0f}')
open(f'{ROOT}/outputs/audit2_numbers.txt', 'w').write('\n'.join(L) + '\n')
