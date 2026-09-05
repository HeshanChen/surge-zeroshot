"""Spatial-block bootstrap confidence intervals for the headline comparisons (audit response 2026-09-05).

Gauges are spatially correlated, so per-gauge Wilcoxon p-values overstate certainty. Here the resampling unit
is a block of gauges (a coarse ocean region, and separately the country code), redrawn with replacement 4,000
times; skill = 1 - mean(model RMSE)/mean(persistence RMSE) over the resampled gauges, reported with the 2.5 and
97.5 percentiles. Paired differences (model minus reference RMSE) and win fractions get the same treatment.
Usage: python3 scripts/bootstrap_cis.py [suffix]   (suffix '' = all-window evals, '_cont' = continuous-window evals)
-> outputs/bootstrap_cis{suffix}.txt"""
import sys, numpy as np, pandas as pd
ROOT = '/Users/heshan/Desktop/surge_fm'
SUF = sys.argv[1] if len(sys.argv) > 1 else ''
rng = np.random.default_rng(0); B = 4000
sa = pd.read_csv(f'{ROOT}/catalog/static_attributes.csv').set_index('name')

def region(n):
    la, lo = sa.loc[n, 'lat'], sa.loc[n, 'lon']
    if 50 <= la <= 72 and 5 <= lo <= 32: return 'Baltic/North Sea'
    if 35 <= la <= 62 and -12 <= lo < 5: return 'W Europe/Iberia'
    if 24 <= la <= 50 and -82 <= lo <= -60: return 'US East/Canada'
    if 30 <= la <= 62 and -130 <= lo <= -115: return 'US/Canada West'
    if 20 <= la <= 50 and 120 <= lo <= 150: return 'Japan/E Asia'
    if -50 <= la <= -10 and 110 <= lo <= 180: return 'Australia/NZ'
    if -40 <= la <= 40 and -20 <= lo <= 65: return 'Africa/Indian Oc'
    if la < 0 and -90 <= lo <= -30: return 'S America'
    if la < 20 and 95 <= lo <= 130: return 'SE Asia'
    if -30 <= la <= 30 and (lo > 150 or lo < -100): return 'Pacific islands'
    return 'other'
def country(n): return n.split('-')[-2]

def block_boot(df, stat, blockfn):
    blocks = df.groupby(df.index.map(blockfn)).indices; keys = list(blocks)
    out = []
    for _ in range(B):
        pick = rng.choice(len(keys), len(keys), replace=True)
        idx = np.concatenate([blocks[keys[k]] for k in pick]); out.append(stat(df.iloc[idx]))
    lo, hi = np.percentile(out, [2.5, 97.5]); return stat(df), lo, hi, len(keys)

def skill(m, p): return lambda d: 100*(1-d[m].mean()/d[p].mean())
def diff(a, b): return lambda d: (d[a]-d[b]).mean()
def winfrac(a, b): return lambda d: 100*(d[a] < d[b]).mean()

lines = [f'SPATIAL-BLOCK BOOTSTRAP CIs (B={B}; evals{SUF or " (all windows)"})', '']
def report(title, df, stats):
    lines.append(f'--- {title} (n={len(df)} gauges) ---')
    for lab, st in stats:
        for bname, bfn in [('region blocks', region), ('country blocks', country)]:
            v, lo, hi, nb = block_boot(df, st, bfn)
            lines.append(f'  {lab:44s} {v:7.2f}  95% CI [{lo:7.2f}, {hi:7.2f}]   ({nb} {bname})')
    lines.append('')

def load(tag):
    try: return pd.read_csv(f'{ROOT}/outputs/eval_full_{tag}{SUF}.csv').set_index('stn')
    except FileNotFoundError: return None

d = load('lstmq_v2final')
if d is not None:
    st = [('pooled skill vs persistence (%)', skill('rmse_p', 'prmse_p')), ('8-h skill vs persistence (%)', skill('r8', 'p8')),
          ('gauges beating persistence, pooled (%)', winfrac('rmse_p', 'prmse_p'))]
    if 's8_m' in d.columns:
        dd = d.dropna(subset=['s8_m']); report('Deployment, 84 marine test gauges', d, st)
        report('Deployment, storm windows at 8 h', dd, [('storm 8-h skill vs persistence (%)', skill('s8_m', 's8_p'))])
        if 'pkcov' in d.columns: report('Deployment, q99 covers the true storm peak', d.dropna(subset=['pkcov']), [('fraction of p99.9 peaks under q99 (%)', lambda x: 100*np.average(x.pkcov, weights=x.n_storm))])
    else: report('Deployment, 84 marine test gauges', d, st)

for rot, tag in [('Japan+Pacific', 'lstmq_v2rotjp'), ('Europe', 'lstmq_v2roteu'), ('North America', 'lstmq_v2rotna'), ('Oceania (median run)', 'lstmq_v2rotocr2')]:
    d = load(tag)
    if d is not None: report(f'Rotation {rot}', d, [('pooled skill (%)', skill('rmse_p', 'prmse_p')), ('8-h skill (%)', skill('r8', 'p8'))])

# Europe corpus intervention: paired 593 vs 299 on the shared gauges
a, b = load('lstmq_v2roteu'), load('lstmq_v2eu299')
if a is not None and b is not None:
    j = a[['r8', 'rmse_p', 'p8', 'prmse_p']].join(b[['r8', 'rmse_p']], rsuffix='_299', how='inner')
    report('Europe intervention, 593-gauge minus 299-gauge corpus', j,
           [('8-h skill 593 (%)', skill('r8', 'p8')), ('8-h skill 299 (%)', skill('r8_299', 'p8')),
            ('8-h RMSE difference 593-299 (cm)', diff('r8', 'r8_299')), ('gauges where 593 has lower 8-h RMSE (%)', winfrac('r8', 'r8_299'))])

# composition ablation at fixed count
a, b = load('lstmq_v2c298strat'), load('lstmq_v2c298usjp')
if a is not None and b is not None:
    j = a[['r8', 'rmse_p', 'p8', 'prmse_p']].join(b[['r8', 'rmse_p']], rsuffix='_usjp', how='inner')
    report('Composition ablation, 49-country minus 2-country 298', j,
           [('8-h skill 49-country (%)', skill('r8', 'p8')), ('8-h skill 2-country (%)', skill('r8_usjp', 'p8')),
            ('8-h RMSE difference (cm)', diff('r8', 'r8_usjp')), ('gauges where 49-country lower (%)', winfrac('r8', 'r8_usjp'))])

# GTSM tiers
try:
    g = pd.read_csv(f'{ROOT}/outputs/eval_gtsm_symmetric{SUF}.csv').set_index('stn')
    gs = g.dropna(subset=['storm8_gf', 'storm8_ours'])
    report('Tier A: gauge-free minus GTSM (high-passed)', g,
           [('8-h RMSE difference gf-GTSM (cm)', diff('rmse8_gf', 'rmse8_gtsm_hp')), ('48-h RMSE difference (cm)', diff('rmse48_gf', 'rmse48_gtsm_hp')),
            ('gauges where gauge-free lower at 8 h (%)', winfrac('rmse8_gf', 'rmse8_gtsm_hp'))])
    report('Tier A storm windows: gauge-free minus GTSM', gs,
           [('8-h storm RMSE difference (cm)', diff('storm8_gf', 'storm8_gtsm_hp')), ('48-h storm RMSE difference (cm)', diff('storm48_gf', 'storm48_gtsm_hp')),
            ('peak capture difference gf-GTSM', lambda d: (d.pkcap_gf-d.pkcap_gtsm_hp).mean())])
    report('Tier B: forecast minus GTSM + 25-h bias correction', g,
           [('8-h RMSE difference (cm)', diff('rmse8_ours', 'rmse8_anch_hp_lp')), ('gauges where forecast lower at 8 h (%)', winfrac('rmse8_ours', 'rmse8_anch_hp_lp')),
            ('48-h RMSE difference (cm)', diff('rmse48_ours', 'rmse48_anch_hp_lp'))])
    report('Tier B storm windows: forecast minus GTSM + 25-h bias correction', gs,
           [('8-h storm RMSE difference (cm)', diff('storm8_ours', 'storm8_anch_hp_lp')), ('48-h storm RMSE difference (cm)', diff('storm48_ours', 'storm48_anch_hp_lp')),
            ('peak capture difference', lambda d: (d.pkcap_ours-d.pkcap_anch_hp_lp).mean())])
except FileNotFoundError: pass

txt = '\n'.join(lines); open(f'{ROOT}/outputs/bootstrap_cis{SUF}.txt', 'w').write(txt+'\n'); print(txt)
