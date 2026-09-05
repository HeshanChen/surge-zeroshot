"""Split files for the 2026-09-04 audit re-runs (docs/protocol_symmetry_audit.md). Deterministic, seed 0.

exp_split_eu299v.csv   Europe corpus intervention, small arm under the clean protocol: the original 299-gauge
                       Europe-free training set unchanged; test = the 234-gauge Europe rotation test set
                       (superset of the original 76); val = 40 gauges country-stratified from the 334 extra
                       Europe-free gauges of the 633 corpus (non-test, never in this arm's training).
exp_split_c298strat.csv  Composition ablation, diverse arm: 298 gauges country-stratified (shuffle-trim) from the
                       760-gauge final pool; val/test/test_xdom identical to exp_split_final.csv.
exp_split_c298usjp.csv   Composition ablation, low-diversity arm: every USA and Japan gauge of the same 760 pool
                       (184 + 114 = 298, so count is matched exactly without sampling); same val/test.
"""
import pandas as pd, numpy as np
ROOT = '/Users/heshan/Desktop/surge_fm'
rng = np.random.default_rng(0)
sa = pd.read_csv(f'{ROOT}/catalog/static_attributes.csv').set_index('name')
fin = pd.read_csv(f'{ROOT}/catalog/exp_split_final.csv')
eu = pd.read_csv(f'{ROOT}/catalog/exp_split_eu.csv')
eus = pd.read_csv(f'{ROOT}/catalog/exp_split_eu_scale.csv')
euv2 = pd.read_csv(f'{ROOT}/catalog/exp_split_eu_v2.csv')
country = lambda n: n.split('-')[-2]

def stratified(names, k, rng):
    """country-proportional allocation, shuffle within country, trim/pad to exactly k (the ladder's recipe)"""
    df = pd.DataFrame({'name': names, 'c': [country(n) for n in names]})
    parts = []
    for c, g in df.groupby('c'):
        m = max(1, int(round(len(g)*k/len(df))))
        parts.append(g.sample(min(m, len(g)), random_state=int(rng.integers(1e9))))
    sub = pd.concat(parts).sample(frac=1, random_state=int(rng.integers(1e9)))
    if len(sub) > k: sub = sub.head(k)
    elif len(sub) < k:
        rest = df[~df.name.isin(sub.name)].sample(k-len(sub), random_state=int(rng.integers(1e9))); sub = pd.concat([sub, rest])
    return sub.name.tolist()

# 1. Europe 299 arm with a non-test val fold
tr299 = [n for n in eu[eu.fold == 'train'].name if n in sa.index]
te234 = [n for n in euv2[euv2.fold == 'test'].name if n in sa.index]
extra = [n for n in eus[eus.fold == 'train'].name if n in sa.index and n not in set(tr299) and n not in set(te234)]
assert len(tr299) == 299, len(tr299)
val40 = stratified(extra, 40, rng)
assert not set(val40) & set(tr299) and not set(val40) & set(te234)
out = pd.DataFrame({'name': tr299 + val40 + te234, 'fold': ['train']*len(tr299) + ['val']*len(val40) + ['test']*len(te234)})
out.to_csv(f'{ROOT}/catalog/exp_split_eu299v.csv', index=False)
print('eu299v', out.fold.value_counts().to_dict(), '| val from', len(extra), 'extra Europe-free gauges')

# 2./3. Composition ablation at 298 from the 760 pool
pool = [n for n in fin[fin.fold == 'train'].name if n in sa.index]
keep = fin[fin.fold != 'train']
strat = stratified(pool, 298, rng)
usjp = [n for n in pool if country(n) in ('usa', 'jpn')]
assert len(strat) == 298 and len(usjp) == 298, (len(strat), len(usjp))
for tag, tr in [('c298strat', strat), ('c298usjp', usjp)]:
    out = pd.concat([pd.DataFrame({'name': tr, 'fold': 'train'}), keep[['name', 'fold']]])
    out.to_csv(f'{ROOT}/catalog/exp_split_{tag}.csv', index=False)
    cs = pd.Series([country(n) for n in tr]).value_counts()
    print(tag, out.fold.value_counts().to_dict(), '| countries', len(cs), '| top', cs.head(4).to_dict())
