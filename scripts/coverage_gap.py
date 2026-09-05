"""Coverage-gap indicator for the Discussion: how much of the world's low-lying coastal population lives in countries
without a national operational storm-surge forecast service, and how many gauges of our corpus sit in such countries.
Inputs: World Bank indicators fetched 2026-09-05 (EN.POP.EL5M.ZS = share of population living below 5 m elevation,
latest value, CIESIN-derived; SP.POP.TOTL for 2023) in data/raw/coverage_gap/, and the sourced country classification
catalog/surge_forecast_systems.csv (status: national / regional_only / none_found).
Usage: python3 scripts/coverage_gap.py -> outputs/coverage_gap.txt + outputs/coverage_gap_countries.csv"""
import json, numpy as np, pandas as pd
ROOT = '/Users/heshan/Desktop/surge_fm'; D = f'{ROOT}/data/raw/coverage_gap'
def wb(path):
    recs = json.load(open(path))[1]
    return pd.DataFrame([dict(iso3=r['countryiso3code'], name=r['country']['value'], year=r['date'], value=r['value']) for r in recs if r['countryiso3code']])
share = wb(f'{D}/wb_pop_below5m_share.json').rename(columns={'value': 'share_below5m', 'year': 'share_year'})
pop = wb(f'{D}/wb_pop_total_2023.json').rename(columns={'value': 'pop2023'})[['iso3', 'pop2023']]
meta = pd.DataFrame([dict(iso3=c['id'], region=c['region']['value'], income=c['incomeLevel']['value']) for c in json.load(open(f'{D}/wb_countries.json'))[1]])
df = share.merge(pop, on='iso3', how='inner').merge(meta, on='iso3', how='left')
df = df[(df.region != 'Aggregates') & df.share_below5m.notna() & df.pop2023.notna()].copy()
df['pop_below5m'] = df.share_below5m / 100 * df.pop2023
cls = pd.read_csv(f'{ROOT}/catalog/surge_forecast_systems.csv')
df = df.merge(cls[['iso3', 'status', 'agency', 'system']], on='iso3', how='left')
df['status'] = df.status.fillna('not_assessed')
# corpus gauges per country (ISO3 from the GESLA-derived names: ...-<cc>-<provider>); country codes in the names are ISO3 lower-case
sa = pd.read_csv(f'{ROOT}/catalog/static_attributes.csv'); sp = pd.read_csv(f'{ROOT}/catalog/exp_split_final.csv').set_index('name')
sa['iso3'] = [n.split('-')[-2].upper() for n in sa.name]; sa['fold'] = [sp.fold.get(n, 'unused') for n in sa.name]
sa = sa[sa.fold.isin(['train', 'val', 'test'])]                    # the 900 gauges that trained, selected, or tested the model
gcount = sa.groupby('iso3').size().rename('corpus_gauges'); tcount = sa[sa.fold == 'test'].groupby('iso3').size().rename('test_gauges')
df = df.merge(gcount, on='iso3', how='left').merge(tcount, on='iso3', how='left'); df['corpus_gauges'] = df.corpus_gauges.fillna(0).astype(int); df['test_gauges'] = df.test_gauges.fillna(0).astype(int)
marine = pd.read_csv(f'{ROOT}/outputs/eval_full_lstmq_v2final.csv').stn; sa_m = sa[sa.name.isin(marine)]
tot = df.pop_below5m.sum(); lines = [f'COVERAGE GAP INDICATOR (World Bank EN.POP.EL5M.ZS x SP.POP.TOTL 2023; classification catalog/surge_forecast_systems.csv)',
                                     f'countries with data: {len(df)}; population below 5 m elevation: {tot/1e6:.0f} million']
for st, g in df.groupby('status'):
    lines.append(f'  {st:14s} countries {len(g):3d}  pop below 5 m {g.pop_below5m.sum()/1e6:7.1f} M ({100*g.pop_below5m.sum()/tot:4.1f}%)  split gauges {g.corpus_gauges.sum():4d}  test gauges {g.test_gauges.sum():3d}')
nonat = df[df.status.isin(['regional_only', 'none_found'])]
lines.append(f'without a national service (regional_only + none_found): {len(nonat)} countries, {nonat.pop_below5m.sum()/1e6:.0f} M people below 5 m ({100*nonat.pop_below5m.sum()/tot:.0f}%), {nonat.corpus_gauges.sum()} corpus gauges')
mt = sa_m.merge(df[['iso3', 'status']], on='iso3', how='left'); mt['status'] = mt.status.fillna('not_assessed')
lines.append(f'marine test gauges (84) by service status: ' + ', '.join(f'{k} {v}' for k, v in mt.status.value_counts().items()))
lines.append('marine test gauges in countries without a national service: ' + ', '.join(sorted(n.split('-')[0] + f' ({i})' for n, i in zip(mt[mt.status.isin(['regional_only', 'none_found'])].name, mt[mt.status.isin(['regional_only', 'none_found'])].iso3))))
lines.append('largest low-lying populations without a national service:')
for _, r in nonat.sort_values('pop_below5m', ascending=False).head(15).iterrows():
    lines.append(f'  {r["name"]:32s} {r.pop_below5m/1e6:6.1f} M below 5 m  ({r.status}; corpus gauges {r.corpus_gauges})')
na = df[df.status == 'not_assessed']; lines.append(f'not assessed: {len(na)} countries, {na.pop_below5m.sum()/1e6:.1f} M below 5 m')
txt = '\n'.join(lines); print(txt); open(f'{ROOT}/outputs/coverage_gap.txt', 'w').write(txt + '\n')
df.sort_values('pop_below5m', ascending=False).to_csv(f'{ROOT}/outputs/coverage_gap_countries.csv', index=False)

# SI table: the 20 largest low-lying populations without a national service, plus totals by status
rows = []
for _, r in nonat.sort_values('pop_below5m', ascending=False).head(20).iterrows():
    ag = str(r.agency) if isinstance(r.agency, str) else ''
    rows.append(f"{r['name'].split(',')[0]} & {r.pop_below5m/1e6:.1f} & {r.status.replace('_', ' ')} & {ag[:60]} & {int(r.corpus_gauges)}\\\\")
tex = ['\\footnotesize\\setlength{\\tabcolsep}{4pt}\\begin{tabular}{lrlp{6.2cm}c}\\toprule', 'Country & pop.\\ below 5 m (M) & status & agency & corpus gauges\\\\\\midrule'] + rows + ['\\bottomrule\\end{tabular}']
open(f'{ROOT}/outputs/coverage_gap_table.tex', 'w').write('\n'.join(tex) + '\n')
