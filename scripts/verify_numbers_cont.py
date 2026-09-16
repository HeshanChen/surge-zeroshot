"""Audit gate for the September 2026 revision (Nature Water draft): every headline number quoted on continuous
evaluation windows is asserted directly from the committed artifacts under outputs/. Companion to
verify_numbers.py, which gates the July 2026 draft numbers (all windows). Run from the repository root:
    python scripts/verify_numbers_cont.py      # exit 0 = every number reproduces"""
import os as _os
_ROOT = _os.environ.get('SURGE_ROOT') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import os, re, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
O = f'{ROOT}/outputs'
FAIL = []
def check(name, got, want, tol=0.0):
    ok = abs(got - want) <= tol if tol else got == want
    print(f'{"ok " if ok else "FAIL"} {name:70s} got {got} want {want}')
    if not ok: FAIL.append(name)
def csv(tag): return pd.read_csv(f'{O}/eval_full_{tag}.csv')
def log(path): return open(f'{O}/{path}').read()
def sk(d, m='rmse_p', p='prmse_p'): return 100*(1-d[m].mean()/d[p].mean())
def grab(t, pat, cast=float): m = re.search(pat, t); assert m, pat; return cast(m.group(1))

# 1. deployment, continuous windows (Results 1, Table 1, abstract)
d = csv('lstmq_v2final_cont'); a = csv('lstmq_v2final')
check('deployment gauges', len(d), 84)
check('deployment wins, continuous', int((d.rmse_p < d.prmse_p).sum()), 81)
check('deployment wins, all windows', int((a.rmse_p < a.prmse_p).sum()), 82)
check('deployment pooled skill, continuous (%)', round(sk(d), 1), 37.3, 0.06)
check('deployment 8-h skill, continuous (%)', round(sk(d, 'r8', 'p8'), 1), 30.3, 0.06)
check('deployment NNSE@8h, continuous', round(d.n8.mean(), 3), 0.782, 0.0006)
check('deployment pooled skill, all windows (%)', round(sk(a), 1), 39.5, 0.06)
t = log('eval_full_lstmq_v2final_cont.log')
check('storm per-lead skill @8h, continuous (%)', grab(t, r'L=\s*8h: mdl.*?skill\s*([+-]?\d+)%', int), 40)
check('storm per-lead skill @4h, continuous (%)', grab(t, r'L=\s*4h: mdl.*?skill\s*([+-]?\d+)%', int), 32)
check('point peak capture, continuous', grab(t, r'model\s*:.*?capture ([\d.]+)'), 0.59, 0.001)
check('event-level peak coverage, continuous (%)', grab(t, r'EVENT-LEVEL peak coverage.*?in ([\d.]+)% of'), 41.3, 0.06)
check('q99 hourly coverage overall, continuous', grab(t, r'q99 ([\d.]+) \(nominal'), 0.984, 0.0006)
check('tail coverage, continuous', grab(t, r'TAIL coverage.*?q99 ([\d.]+)'), 0.753, 0.0006)
pl = pd.read_csv(f'{O}/perlead_full_cont.csv')
check('per-lead skill @1h, continuous (%)', round(float(pl.skill_full.iloc[0])), 11)
check('per-lead skill @48h, continuous (%)', round(float(pl.skill_full.iloc[47])), 40)
check('per-lead skill all positive (leads)', int((pl.skill_full > 0).sum()), 48)

# 2. rotations (Results 2, ED Table 1)
rot = {k: csv(f'lstmq_v2{k}_cont') for k in ('rotjp', 'roteu', 'rotna', 'rotocr2')}
for k, n, want in (('rotjp', 77, 29.8), ('roteu', 231, 10.7), ('rotna', 214, 9.6), ('rotocr2', 111, 7.9)):
    check(f'rotation {k} gauges, continuous', len(rot[k]), n); check(f'rotation {k} pooled skill, continuous (%)', round(sk(rot[k]), 1), want, 0.06)
u = pd.concat(rot.values()); check('rotation union gauges', len(u), 633); check('rotation union wins', int((u.rmse_p < u.prmse_p).sum()), 477)
check('rotation union pooled skill (%)', round(sk(u), 1), 11.2, 0.06)
oc = [round(sk(csv(f'lstmq_v2{k}_cont')), 1) for k in ('rotoc', 'rotocr2', 'rotocr3')]
check('Oceania median run is r2 (continuous)', sorted(oc)[1] == oc[1], True)

# 3. Europe intervention and composition ablation (Results 2 and 3)
eu, eu299 = rot['roteu'].set_index('stn'), csv('lstmq_v2eu299_cont').set_index('stn'); j = eu.join(eu299, rsuffix='_b', how='inner')
check('Europe 299 arm pooled skill (%)', round(sk(eu299), 1), -0.4, 0.06); check('Europe 593 better at 8 h (gauges)', int((j.r8 < j.r8_b).sum()), 186)
cs, cu = csv('lstmq_v2c298strat_cont').set_index('stn'), csv('lstmq_v2c298usjp_cont').set_index('stn'); j = cs.join(cu, rsuffix='_b', how='inner')
check('composition 49-country 8-h skill (%)', round(sk(cs, 'r8', 'p8'), 1), 23.8, 0.06); check('composition two-country 8-h skill (%)', round(sk(cu, 'r8', 'p8'), 1), -1.7, 0.06)
check('composition 49-country lower at 8 h (gauges)', int((j.r8 < j.r8_b).sum()), 67)

# 4. scaling ladder (Results 3, ED Table 2)
lad = [round(sk(csv(f'lstmq_v2{k}_cont')), 1) for k in ('g64', 'g128', 'g256', 'g384', 'g512', 'g640', 'final')]
for v, w in zip(lad, (19.1, 23.7, 27.9, 31.5, 34.7, 36.4, 37.3)): check(f'ladder pooled skill {w}', v, w, 0.06)
check('ladder monotone in pooled skill', all(np.diff(lad) > 0), True)
check('log-linear slope per doubling (%)', round(float(np.polyfit(np.log2([64, 128, 256, 384, 512, 640, 760]), lad, 1)[0]), 1), 5.2, 0.06)

# 5. GTSM head-to-head, continuous (Table 2b)
g = pd.read_csv(f'{O}/eval_gtsm_symmetric_cont.csv'); check('GTSM gauges', len(g), 80)
for col, want in (('rmse8_ours', 4.17), ('rmse8_anch_hp_lp', 5.47), ('rmse8_gf', 6.74), ('rmse8_gtsm_hp', 6.21), ('rmse48_ours', 5.95), ('rmse48_anch_hp_lp', 6.34)):
    check(f'GTSM mean {col} (cm)', round(g[col].mean(), 2), want, 0.006)
check('tier B forecast lower at 8 h (gauges)', int((g.rmse8_ours < g.rmse8_anch_hp_lp).sum()), 66)
check('tier A gauge-free lower at 8 h (gauges)', int((g.rmse8_gf < g.rmse8_gtsm_hp).sum()), 38)
gs = g.dropna(subset=['storm8_ours', 'storm8_anch_hp_lp']); check('storm gauges with >=5 windows', len(gs), 61)
tg = log('eval_gtsm_symmetric_cont.log')
check('tier B storm 8h ours (cm)', grab(tg, r'p99.9 storm windows @8h : ours ([\d.]+) \| anch_hp_lp'), 7.88, 0.006)
check('tier B storm 8h GTSM+bc (cm)', grab(tg, r'p99.9 storm windows @8h : ours [\d.]+ \| anch_hp_lp ([\d.]+)'), 7.70, 0.006)
check('tier B peak capture ours', grab(tg, r'peak capture.*?: ours ([\d.]+) \| anch_hp_lp'), 0.62, 0.001)
check('tier B peak capture GTSM+bc', grab(tg, r'peak capture.*?: ours [\d.]+ \| anch_hp_lp ([\d.]+)'), 0.66, 0.001)
check('tier A storm 8h gauge-free (cm)', grab(tg, r'p99.9 storm windows @8h : gf ([\d.]+) \| gtsm_hp'), 11.79, 0.006)
check('tier A storm 8h GTSM (cm)', grab(tg, r'p99.9 storm windows @8h : gf [\d.]+ \| gtsm_hp ([\d.]+)'), 8.75, 0.006)

# 6. gauge-free ladder and zero-water-level ablation (Table 2a)
for tag, want in (('v2fonly_eot2', 0.593), ('v2fonly_predscale', 0.598), ('v2fonly', 0.618), ('v2final', 0.522)):
    f = pd.read_csv(f'{O}/eval_forcing_only_{tag}_cont.csv'); check(f'gauge-free NNSE@8h {tag}, continuous', round(f.nnse8.mean(), 3), want, 0.0006)

# 7. factor study (ED Table 3)
l298, v7e = csv('lstmq_v2n298_cont').set_index('stn'), csv('v7e_n298v_cont').set_index('stn'); j = l298.join(v7e, rsuffix='_v', how='inner')
check('factor LSTM lower 8-h RMSE than v7e (gauges)', int((j.r8 < j.r8_v).sum()), 71)
check('factor LSTM pooled skill (%)', round(sk(l298)), 15); check('factor v7e pooled skill (%)', round(sk(v7e)), 7)
tc = log('eval_chronos_bolt-small_masked_cont.log')
check('Chronos pooled skill (%)', grab(tc, r'pooled: RMSE.*?skill\s*([+-]?\d+)%', int), -1); check('Chronos 8-h skill (%)', grab(tc, r'L= 8h : RMSE.*?skill\s*([+-]?\d+)%', int), 5)

# 8. pinball and GEFS
tp = log('eval_pinball_cont.log')
check('pinball q90 skill (%)', grab(tp, r'q90:.*?skill ([+-][\d.]+)%'), 44.7, 0.06); check('pinball q99 skill (%)', grab(tp, r'q99:.*?skill ([+-][\d.]+)%'), 47.2, 0.06)
ge = pd.read_csv(f'{O}/eval_gefs_v2.csv'); check('GEFS cases', len(ge), 9)
check('GEFS mean RMSE reanalysis (cm)', round(ge.rmse_era5.mean(), 1), 37.1, 0.06); check('GEFS mean RMSE GEFS (cm)', round(ge.rmse_gefs.mean(), 1), 48.2, 0.06)
check('GEFS cases beating persistence', int((ge.rmse_gefs < ge.rmse_per).sum()), 8); check('GEFS peaks inside window', int(ge.peak_in_window.sum()), 9)

# 8b. season-scale real-forecast check (Results 1, ED Table 4, ED Fig. 4)
se = pd.read_csv(f'{O}/eval_gefs_season.csv'); check('season gauges', len(se), 79); check('season windows', int(se.n_win.sum()), 35228); check('season storm windows', int(se.n_storm.sum()), 283)
def gs(m, p): return round(100*(1-se[m].mean()/se[p].mean()), 1)
check('season pooled skill reanalysis (%)', gs('rmse_p_era5', 'rmse_p_pers'), 37.3, 0.06); check('season pooled skill GEFS (%)', gs('rmse_p_gefs', 'rmse_p_pers'), 30.1, 0.06)
check('season 8-h skill reanalysis (%)', gs('r8_era5', 'r8_pers'), 28.2, 0.06); check('season 8-h skill GEFS (%)', gs('r8_gefs', 'r8_pers'), 24.2, 0.06)
check('season 48-h skill GEFS (%)', gs('r48_gefs', 'r48_pers'), 32.9, 0.06)
check('season GEFS wins pooled', int((se.rmse_p_gefs < se.rmse_p_pers).sum()), 72); check('season reanalysis wins pooled', int((se.rmse_p_era5 < se.rmse_p_pers).sum()), 74)
ts = log('eval_gefs_season.log')
check('season storm RMSE reanalysis (cm)', grab(ts, r'storm-window RMSE \(all leads\): reanalysis ([\d.]+)'), 13.4, 0.06); check('season storm RMSE GEFS (cm)', grab(ts, r'storm-window RMSE \(all leads\): reanalysis [\d.]+ \| GEFS ([\d.]+)'), 15.4, 0.06)
check('season peak capture GEFS', grab(ts, r'peak capture at the true peak hour.*?GEFS ([\d.]+)'), 0.69, 0.001)
for y, want in (('2017', 30.2), ('2012', 30.5)):
    sy = pd.read_csv(f'{O}/eval_gefs_season_{y}.csv'); check(f'season {y} pooled skill GEFS (%)', round(100*(1-sy.rmse_p_gefs.mean()/sy.rmse_p_pers.mean()), 1), want, 0.06)
g3 = pd.read_csv(f'{O}/eval_gefs_v3.csv'); check('nine-case GEFS mean RMSE, corrected buckets (cm)', round(g3.rmse_gefs.mean(), 1), 49.3, 0.06); check('nine-case GEFS peak capture', round(g3.cap_gefs.mean(), 2), 0.36, 0.006)

# 9. data provenance (Methods: dataset chain, fold sizes, join share, seismic mask, rotation test-set composition, duplicates; Supplementary Table 3)
import os
def first(*paths):
    for q in paths:
        if os.path.exists(q): return q
    raise FileNotFoundError(paths)
sa = pd.read_csv(f'{ROOT}/catalog/static_attributes.csv').set_index('name'); split = pd.read_csv(f'{ROOT}/catalog/exp_split_final.csv')
uni = pd.read_csv(first(f'{O}/audit_universe_loadability.csv')); check('joined gauges (observations and forcing)', len(uni), 1054)
load = uni[uni.n_joined >= 3000].stn; check('gauges with at least 3,000 joined hours', len(load), 1047)
for fold, want, want_eff in (('train', 760, 756), ('val', 40, 39), ('test', 84, 84), ('test_xdom', 16, 15)):
    names = split[split.fold == fold].name; check(f'fold {fold} nominal', len(names), want); check(f'fold {fold} effective', int(names.isin(load).sum()), want_eff)
for f, want in (('g64', 63), ('g128', 126), ('g256', 254), ('g384', 380), ('g512', 508), ('g640', 636)):
    sp2 = pd.read_csv(f'{ROOT}/catalog/exp_split_{f}.csv'); check(f'ladder {f} effective training gauges', int(sp2[sp2.fold == 'train'].name.isin(load).sum()), want)
for f, want in (('jp_v2', 592), ('eu_v2', 587), ('na_v2', 606), ('oc_v2', 785)):
    sp2 = pd.read_csv(f'{ROOT}/catalog/exp_split_{f}.csv'); check(f'rotation {f} effective training gauges', int(sp2[sp2.fold == 'train'].name.isin(load).sum()), want)
lat = sa.lat.astype(float); lon = pd.Series(np.where(sa.lon.values > 180, sa.lon.values - 360, sa.lon.values), index=sa.index)
def hav(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2)); return 6371*2*np.arcsin(np.sqrt(np.sin((la2-la1)/2)**2 + np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2))
test = set(split[split.fold.isin(['test', 'test_xdom'])].name); pool = set(split[split.fold.isin(['train', 'val'])].name); joined = sorted(uni.stn)
tl, tlo = lat[list(test)].values, lon[list(test)].values
rest = [n for n in joined if n not in test]; buf = [n for n in rest if ((np.abs(lat[n] - tl) <= 0.5) & (np.abs(lon[n] - tlo) <= 0.5)).any()]
check('non-test joined gauges inside a 0.5-degree box of a test gauge', len(buf), 127)
rest2 = [n for n in rest if n not in set(buf)]; left = [n for n in rest2 if n not in pool]; check('joined gauges left after test, buffer and pool', len(left), 27)
paired = [n for n in left if any(m in pool and abs(lat[m] - lat[n]) <= 0.02 and abs(lon[m] - lon[n]) <= 0.02 for m in rest2)]
check('dropped members of 0.02-degree pairs whose partner is in the pool', len(paired), 23); check('gauges the catalogue cleaning had excluded', len(left) - len(paired), 4)
clean = set(pd.read_csv(f'{ROOT}/catalog/clean_stations.csv').name); check('those four are absent from clean_stations.csv', int(sum(n not in clean for n in left if n not in paired)), 4)
npairs = sum(1 for i, a in enumerate(joined) for b in joined[i+1:] if abs(lat[a] - lat[b]) <= 0.02 and abs(lon[a] - lon[b]) <= 0.02); check('0.02-degree pairs among all joined gauges', npairs, 26)
js = pd.read_csv(first(f'{O}/join_share_audit.csv', f'{O}/join_share_audit_2026-09-14.csv')); ld = js[js.joined_h >= 3000]
check('median share of the surge record that joins (%)', round(100*ld.share_all.median()), 44); check('join share lower quartile (%)', round(100*ld.share_all.quantile(.25)), 27)
check('join share upper quartile (%)', round(100*ld.share_all.quantile(.75)), 73); check('join share minimum (%)', round(100*ld.share_all.min()), 1)
aff = ld[ld.forcing_gap_post >= 0.23]; check('gauges losing at least 23% of post-2000 hours at the join', len(aff), 251)
check('largest post-2000 loss (%)', round(100*aff.forcing_gap_post.max()), 78); check('median post-2000 loss among them (%)', round(100*aff.forcing_gap_post.median()), 54)
check('largest post-2000 loss among the other gauges (%)', round(100*ld[ld.forcing_gap_post < 0.23].forcing_gap_post.max()), 20)
wc = pd.read_csv(first(f'{O}/audit_window_count.csv')); wc = wc[wc.stn.isin(split.name)]; share = wc.n_masked_out / wc.n_joined
check('seismic mask median share of hours (%)', round(100*share.median(), 1), 0.8); check('seismic mask share Japan (%)', round(100*share[wc.country == 'jpn'].median(), 1), 2.2); check('seismic mask share Germany (%)', round(100*share[wc.country == 'deu'].median(), 1), 0.1)
sx = pd.read_csv(f'{O}/seismic_exceedance_audit.csv'); check('gauges losing more than 10% of p99.9 exceedance hours', int((sx.frac_p999_masked > 0.10).sum()), 16); check('largest such share (%)', round(100*sx.frac_p999_masked.max()), 45)
dom = pd.read_csv(first(f'{ROOT}/catalog/rotation_test_domain.csv', f'{O}/rotation_test_domain_2026-09-14.csv')); dom['nonmarine'] = dom.nonmarine_class.fillna('').astype(str).str.len() > 0
for rot, want in (('North America', 80), ('Europe', 7), ('Oceania', 1), ('Japan', 0)): check(f'non-marine gauges in the {rot} rotation test set', int(dom[dom.rotation == rot].nonmarine.sum()), want)
qc = pd.read_csv(f'{ROOT}/catalog/processing_qc.csv').set_index('name')
union = {'all': [0, 0], 'marine': [0, 0], 'dedup': [0, 0]}; pairs_per = {}
for rot, tag in (('Japan', 'lstmq_v2rotjp'), ('Europe', 'lstmq_v2roteu'), ('North America', 'lstmq_v2rotna'), ('Oceania', 'lstmq_v2rotocr2')):
    r = csv(f'{tag}_cont'); nm = set(dom[(dom.rotation == rot) & dom.nonmarine].stn); m = r[~r.stn.isin(nm)]
    names = r.stn.tolist(); drop = set(); npair = 0
    for i, a in enumerate(names):
        for b in names[i+1:]:
            if hav(lat[a], lon[a], lat[b], lon[b]) < 2.0: npair += 1; drop.add(a if qc.gap_frac.get(a, 0) >= qc.gap_frac.get(b, 0) else b)
    pairs_per[rot] = npair; dd = r[~r.stn.isin(drop)]
    for k, x in (('all', r), ('marine', m), ('dedup', dd)): union[k][0] += int((x.rmse_p < x.prmse_p).sum()); union[k][1] += len(x)
    if rot == 'North America':
        check('North America marine-only pooled skill (%)', round(sk(m)), 16); check('North America marine-only 8-h skill (%)', round(sk(m, 'r8', 'p8')), 14)
        check('North America marine-only NNSE@8h', round(m.n8.mean(), 3), 0.713); check('North America marine-only wins', int((m.rmse_p < m.prmse_p).sum()), 110); check('North America marine-only gauges', len(m), 134)
        check('North America non-marine pooled skill (%)', round(sk(r[r.stn.isin(nm)])), -4)
    else: check(f'{rot} pooled skill change when restricted to marine gauges (points, <=0.2)', round(abs(sk(m) - sk(r)), 1) <= 0.2, True)
    check(f'{rot} pooled skill change after dropping duplicate pairs (points, <=0.2)', round(abs(sk(dd) - sk(r)), 1) <= 0.2, True)
check('duplicate pairs within 2 km inside rotation test sets (Europe)', pairs_per['Europe'], 7); check('duplicate pairs (North America)', pairs_per['North America'], 2); check('duplicate pairs (Oceania)', pairs_per['Oceania'], 7)
check('rotation union wins', union['all'][0], 477); check('rotation union gauges', union['all'][1], 633)
check('rotation union wins, marine gauges only', union['marine'][0], 414); check('rotation union gauges, marine only', union['marine'][1], 545)
check('rotation union wins after dropping duplicates', union['dedup'][0], 465); check('rotation union gauges after dropping duplicates', union['dedup'][1], 617)
ta = pd.read_csv(f'{O}/tsunami_mask_annex.csv').set_index('event')
for ev, hrs, gauges, status in (('Tohoku 2011', 105, 354, 'removed'), ('Sumatra 2004', 138, 276, 'removed'), ('Sandy 2012', 0, 76, 'kept'), ('Haiyan 2013', 0, 0, 'kept')):
    check(f'seismic mask annex {ev}: hours removed', int(ta.loc[ev, 'hours_removed']), hrs); check(f'seismic mask annex {ev}: gauges inside the radius', int(ta.loc[ev, 'gauges_in_radius_of_largest']), gauges); check(f'seismic mask annex {ev}: largest surge', ta.loc[ev, 'peak_status'], status)

print(f'\n{len(FAIL)} failures' if FAIL else '\nALL CHECKS PASS'); sys.exit(1 if FAIL else 0)
