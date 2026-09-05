"""Audit gate for the September 2026 revision (Nature Water draft): every headline number quoted on continuous
evaluation windows is asserted directly from the committed artifacts under outputs/. Companion to
verify_numbers.py, which gates the July preprint numbers (all windows). Run from the repository root:
    python scripts/verify_numbers_cont.py      # exit 0 = every number reproduces"""
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

print(f'\n{len(FAIL)} failures' if FAIL else '\nALL CHECKS PASS'); sys.exit(1 if FAIL else 0)
