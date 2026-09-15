"""Provenance verifier: asserts every headline number in the paper directly from the
committed evaluation artifacts. Run `python scripts/verify_numbers.py` — exit 0 means
every checked claim reproduces from the artifacts in outputs/.
This script ships with the publication repository as its audit gate."""
import os as _os
_ROOT = _os.environ.get('SURGE_ROOT') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import re, sys
import pandas as pd, numpy as np

O = 'outputs'
PASS, FAIL = [], []

def check(name, got, want, tol=0.0):
    ok = (abs(got - want) <= tol) if tol else (got == want)
    (PASS if ok else FAIL).append(f'{"PASS" if ok else "FAIL"}  {name}: got {got}, paper {want}' + (f' (tol {tol})' if tol else ''))

def sk(df, m='rmse_p', p='prmse_p'):
    return 100*(1 - df[m].mean()/df[p].mean())

def load(tag):
    return pd.read_csv(f'{O}/eval_full_lstmq_{tag}.csv')

# ---------- 1. deployment ----------
d = load('v2final')
check('deploy n', len(d), 84)
check('deploy pooled %', round(sk(d)), 39)
check('deploy @8h %', round(sk(d, 'r8', 'p8')), 31)
check('deploy NNSE@8h', round(d.n8.mean(), 3), 0.786)
check('deploy wins', int((d.rmse_p < d.prmse_p).sum()), 82)

# ---------- 2. ladder ----------
for tag, want in [('v2g64', 21), ('v2g128', 26), ('v2g256', 30), ('v2g384', 34),
                  ('v2g512', 37), ('v2g640', 38), ('v2final', 39)]:
    check(f'ladder {tag} pooled %', round(sk(load(tag))), want)

# ---------- 3. seed replicate bound ----------
a, b = load('v2g512'), load('v2g512r2')
check('seed replicate |d pooled|<=0.7', round(abs(sk(a)-sk(b)), 1) <= 0.7, True)
check('seed replicate |d @8h|<=0.3', round(abs(sk(a,'r8','p8')-sk(b,'r8','p8')), 1) <= 0.3, True)

# ---------- 4. rotations ----------
for tag, wp, w8, wn in [('v2rotjp', 35, 26, 0.808), ('v2roteu', 13, 12, 0.719),
                        ('v2rotna', 12, 10, 0.687), ('v2rotocr2', 11, 10, 0.731)]:
    r = load(tag)
    check(f'rot {tag} pooled %', round(sk(r)), wp)
    check(f'rot {tag} @8h %', round(sk(r, 'r8', 'p8')), w8)
    check(f'rot {tag} NNSE', round(r.n8.mean(), 3), wn)
runs = [round(sk(load(t)), 1) for t in ('v2rotoc', 'v2rotocr2', 'v2rotocr3')]
check('oc median run is r2', sorted(runs)[1] == round(sk(load('v2rotocr2')), 1), True)
u = pd.concat([load(t) for t in ('v2rotjp', 'v2roteu', 'v2rotna', 'v2rotocr2')])
check('union n', len(u), 635)
check('union wins', int((u.rmse_p < u.prmse_p).sum()), 510)
check('union pooled %', round(sk(u)), 14)
check('union @8h %', round(sk(u, 'r8', 'p8')), 12)

# ---------- 5. storm ranges from logs ----------
def storm_range(tag):
    log = open(f'{O}/eval_full_lstmq_{tag}.log').read()
    sec = log.split('[B] Lead-time skill on p99.9 windows:')[1]
    v = [int(x) for x in re.findall(r'skill\s+([+-]\d+)%', sec)[:5]]
    return min(v), max(v)
for tag, lo, hi in [('v2rotjp', 31, 37), ('v2roteu', 8, 13), ('v2rotna', 16, 19), ('v2rotocr2', -1, 11)]:
    g = storm_range(tag)
    check(f'storm range {tag}', g, (lo, hi))
check('deploy storm per-lead range', storm_range('v2final'), (29, 36))

dlog = open(f'{O}/eval_full_lstmq_v2final.log').read()
hw = [int(x) for x in re.findall(r'high-water-hour RMSE.*?skill\s+([+-]\d+)%', dlog)]
if hw: check('deploy high-water skill in 43..48', min(hw) >= 43 and max(hw) <= 48, True)
cap = float(re.search(r'capture ([\d.]+)\n  persist', dlog).group(1))
check('deploy peak capture', round(cap, 2), 0.60)
tail = float(re.search(r'TAIL coverage.*?: q99 ([\d.]+)', dlog).group(1))
check('deploy tail coverage', round(tail, 2), 0.78)

# ---------- 6. gauge-free ladder ----------
for f, want in [('eval_forcing_only_v2fonly.csv', 0.631), ('eval_forcing_only_v2fonly_predscale.csv', 0.611),
                ('eval_forcing_only_v2fonly_eot2.csv', 0.606)]:
    g = pd.read_csv(f'{O}/{f}')
    check(f'gauge-free {f.split("_v2fonly")[1] or "measured"} NNSE', round(g.nnse8.mean(), 3), want)
g = pd.read_csv(f'{O}/eval_forcing_only_v2fonly_eot2.csv')
check('gauge-free storm peak capture', round(g.pk_cap.mean(), 2), 0.50)
check('gauge-free q99 envelope at peak', round(g.q99_cap.mean(), 2), 0.98)
check('gauge-free cov99', round(g.cov99.mean(), 3), 0.983)

# ---------- 7. GTSM head-to-head ----------
gl = open(f'{O}/eval_gtsm.log').read()
m = re.search(r'@8h RMSE: ours ([\d.]+) \| GTSM ([\d.]+) \| pers ([\d.]+)', gl)
check('GTSM @8h ours cm', round(float(m.group(1)), 1), 4.4)
check('GTSM @8h gtsm cm', round(float(m.group(2)), 1), 8.3)
check('GTSM wins', gl.count('79/80') >= 1, True)
m = re.search(r'@48h: ours ([\d.]+) \| GTSM ([\d.]+) \| pers ([\d.]+)', gl)
check('GTSM storm48 ours', round(float(m.group(1)), 1), 13.2)
check('GTSM storm48 gtsm', round(float(m.group(2)), 1), 12.5)
check('GTSM storm48 pers', round(float(m.group(3)), 1), 27.6)

# ---------- 8. GEFS spot check ----------
ge = pd.read_csv(f'{O}/eval_gefs.csv')
check('GEFS n storms', len(ge), 6)
check('GEFS era5 RMSE', round(ge.rmse_era5.mean(), 1), 21.5)
check('GEFS gefs RMSE', round(ge.rmse_gefs.mean(), 1), 38.2)
check('GEFS pers RMSE', round(ge.rmse_per.mean(), 1), 58.8)
check('GEFS capture era5/gefs', (round(ge.cap_era5.mean(), 2), round(ge.cap_gefs.mean(), 2)), (0.71, 0.34))
check('GEFS beats pers', int((ge.rmse_gefs < ge.rmse_per).sum()), 5)

# ---------- 9. trailing causality ----------
tl = open(f'{O}/eval_trailing.log').read()
m = re.search(r'trailing\s+n=84 pooled \+([\d.]+)% @8h \+([\d.]+)% NNSE ([\d.]+) cap ([\d.]+) storm@8h \+([\d.]+)%', tl)
check('trailing pooled', round(float(m.group(1))), 36)
check('trailing NNSE', round(float(m.group(3)), 3), 0.818)
check('trailing capture', round(float(m.group(4)), 2), 0.71)
check('trailing storm@8h', round(float(m.group(5))), 40)

# ---------- 10. pinball proper score ----------
pb = pd.read_csv(f'{O}/eval_pinball.csv')
check('pinball q90 skill %', round(100*(1-pb.pb90.mean()/pb.pb90_clim.mean()), 1), 45.4)
check('pinball q99 skill %', round(100*(1-pb.pb99.mean()/pb.pb99_clim.mean()), 1), 46.2)
check('pinball q90 wins', int((pb.pb90 < pb.pb90_clim).sum()), 82)
check('pinball q99 wins', int((pb.pb99 < pb.pb99_clim).sum()), 77)

# ---------- 11. lake/marine ablations ----------
mv = load('v2marine')
check('marine-706 |d pooled|<=0.7', abs(round(sk(mv)-sk(d), 1)) <= 0.7, True)
m640, m640m = load('v2g640'), load('v2g640marine')
check('matched-640 |d pooled|<=0.7', abs(round(sk(m640)-sk(m640m), 1)) <= 0.7, True)

# ---------- 12. xdom ----------
x = load('v2final_xdom')
check('xdom evaluated', len(x), 15)
check('xdom wins', int((x.rmse_p < x.prmse_p).sum()), 14)

# ---------- 13. per-lead full curve ----------
pl = pd.read_csv(f'{O}/perlead_full.csv')
fd = pl[pl.series == 'full'] if 'series' in pl.columns else pl
scol = [c for c in fd.columns if 'skill' in c][0]
check('per-lead all 48 positive', bool((fd[scol] > 0).all()), True)

print('\n'.join(PASS))
print()
if FAIL:
    print('\n'.join(FAIL))
    print(f'\n{len(FAIL)} FAILURES / {len(PASS)+len(FAIL)} checks')
    sys.exit(1)
print(f'ALL {len(PASS)} CHECKS PASS — every audited paper number reproduces from committed artifacts')
