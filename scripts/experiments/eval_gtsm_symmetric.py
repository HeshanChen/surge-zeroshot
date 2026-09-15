"""Symmetric physics head-to-head (audit of eval_gtsm.py's information asymmetry).

eval_gtsm.py compares the gauge-fed, persistence-anchored flagship against a GTSM hindcast that
assimilates nothing. At 8 h the observed water level at t0 is worth more than the physics
(persistence alone beats GTSM at 60/80 gauges), so "79/80 vs GTSM" mostly measures access to
the observation, not the model. This script scores the two information-matched pairings:

  tier A (no local observation):  gauge-free model (v2fonly, EOT20 statics, sigma_hat)  vs  GTSM
  tier B (observation at t0):     flagship (v2final, anchored)  vs  GTSM + persistence-of-error
                                  correction  y(t0) + [GTSM(t0+l) - GTSM(t0)]  (tau=inf, 24 h, 12 h)

and removes the target-definition asymmetry: our surge target has a centered 30-day moving mean
removed (clean_surge.highpass_days=30); the GTSM series in eval_gtsm.py only had its 2010-2018
mean removed. Here GTSM is high-passed with the same 720-h centered window (gtsm_hp); the raw
mean-aligned version (gtsm_al) is kept for reference. Same 80 gauges, same 48-h windows,
same seismic mask as eval_gtsm.py.
-> outputs/eval_gtsm_symmetric.csv (per gauge) + outputs/eval_gtsm_symmetric.log"""
import os as _os
_ROOT = _os.environ.get('SURGE_ROOT') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
import sys, warnings; warnings.filterwarnings('ignore')
import torch, pandas as pd, numpy as np
from scipy.stats import binomtest, wilcoxon
sys.path.insert(0, f'{_ROOT}/src')
from models.baseline_lstm import GlobalLSTM
from data.dataset_v0 import seismic_mask
ROOT = _ROOT
Tctx, W, H = 208, 256, 48
DIST_GATE_KM = 25.0
import argparse
_ap = argparse.ArgumentParser(); _ap.add_argument('--continuous_only', action='store_true', help='keep only windows whose 256 rows are consecutive hours (audit 2026-09-05)')
_args = _ap.parse_args(); SUF = '_cont' if _args.continuous_only else ''

sa = pd.read_csv(f'{ROOT}/catalog/static_attributes.csv').set_index('name')
sp = pd.read_csv(f'{ROOT}/catalog/exp_split_final.csv')
tr = [n for n in sp[sp.fold == 'train'].name if n in sa.index]
te = [n for n in sp[sp.fold == 'test'].name if n in sa.index]
mp = pd.read_csv(f'{ROOT}/data/raw/gtsm/gtsm_mapping.csv').set_index('stn')
G = pd.read_parquet(f'{ROOT}/data/raw/gtsm/gtsm_hourly_2010_2018.parquet')
G.columns = [int(c) for c in G.columns]
eot = pd.read_csv(f'{ROOT}/outputs/eot20_statics_test.csv').set_index('stn')
sig = pd.read_csv(f'{ROOT}/outputs/sigma_hat_eot_test.csv').set_index('stn')['sigma_hat'].to_dict()

def sfeat(n, override=False):
    r = sa.loc[n]; la, lo = np.radians(float(r.lat)), np.radians(float(r.lon))
    rng, ff = float(r.tidal_range_m), float(r.form_factor)
    if override and n in eot.index and np.isfinite(eot.loc[n, 'range_eot']):
        rng, ff = float(eot.loc[n, 'range_eot']), float(eot.loc[n, 'ff_eot'])
    return np.array([np.cos(la)*np.cos(lo), np.cos(la)*np.sin(lo), np.sin(la), rng, ff], dtype='float32')
arr = np.stack([sfeat(n) for n in tr]); smu = arr.mean(0); ssd = arr.std(0)+1e-6
SF_true = {n: (sfeat(n)-smu)/ssd for n in te}
SF_eot = {n: (sfeat(n, True)-smu)/ssd for n in te}

m_fl = GlobalLSTM(n_out=3); m_fl.load_state_dict(torch.load(f'{ROOT}/models/deploy_760_best.pt', map_location='cpu')); m_fl.eval()
m_gf = GlobalLSTM(n_out=3); m_gf.load_state_dict(torch.load(f'{ROOT}/models/gaugefree_760_best.pt', map_location='cpu')); m_gf.eval()

def load_with_time(name):
    s = pd.read_parquet(f'{ROOT}/data/processed/{name}.parquet')['surge']; s = s.where(s.abs() < 4)
    e = pd.read_parquet(f'{ROOT}/data/raw/era5/{name}.parquet')
    if e.index.tz is None: e.index = e.index.tz_localize('UTC')
    for c in ['wind_speed_10m', 'wind_direction_10m', 'pressure_msl', 'precipitation']:
        e[c] = pd.to_numeric(e[c], errors='coerce')
    spd = e['wind_speed_10m'].values; d = np.deg2rad(e['wind_direction_10m'].values)
    e2 = pd.DataFrame({'wu': spd*np.sin(d), 'wv': spd*np.cos(d), 'mslp': e['pressure_msl'].values, 'precip': e['precipitation'].values}, index=e.index)
    df = pd.DataFrame({'surge': s}).join(e2, how='inner').dropna()
    if len(df):
        r = sa.loc[name]; df = df[~seismic_mask(float(r.lat), float(r.lon), df.index)]
    return df

def run_model(m, ctx, ff, static, anchor, scale):
    ps = []
    with torch.no_grad():
        for j in range(0, len(ctx), 512):
            b = min(512, len(ctx)-j)
            _, o = m.predict_window(torch.tensor(ctx[j:j+b]).float(), torch.tensor(ff[j:j+b]).float(), H,
                                    torch.tensor(static)[None].expand(b, -1).float(), torch.tensor(anchor[j:j+b]).float())
            ps.append(o.cpu())
    return torch.cat(ps).numpy()*scale*100          # (N,48,3) cm

def rmse_lead(p, t): return np.sqrt(((p-t)**2).mean(0))
def nnse(p, t):
    ns = 1-((p-t)**2).mean()/max(((t-t.mean())**2).mean(), 1e-9); return 1/(2-ns)

KEYS = ['ours', 'gf', 'pers', 'gtsm_al', 'gtsm_hp', 'anch_inf', 'anch_24', 'anch_12', 'anch_lp', 'anch_hp_lp', 'anch_hp_lp208']
import os; os.makedirs(f'{ROOT}/outputs/gtsm_sym_preds', exist_ok=True)
FULL = {k: [] for k in KEYS}; STORM = {k: [] for k in KEYS}; PK = {k: [] for k in KEYS}
rows = []
for k_i, name in enumerate(te):
    df = load_with_time(name)
    df = df[(df.index >= pd.Timestamp('2010-01-01', tz='UTC')) & (df.index <= pd.Timestamp('2018-12-31 23:00', tz='UTC'))]
    if len(df) < 3000: continue
    gi = int(mp.loc[name, 'gtsm_idx'])
    gfull = G[gi]
    ghp_full = gfull - gfull.rolling(720, center=True, min_periods=360).mean()      # same 30-d centered high-pass as the target
    g = gfull.reindex(df.index); ghp = ghp_full.reindex(df.index)
    ok = g.notna() & ghp.notna(); df = df[ok]; g = g[ok]; ghp = ghp[ok]
    if len(df) < 3000: continue
    g_al = (g - (g.mean() - df.surge.mean())).values.astype('float32')
    g_hp = (ghp - (ghp.mean() - df.surge.mean())).values.astype('float32')
    a = df.values.astype('float32')
    tstd = float(a[:, 0].std())+1e-6
    af = a.copy(); af[:, 1:] = (af[:, 1:]-af[:, 1:].mean(0))/(af[:, 1:].std(0)+1e-6)
    starts = np.arange(0, len(af)-W, H)
    tt = df.index.values.astype('datetime64[h]').astype('int64')
    cont = (tt[starts+W-1]-tt[starts]) == (W-1)
    if _args.continuous_only: starts = starts[cont]
    if len(starts) < 20: continue
    starts = list(starts)
    ws = np.stack([af[st:st+W] for st in starts])
    raw = np.stack([a[st:st+W, 0] for st in starts])
    mu = ws[:, :Tctx, 0].mean(1, keepdims=True); sd = ws[:, :Tctx, 0].std(1, keepdims=True)+1e-6
    ctx_fl = np.concatenate([((ws[:, :Tctx, 0]-mu)/sd)[:, :, None], ws[:, :Tctx, 1:]], 2).transpose(0, 2, 1)
    ctx_gf = np.concatenate([np.zeros_like(ws[:, :Tctx, :1]), ws[:, :Tctx, 1:]], 2).transpose(0, 2, 1)
    ff = ws[:, Tctx:, 1:].transpose(0, 2, 1)
    last = raw[:, Tctx-1]
    tru = raw[:, Tctx:]*100                                   # (N,48) cm
    P = {}
    P['ours'] = run_model(m_fl, ctx_fl, ff, SF_true[name], last/tstd, tstd)[..., 0]
    P['gf'] = run_model(m_gf, ctx_gf, ff, SF_eot[name], np.zeros(len(ws), dtype='float32'), sig.get(name, tstd))[..., 0]
    P['pers'] = np.repeat((last*100)[:, None], H, 1)
    P['gtsm_al'] = np.stack([g_al[st+Tctx:st+W] for st in starts])*100
    P['gtsm_hp'] = np.stack([g_hp[st+Tctx:st+W] for st in starts])*100
    g0 = np.array([g_al[st+Tctx-1] for st in starts])*100    # GTSM at issue time t0
    err0 = (last*100) - g0                                    # observed error at t0
    lead = np.arange(1, H+1, dtype='float32')
    P['anch_inf'] = P['gtsm_al'] + err0[:, None]
    P['anch_24'] = P['gtsm_al'] + err0[:, None]*np.exp(-lead/24.0)[None]
    P['anch_12'] = P['gtsm_al'] + err0[:, None]*np.exp(-lead/12.0)[None]
    # bias correction with the mean error over the last 25 h (removes the semidiurnal component of y - GTSM)
    ctx_err_al = np.stack([raw[i, Tctx-25:Tctx]*100 - g_al[st+Tctx-25:st+Tctx]*100 for i, st in enumerate(starts)])
    ctx_err_hp = np.stack([raw[i, Tctx-25:Tctx]*100 - g_hp[st+Tctx-25:st+Tctx]*100 for i, st in enumerate(starts)])
    P['anch_lp'] = P['gtsm_al'] + ctx_err_al.mean(1)[:, None]
    P['anch_hp_lp'] = P['gtsm_hp'] + ctx_err_hp.mean(1)[:, None]
    ctx_err_hp208 = np.stack([raw[i, :Tctx]*100 - g_hp[st:st+Tctx]*100 for i, st in enumerate(starts)])
    P['anch_hp_lp208'] = P['gtsm_hp'] + ctx_err_hp208.mean(1)[:, None]
    np.savez_compressed(f'{ROOT}/outputs/gtsm_sym_preds/{name}{SUF}.npz', tru=tru, last=last*100, g0=g0,
                        ctx_err_al=ctx_err_al, ctx_err_hp=ctx_err_hp, **{k: P[k] for k in ['ours', 'gf', 'gtsm_al', 'gtsm_hp']})
    thr = np.quantile(tru, 0.999); wm = tru.max(1) >= thr
    ipk = tru[wm].argmax(1); ar = np.arange(wm.sum())
    row = dict(stn=name, n_win=len(ws), n_storm=int(wm.sum()), dist_km=float(mp.loc[name, 'dist_km']))
    for k in KEYS:
        e = rmse_lead(P[k], tru); FULL[k].append(e)
        row[f'rmse8_{k}'] = float(e[7]); row[f'rmse24_{k}'] = float(e[23]); row[f'rmse48_{k}'] = float(e[47])
        row[f'nnse8_{k}'] = float(nnse(P[k][:, 7], tru[:, 7]))
        if wm.sum() >= 5:
            STORM[k].append(rmse_lead(P[k][wm], tru[wm]))
            PK[k].append(float(P[k][wm][ar, ipk].sum()/(tru[wm][ar, ipk].sum()+1e-6)))
            row[f'storm8_{k}'] = float(STORM[k][-1][7]); row[f'storm48_{k}'] = float(STORM[k][-1][47]); row[f'pkcap_{k}'] = PK[k][-1]
    rows.append(row)
    if (k_i+1) % 10 == 0: print(f'  {k_i+1}/{len(te)}', flush=True)

d = pd.DataFrame(rows); d.to_csv(f'{ROOT}/outputs/eval_gtsm_symmetric{SUF}.csv', index=False)

def pval_wins(a, b):
    w = int((a < b).sum()); n = int(len(a)); return w, n, binomtest(w, n, 0.5).pvalue
def block(title, cand, ref, sub):
    L = [f'--- {title}  (n={len(sub)}) ---']
    for lead_c, lab in [(8, 'rmse8'), (24, 'rmse24'), (48, 'rmse48')]:
        L.append(f'  full-dist @{lead_c:>2}h RMSE: {cand} {sub[f"{lab}_{cand}"].mean():.2f} | {ref} {sub[f"{lab}_{ref}"].mean():.2f} cm'
                 f'   (median {sub[f"{lab}_{cand}"].median():.2f} | {sub[f"{lab}_{ref}"].median():.2f})')
    L.append(f'  NNSE@8h mean: {cand} {sub[f"nnse8_{cand}"].mean():.3f} | {ref} {sub[f"nnse8_{ref}"].mean():.3f}')
    w, n, p = pval_wins(sub[f'rmse8_{cand}'], sub[f'rmse8_{ref}'])
    pw = wilcoxon(sub[f'rmse8_{cand}'], sub[f'rmse8_{ref}']).pvalue
    L.append(f'  gauges where {cand} beats {ref} @8h: {w}/{n}  (sign p={p:.1e}, Wilcoxon p={pw:.1e})')
    w48, n48, p48 = pval_wins(sub[f'rmse48_{cand}'], sub[f'rmse48_{ref}'])
    L.append(f'  gauges where {cand} beats {ref} @48h: {w48}/{n48}  (sign p={p48:.1e})')
    ss = sub.dropna(subset=[f'storm8_{cand}', f'storm8_{ref}'])
    L.append(f'  p99.9 storm windows @8h : {cand} {ss[f"storm8_{cand}"].mean():.2f} | {ref} {ss[f"storm8_{ref}"].mean():.2f} cm   (n={len(ss)} gauges)')
    L.append(f'  p99.9 storm windows @48h: {cand} {ss[f"storm48_{cand}"].mean():.2f} | {ref} {ss[f"storm48_{ref}"].mean():.2f} cm')
    L.append(f'  peak capture at true peak hour (storm windows, pooled): {cand} {ss[f"pkcap_{cand}"].mean():.2f} | {ref} {ss[f"pkcap_{ref}"].mean():.2f}')
    return L

out = [f'SYMMETRIC GTSM HEAD-TO-HEAD (n={len(d)} gauges, 2010-2018 overlap, identical 48h windows, seismic-masked; {"CONTINUOUS windows only" if _args.continuous_only else "all windows"})', '']
out.append('== 0. Reproduction of eval_gtsm.py (flagship vs raw mean-aligned GTSM vs persistence) ==')
out.append(f'  @8h  ours {d.rmse8_ours.mean():.2f} | GTSM_al {d.rmse8_gtsm_al.mean():.2f} | pers {d.rmse8_pers.mean():.2f}   [eval_gtsm.log: 4.44 | 8.33 | 6.43]')
out.append(f'  @48h ours {d.rmse48_ours.mean():.2f} | GTSM_al {d.rmse48_gtsm_al.mean():.2f} | pers {d.rmse48_pers.mean():.2f}   [6.35 | 8.23 | 11.25]')
w, n, p = pval_wins(d.rmse8_pers, d.rmse8_gtsm_al)
out.append(f'  DIAGNOSTIC: persistence alone beats GTSM_al @8h at {w}/{n} gauges (p={p:.1e}); @24h {int((d.rmse24_pers<d.rmse24_gtsm_al).sum())}/{n}; @48h {int((d.rmse48_pers<d.rmse48_gtsm_al).sum())}/{n}')
out.append('')
out.append('== 1. Target-definition symmetry: GTSM high-passed like the target (720-h centered mean removed) ==')
for lab in ['rmse8', 'rmse24', 'rmse48']:
    out.append(f'  {lab}: GTSM_al {d[f"{lab}_gtsm_al"].mean():.2f} -> GTSM_hp {d[f"{lab}_gtsm_hp"].mean():.2f} cm')
out.append(f'  NNSE@8h: GTSM_al {d.nnse8_gtsm_al.mean():.3f} -> GTSM_hp {d.nnse8_gtsm_hp.mean():.3f}')
out.append('')
out.append('== 2. TIER A, no local observation on either side: gauge-free model vs GTSM_hp ==')
out += block('gauge-free (EOT20 statics, sigma_hat) vs GTSM_hp', 'gf', 'gtsm_hp', d)
out += block('gauge-free vs GTSM_al (raw alignment, as in the paper)', 'gf', 'gtsm_al', d)
out.append('')
out.append('== 3. TIER B, observation at t0 on both sides: flagship vs GTSM + persistence-of-error ==')
out += block('flagship vs GTSM anchored, tau=inf (y0 + GTSM increment)', 'ours', 'anch_inf', d)
out += block('flagship vs GTSM anchored, tau=24h', 'ours', 'anch_24', d)
out += block('flagship vs GTSM anchored, tau=12h', 'ours', 'anch_12', d)
out += block('flagship vs GTSM + 25-h mean-error bias correction (raw GTSM)', 'ours', 'anch_lp', d)
out += block('flagship vs GTSM_hp + 25-h mean-error bias correction  [strongest physics+obs baseline]', 'ours', 'anch_hp_lp', d)
out += block('flagship vs GTSM_hp + 208-h mean-error correction  [same history length as the learned model]', 'ours', 'anch_hp_lp208', d)
out += block('GTSM anchored (tau=inf) vs persistence  [does physics add to the observation?]', 'anch_inf', 'pers', d)
out += block('GTSM_hp + 25-h bias correction vs persistence', 'anch_hp_lp', 'pers', d)
out.append('')
sub = d[d.dist_km <= DIST_GATE_KM]
out.append(f'== 4. Sensitivity: drop gauges whose nearest GTSM point is > {DIST_GATE_KM:.0f} km away ({len(d)-len(sub)} dropped: {", ".join(d[d.dist_km > DIST_GATE_KM].stn)}) ==')
out += block('gauge-free vs GTSM_hp', 'gf', 'gtsm_hp', sub)
out += block('flagship vs GTSM anchored tau=inf', 'ours', 'anch_inf', sub)
out += block('flagship vs GTSM_hp + 25-h bias correction', 'ours', 'anch_hp_lp', sub)
out.append('')
out.append('== 5. Lead-resolved mean RMSE (cm), all gauges ==')
out.append('  lead  ' + ' '.join(f'{k:>8}' for k in KEYS))
for l in [1, 2, 4, 6, 8, 12, 18, 24, 36, 48]:
    out.append(f'  {l:>4}h ' + ' '.join(f'{np.mean(FULL[k], 0)[l-1]:8.2f}' for k in KEYS))
out.append('  storm windows (p99.9), mean over gauges with >=5 windows:')
for l in [4, 8, 12, 24, 36, 48]:
    out.append(f'  {l:>4}h ' + ' '.join(f'{np.mean(STORM[k], 0)[l-1]:8.2f}' for k in KEYS))
txt = '\n'.join(out)
open(f'{ROOT}/outputs/eval_gtsm_symmetric{SUF}.log', 'w').write(txt+'\n')
print(txt)
