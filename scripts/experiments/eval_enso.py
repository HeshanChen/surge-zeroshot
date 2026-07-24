"""ENSO-phase and season stratification of zero-shot skill (Andrew's hypothesis):
does the Oceania rotation (no-Australia training) lose disproportionately in particular
ENSO phases, relative to the global deployment control?
Per-window skill stratified by the ONI phase of the window's issue month.
-> outputs/eval_enso_<tag>.csv + combined log"""
import sys, warnings; warnings.filterwarnings('ignore')
import torch, pandas as pd, numpy as np
sys.path.insert(0, '/Users/heshan/Desktop/surge_fm/src')
from models.baseline_lstm import GlobalLSTM
from data.dataset_v0 import seismic_mask
ROOT = '/Users/heshan/Desktop/surge_fm'
Tctx, W, H = 208, 256, 48

# ---- ONI phases: season -> center month ----
SEAS = {'DJF':1,'JFM':2,'FMA':3,'MAM':4,'AMJ':5,'MJJ':6,'JJA':7,'JAS':8,'ASO':9,'SON':10,'OND':11,'NDJ':12}
oni = {}
for ln in open(f'{ROOT}/data/raw/oni.txt').read().strip().split('\n')[1:]:
    p = ln.split()
    if len(p) == 4 and p[0] in SEAS:
        y, m, a = int(p[1]), SEAS[p[0]], float(p[3])
        oni[(y, m)] = 'EN' if a >= 0.5 else ('LN' if a <= -0.5 else 'NEU')
MET_SEASON = {12:'DJF',1:'DJF',2:'DJF',3:'MAM',4:'MAM',5:'MAM',6:'JJA',7:'JJA',8:'JJA',9:'SON',10:'SON',11:'SON'}

sa = pd.read_csv(f'{ROOT}/catalog/static_attributes.csv').set_index('name')

def run(ckpt, split_csv, tag):
    sp = pd.read_csv(f'{ROOT}/catalog/{split_csv}')
    tr = [n for n in sp[sp.fold == 'train'].name if n in sa.index]
    te = [n for n in sp[sp.fold == 'test'].name if n in sa.index]
    def sfeat(n):
        r = sa.loc[n]; la, lo = np.radians(float(r.lat)), np.radians(float(r.lon))
        return np.array([np.cos(la)*np.cos(lo), np.cos(la)*np.sin(lo), np.sin(la), float(r.tidal_range_m), float(r.form_factor)], dtype='float32')
    arr = np.stack([sfeat(n) for n in tr]); smu = arr.mean(0); ssd = arr.std(0)+1e-6
    m = GlobalLSTM(n_out=3); m.load_state_dict(torch.load(f'{ROOT}/outputs/{ckpt}', map_location='cpu')); m.eval()
    rows = []
    for k, name in enumerate(te):
        try:
            s = pd.read_parquet(f'{ROOT}/data/processed/{name}.parquet')['surge']; s = s.where(s.abs() < 4)
            e = pd.read_parquet(f'{ROOT}/data/raw/era5/{name}.parquet')
            if e.index.tz is None: e.index = e.index.tz_localize('UTC')
            for c in ['wind_speed_10m','wind_direction_10m','pressure_msl','precipitation']:
                e[c] = pd.to_numeric(e[c], errors='coerce')
            spd = e['wind_speed_10m'].values; dd = np.deg2rad(e['wind_direction_10m'].values)
            e2 = pd.DataFrame({'wu':spd*np.sin(dd),'wv':spd*np.cos(dd),'mslp':e['pressure_msl'].values,'precip':e['precipitation'].values}, index=e.index)
            df = pd.DataFrame({'surge':s}).join(e2, how='inner').dropna()
            r = sa.loc[name]; df = df[~seismic_mask(float(r.lat), float(r.lon), df.index)]
        except Exception: continue
        if len(df) < 3000: continue
        a = df.values.astype('float32'); tstd = float(a[:, 0].std())+1e-6
        af = a.copy(); af[:, 1:] = (af[:, 1:]-af[:, 1:].mean(0))/(af[:, 1:].std(0)+1e-6)
        starts = list(range(0, len(af)-W, H))
        t0s = df.index[[st+Tctx-1 for st in starts]]
        ws = np.stack([af[st:st+W] for st in starts]); raw = np.stack([a[st:st+W, 0] for st in starts])
        mu = ws[:, :Tctx, 0].mean(1, keepdims=True); sd = ws[:, :Tctx, 0].std(1, keepdims=True)+1e-6
        ctx = np.concatenate([((ws[:, :Tctx, 0]-mu)/sd)[:, :, None], ws[:, :Tctx, 1:]], 2).transpose(0, 2, 1)
        ff = ws[:, Tctx:, 1:].transpose(0, 2, 1); last = raw[:, Tctx-1]
        ps = []
        with torch.no_grad():
            for j in range(0, len(ws), 512):
                _, o = m.predict_window(torch.tensor(ctx[j:j+512]).float(), torch.tensor(ff[j:j+512]).float(), H,
                                        torch.tensor((sfeat(name)-smu)/ssd)[None].expand(min(512, len(ws)-j), -1).float(),
                                        torch.tensor(last[j:j+512]/tstd).float())
                ps.append(o[..., 0].cpu())
        pred = torch.cat(ps).numpy()*tstd*100
        tru = raw[:, Tctx:]*100; per = (last*100)[:, None]
        se_m = ((pred-tru)**2).mean(1); se_p = ((per-tru)**2).mean(1)
        s8_m = (pred[:, 7]-tru[:, 7])**2; s8_p = (per[:, 0]-tru[:, 7])**2
        for i, t0 in enumerate(t0s):
            ph = oni.get((t0.year, t0.month))
            if ph is None: continue
            rows.append((name, ph, MET_SEASON[t0.month], se_m[i], se_p[i], s8_m[i], s8_p[i]))
        if (k+1) % 25 == 0: print(f'  [{tag}] {k+1}/{len(te)}', flush=True)
    d = pd.DataFrame(rows, columns=['stn','phase','season','se_m','se_p','s8_m','s8_p'])
    d.to_csv(f'{ROOT}/outputs/eval_enso_{tag}.csv', index=False)
    return d

def agg(d, key):
    out = []
    for g, gr in d.groupby(key):
        pg = gr.groupby('stn').agg(m=('se_m','mean'), p=('se_p','mean'), m8=('s8_m','mean'), p8=('s8_p','mean'), n=('se_m','size'))
        pg = pg[pg.n >= 20]
        sk = 100*(1 - np.sqrt(pg.m).mean()/np.sqrt(pg.p).mean())
        sk8 = 100*(1 - np.sqrt(pg.m8).mean()/np.sqrt(pg.p8).mean())
        out.append(f'  {g:4s}: pooled {sk:+.1f}%  @8h {sk8:+.1f}%  (gauges {len(pg)}, windows {int(pg.n.sum())})')
    return '\n'.join(out)

ENCL = ('west_channel_pile','hovell_pile','point_richards_corio_bay','melbourne_williamstown','queenscliff','lakes_entrance_inner_bullock_island')
lines = []
for ckpt, split, tag in [('baseline_lstmq_v2rotocr2_best.pt', 'exp_split_oc_v2.csv', 'ocrot'),
                          ('baseline_lstmq_v2final_best.pt', 'exp_split_final.csv', 'deploy')]:
    d = run(ckpt, split, tag)
    lines.append(f'== {tag} ==  ENSO phase:'); lines.append(agg(d, 'phase'))
    lines.append('  season:'); lines.append(agg(d, 'season'))
    if tag == 'ocrot':
        d['encl'] = ['enclosed' if s.startswith(ENCL) else 'open' for s in d.stn]
        for sub in ('enclosed','open'):
            lines.append(f'  [{sub}] ENSO phase:'); lines.append(agg(d[d.encl == sub], 'phase'))
    print('\n'.join(lines[-6:]), flush=True)
open(f'{ROOT}/outputs/eval_enso.log','w').write('\n'.join(lines)+'\n')
print('ENSO EVAL DONE', flush=True)
