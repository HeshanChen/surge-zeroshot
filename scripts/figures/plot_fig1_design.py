"""Figure 1 (study design, Nature style, 183mm, THREE stacked rows):
(a) global gauge map, equal-aspect, blue train dots / red test dots on real coastlines;
(b) architecture schematic, FULL-WIDTH row, large boxes, no overflow;
(c) zero-shot output example, full-width short row.
-> outputs/fig1_design.{pdf,png}"""
import os as _os
_ROOT = _os.environ.get('SURGE_ROOT') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
import sys, json, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, f'{_ROOT}/scripts')
sys.path.insert(0, f'{_ROOT}/src')
import matplotlib; matplotlib.use('Agg')
from pubstyle import apply, save_pub, PAL, W2, panel_label
apply()
import pandas as pd, numpy as np, torch
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon as MplPolygon
ROOT = _ROOT

figm, axm = plt.subplots(figsize=(W2, 3.55))
figc, axc = plt.subplots(figsize=(W2*0.55, 2.2))

# ---------- (a) map: equal aspect, red/blue dots ----------
land = json.load(open(f'{ROOT}/catalog/ne_110m_land.geojson'))
for feat in land['features']:
    g = feat['geometry']
    rings = g['coordinates'][:1] if g['type'] == 'Polygon' else [poly[0] for poly in g['coordinates']]
    for ring in rings:
        axm.add_patch(MplPolygon(ring, closed=True, fc='#EDEAE3', ec='#C4BFB2', lw=0.3, zorder=0))
sa = pd.read_csv(f'{ROOT}/catalog/static_attributes.csv').set_index('name')
sp = pd.read_csv(f'{ROOT}/catalog/exp_split_final.csv')
NONMAR_NAMES = ('rak_zuid','trois_rivieres','batiscan','lac_saint_pierre','sorel','port_saint_francois',
                'becancour','champlain_qc','deschaillons','portneuf_qc','neuville')
def nonmarine(n):
    la, lo_ = float(sa.loc[n,'lat']), float(sa.loc[n,'lon'])
    return (41.0 <= la <= 49.5 and -93.5 <= lo_ <= -75.5) or n.startswith(NONMAR_NAMES)
lo = lambda x: x if x <= 180 else x - 360
def draw(names, color, dot_lab, x_lab, ds=4, xs=9):
    mar = [n for n in names if not nonmarine(n)]; non = [n for n in names if nonmarine(n)]
    axm.scatter([lo(sa.loc[n,'lon']) for n in mar], [sa.loc[n,'lat'] for n in mar],
                s=ds, c=color, alpha=0.85, lw=0, zorder=2, label=f'{dot_lab} ({len(mar)})')
    if non:
        axm.scatter([lo(sa.loc[n,'lon']) for n in non], [sa.loc[n,'lat'] for n in non],
                    s=xs, c=color, marker='x', lw=0.6, zorder=3, label=f'{x_lab} ({len(non)})')
tr = [n for n in sp[sp.fold=='train'].name if n in sa.index]
va = [n for n in sp[sp.fold=='val'].name if n in sa.index]
te = [n for n in sp[sp.fold=='test'].name if n in sa.index]
axm.scatter([lo(sa.loc[n,'lon']) for n in tr], [sa.loc[n,'lat'] for n in tr],
            s=4, c=PAL['blue'], alpha=0.85, lw=0, zorder=2, label=f'train ({len(tr)})')
axm.scatter([lo(sa.loc[n,'lon']) for n in va], [sa.loc[n,'lat'] for n in va],
            s=7, c=PAL['green'], lw=0, zorder=3, label=f'validation ({len(va)})')
axm.scatter([lo(sa.loc[n,'lon']) for n in te], [sa.loc[n,'lat'] for n in te],
            s=6, c='#D62728', lw=0, zorder=4, label=f'marine test, every continent ({len(te)})')
axm.set_xlim(-180, 180); axm.set_ylim(-90, 90)
axm.set_aspect('equal', adjustable='box')
axm.set_xticks(range(-150, 151, 50)); axm.set_yticks(range(-80, 81, 40))
axm.set_xlabel('longitude ($^\\circ$)', labelpad=1); axm.set_ylabel('latitude ($^\\circ$)')
axm.legend(loc='lower left', markerscale=1.2, handletextpad=0.3, borderaxespad=0.2)
axm.set_title('Pre-registered global split: deduplicated, seismic-masked, 0.5$^\\circ$ buffer', pad=4)


# ---------- (c) example: full-width short row ----------
from models.baseline_lstm import GlobalLSTM
import data.dataset_v0 as dv
name = 'aburatsu-354a-jpn-uhslc'
a = dv.load_station(name); tstd = float(a[:, 0].std()) + 1e-6
af = a.copy(); af[:, 1:] = (af[:, 1:] - af[:, 1:].mean(0)) / (af[:, 1:].std(0) + 1e-6)
Tctx, W = 208, 256
starts = list(range(0, len(a) - W, 48))
tru_all = np.stack([a[st + Tctx:st + W, 0] for st in starts])
wi = int(np.argmax(tru_all.max(1))); st = starts[wi]; w = af[st:st + W]
mu = w[:Tctx, 0].mean(); sd = w[:Tctx, 0].std() + 1e-6
ctx = np.concatenate([((w[:Tctx, 0] - mu) / sd)[:, None], w[:Tctx, 1:]], 1).T[None]
ff = w[Tctx:, 1:].T[None]
splt = pd.read_csv(f'{ROOT}/catalog/exp_split_jp_v2.csv')
trn = [n for n in splt[splt.fold == 'train'].name if n in sa.index]
def sfeat(n):
    r = sa.loc[n]; la_, lo_ = np.radians(float(r.lat)), np.radians(float(r.lon))
    return np.array([np.cos(la_) * np.cos(lo_), np.cos(la_) * np.sin(lo_), np.sin(la_),
                     float(r.tidal_range_m), float(r.form_factor)], dtype='float32')
arrst = np.stack([sfeat(n) for n in trn]); smu_, ssd_ = arrst.mean(0), arrst.std(0) + 1e-6
sfv = (sfeat(name) - smu_) / ssd_
m = GlobalLSTM(n_out=3); m.load_state_dict(torch.load(f'{ROOT}/models/rot_japan_best.pt', map_location='cpu')); m.eval()
with torch.no_grad():
    _, out = m.predict_window(torch.tensor(ctx).float(), torch.tensor(ff).float(), 48,
                              torch.tensor(sfv)[None].float(), torch.tensor([a[st + Tctx - 1, 0] / tstd]).float())
o = out[0].numpy() * tstd * 100
hx = np.arange(1, 49); tru_h = a[st + Tctx:st + W, 0] * 100
axc.fill_between(hx, o[:, 1], o[:, 2], color=PAL['blue'], alpha=0.18, lw=0, label='q90 to q99 envelope')
axc.plot(hx, tru_h, c='k', lw=1.0, label='observed', zorder=3)
axc.plot(hx, o[:, 0], c=PAL['blue'], lw=1.0, label='point forecast', zorder=2)
axc.axhline(a[st + Tctx - 1, 0] * 100, c=PAL['grey'], ls='--', lw=0.7, label='persistence')
axc.set_xlabel('lead time (h)', labelpad=1); axc.set_ylabel('surge (cm)')
axc.set_xlim(0, 48); axc.set_ylim(min(tru_h.min(), o[:, 0].min()) - 8, max(tru_h.max(), o[:, 2].max()) + 30)
axc.legend(loc='upper right', ncol=1, handlelength=1.5, borderaxespad=0.3, labelspacing=0.3)
axc.set_title('Zero-shot forecast at a held-out gauge (typhoon)', pad=3)


figm.tight_layout(pad=0.3); save_pub(figm, f'{ROOT}/outputs/fig1a_map')
figc.tight_layout(pad=0.3); save_pub(figc, f'{ROOT}/outputs/fig1c_example')
print('wrote outputs/fig1a_map + fig1c_example (pdf+png)')
