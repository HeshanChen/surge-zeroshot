"""Appendix zoom figure: the two regions where non-marine gauges cluster.
(a) Great Lakes + St Lawrence  (b) NW European estuaries. Same encoding as the main map:
color = role (train/val/test), cross = non-marine. 50m coastline + lakes overlay.
-> outputs/figS1_zoom.{pdf,png}"""
import os as _os
_ROOT = _os.environ.get('SURGE_ROOT') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
import sys, json, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, f'{_ROOT}/scripts')
import matplotlib; matplotlib.use('Agg')
from pubstyle import apply, save_pub, PAL, W2, panel_label
apply()
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
ROOT = _ROOT

sa = pd.read_csv(f'{ROOT}/catalog/static_attributes.csv').set_index('name')
sp = pd.read_csv(f'{ROOT}/catalog/exp_split_final.csv')
NONMAR_NAMES = ('rak_zuid','trois_rivieres','batiscan','lac_saint_pierre','sorel','port_saint_francois',
                'becancour','champlain_qc','deschaillons','portneuf_qc','neuville')
def nonmarine(n, fold):
    if fold == 'test_xdom': return True
    la, lo_ = float(sa.loc[n,'lat']), float(sa.loc[n,'lon'])
    return (41.0 <= la <= 49.5 and -93.5 <= lo_ <= -75.5) or n.startswith(NONMAR_NAMES)
lo = lambda x: x if x <= 180 else x - 360

def draw_geo(ax, x0, x1, y0, y1):
    for fname, fc, ec, z in [('ne_50m_land.geojson', '#EDEAE3', '#C4BFB2', 0),
                             ('ne_50m_lakes.geojson', 'white', '#C4BFB2', 1)]:
        gj = json.load(open(f'{ROOT}/catalog/{fname}'))
        for feat in gj['features']:
            g = feat['geometry']
            polys = [g['coordinates']] if g['type'] == 'Polygon' else g['coordinates']
            for poly in polys:
                ring = poly[0]
                xs = [p[0] for p in ring]; ys = [p[1] for p in ring]
                if max(xs) < x0-2 or min(xs) > x1+2 or max(ys) < y0-2 or min(ys) > y1+2: continue
                ax.add_patch(MplPolygon(ring, closed=True, fc=fc, ec=ec, lw=0.3, zorder=z))
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect('equal', adjustable='box')

def draw_stations(ax, x0, x1, y0, y1):
    handles = {}
    for fold, color, lab in [('train', PAL['blue'], 'train'), ('val', PAL['green'], 'validation'),
                             ('test', '#D62728', 'test (marine)'), ('test_xdom', '#D62728', None)]:
        names = [n for n in sp[sp.fold == fold].name if n in sa.index]
        pts = [(lo(float(sa.loc[n,'lon'])), float(sa.loc[n,'lat']), nonmarine(n, fold)) for n in names]
        pts = [(x, y, nm) for x, y, nm in pts if x0 <= x <= x1 and y0 <= y <= y1]
        for nm, marker, s_, lw_ in [(False, 'o', 14, 0), (True, 'x', 26, 0.9)]:
            sel = [(x, y) for x, y, m in pts if m == nm]
            if not sel: continue
            h = ax.scatter(*zip(*sel), s=s_, c=color, marker=marker, lw=lw_, zorder=3)
            key = (color, marker)
            if key not in handles: handles[key] = h
    return handles

fig, axes = plt.subplots(1, 2, figsize=(W2, 2.9), gridspec_kw=dict(width_ratios=[1.55, 1]))
draw_geo(axes[0], -95, -69, 41, 50.5); draw_stations(axes[0], -95, -69, 41, 50.5)
axes[0].set_title('Great Lakes and St. Lawrence')
axes[0].set_xlabel('longitude ($^\\circ$)', labelpad=1); axes[0].set_ylabel('latitude ($^\\circ$)')
draw_geo(axes[1], -4.5, 11, 46, 55.5); h = draw_stations(axes[1], -4.5, 11, 46, 55.5)
axes[1].set_title('NW European estuaries')
axes[1].set_xlabel('longitude ($^\\circ$)', labelpad=1)
import matplotlib.lines as mlines
leg = [mlines.Line2D([], [], color=PAL['blue'], marker='o', ls='', ms=4, label='train'),
       mlines.Line2D([], [], color=PAL['green'], marker='o', ls='', ms=4, label='validation'),
       mlines.Line2D([], [], color='#D62728', marker='o', ls='', ms=4, label='test (marine)'),
       mlines.Line2D([], [], color='#444444', marker='x', ls='', ms=5, mew=1.0, label='non-marine (any role)')]
axes[0].legend(handles=leg, loc='lower left', handletextpad=0.3, borderaxespad=0.25)
for ax, letter in zip(axes, 'ab'): panel_label(ax, letter, dx=-0.07, dy=1.10)
fig.tight_layout(w_pad=1.6)
save_pub(fig, f'{ROOT}/outputs/figS1_zoom')
print('wrote outputs/figS1_zoom.{pdf,png}')
