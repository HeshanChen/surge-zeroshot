"""Supplementary annex for the seismic (tsunami and seiche) mask, requested in review round 1 (item 65, 2026-09-14).
Four spot-check events, two tsunamis the mask removes and two storms it leaves intact: the M7+ earthquakes inside each
event window, the rule tier each triggers at an example gauge, the hours removed there, how many of the 900 split gauges
lie inside the rule radius of the largest event, and whether the largest surge in the window survives; plus the sixteen
gauges that lose the largest share of their p99.9 exceedance hours (outputs/seismic_exceedance_audit.csv). Everything is
recomputed from the catalogue (catalog/earthquakes_m7.csv), the processed records and the split file with the production
mask function (src/data/dataset_v0.seismic_mask); no model is run.
-> outputs/tsunami_mask_annex.csv, outputs/tsunami_mask_annex_table.tex, outputs/figS3_tsunami_mask.{pdf,png}"""
import os as _os
_ROOT = _os.environ.get('SURGE_ROOT') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
import sys, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.dates as mdates, matplotlib.transforms as mtrans
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import cartopy.crs as ccrs, cartopy.feature as cfeature
from cartopy.geodesic import Geodesic
ROOT = _ROOT
sys.path.insert(0, f'{ROOT}/src'); sys.path.insert(0, f'{ROOT}/scripts')
from data.dataset_v0 import seismic_mask, _quakes
from pubstyle import apply, save_pub, PAL, MM, panel_label
apply()
sa = pd.read_csv(f'{ROOT}/catalog/static_attributes.csv').set_index('name')
sp = pd.read_csv(f'{ROOT}/catalog/exp_split_final.csv'); FOLD = sp.set_index('name').fold
GAUGES = [n for n in sp.name if n in sa.index]; GLAT = sa.loc[GAUGES, 'lat'].values.astype(float); GLON = sa.loc[GAUGES, 'lon'].values.astype(float)
q = _quakes()
TIERS = [(8.0, 8000, 168), (7.5, 3000, 96), (7.0, 1000, 72)]        # magnitude floor, radius (km), hours removed after the event
def hav(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    return 6371*2*np.arcsin(np.sqrt(np.sin((la2-la1)/2)**2 + np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2))
def cls(m):                                                           # magnitude class of an event: (label, radius km, hours)
    for mm, r, h in TIERS:
        if m >= mm: return ('M8.0+' if mm == 8.0 else f'M{mm:.1f} to {mm+0.4:.1f}', r, h)
    return None
def triggers(m, d): c = cls(m); return bool(c and d <= c[1])
COUNTRY = {'jpn': 'Japan', 'tha': 'Thailand', 'usa': 'United States', 'phl': 'Philippines', 'vnm': 'Vietnam', 'mex': 'Mexico', 'vut': 'Vanuatu',
           'fra': 'France', 'nzl': 'New Zealand', 'lva': 'Latvia'}
PLACE = {'numbo_noumea': ('Nouméa', 'New Caledonia (France)'), 'pointe_pitre': ('Pointe-à-Pitre', 'Guadeloupe (France)'), 'pago_bay': ('Pago Bay', 'Guam (United States)'),
         'veracruz_ver': ('Veracruz', None), 'rough_and_ready_island': ('Rough and Ready Island', None), 'ko_taphao_noi': ('Ko Taphao Noi', None)}
def gname(g): b = g.split('-')[0]; return PLACE.get(b, (None,))[0] or b.replace('_', ' ').title()
def country(g): b = g.split('-')[0]; c = g.split('-')[-2]; return (PLACE.get(b, (None, None))[1] if b in PLACE else None) or COUNTRY.get(c, c.upper())
EVENTS = [dict(name='Tohoku 2011', gauge='onahama-ma12-jpn-jodc_jma', t0='2011-03-06', t1='2011-03-20', gauge_off=(-5, -9, 'right'), epi_off=(5, 4, 'left'), radius_az=100),
          dict(name='Sumatra 2004', gauge='ko_taphao_noi-148a-tha-uhslc', t0='2004-12-21', t1='2005-01-04', gauge_off=(5, 5, 'left'), epi_off=(-5, -9, 'right'), radius_az=135),
          dict(name='Sandy 2012', gauge='atlantic_city-8534720-usa-noaa', t0='2012-10-24', t1='2012-11-03', gauge_off=(5, -8, 'left'), epi_off=(-6, -7, 'right'), radius_az=200),
          dict(name='Haiyan 2013', gauge='cebu-379a-phl-uhslc', t0='2013-11-03', t1='2013-11-13', gauge_off=(5, 5, 'left'), epi_off=(5, 4, 'left'))]
rows, panels = [], []
for ev in EVENTS:
    g = ev['gauge']; la, lo = float(sa.loc[g, 'lat']), float(sa.loc[g, 'lon'])
    T0, T1 = pd.Timestamp(ev['t0'], tz='UTC'), pd.Timestamp(ev['t1'], tz='UTC')
    s = pd.read_parquet(f'{ROOT}/data/processed/{g}.parquet')['surge']; s = s[(s.index >= T0) & (s.index <= T1)].dropna()
    full = pd.date_range(T0, T1, freq='h'); sf = s.reindex(full)                 # NaN where the record has no row
    m = seismic_mask(la, lo, s.index)                                             # production mask on the rows that exist
    qq = q[(q.time >= T0) & (q.time <= T1)].copy().sort_values('time')
    qq['dist'] = hav(la, lo, qq.latitude.values, qq.longitude.values); qq['trig'] = [triggers(mm, dd) for mm, dd in zip(qq.mag, qq.dist)]
    trig = qq[qq['trig'].astype(bool)]; strongest = trig.loc[trig.mag.idxmax()] if len(trig) else None
    spans = [(max(T0, t - pd.Timedelta(hours=1)), min(T1, t + pd.Timedelta(hours=cls(mm)[2]))) for t, mm in zip(trig.time, trig.mag)]
    nominal = pd.Series(False, index=full)
    for a, b in spans: nominal.loc[a:b] = True
    if len(qq):
        big = qq.loc[qq.mag.idxmax()]; Rbig = cls(big.mag)[1]
        aff = hav(GLAT, GLON, float(big.latitude), float(big.longitude)) <= Rbig
        aff_any = np.zeros(len(GAUGES), bool)
        for _, e in qq.iterrows(): aff_any |= hav(GLAT, GLON, float(e.latitude), float(e.longitude)) <= cls(e.mag)[1]
    else: big = None; Rbig = 0; aff = np.zeros(len(GAUGES), bool); aff_any = aff.copy()
    ipk = s.abs().idxmax(); peak = float(s.loc[ipk])*100; peak_removed = bool(m.loc[ipk])
    eq = [(t, mm, dd, tr) for t, mm, dd, tr in zip(qq.time, qq.mag, qq.dist, qq.trig)]
    rows.append(dict(event=ev['name'], gauge=g, gauge_label=gname(g), country=country(g), fold=FOLD[g],
                     earthquakes='; '.join(f"{t:%Y-%m-%d %H:%M} UTC M{mm:.1f} {dd:,.0f} km{' (triggers)' if tr else ''}" for t, mm, dd, tr in eq) or 'none',
                     n_m7=len(qq), n_triggering=len(trig),
                     strongest_tier=(f"{cls(strongest.mag)[0]}, {cls(strongest.mag)[1]:,} km, {cls(strongest.mag)[2]} h (M{strongest.mag:.1f} at {strongest.dist:,.0f} km)" if strongest is not None else 'none'),
                     hours_removed=int(m.sum()), hours_in_rule_windows=int(nominal.sum()), rows_present=len(s), hours_in_window=len(full), last_row=str(s.index[-1]),
                     largest_event=(f"M{big.mag:.1f} {big.dist:,.0f} km, radius {Rbig:,} km" if big is not None else 'none'),
                     gauges_in_radius_of_largest=int(aff.sum()), gauges_in_any_radius=int(aff_any.sum()), split_gauges=len(GAUGES),
                     largest_mag=(float(big.mag) if big is not None else np.nan), largest_dist_km=(round(float(big.dist)) if big is not None else np.nan), largest_radius_km=Rbig, largest_hours=(cls(big.mag)[2] if big is not None else 0),
                     strongest_mag=(float(strongest.mag) if strongest is not None else np.nan), strongest_dist_km=(round(float(strongest.dist)) if strongest is not None else np.nan),
                     strongest_class=(cls(strongest.mag)[0] if strongest is not None else 'none'), strongest_radius_km=(cls(strongest.mag)[1] if strongest is not None else 0), strongest_hours=(cls(strongest.mag)[2] if strongest is not None else 0),
                     peak_cm=round(peak, 1), peak_time=str(ipk), peak_status=('removed' if peak_removed else 'kept')))
    panels.append(dict(ev=ev, s=s, sf=sf, m=m, qq=qq, trig=trig, spans=spans, aff=aff, la=la, lo=lo, T0=T0, T1=T1, big=big, Rbig=Rbig, strongest=strongest, ipk=ipk, peak=peak, eq=eq))
d = pd.DataFrame(rows); d.to_csv(f'{ROOT}/outputs/tsunami_mask_annex.csv', index=False)
print(d.drop(columns=['earthquakes']).to_string())
se = pd.read_csv(f'{ROOT}/outputs/seismic_exceedance_audit.csv').sort_values('frac_p999_masked', ascending=False).head(16)
se['fold'] = se.stn.map(FOLD).fillna('pool')
print('gauges with more than 10% of p99.9 exceedance hours removed:', int((pd.read_csv(f'{ROOT}/outputs/seismic_exceedance_audit.csv').frac_p999_masked > 0.10).sum()), '| 16th listed share:', round(100*se.frac_p999_masked.iloc[-1], 1))

# ---------------- LaTeX table ----------------
def esc(x): return str(x).replace('%', '\\%').replace('_', '\\_').replace('&', '\\&')
r0 = d.iloc[0]; r2 = d.iloc[2]; p0 = panels[0]
onahama_note = (f"the Onahama record has no rows after {pd.Timestamp(r0.last_row):%-d %B %H:%M} UTC in this window, so the rule spans "
                f"{r0.hours_in_rule_windows} hours but only {r0.rows_present} rows exist, of which {r0.hours_removed} are removed")
sandy_eq = panels[2]['big']
cap_a = ("Seismic mask: behaviour on four events and the gauges it affects most. The mask removes hours within graded windows of USGS M7+ earthquakes "
         "(M$\\geq$7.0 within 1{,}000 km: 72 h; M$\\geq$7.5 within 3{,}000 km: 96 h; M$\\geq$8.0 within 8{,}000 km: 168 h; Methods). "
         "(a) Two tsunamis the mask removes and two storms it leaves intact. Earthquakes: every M7+ event in the window with its distance to the example gauge; "
         "events inside the rule radius of their magnitude class are in bold. Strongest tier: the magnitude class, radius and duration of the longest window triggered at the gauge. "
         f"Hours removed: rows of the gauge's record inside the mask windows; the count falls below the nominal span where the record has gaps (Ko Taphao Noi lacks {d.iloc[1].hours_in_window - d.iloc[1].rows_present} of the {d.iloc[1].hours_in_window} hours in its window; {onahama_note}). "
         "Gauges affected: the number of the 900 split gauges inside the rule radius of the largest earthquake in the window. "
         "Largest surge: the largest absolute surge in the window at the example gauge and whether the mask removes it. "
         f"During Sandy an M{sandy_eq.mag:.1f} earthquake off Haida Gwaii lay {sandy_eq.dist:,.0f} km from Atlantic City, outside the {cls(sandy_eq.mag)[1]:,} km radius of its class, "
         f"so no hour was removed there while {r2.gauges_in_radius_of_largest} Pacific gauges lost hours; no M7+ earthquake occurred during Haiyan. "
         "(b) The sixteen gauges that lose the largest share of their p99.9 exceedance hours to the mask, with the share and the number of exceedance hours.")
L = ['\\begin{table}[h]\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}', '\\caption{' + cap_a + '}\\label{sitab:tsunami}',
     '\\providecommand{\\arraybackslash}{\\let\\\\\\tabularnewline}',
     '\\makebox[\\textwidth][l]{\\textbf{a}}\\\\[0.2em]',
     '\\begin{tabular}{p{0.075\\textwidth}p{0.12\\textwidth}p{0.25\\textwidth}p{0.13\\textwidth}rrp{0.11\\textwidth}}\\toprule',
     'Event & \\raggedright\\arraybackslash Example gauge (fold) & \\raggedright\\arraybackslash M7+ earthquakes in the window (distance to the gauge) & \\raggedright\\arraybackslash Strongest tier triggered at the gauge & \\multicolumn{1}{c}{\\shortstack[c]{Hours\\\\removed}} & \\multicolumn{1}{c}{\\shortstack[c]{Gauges\\\\affected}} & \\raggedright\\arraybackslash Largest surge in the window\\\\\\midrule']
for r, P in zip(rows, panels):
    eqs = '; '.join((('\\textbf{' if tr else '') + f"{t:%-d %b %H:%M} M{mm:.1f}, {dd:,.0f} km" + ('}' if tr else '')) for t, mm, dd, tr in P['eq']) or 'none'
    st = P['strongest']; tier = f"{cls(st.mag)[0]}: {cls(st.mag)[1]:,} km, {cls(st.mag)[2]} h" if st is not None else 'none'
    L.append(f"\\raggedright\\arraybackslash {esc(r['event'])} & \\raggedright\\arraybackslash {esc(r['gauge_label'])}, {esc(r['country'])} ({r['fold']}) & \\raggedright\\arraybackslash {eqs} & \\raggedright\\arraybackslash {tier} & {r['hours_removed']} & {r['gauges_in_radius_of_largest']} & \\raggedright\\arraybackslash {('$-$' + f'{abs(r[chr(112)+chr(101)+chr(97)+chr(107)+chr(95)+chr(99)+chr(109)]):.0f}') if r['peak_cm'] < 0 else f'{r[chr(112)+chr(101)+chr(97)+chr(107)+chr(95)+chr(99)+chr(109)]:.0f}'}~cm, {r['peak_status']}\\\\")
L += ['\\bottomrule\\end{tabular}\\\\[1.0em]', '\\makebox[\\textwidth][l]{\\textbf{b}}\\\\[0.2em]', '\\begin{tabular}{llrr}\\toprule', 'Gauge & Fold & p99.9 exceedance hours removed & Exceedance hours\\\\\\midrule']
for _, r in se.iterrows():
    L.append(f"{esc(gname(r.stn))}, {esc(country(r.stn))} & {r.fold} & {100*r.frac_p999_masked:.0f}\\% & {int(r.n_exc)}\\\\")
L += ['\\bottomrule\\end{tabular}', '\\end{table}']
open(f'{ROOT}/outputs/tsunami_mask_annex_table.tex', 'w').write('\n'.join(L) + '\n'); print('table written')

# ---------------- figure (house style: pubstyle, legend on top, plain centred titles, no text inside the panels) ----------------
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
V, B, GREY = PAL['verm'], PAL['blue'], PAL['grey']
LAND, COAST = '#EDEAE3', '#C4BFB2'                     # same land palette as Fig. 1a
fig = plt.figure(figsize=(183*MM, 190*MM))
gs = fig.add_gridspec(4, 2, width_ratios=[1.0, 1.7], hspace=0.5, wspace=0.06, left=0.06, right=0.985, top=0.91, bottom=0.045)
def has_scale(scale):
    try: cfeature.LAND.with_scale(scale).geometries().__next__(); return True
    except Exception: return False
gd = Geodesic()
def num(t): return mdates.date2num(t.tz_convert(None))
for i, (r, P) in enumerate(zip(rows, panels)):
    ev, s_, sf, qq, trig, spans, aff, la, lo, T0, T1, big, Rbig = (P[k] for k in ('ev', 's', 'sf', 'qq', 'trig', 'spans', 'aff', 'la', 'lo', 'T0', 'T1', 'big', 'Rbig'))
    # ----- map: azimuthal equidistant, centred on the largest earthquake (its rule radius is an exact circle) or on the gauge
    clon, clat = (float(big.longitude), float(big.latitude)) if big is not None else (lo, la)
    proj = ccrs.AzimuthalEquidistant(central_longitude=clon, central_latitude=clat)
    axm = fig.add_subplot(gs[i, 0], projection=proj)
    bb = axm.get_position(); W_in, H_in = fig.get_size_inches(); side = bb.height*H_in*1.25
    cx, cy = (bb.x0 + bb.x1)/2, (bb.y0 + bb.y1)/2
    axm.set_position([cx - side/W_in/2, cy - side/H_in/2, side/W_in, side/H_in])
    dg = hav(clat, clon, la, lo) if big is not None else 0.0
    E = max(Rbig*1.12, dg*1.35, 2000)*1000
    axm.set_xlim(-E, E); axm.set_ylim(-E, E); axm.set_facecolor('white')
    scale = '110m' if (E > 6.5e6 or not has_scale('50m')) else '50m'
    axm.add_feature(cfeature.LAND.with_scale(scale), facecolor=LAND, edgecolor='none', zorder=1)
    axm.add_feature(cfeature.COASTLINE.with_scale(scale), lw=0.3, edgecolor=COAST, zorder=2)
    axm.scatter(GLON[~aff], GLAT[~aff], s=0.55, color='#b8b8b8', lw=0, transform=ccrs.PlateCarree(), zorder=3)
    if aff.any(): axm.scatter(GLON[aff], GLAT[aff], s=1.5, color=V, lw=0, transform=ccrs.PlateCarree(), zorder=4)
    if big is not None:
        c = np.asarray(gd.circle(lon=float(big.longitude), lat=float(big.latitude), radius=Rbig*1000, n_samples=240))
        axm.plot(c[:, 0], c[:, 1], color=V if bool(qq.loc[qq.mag.idxmax(), 'trig']) else GREY, lw=0.8, ls=(0, (1.2, 1.6)), transform=ccrs.Geodetic(), zorder=5)
    if big is not None:
        axm.plot(float(big.longitude), float(big.latitude), marker='+', ms=5, mew=0.45, color=V if bool(qq.loc[qq.mag.idxmax(), 'trig']) else GREY, lw=0, transform=ccrs.PlateCarree(), zorder=6)
    axm.plot(lo, la, 'o', ms=3.8, color=B, lw=0, transform=ccrs.PlateCarree(), zorder=10)
    for sp_ in axm.spines.values(): sp_.set_linewidth(0.6); sp_.set_edgecolor('#888888')
    panel_label(axm, 'abcd'[i], dx=-0.1, dy=1.06)
    # ----- series
    ax = fig.add_subplot(gs[i, 1]); sc = sf*100
    nom = pd.Series(False, index=sf.index)
    for a, b in spans: nom.loc[a:b] = True
    nv = nom.values.astype(int)
    for a_, b_ in zip(np.where(np.diff(np.r_[0, nv]) == 1)[0], np.where(np.diff(np.r_[nv, 0]) == -1)[0]):
        ax.axvspan(num(sf.index[a_]), num(sf.index[b_]), color=V, alpha=0.15, lw=0, zorder=1)
    miss = sc.isna().values.astype(int)
    for a_, b_ in zip(np.where(np.diff(np.r_[0, miss]) == 1)[0], np.where(np.diff(np.r_[miss, 0]) == -1)[0]):
        ax.axvspan(num(sc.index[a_]), num(sc.index[b_]), ymin=0, ymax=0.035, color=GREY, alpha=0.8, lw=0, zorder=2)
    ax.plot(mdates.date2num(sc.index.tz_convert(None)), sc.values, color='k', lw=1.0, zorder=4)
    lo_, hi_ = np.nanmin(sc.values), np.nanmax(sc.values); rng = hi_ - lo_
    ax.set_ylim(lo_ - 0.12*rng, hi_ + 0.15*rng); ax.set_xlim(num(T0), num(T1))
    for _, e in qq.iterrows():
        ax.axvline(num(e.time), color=V if e.trig else GREY, lw=0.8, ls='--', zorder=3)
    ax.grid(alpha=0.25, lw=0.4)
    ax.set_ylabel('surge (cm)'); ax.xaxis.set_major_locator(mdates.DayLocator(interval=3)); ax.xaxis.set_minor_locator(mdates.DayLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter('%-d %b'))
    ax.tick_params(axis='x', which='minor', length=1.5)
    ax.set_title(f"{gname(ev['gauge'])}, {country(ev['gauge'])}   {ev['name']}", fontsize=6.5)
handles = [Line2D([], [], color='k', lw=1.0), Patch(fc=V, alpha=0.25, ec='none'), Patch(fc=GREY, alpha=0.8, ec='none'),
           Line2D([], [], color=V, lw=0.8, ls='--'), Line2D([], [], color=GREY, lw=0.8, ls='--')]
labels = ['observed surge', 'hours removed by the mask', 'no data', 'M7+ earthquake inside its rule radius', 'M7+ earthquake outside its rule radius']
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=3, handlelength=2.0, columnspacing=2.0)
save_pub(fig, f'{ROOT}/outputs/figS3_tsunami_mask'); print('figure written')
