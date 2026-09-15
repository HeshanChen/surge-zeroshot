"""Season-scale GEFSv12 reforecast forcing at the 84 marine test gauges: for every 00 UTC control-member cycle in the
requested years, fetch the 10 m u/v wind, MSL pressure and 3-h precipitation records for forecast hours 3..48 by HTTP
byte range from the public AWS bucket, decode them in memory with ecCodes, sample the nearest 0.25-degree grid point
for each gauge, and save one small npz per cycle: data/raw/gefs_season/{YYYYMMDD00}.npz with arrays
u, v, msl, apcp of shape (16 hours, n_gauges), plus hours, gauges, and the APCP step ranges. Nothing else is kept on
disk (a full year of global records would be ~17 GB). Resumable: cycles with an existing npz are skipped.
Usage: python3 scripts/fetch_gefs_season.py 2017 2012 [--workers 16]     (log: outputs/gefs_season_fetch.log)
Written 2026-09-05 for the season-scale real-forecast evaluation (scripts/eval_gefs_season.py)."""
import os as _os
_ROOT = _os.environ.get('SURGE_ROOT') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
import sys, os, re, time, io, urllib.request, concurrent.futures as cf
import numpy as np, pandas as pd
import eccodes as ec
ROOT = _ROOT; OUT = f'{ROOT}/data/raw/gefs_season'; os.makedirs(OUT, exist_ok=True)
BASE = 'https://noaa-gefs-retrospective.s3.amazonaws.com/GEFSv12/reforecast'
VARS = {'ugrd_hgt': ('UGRD', '10 m above ground'), 'vgrd_hgt': ('VGRD', '10 m above ground'),
        'pres_msl': ('PRES', 'mean sea level'), 'apcp_sfc': ('APCP', 'surface')}
HOURS = list(range(3, 49, 3))
LOG = open(f'{ROOT}/outputs/gefs_season_fetch.log', 'a')
def log(msg): print(msg, flush=True); LOG.write(msg + '\n'); LOG.flush()

WORKERS = int(sys.argv[sys.argv.index('--workers') + 1]) if '--workers' in sys.argv else 16
years = [int(a) for a in sys.argv[1:] if a.isdigit() and len(a) == 4]   # four-digit years only (a flag value such as 16 must not become a year)

sa = pd.read_csv(f'{ROOT}/catalog/static_attributes.csv').set_index('name')
gauges = list(pd.read_csv(f'{ROOT}/outputs/eval_full_lstmq_v2final.csv').stn)
lat = np.array([sa.loc[n, 'lat'] for n in gauges]); lon = np.array([sa.loc[n, 'lon'] for n in gauges]) % 360
# nearest grid point on the 0.25-degree global grid (first point 90N, 0E; 721 x 1440)
ii = np.clip(np.round((90.0 - lat) / 0.25).astype(int), 0, 720); jj = np.round(lon / 0.25).astype(int) % 1440
IDX = ii * 1440 + jj

def fetch(url, rng=None, tries=5):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={'Range': f'bytes={rng[0]}-{rng[1]}'} if rng else {})
            with urllib.request.urlopen(req, timeout=120) as r: data = r.read()
            if rng and len(data) != rng[1] - rng[0] + 1: raise IOError(f'short read {len(data)}')
            return data
        except Exception as e:
            if k == tries - 1: raise
            time.sleep(2 * (k + 1))

def fhour(desc):
    m = re.search(r'(\d+)-(\d+) hour acc fcst', desc)
    if m: return int(m.group(2))
    m = re.search(r'(\d+) hour fcst', desc); return int(m.group(1)) if m else None

def records(init, var, name, level):
    """(fh, byte range, step description) for the wanted forecast hours of one variable file"""
    url = f'{BASE}/{init[:4]}/{init}/c00/Days%3A1-10/{var}_{init}_c00.grib2'
    idx = fetch(url + '.idx').decode().strip().split('\n'); recs = []
    for i, line in enumerate(idx):
        p = line.split(':'); off = int(p[1])
        if p[3] == name and p[4] == level and fhour(p[5]) in HOURS:
            nxt = int(idx[i + 1].split(':')[1]) - 1 if i + 1 < len(idx) else off + 60_000_000
            recs.append((fhour(p[5]), (off, nxt), p[5]))
    return url, recs

def decode_sample(data):
    gid = ec.codes_new_from_message(data)
    try:
        assert ec.codes_get(gid, 'Ni') == 1440 and ec.codes_get(gid, 'Nj') == 721
        assert abs(ec.codes_get(gid, 'latitudeOfFirstGridPointInDegrees') - 90.0) < 1e-6
        vals = ec.codes_get_values(gid); return vals[IDX].astype('float32')
    finally: ec.codes_release(gid)

def one_cycle(init):
    dst = f'{OUT}/{init}.npz'
    if os.path.exists(dst): return 'skip'
    plan = {}
    for var, (name, level) in VARS.items(): plan[var] = records(init, var, name, level)
    tasks = [(var, fh, url, rng, desc) for var, (url, recs) in plan.items() for fh, rng, desc in recs]
    if any(len(plan[v][1]) != len(HOURS) for v in VARS): return f'incomplete index ({[len(plan[v][1]) for v in VARS]})'
    out = {v: np.full((len(HOURS), len(gauges)), np.nan, 'float32') for v in VARS}; steps = {}
    with cf.ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(fetch, url, rng): (var, fh, desc) for var, fh, url, rng, desc in tasks}
        for f in cf.as_completed(futs):
            var, fh, desc = futs[f]; data = f.result()
            out[var][HOURS.index(fh)] = decode_sample(data)
            if var == 'apcp_sfc': steps[fh] = desc
    np.savez_compressed(dst, u=out['ugrd_hgt'], v=out['vgrd_hgt'], msl=out['pres_msl'], apcp=out['apcp_sfc'],
                        hours=np.array(HOURS), gauges=np.array(gauges), apcp_steps=np.array([steps[h] for h in HOURS]))
    return 'ok'

for y in years:
    inits = [d.strftime('%Y%m%d00') for d in pd.date_range(f'{y}-01-01', f'{y}-12-31', freq='D')]
    t0 = time.time(); n_ok = n_skip = n_fail = 0
    for k, init in enumerate(inits):
        try: r = one_cycle(init)
        except Exception as e: r = f'FAIL {str(e)[:80]}'
        if r == 'ok': n_ok += 1
        elif r == 'skip': n_skip += 1
        else: n_fail += 1; log(f'  {init}: {r}')
        if (k + 1) % 10 == 0 or k + 1 == len(inits):
            log(f'{y}: {k+1}/{len(inits)} cycles ({n_ok} fetched, {n_skip} existing, {n_fail} failed) {time.time()-t0:.0f} s')
    log(f'YEAR {y} DONE')
log('GEFS SEASON FETCH DONE')
