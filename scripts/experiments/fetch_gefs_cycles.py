"""Fetch GEFSv12 reforecast control-member records (10 m u/v wind, MSL pressure, 3-h precipitation) for given
storm/issue-time pairs by HTTP byte range from the public AWS bucket, one small grib2 file per forecast hour,
in the layout scripts/eval_gefs.py reads: data/raw/gefs/{short}_{YYYYMMDD00}_{var}_f{hhh}.grib2.
The reforecast is issued at 00 UTC only; forecast hours 3..48 (3-hourly), Days:1-10 files.
Usage: python3 scripts/fetch_gefs_cycles.py short:YYYYMMDD00 [short:YYYYMMDD00 ...]
       python3 scripts/fetch_gefs_cycles.py --from-events short [short ...]   (issue time = (peak - 36 h).floor(day), as in eval_gefs.py)
Audit 2026-09-05: written so that the storm list and issue times are reproducible from code (the original fetch had no script)."""
import sys, os, re, urllib.request, pandas as pd
ROOT = '/Users/heshan/Desktop/surge_fm'; OUT = f'{ROOT}/data/raw/gefs'
BASE = 'https://noaa-gefs-retrospective.s3.amazonaws.com/GEFSv12/reforecast'
VARS = {'ugrd_hgt': ('UGRD', '10 m above ground'), 'vgrd_hgt': ('VGRD', '10 m above ground'),
        'pres_msl': ('PRES', 'mean sea level'), 'apcp_sfc': ('APCP', 'surface')}
HOURS = list(range(3, 49, 3))

def fetch(url, rng=None, tries=4):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={'Range': f'bytes={rng[0]}-{rng[1]}'} if rng else {})
            with urllib.request.urlopen(req, timeout=180) as r: data = r.read()
            if rng and rng[1] != '' and len(data) != rng[1]-rng[0]+1: raise IOError(f'short read {len(data)} of {rng[1]-rng[0]+1}')
            return data
        except Exception as e:
            if k == tries-1: raise
            import time; time.sleep(3*(k+1))

def fhour(desc):
    m = re.search(r'(\d+)-(\d+) hour acc fcst', desc)
    if m: return int(m.group(2))
    m = re.search(r'(\d+) hour fcst', desc)
    return int(m.group(1)) if m else None

def get(short, init):
    year = init[:4]; n_new = 0
    for var, (name, level) in VARS.items():
        url = f'{BASE}/{year}/{init}/c00/Days%3A1-10/{var}_{init}_c00.grib2'
        try: idx = fetch(url+'.idx').decode().strip().split('\n')
        except Exception as e: print(f'  {short} {init} {var}: idx FAIL {str(e)[:80]}', flush=True); continue
        recs = []
        for i, line in enumerate(idx):
            p = line.split(':'); off = int(p[1]); nxt = int(idx[i+1].split(':')[1])-1 if i+1 < len(idx) else ''
            if p[3] == name and p[4] == level:
                fh = fhour(p[5])
                if fh in HOURS: recs.append((fh, off, nxt))
        for fh, a, b in recs:
            dst = f'{OUT}/{short}_{init}_{var}_f{fh:03d}.grib2'
            if os.path.exists(dst) and os.path.getsize(dst) == (b-a+1 if b != '' else os.path.getsize(dst)): continue
            try:
                open(dst, 'wb').write(fetch(url, (a, b if b != '' else a+50_000_000))); n_new += 1
            except Exception as e: print(f'  {short} {init} {var} f{fh:03d}: FAIL {str(e)[:80]}', flush=True)
        print(f'  {short} {init} {var}: {len(recs)} records ({len([1 for fh,_,_ in recs])} hours) ready', flush=True)
    return n_new

args = sys.argv[1:]
pairs = []; from_events = False
ev = None
for a in args:
    if a == '--from-events': from_events = True; continue
    if from_events:
        if ev is None: ev = pd.read_csv(f'{ROOT}/outputs/gfs_events.csv', parse_dates=['peak_time'])
        r = ev[ev.stn.str.startswith(a)].iloc[0]
        pairs.append((a, (r.peak_time - pd.Timedelta(hours=36)).floor('D').strftime('%Y%m%d00')))
    else:
        pairs.append(tuple(a.split(':')))
for short, init in pairs:
    print(f'{short} {init}:', flush=True); n = get(short, init); print(f'  {n} new files', flush=True)
print('GEFS FETCH DONE', flush=True)
