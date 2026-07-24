"""Fetch ERA5 forcing @ each station via Open-Meteo archive (perfect-prog).
Window START_YEAR.., 3 threads, resume(skip existing), large 429 backoff.
  python fetch_era5.py            # all final_stations
  python fetch_era5.py sub 400    # country-stratified subset of 400
  python fetch_era5.py 50         # first 50"""
import sys, os, time, json, warnings; warnings.filterwarnings('ignore')
import pandas as pd, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
ROOT='/Users/heshan/Desktop/surge_fm'; OUT=f'{ROOT}/data/raw/era5'; os.makedirs(OUT, exist_ok=True)
VARS='wind_speed_10m,wind_direction_10m,pressure_msl,precipitation'; START_YEAR=2000
def fetch(name):
    out=f'{OUT}/{name}.parquet'
    if os.path.exists(out) and os.path.getsize(out)>0: return ('skip',name)
    try:
        raw=pd.read_parquet(f'{ROOT}/data/raw/gesla/{name}.parquet', columns=['lat','lon','year'])
        lat,lon=float(raw.lat.iloc[0]),float(raw.lon.iloc[0]); y0=max(START_YEAR,int(raw.year.min())); y1=int(raw.year.max())
        if y1<y0: return ('tooshort',name)
        url=(f'https://archive-api.open-meteo.com/v1/archive?latitude={lat:.4f}&longitude={lon:.4f}'
             f'&start_date={y0}-01-01&end_date={y1}-12-31&hourly={VARS}&timezone=GMT&wind_speed_unit=ms')
        for k in range(8):
            try:
                with urllib.request.urlopen(url, timeout=180) as r: d=json.load(r); break
            except urllib.error.HTTPError as e:
                if e.code in (429,503): time.sleep(30*(k+1)); continue
                return (f'http{e.code}',name)
        else: return ('ratelimited',name)
        h=pd.DataFrame(d['hourly']); h['time']=pd.to_datetime(h['time']); h.set_index('time').to_parquet(out)
        return ('ok',name)
    except Exception as e: return (f'err:{str(e)[:35]}',name)
def subset(names,n):
    fs=pd.DataFrame({'name':names}); fs['c']=fs['name'].str.split('-').str[-2]; frac=n/len(fs)
    s=fs.groupby('c',group_keys=False).apply(lambda g: g.sample(max(1,round(len(g)*frac)),random_state=0))
    return s['name'].head(n).tolist()
if __name__=='__main__':
    names=pd.read_csv(f'{ROOT}/catalog/final_stations.csv')['name'].tolist()
    a=sys.argv[1:]
    if a and a[0]=='clean': names=pd.read_csv(f'{ROOT}/catalog/clean_stations.csv')['name'].tolist(); print(f'clean station set: {len(names)} independent stations',flush=True)
    elif a and a[0].startswith('sub'): names=subset(names,int(a[1])); pd.Series(names).to_csv(f'{ROOT}/catalog/subset_stations.csv',index=False,header=['name']); print(f'subset {len(names)} (country-stratified) -> catalog/subset_stations.csv',flush=True)
    elif a: names=names[:int(a[0])]
    print(f'fetching ERA5 {START_YEAR}+ for {len(names)} stations (3 threads)...',flush=True)
    t0=time.time(); res=[]
    with ThreadPoolExecutor(max_workers=3) as ex:
        for i,r in enumerate(ex.map(fetch,names)):
            res.append(r)
            if (i+1)%50==0: print(f'  {i+1}/{len(names)} {time.time()-t0:.0f}s {dict(Counter(s for s,_ in res))}',flush=True)
    print('FINAL:',dict(Counter(s for s,_ in res)),f'in {time.time()-t0:.0f}s',flush=True)
