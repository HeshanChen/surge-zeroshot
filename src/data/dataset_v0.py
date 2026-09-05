"""v0 multi-stream dataset: align surge + ERA5 -> C=5 [surge,wind_u,wind_v,mslp,precip], clip |surge|>4,
window (ctx+tgt), RevIN per-window (ctx stats), target normalized by per-station train std.
SEISMIC FILTER (2026-07-02): tsunami/seiche hours are masked from the joined record (non-meteorological
events contaminate percentile-selected extremes; 1952 Aburatsubo case). Rule (USGS M>=7 catalog):
  M>=7.0 within 1000km -> mask [t-1h, t+72h] | M>=7.5 within 3000km -> +96h | M>=8.0 within 8000km -> +168h
Toggle via MASK_SEISMIC=False for ablation."""
import torch, pandas as pd, numpy as np, glob, os
ROOT='/Users/heshan/Desktop/surge_fm'
MASK_SEISMIC=True
_EQ=None; _SA=None
def _quakes():
    global _EQ
    if _EQ is None:
        q=pd.read_csv(f'{ROOT}/catalog/earthquakes_m7.csv', usecols=['time','latitude','longitude','mag'])
        q['time']=pd.to_datetime(q.time, utc=True, format='ISO8601')
        _EQ=q.dropna(subset=['latitude','longitude','mag'])
    return _EQ
def seismic_mask(lat, lon, index):
    """boolean Series over index: True = hour contaminated by a qualifying earthquake/tsunami window."""
    q=_quakes()
    la1,lo1,la2,lo2=map(np.radians,(lat,lon,q.latitude.values,q.longitude.values))
    d=6371*2*np.arcsin(np.sqrt(np.sin((la2-la1)/2)**2+np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2))
    hrs=np.where(q.mag.values>=8.0,168,np.where(q.mag.values>=7.5,96,72))
    sel=((q.mag.values>=8.0)&(d<=8000))|((q.mag.values>=7.5)&(d<=3000))|((q.mag.values>=7.0)&(d<=1000))
    m=pd.Series(False,index=index)
    for t,h in zip(q.time[sel],hrs[sel]):
        m.loc[t-pd.Timedelta(hours=1):t+pd.Timedelta(hours=int(h))]=True
    return m
def load_station(name, return_time=False):
    """return_time=True also returns the hourly timestamps (int64 hours since epoch) of the kept rows, so callers
    can detect windows that span a dropped gap (rows are concatenated positionally after dropna and masking)."""
    s=pd.read_parquet(f'{ROOT}/data/processed/{name}.parquet')['surge']; s=s.where(s.abs()<4)
    e=pd.read_parquet(f'{ROOT}/data/raw/era5/{name}.parquet')
    if e.index.tz is None: e.index=e.index.tz_localize('UTC')
    for c in ['wind_speed_10m','wind_direction_10m','pressure_msl','precipitation']:
        e[c]=pd.to_numeric(e[c],errors='coerce')                 # some downloads carry None (partial API nulls)
    sp=e['wind_speed_10m'].values; d=np.deg2rad(e['wind_direction_10m'].values)
    e2=pd.DataFrame({'wu':sp*np.sin(d),'wv':sp*np.cos(d),'mslp':e['pressure_msl'].values,'precip':e['precipitation'].values},index=e.index)
    df=pd.DataFrame({'surge':s}).join(e2,how='inner').dropna()
    if MASK_SEISMIC and len(df):
        global _SA
        if _SA is None: _SA=pd.read_csv(f'{ROOT}/catalog/static_attributes.csv').set_index('name')
        if name in _SA.index:
            df=df[~seismic_mask(float(_SA.loc[name,'lat']), float(_SA.loc[name,'lon']), df.index)]
    if len(df)<3000: return (None, None) if return_time else None
    a=df[['surge','wu','wv','mslp','precip']].values.astype('float32')
    if return_time: return a, df.index.values.astype('datetime64[h]').astype('int64')
    return a
def available_stations():
    names=pd.read_csv(f'{ROOT}/catalog/clean_stations.csv')['name']
    era5=set(os.path.basename(f)[:-8] for f in glob.glob(f'{ROOT}/data/raw/era5/*.parquet'))
    return [n for n in names if n in era5]
class SurgeV0(torch.utils.data.Dataset):
    def __init__(s,names,Tctx=208,Ttgt=48):
        s.Tctx,s.Ttgt,s.W=Tctx,Ttgt,Tctx+Ttgt; s.items=[]
        for n in names:
            a=load_station(n)
            if a is None: continue
            tstd=float(a[:,0].std())+1e-6
            for st in range(0,len(a)-s.W,s.W): s.items.append((a[st:st+s.W],tstd))
    def __len__(s): return len(s.items)
    def __getitem__(s,i):
        w,tstd=s.items[i]; mu=w[:s.Tctx].mean(0); sd=w[:s.Tctx].std(0)+1e-6; wn=(w-mu)/sd
        return torch.tensor(wn[:s.Tctx].T), torch.tensor(wn[s.Tctx:,1:].T), torch.tensor(w[s.Tctx:,0]/tstd)
