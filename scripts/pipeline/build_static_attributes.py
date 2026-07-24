"""Static-attribute table (Caravan-for-coasts) for FiLM conditioning — the 'already-have' half.
Tidal constituents -> form factor + range (utide on raw sea_level, recent 10yr); lat/lon; surge climatology.
GEBCO-derived (slope/shelf width/depth) added later."""
import sys, os, warnings; warnings.filterwarnings('ignore')
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): os.environ[v]='1'
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
ROOT='/Users/heshan/Desktop/surge_fm'; sys.path.insert(0,f'{ROOT}/src')
def proc(name):
    import pandas as pd; sys.path.insert(0,f'{ROOT}/src')
    from utide import solve; from data.clean_surge import to_hourly, _td64, _amp
    try:
        raw=pd.read_parquet(f'{ROOT}/data/raw/gesla/{name}.parquet', columns=['date_time','sea_level','use_flag','lat','lon'])
        raw=raw.set_index(pd.to_datetime(raw['date_time'])); lat,lon=float(raw.lat.iloc[0]),float(raw.lon.iloc[0])
        wl,_=to_hourly(raw['sea_level'].where(raw['use_flag']==1)); wl=wl.dropna()
        wl=wl[wl.index>=wl.index.max()-pd.Timedelta(days=3650)]               # recent 10yr for speed
        c=solve(_td64(wl.index), wl.values, lat=lat, method='ols', conf_int='none', trend=False, verbose=False)
        A={k:(_amp(c,k) or 0.0) for k in ['M2','S2','N2','K1','O1']}
        F=(A['K1']+A['O1'])/max(A['M2']+A['S2'],1e-6)
        surge=pd.read_parquet(f'{ROOT}/data/processed/{name}.parquet')['surge'].dropna()
        return dict(name=name, lat=round(lat,4), lon=round(lon,4),
                    M2_cm=round(A['M2']*100,1), S2_cm=round(A['S2']*100,1), K1_cm=round(A['K1']*100,1), O1_cm=round(A['O1']*100,1),
                    form_factor=round(F,3), tidal_range_m=round(2*(A['M2']+A['S2']+A['K1']+A['O1']),3),
                    surge_std_cm=round(surge.std()*100,1), surge_p99_cm=round(surge.quantile(0.99)*100,1), surge_max_m=round(surge.max(),2))
    except Exception as e: return dict(name=name, error=str(e)[:60])
if __name__=='__main__':
    names=pd.read_csv(f'{ROOT}/catalog/final_stations.csv')['name'].tolist()
    if len(sys.argv)>1: names=names[:int(sys.argv[1])]
    print(f'building static attributes for {len(names)} stations...', flush=True)
    with ProcessPoolExecutor(max_workers=10) as ex: res=list(ex.map(proc, names, chunksize=8))
    df=pd.DataFrame(res); df.to_csv(f'{ROOT}/catalog/static_attributes.csv', index=False)
    ok=df[df['error'].isna()] if 'error' in df.columns else df
    print(f'done: {len(ok)}/{len(names)} ok -> catalog/static_attributes.csv')
    if 'form_factor' in ok.columns:
        print(f"tide type: semidiurnal(F<0.25)={int((ok.form_factor<0.25).sum())} | mixed(0.25-1.5)={int(((ok.form_factor>=0.25)&(ok.form_factor<1.5)).sum())} | diurnal(F>=1.5)={int((ok.form_factor>=1.5).sum())}")
