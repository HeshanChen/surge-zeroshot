"""Batch de-tide GESLA selected stations with the validated clean_surge pipeline.
  python scripts/batch_detide.py [N]        # optional: limit to first N stations (pilot)
Outputs: data/processed/<name>.parquet (clean storm surge) + catalog/processing_qc.csv (per-station QC).
Multiprocessing (utide is CPU-bound). Each station passes the same gates validated on Battery +
non-US regimes (segmented de-tide, gross/spike removal, length/gap rejection)."""
import os as _os
_ROOT = _os.environ.get('SURGE_ROOT') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
import sys, os, warnings; warnings.filterwarnings('ignore')
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v]='1'                      # 1 BLAS thread per worker -> no oversubscription across processes
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

ROOT=_ROOT
OUT=f'{ROOT}/data/processed'
sys.path.insert(0, f'{ROOT}/src')

def process(name):
    import warnings; warnings.filterwarnings('ignore')
    import os, pandas as pd
    sys.path.insert(0, f'{ROOT}/src')
    from data.clean_surge import compute_storm_surge
    local=f'{ROOT}/data/raw/gesla/{name}.parquet'
    src=local if os.path.exists(local) else 's3://gesla-dataset/parquet_files/'+name+'.parquet'
    try:
        df=pd.read_parquet(src, columns=['date_time','sea_level','use_flag','lat'],
                           storage_options=({'anon':True} if str(src).startswith('s3') else None))
        df=df.set_index(pd.to_datetime(df['date_time'])); lat=float(df['lat'].iloc[0])
        surge, qc=compute_storm_surge(df['sea_level'].where(df['use_flag']==1), lat=lat)
        qc['name']=name; qc['lat']=lat
        if surge is not None:
            pd.DataFrame({'surge':surge}).dropna().to_parquet(f'{OUT}/{name}.parquet')
        return qc
    except Exception as e:
        return {'name':name, 'status':'error', 'reason':str(e)[:100]}

if __name__=='__main__':
    os.makedirs(OUT, exist_ok=True)
    sel=pd.read_csv(f'{ROOT}/catalog/selected_stations.csv')
    names=sel['name'].tolist()
    if len(sys.argv)>1: names=names[:int(sys.argv[1])]
    print(f'processing {len(names)} stations on {os.cpu_count()} cores...', flush=True)
    with ProcessPoolExecutor(max_workers=12) as ex:
        res=list(ex.map(process, names, chunksize=4))
    qc=pd.DataFrame(res); qc.to_csv(f'{ROOT}/catalog/processing_qc.csv', index=False)
    ok=int((qc.get('status')=='ok').sum()); rej=int((qc.get('status')=='rejected').sum()); err=int((qc.get('status')=='error').sum())
    print(f'done: {ok} ok / {rej} rejected / {err} error  (of {len(names)})')
    if 'M2_leak_cm' in qc.columns:
        g=qc[qc.get('status')=='ok']
        print(f'  median M2_leak={g.M2_leak_cm.median():.2f}cm  median lowfreq={g.lowfreq_frac_pct.median():.1f}%  median std={g.std_cm.median():.1f}cm')
    print('-> data/processed/*.parquet + catalog/processing_qc.csv')
