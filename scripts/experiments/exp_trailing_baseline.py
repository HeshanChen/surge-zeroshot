"""Causal-preprocessing robustness: recompute the 84 marine test gauges' surge with a TRAILING
30-day baseline and trailing despike windows (causal=True), then re-evaluate the delivered
checkpoint zero-shot on the causally-defined surge. Compares against the centered definition.
-> data/processed_trailing/*.parquet + outputs/trailing_qc.csv"""
import os as _os
_ROOT = _os.environ.get('SURGE_ROOT') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
import sys, os, warnings; warnings.filterwarnings('ignore')
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v]='1'
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
ROOT=_ROOT
OUT=f'{ROOT}/data/processed_trailing'
sys.path.insert(0, f'{ROOT}/src')

def process(name):
    import warnings; warnings.filterwarnings('ignore')
    import os, sys, pandas as pd
    sys.path.insert(0, f'{ROOT}/src')
    from data.clean_surge import compute_storm_surge
    try:
        df=pd.read_parquet(f'{ROOT}/data/raw/gesla/{name}.parquet', columns=['date_time','sea_level','use_flag','lat'])
        df=df.set_index(pd.to_datetime(df['date_time'])); lat=float(df['lat'].iloc[0])
        surge, qc=compute_storm_surge(df['sea_level'].where(df['use_flag']==1), lat=lat, causal=True)
        qc['name']=name
        if surge is not None:
            pd.DataFrame({'surge':surge}).dropna().to_parquet(f'{OUT}/{name}.parquet')
        return qc
    except Exception as e:
        return {'name':name,'status':'error','reason':str(e)[:100]}

if __name__=='__main__':
    os.makedirs(OUT, exist_ok=True)
    sa=pd.read_csv(f'{ROOT}/catalog/static_attributes.csv').set_index('name')
    sp=pd.read_csv(f'{ROOT}/catalog/exp_split_final.csv')
    te=[n for n in sp[sp.fold=='test'].name if n in sa.index]
    todo=[n for n in te if not os.path.exists(f'{OUT}/{n}.parquet')]
    print(f'{len(todo)}/{len(te)} to process', flush=True)
    with ProcessPoolExecutor(max_workers=10) as ex:
        qcs=list(ex.map(process, todo))
    pd.DataFrame(qcs).to_csv(f'{ROOT}/outputs/trailing_qc.csv', index=False)
    ok=sum(1 for q in qcs if q.get('status')=='ok')
    print(f'TRAILING DETIDE DONE: {ok}/{len(todo)} ok', flush=True)
