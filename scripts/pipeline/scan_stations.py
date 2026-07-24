"""Scan all GESLA stations via parquet FOOTER statistics (no data-body reads).
Outputs catalog/station_index.csv (all) + selected_stations.csv (coastal, >=15yr, dense)."""
import warnings; warnings.filterwarnings('ignore')
import s3fs, pandas as pd, pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor
fs=s3fs.S3FileSystem(anon=True); files=fs.ls('gesla-dataset/parquet_files')
def info(f):
    name=f.split('/')[-1][:-8]; p=name.split('-'); src=p[-1]; ctry=p[-2] if len(p)>=2 else '?'
    try:
        md=pq.ParquetFile(f, filesystem=fs).metadata
        names=[md.schema.column(i).name for i in range(md.num_columns)]; yi=names.index('year')
        gs=[md.row_group(g).column(yi).statistics for g in range(md.num_row_groups)]
        ymin=min(s.min for s in gs if s); ymax=max(s.max for s in gs if s)
        return (name,src,ctry,int(ymin),int(ymax),int(ymax-ymin+1),md.num_rows)
    except Exception:
        return (name,src,ctry,None,None,None,None)
with ThreadPoolExecutor(max_workers=32) as ex:
    rows=list(ex.map(info, files))
df=pd.DataFrame(rows,columns=['name','source','country','yr_start','yr_end','yr_span','n_rows'])
df.to_csv('/Users/heshan/Desktop/surge_fm/catalog/station_index.csv',index=False)
RIVER={'usgs','sfwmd','nwfwmd'}
df['coastal']=~df.source.isin(RIVER)
df['ok']=df.coastal & (df.yr_span.fillna(0)>=15) & (df.n_rows.fillna(0)>=15*6000)
print('scanned',len(df),'| failed',int(df.yr_span.isna().sum()))
print('coastal(non-river)',int(df.coastal.sum()),'| SELECTED(coastal & >=15yr & dense)',int(df.ok.sum()))
print('\nyr_span of selected:'); print(df[df.ok].yr_span.describe().round(1).to_string())
print('\nselected top countries:'); print(df[df.ok].country.value_counts().head(15).to_string())
df[df.ok].to_csv('/Users/heshan/Desktop/surge_fm/catalog/selected_stations.csv',index=False)
print('\nDONE -> catalog/station_index.csv + selected_stations.csv')
