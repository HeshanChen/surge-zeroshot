"""Download ALL GESLA station parquet files from the public S3 bucket to local disk."""
import warnings; warnings.filterwarnings('ignore')
import s3fs, os, time
from concurrent.futures import ThreadPoolExecutor
fs=s3fs.S3FileSystem(anon=True)
files=fs.ls('gesla-dataset/parquet_files')
LOCAL='/Users/heshan/Desktop/surge_fm/data/raw/gesla'; os.makedirs(LOCAL, exist_ok=True)
def dl(f):
    out=os.path.join(LOCAL, f.split('/')[-1])
    if os.path.exists(out) and os.path.getsize(out)>0: return os.path.getsize(out)
    try: fs.get(f, out); return os.path.getsize(out)
    except Exception: return -1
t0=time.time(); print(f'downloading {len(files)} files to {LOCAL} ...', flush=True)
with ThreadPoolExecutor(max_workers=24) as ex:
    sizes=list(ex.map(dl, files))
ok=[s for s in sizes if s>0]; err=sum(1 for s in sizes if s==-1)
print(f'DONE: {len(ok)} files, {round(sum(ok)/1e9,2)} GB, {err} errors, {round(time.time()-t0)}s', flush=True)
