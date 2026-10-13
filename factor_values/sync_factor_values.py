"""Version and package newly computed factor values; no IC or raw-data upload.

The upstream calculator supplies its immutable approved result manifest.
TUSHARE_TOKEN is consumed by the upstream calculator, never by this publisher.
"""
from pathlib import Path
import argparse,hashlib,json,shutil,zipfile,datetime

PROTECTED={'TianfengFactors','agent3','confidential','confidential_reports'}

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(8*1024*1024):h.update(b)
 return h.hexdigest()

def sync(manifest_path,scope_path,out):
 manifest=json.loads(manifest_path.read_text());scope=json.loads(scope_path.read_text())
 ids={x['factor_id'] for x in scope['factors']}
 if len(ids)!=247:raise ValueError('247-factor scope required')
 parts=manifest.get('partitions',[])
 if not parts:raise ValueError('No computed partitions supplied')
 if manifest.get('financial_version_policy')!='latest_available_pit_v1':raise ValueError('PIT policy missing or incompatible')
 if manifest.get('data_quality_status')!='APPROVED':raise ValueError('Approved upstream data-quality gate required')
 import pyarrow.parquet as pq
 checked=[];seen=set()
 for r in parts:
  fid=r['factor_id'];year=int(r['year']);key=(fid,year)
  if fid not in ids or key in seen:raise ValueError('Unknown or duplicate factor/year')
  seen.add(key);p=Path(r['path']).resolve()
  if PROTECTED.intersection(p.parts) or p.is_symlink():raise ValueError('Forbidden input')
  if sha(p)!=r['sha256']:raise ValueError('Immutable result hash mismatch')
  if not {'date','security_id','value'}<=set(pq.read_schema(p).names):raise ValueError('Unexpected factor schema')
  if not r.get('formula_version') or not r.get('source_version') or not r.get('data_cutoff'):raise ValueError('Version lineage required per factor')
  checked.append((r,p))
 stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');dest=out/stamp;dest.mkdir(parents=True,exist_ok=False)
 safe=[]
 for r,p in checked:
  rel=Path('values')/r['factor_id']/f"year={r['year']}"/'values.parquet';target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
  if sha(target)!=r['sha256']:raise ValueError('Copy hash mismatch')
  safe.append({k:r[k] for k in ['factor_id','year','sha256','formula_version','source_version','data_cutoff']}|{'path':rel.as_posix()})
 result={'factor_scope_count':247,'updated_factor_count':len({r['factor_id'] for r in safe}),'financial_version_policy':'latest_available_pit_v1','partitions':safe}
 (dest/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));archive=out/f'factor-values-increment-{stamp}.zip'
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_STORED,allowZip64=True) as z:
  for p in sorted(dest.rglob('*')):
   if p.is_file():z.write(p,p.relative_to(dest))
 print(json.dumps({'updated_factor_count':result['updated_factor_count'],'archive':str(archive),'sha256':sha(archive),'notes':'Input remains frozen; no IC computed; prior versions preserved.'}))
 return result

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--scope',type=Path,default=Path(__file__).with_name('factor_scope.json'));p.add_argument('--output',type=Path,required=True);a=p.parse_args();sync(a.manifest,a.scope,a.output)
