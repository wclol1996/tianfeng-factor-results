"""Pledge statistics adapter; statistical end date is never availability."""
import hashlib,json
import numpy as np
import pandas as pd
try:
 from .tushare_financial import available_time,_timestamp
except ImportError:
 from financial_adapter import available_time,_timestamp

def normalize_pledge(raw,calendar):
 """Return engine-compatible long data and quarantine, keeping source metadata.

 pledge_stat has end_date but no native announcement field. Optional ann_date
 must be accompanied by its announcement_source. Otherwise first observed_at
 is the earliest supported availability, never retroactive end_date.
 """
 if not {'ts_code','end_date','pledge_ratio'}<=set(raw):raise ValueError('pledge keys/ratio missing')
 rows=[];bad=[]
 for i,r in raw.iterrows():
  try:
   if pd.isna(r.ts_code) or pd.isna(r.end_date):raise ValueError('missing pledge keys')
   times=[]
   if pd.notna(r.get('ann_date')):
    if not r.get('announcement_source'):raise ValueError('announcement date lacks source evidence')
    times.append(available_time(r.ann_date,calendar))
   if pd.notna(r.get('source_observed_at')):times.append(available_time(r.source_observed_at,calendar))
   if not times:raise ValueError('availability missing; statistical end_date cannot substitute')
   available=max(times);period=_timestamp(r.end_date).normalize()
   if available<period:raise ValueError('availability precedes statistics end')
   value=float(r.pledge_ratio)
   if not np.isfinite(value) or not 0<=value<=100:raise ValueError('pledge percentage outside 0..100')
   metadata=json.dumps({k:str(v) for k,v in r.items()},ensure_ascii=False,sort_keys=True)
   rows.append({'security_id':str(r.ts_code),'report_period':period,'available_at':available,'revision_at':available,'field':'股东质押比例','value':value*.01,'source_api':'pledge_stat','source_field':'pledge_ratio','source_unit':'percent','multiplier':.01,'source_record_id':hashlib.sha256(metadata.encode()).hexdigest(),'source_metadata_json':metadata,'period_semantics':'point','accounting_method':'A'})
  except (ValueError,TypeError) as e:bad.append({'row_index':i,'reason':str(e)})
 out=pd.DataFrame(rows)
 if not out.empty and out.duplicated(['security_id','report_period','available_at','field']).any():raise ValueError('duplicate pledge records at same availability')
 return out,pd.DataFrame(bad)
