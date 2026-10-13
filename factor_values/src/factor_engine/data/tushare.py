"""Strict local Tushare adapter. No network, approvals, or supplier factor reuse."""
from pathlib import Path
from datetime import date
import json
import hashlib
import pandas as pd
import numpy as np
from factor_engine.data.base import DataProvider, FactorContext

PRICE_MODES={'raw':'','backward':'_hfq','forward':'_qfq'}
UNITS={'volume':('vol',100.,'shares'),'amount':('amount',1000.,'CNY'),'turnover_rate':('turnover_rate',.01,'ratio'),'total_shares':('total_share',10000.,'shares'),'float_shares':('float_share',10000.,'shares'),'free_float_shares':('free_share',10000.,'shares'),'total_market_cap':('total_mv',10000.,'CNY'),'float_market_cap':('circ_mv',10000.,'CNY')}

def provider_contract():
 return {'version':1,'prices':PRICE_MODES,'units':{k:{'source_field':v[0],'multiplier':v[1],'canonical_unit':v[2]} for k,v in UNITS.items()},'daily_schema':['date','security_id','open','high','low','close','price_adjustment','raw_open','raw_high','raw_low','raw_close',*UNITS],'calendar_schema':['date'],'security_master_schema':['security_id','list_date','delist_date'],'index_daily_schema':['date','index_id','close'],'joins':'date/security_id one-to-one; preserve quote left side; reject unmatched basic rows','missing_dates':'calendar cross securities with NaN; never ffill','excluded':['supplier technical factors','pb','derived vwap','price-inferred limit flags','PIT financials'],'human_approved':False,'optional_daily_fields':['pe_ttm','ps_ttm','is_limit_up','is_limit_down'],'context_tables':['financial_observations.parquet','consensus_observations.parquet','events/financial_report_disclosures.parquet','events/earnings_forecasts.parquet'],'limit_status_source':'daily_basic.limit_status 0..6; missing remains missing; no no-limit flag inferred'}

def _mode(adjustment):
 mode=getattr(adjustment,'value',adjustment)
 if mode not in PRICE_MODES:raise ValueError('unknown adjustment')
 return mode

def _keys(frame,date_name='date',id_name='security_id'):
 frame=frame.copy()
 if not {date_name,id_name}<=set(frame):raise ValueError('missing date/security key columns')
 # YYYYMMDD integer input must never be interpreted as Unix nanoseconds.
 frame[date_name]=pd.to_datetime(frame[date_name].astype(str),errors='raise').dt.normalize()
 if frame[[date_name,id_name]].isna().any().any():raise ValueError('null keys')
 frame[id_name]=frame[id_name].astype(str)
 if frame.duplicated([date_name,id_name]).any():raise ValueError('duplicate date/security keys')
 return frame

def normalize_daily_sources(stk_factor_pro, daily_basic=None, *, adjustment='backward'):
 """Use documented direct OHLC and explicit units. Return canonical MultiIndex.

 Quotes are authoritative for vol/amount/turnover. Basic supplies daily shares
 and market caps; overlapping turnover must agree within source rounding.
 """
 mode=_mode(adjustment);q=_keys(stk_factor_pro.rename(columns={'trade_date':'date','ts_code':'security_id'}))
 suffix=PRICE_MODES[mode];price_fields=[x+suffix for x in ('open','high','low','close')]
 required=set(price_fields)|{'open','high','low','close','vol','amount'}
 if not required<=set(q):raise ValueError('missing source fields: '+str(sorted(required-set(q))))
 if q[price_fields].isna().any().any():raise ValueError('missing requested adjustment price')
 q=q.set_index(['date','security_id']).sort_index();out=pd.DataFrame(index=q.index)
 for canonical,source in zip(('open','high','low','close'),price_fields):out[canonical]=pd.to_numeric(q[source],errors='raise')
 for name in ('open','high','low','close'):out['raw_'+name]=pd.to_numeric(q[name],errors='raise')
 if not np.isfinite(out.to_numpy()).all() or (out<=0).any().any():raise ValueError('invalid OHLC values')
 if ((out.low>out[['open','close']].min(axis=1))|(out.high<out[['open','close']].max(axis=1))).any():raise ValueError('invalid OHLC bounds')
 basic=None
 if daily_basic is not None:
  basic=_keys(daily_basic.rename(columns={'trade_date':'date','ts_code':'security_id'})).set_index(['date','security_id']).sort_index()
  if not basic.index.difference(q.index).empty:raise ValueError('basic contains unmatched quote keys')
  if 'turnover_rate' in q and 'turnover_rate' in basic:
   aligned=basic['turnover_rate'].reindex(q.index);both=q.turnover_rate.notna()&aligned.notna()
   if ((q.loc[both,'turnover_rate'].astype(float)-aligned[both].astype(float)).abs()>.00011).any():raise ValueError('turnover source conflict')
 for canonical,(source,multiplier,unit) in UNITS.items():
  src=q if source in q else basic
  if src is not None and source in src:
   values=pd.to_numeric(src[source],errors='raise').reindex(q.index)*multiplier
   if np.isinf(values).any() or (values.dropna()<0).any():raise ValueError('invalid '+canonical)
   out[canonical]=values
 if basic is not None:
  for field in ['pe_ttm','ps_ttm']:
   if field in basic:
    values=pd.to_numeric(basic[field],errors='raise').reindex(out.index)
    if np.isinf(values).any():raise ValueError('invalid '+field)
    out[field]=values
  if 'limit_status' in basic:
   try:
    from .tushare_features import limit_flags
   except ImportError:
    from source_features import limit_flags
   flags=limit_flags(basic.limit_status.reindex(out.index))
   out=out.join(flags)
 out['price_adjustment']=mode
 return out

class TushareParquetProvider(DataProvider):
 def __init__(self,data_root,eligible_factor_ids,required_fields_by_factor,*,exclude_st=False,book_equity_field=None,financial_source_policy=None):
  self.root=Path(data_root).resolve();self.eligible=set(eligible_factor_ids);self.required={k:set(v) for k,v in required_fields_by_factor.items()}
  if exclude_st:raise ValueError('historical ST gate unavailable; cannot enable exclude_st')
  if not self.eligible<=set(self.required):raise ValueError('each eligible factor needs explicit required fields')
  manifest=self.root/'manifest.json'
  if not manifest.is_file():raise FileNotFoundError('data manifest required')
  self._manifest_digest=hashlib.sha256(manifest.read_bytes()).hexdigest()
  entries=json.loads(manifest.read_text()).get('files',[])
  self._files={x['path']:x['sha256'] for x in entries}
  if len(self._files)!=len(entries):raise ValueError('duplicate manifest paths')
  self._cache={}
  self.book_equity_field=book_equity_field
  self.financial_source_policy=dict(financial_source_policy or {})
 def _read(self,path):
  path=Path(path);rel=path.relative_to(self.root).as_posix()
  if not path.resolve().is_relative_to(self.root):raise ValueError('data path escapes root')
  if rel not in self._files:raise ValueError('unmanifested data path '+rel)
  if hashlib.sha256((self.root/'manifest.json').read_bytes()).hexdigest()!=self._manifest_digest:raise ValueError('manifest changed during provider session')
  content=path.read_bytes()
  if hashlib.sha256(content).hexdigest()!=self._files[rel]:raise ValueError('data integrity mismatch '+rel)
  import io
  return pd.read_parquet(io.BytesIO(content))
 def trading_calendar(self,start,end):
  f=self._read(self.root/'calendar.parquet')
  if 'date' not in f:raise ValueError('calendar requires date')
  dates=pd.DatetimeIndex(pd.to_datetime(f.date.astype(str))).normalize()
  if dates.has_duplicates or dates.isna().any():raise ValueError('invalid calendar')
  return dates[(dates>=pd.Timestamp(start))&(dates<=pd.Timestamp(end))].sort_values()
 def available_history_start(self,spec):
  dates=self.trading_calendar(date.min,date.max)
  if dates.empty:raise ValueError('empty approved source calendar')
  required={'daily/'+day.strftime('%Y%m%d')+'.parquet' for day in dates}
  daily_paths={path for path in self._files if path.startswith('daily/') and path.endswith('.parquet')}
  if required!=daily_paths:raise ValueError('source manifest does not cover declared calendar history exactly')
  return dates[0].date()
 def security_master(self,start,end):
  f=self._read(self.root/'security_master.parquet').copy()
  if not {'security_id','list_date','delist_date'}<=set(f):raise ValueError('incomplete master schema')
  if f.security_id.isna().any() or f.security_id.duplicated().any():raise ValueError('invalid master keys')
  f.security_id=f.security_id.astype(str)
  for c in ['list_date','delist_date']:f[c]=pd.to_datetime(f[c].replace('',None).astype('string'),errors='raise')
  if f.list_date.isna().any():raise ValueError('missing list date')
  return f
 def _daily(self,start,end):
  key=(str(start),str(end))
  calendar=self.trading_calendar(start,end);parts=[]
  for day in calendar:
   p=self.root/'daily'/f'{day:%Y%m%d}.parquet'
   if not p.exists():raise FileNotFoundError('missing calendar partition '+str(p))
   f=_keys(self._read(p))
   if not f.date.eq(day).all():raise ValueError('partition date mismatch')
   parts.append(f)
  if not parts:raise ValueError('no calendar data')
  f=pd.concat(parts,ignore_index=True)
  if f.duplicated(['date','security_id']).any():raise ValueError('duplicate daily keys')
  f=f.set_index(['date','security_id']).sort_index();master=self.security_master(start,end)
  if not set(f.index.get_level_values('security_id'))<=set(master.security_id):raise ValueError('quote security missing from master')
  # Include every market date, even days without a quote for a security.
  idx=pd.MultiIndex.from_product([calendar,sorted(master.security_id)],names=['date','security_id'])
  f=f.reindex(idx);self._cache={key:f};return f.copy()
 def security_status(self,start,end):
  f=self._daily(start,end);observed=f.close.notna()&f.get('volume',pd.Series(np.nan,index=f.index)).gt(0)
  result=pd.DataFrame(index=f.index)
  result['is_trading']=observed
  result['is_suspended']=pd.Series(pd.NA,index=f.index,dtype='boolean');result.loc[observed,'is_suspended']=False
  result['is_st']=pd.Series(pd.NA,index=f.index,dtype='boolean')
  return result.reset_index()
 def index_constituents(self,index_id,start,end):
  return pd.DataFrame(columns=['security_id','effective_from','effective_to'])
 def load_factor_context(self,spec,start,end,calculation_dates,price_adjustment,benchmark_id):
  if spec.factor_id not in self.eligible:raise ValueError('factor is not explicitly eligible')
  if spec.implementation in {'intraday_microstructure','apm'}:raise ValueError('high frequency disabled')
  f=self._daily(start,end);mode=_mode(price_adjustment)
  if 'price_adjustment' not in f:raise ValueError('missing adjustment lineage')
  observed=f.close.notna()
  if not f.loc[observed,'price_adjustment'].eq(mode).all():raise ValueError('requested adjustment differs from partition')
  needs_index='index_close' in self.required[spec.factor_id]
  context_requirements={x for x in self.required[spec.factor_id] if ':' in x}
  need=(self.required[spec.factor_id]-{'index_close'}-context_requirements)|{'close','volume'}
  financial=self._observations('financial_observations.parquet',end)
  if not financial.empty:
   for field,source in self.financial_source_policy.items():
    financial=financial[~financial.field.eq(field)|financial.source_api.eq(source)]
   if spec.implementation=='financial_statement' and 'period_semantics' in financial:
    financial=financial[~financial.period_semantics.eq('single')].copy()
  if {'book_equity','book_to_price'} & need:
   if not self.book_equity_field:raise ValueError('explicit reviewed book equity field required')
   try:
    from .tushare_features import attach_book_equity
   except ImportError:
    from source_features import attach_book_equity
   f=attach_book_equity(f,financial,equity_field=self.book_equity_field)
  if not need<=set(f):raise ValueError('missing canonical fields '+str(sorted(need-set(f))))
  strict_need=need-{'pe_ttm','ps_ttm','is_limit_up','is_limit_down','book_equity','book_to_price'}
  if f.loc[observed,sorted(strict_need)].isna().any().any():raise ValueError('observed quotes missing required canonical values')
  idx=pd.DataFrame();path=self.root/'index_daily.parquet'
  if path.exists():
   idx=_keys(self._read(path),id_name='index_id')
   idx=idx[(idx.index_id==benchmark_id)&idx.date.between(pd.Timestamp(start),pd.Timestamp(end))]
   if not idx.empty:
    idx=idx.set_index('date').reindex(self.trading_calendar(start,end));idx['index_id']=benchmark_id;idx=idx.reset_index(names='date')
  if needs_index:
   if idx.empty or 'close' not in idx:raise ValueError('required index close unavailable')
   prices=pd.to_numeric(idx['close'],errors='raise')
   if not np.isfinite(prices.to_numpy()).all() or not prices.gt(0).all():raise ValueError('required index calendar coverage/close invalid')
  consensus=self._observations('consensus_observations.parquet',end)
  events={}
  for event_key in dict.fromkeys(['financial_report_disclosures','earnings_forecasts',*(spec.parameters.get('event_keys',[]) if spec.implementation=='event_price' else [])]):
   rel='events/'+event_key+'.parquet'
   if rel in self._files:
    event=self._read(self.root/rel)
    if not {'security_id','announcement_at','available_at'}<=set(event):raise ValueError('incomplete event contract')
    event['available_at']=pd.to_datetime(event.available_at,errors='raise')
    if event.available_at.isna().any():raise ValueError('event missing availability')
    events[event_key]=event[event.available_at<=pd.Timestamp(end)+pd.Timedelta(hours=15)].copy()
  for requirement in context_requirements:
   kind,name=requirement.split(':',1)
   if kind=='financial':table=financial
   elif kind=='consensus':table=consensus
   elif kind=='event':
    if name not in events or events[name].empty:raise ValueError('required event missing '+name)
    continue
   else:raise ValueError('unknown context requirement '+kind)
   if table.empty or name not in set(table.field):raise ValueError('required observation field missing '+requirement)
  if not financial.empty and spec.implementation in {'financial_statement','fundamental_formula','delta_roe_rank_gauss','intangible_value'}:
   try:
    from .tushare_financial import prepare_for_engine
   except ImportError:
    from financial_adapter import prepare_for_engine
   financial=prepare_for_engine(financial,spec)
  return FactorContext(calculation_dates=pd.DatetimeIndex(calculation_dates),daily=f.drop(columns='price_adjustment'),index_daily=idx,financial_observations=financial,consensus_observations=consensus,events=events)

 def _observations(self,relative,end):
  if relative not in self._files:return pd.DataFrame()
  f=self._read(self.root/relative)
  required={'security_id','available_at','field','value','source_api','source_record_id'}
  if relative.startswith('financial'):required.add('report_period')
  if not required<=set(f):raise ValueError('incomplete observation contract '+relative)
  f['available_at']=pd.to_datetime(f.available_at,errors='raise')
  if f.available_at.isna().any():raise ValueError('missing observation availability')
  revision=pd.to_datetime(f.get('revision_at',pd.Series(pd.NaT,index=f.index)),errors='raise')
  if (revision.notna() & revision.lt(f.available_at)).any():raise ValueError('revision predates availability')
  effective=revision.fillna(f.available_at)
  if 'report_period' in f and relative.startswith('financial'):
   f['report_period']=pd.to_datetime(f.report_period,errors='raise')
   if (f.report_period>effective.dt.normalize()).any():raise ValueError('report period after availability')
  f=f[effective<=pd.Timestamp(end)+pd.Timedelta(hours=15)].copy()
  return f


if __name__=='__main__':
 Path(__file__).with_name('provider_contract.json').write_text(json.dumps(provider_contract(),ensure_ascii=False,indent=2))
