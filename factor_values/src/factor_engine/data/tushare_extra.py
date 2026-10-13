"""Narrow documented auxiliary adapters; no guessed investor/forecast semantics."""
import pandas as pd
import numpy as np

def normalize_large_order_rate(raw,source_api,*,definition_evidence=None):
 """Adapt THS/DC disclosed net large-order rate only with report-definition evidence.

 Evidence must identify how the source large-order threshold and denominator
 match the requested factor. This function does not approve a mapping itself.
 """
 if source_api not in {'moneyflow_ths','moneyflow_dc'}:raise ValueError('unsupported rate source')
 if not isinstance(definition_evidence,dict) or not all(definition_evidence.get(k) for k in ['report_reference','threshold_definition','denominator_definition']):raise ValueError('factor/source definition evidence required')
 needed={'ts_code','trade_date','buy_lg_amount_rate'}
 if not needed<=set(raw):raise ValueError('missing rate source columns')
 f=raw[list(needed)].copy();f['date']=pd.to_datetime(f.trade_date.astype(str),errors='raise').dt.normalize();f['security_id']=f.ts_code.astype(str)
 if f.duplicated(['date','security_id']).any() or f[['ts_code','trade_date']].isna().any().any():raise ValueError('duplicate or null keys')
 if source_api=='moneyflow_dc' and (f.date<pd.Timestamp('2023-09-11')).any():raise ValueError('DC documented coverage starts 2023-09-11')
 values=pd.to_numeric(f.buy_lg_amount_rate,errors='raise')*.01
 if np.isinf(values).any():raise ValueError('nonfinite rate')
 out=f[['date','security_id']].copy();out['large_order_inflow_rate']=values
 out['source_api']=source_api;out['source_field']='buy_lg_amount_rate';out['source_unit']='percent';out['multiplier']=.01
 out.attrs['definition_evidence']=dict(definition_evidence)
 return out.set_index(['date','security_id']).sort_index()

def intangible_rd_inputs(observations):
 """Rename already-PIT-normalized verified annual R&D total; retain all lineage."""
 required={'field','source_api','source_field','report_period','available_at','value'}
 if not required<=set(observations):raise ValueError('PIT and source lineage required')
 f=observations.copy();selected=f.field.eq('rd_total')
 if (selected&~(f.source_api.eq('fina_indicator')&f.source_field.eq('rd_exp'))).any():raise ValueError('rd_total source is not documented investment total')
 periods=pd.to_datetime(f.report_period)
 if (selected&~periods.dt.strftime('%m-%d').eq('12-31')).any():raise ValueError('ICV requires annual R&D totals')
 if f.loc[selected,'available_at'].isna().any():raise ValueError('R&D PIT availability missing')
 f.loc[selected,'field']='rd_investment_total'
 return f
