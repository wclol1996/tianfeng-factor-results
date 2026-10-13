"""Point-in-time financial/event adapters; never infer a revision's original date."""
import json
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np

# Units are CNY except EPS (CNY/share); source monetary statements are yuan.
FIELDS={
 'income':{'revenue':'operating_revenue','total_revenue':'total_revenue','oper_cost':'operating_cost','operate_profit':'operating_profit','n_income':'net_profit','n_income_attr_p':'net_profit_parent','basic_eps':'eps','ebit':'ebit','ebitda':'ebitda','admin_exp':'management_expense','sell_exp':'selling_expense','fin_exp':'financial_expense','rd_exp':'rd_expense','fin_exp_int_exp':'interest_expense'},
 'balancesheet':{'total_assets':'total_assets','total_liab':'total_liabilities','total_hldr_eqy_exc_min_int':'equity_parent','inventories':'inventory','fix_assets':'fixed_assets','oth_receiv':'other_receivables','notes_receiv':'notes_receivable','accounts_receiv':'accounts_receivable','accounts_receiv_bill':'accounts_receivable_combined','prepayment':'prepayments','contract_assets':'contract_assets','oth_cur_assets':'other_current_assets','notes_payable':'notes_payable','acct_payable':'accounts_payable','accounts_pay':'accounts_payable_combined','payroll_payable':'employee_compensation_payable','taxes_payable':'taxes_payable','oth_payable':'other_payables','undistr_porfit':'undistributed_profit','surplus_rese':'surplus_reserve','contract_liab':'contract_liabilities','adv_receipts':'advance_receipts','money_cap':'cash','minority_int':'minority_interest','oth_eqt_tools':'other_equity_instruments','cap_rese':'capital_reserve','lt_borr':'long_term_borrowing','bond_payable':'bonds_payable','non_cur_liab_due_1y':'noncurrent_liabilities_due_within_one_year','goodwill':'goodwill','total_cur_liab':'total_current_liabilities','total_cur_assets':'total_current_assets'},
 'cashflow':{'c_cash_equ_end_period':'cash_end','n_cashflow_act':'operating_cash_flow','c_fr_oth_operate_a':'cash_received_other_operating','oth_cash_pay_oper_act':'cash_paid_other_operating','c_paid_to_for_empl':'cash_paid_employees','n_incr_cash_cash_equ':'cash_net_increase','c_pay_acq_const_fiolta':'capital_expenditure_cash_paid','c_fr_sale_sg':'cash_received_sales','n_recp_disp_fiolta':'cash_received_disposal_long_term_assets'},
}
FIELDS['fina_indicator']={'profit_dedt':'net_profit_parent_excl_nr','daa':'depreciation_amortization','interestdebt':'interest_bearing_debt','current_exint':'non_interest_current_liabilities','quick_ratio':'quick_ratio','current_ratio':'current_ratio','fcff':'fcff','fcfe':'fcfe','ebit':'ebit','ebitda':'ebitda','rd_exp':'rd_investment_total','eps':'eps','bps':'book_value_per_share','ocfps':'operating_cash_flow_per_share','q_eps':'eps_quarter','q_dtprofit':'net_profit_parent_excl_nr_quarter','q_roe':'roe_quarter','q_npta':'roa_quarter','roe':'roe','roa':'roa','grossprofit_margin':'gross_margin','debt_to_assets':'debt_to_assets','assets_turn':'assets_turnover','inv_turn':'inventory_turnover','ar_turn':'receivable_turnover'}
FIELDS['balancesheet'].update({'st_borr':'short_term_borrowing','total_ncl':'total_noncurrent_liabilities','oth_cur_liab':'other_current_liabilities','oth_ncl':'other_noncurrent_liabilities','trad_asset':'trading_financial_assets','fa_avail_for_sale':'available_for_sale_financial_assets','deriv_assets':'derivative_financial_assets','cost_fin_assets':'amortized_cost_financial_assets','fair_value_fin_assets':'fvoci_financial_assets','oth_illiq_fin_assets':'other_noncurrent_financial_assets','int_receiv':'interest_receivable','div_receiv':'dividends_receivable','int_payable':'interest_payable','div_payable':'dividends_payable','oth_pay_total':'other_payables_total','oth_rcv_total':'other_receivables_total','fix_assets_total':'fixed_assets_total','cip':'construction_in_progress','cip_total':'construction_in_progress_total','intan_assets':'intangible_assets','r_and_d':'development_expenditure_balance','lease_liab':'lease_liabilities','total_share':'total_shares'})
FIELDS['cashflow'].update({'c_cash_equ_beg_period':'cash_begin','end_bal_cash':'cash_balance_end','end_bal_cash_equ':'cash_equivalents_end','n_cashflow_inv_act':'investing_cash_flow','n_cash_flows_fnc_act':'financing_cash_flow','c_paid_goods_s':'cash_paid_goods_services','c_paid_for_taxes':'cash_paid_taxes','depr_fa_coga_dpba':'depreciation_fixed_assets','amort_intang_assets':'amortization_intangible_assets','lt_amort_deferred_exp':'amortization_long_term_deferred_expenses'})
PERCENT_FIELDS={'q_roe','q_npta','roe','roa','grossprofit_margin','debt_to_assets'}
INDICATOR_FLOW_FIELDS={'profit_dedt','daa','fcff','fcfe','ebit','ebitda','rd_exp'}
RATIO_FIELDS={'quick_ratio','current_ratio','assets_turn','inv_turn','ar_turn'}
PER_SHARE_FIELDS={'basic_eps','eps','bps','ocfps','q_eps'}

def source_record_id(row,api):
 return hashlib.sha256(json.dumps({'api':api,'record':{k:str(v) for k,v in row.items()}},sort_keys=True,ensure_ascii=False).encode()).hexdigest()

COLUMNS=['security_id','report_period','available_at','revision_at','field','value','source_api','source_field','source_record_id','source_unit','multiplier','period_semantics']

def _timestamp(value):
 if pd.isna(value) or str(value).strip()=='':return pd.NaT
 text=str(value);return pd.to_datetime(text,format='%Y%m%d') if len(text)==8 and text.isdigit() else pd.Timestamp(text)

def available_time(value,calendar):
 """Date precision -> next actual trading day at midnight; timestamps preserved."""
 ts=_timestamp(value)
 if pd.isna(ts):raise ValueError('missing availability date')
 if ts.tzinfo is not None:ts=ts.tz_convert('Asia/Shanghai').tz_localize(None)
 text=str(value)
 date_only=(len(text)==8 and text.isdigit()) or (len(text)==10 and ':' not in text) or ts==ts.normalize()
 if not date_only:return ts
 days=pd.DatetimeIndex(calendar).normalize().sort_values().unique();future=days[days>ts.normalize()]
 if not len(future):raise ValueError('calendar does not cover next trading day')
 return future[0]

def normalize_financial(raw,api,calendar,*,fields=None):
 """Return (long observations, quarantine).

 Statement type5 preserves pre-adjustment data; type4 requires credible actual
 publication time. Type1 and financial indicators need verified original
 provenance or actual revision/first-observed timestamp. Stable IDs are hashed
 from source records when absent. No assumption from update_flag.
 """
 if api not in FIELDS:raise ValueError('unsupported/ambiguous financial API')
 mapping=FIELDS[api]
 # Ambiguous stock units must not quarantine unrelated balance-sheet fields.
 # The field remains available through explicit fields=['total_share'] plus evidence.
 requested=list(fields) if fields is not None else [x for x in mapping if not (api=='balancesheet' and x=='total_share')]
 if not set(requested)<=set(mapping):raise ValueError('unknown or semantically ambiguous field')
 if not {'ts_code','end_date'}<=set(raw):raise ValueError('financial key columns missing')
 if 'source_record_id' in raw and raw.source_record_id.dropna().duplicated().any():raise ValueError('duplicate source_record_id')
 observations=[];quarantine=[];seen=set()
 for index,row in raw.iterrows():
  row_start=len(observations)
  try:
   if pd.isna(row.ts_code) or not str(row.ts_code).strip():raise ValueError('missing security identity')
   role=row.get('record_role');record_id=row.get('source_record_id')
   if not record_id or pd.isna(record_id):record_id=source_record_id(row,api)
   if record_id in seen:raise ValueError('duplicate source_record_id')
   seen.add(record_id)
   report_type=str(row.get('report_type',''))
   original=row.get('f_ann_date');original=original if pd.notna(original) and str(original) else row.get('ann_date')
   revision=pd.NaT
   if api!='fina_indicator' and report_type=='5':
    effective=available_time(original,calendar)
   elif api!='fina_indicator' and report_type=='4':
    actual=_timestamp(row.get('f_ann_date'));ann=_timestamp(row.get('ann_date'));period_date=_timestamp(row.end_date)
    if pd.isna(actual) or actual.year<=period_date.year or (pd.notna(ann) and actual<ann):raise ValueError('adjusted type4 missing credible current-year actual publication')
    effective=available_time(row.f_ann_date,calendar);revision=effective
   elif api=='fina_indicator' or report_type=='1':
    provenance=row.get('provenance_note')
    if role=='original' and provenance and pd.notna(provenance):effective=available_time(original,calendar)
    elif pd.notna(row.get('revision_published_at')):
     effective=max(available_time(original,calendar),available_time(row.revision_published_at,calendar));revision=effective
    elif pd.notna(row.get('source_observed_at')):
     effective=max(available_time(original,calendar),available_time(row.source_observed_at,calendar));revision=effective
    else:raise ValueError('latest/indicator record needs revision publication or observed timestamp; not original by default')
   else:raise ValueError('unsupported report type; only cumulative consolidated 1/4/5')
   period=_timestamp(row['end_date']).normalize()
   if period.month not in (3,6,9,12) or period!=period.to_period('Q').end_time.normalize():raise ValueError('non-quarter report period')
   if period>effective.normalize():raise ValueError('report period after availability')
   for source in requested:
    if source not in row or pd.isna(row[source]):continue
    if api=='balancesheet' and source=='total_share' and row.get('total_share_source_unit')!='shares':raise ValueError('balancesheet total_share unit not specified in archive; verified shares metadata required')
    value=float(row[source])
    if not np.isfinite(value):raise ValueError('nonfinite financial value')
    observations.append(dict(security_id=str(row.ts_code),report_period=period,available_at=effective,revision_at=revision,field=mapping[source],value=value*(.01 if api=='fina_indicator' and source in PERCENT_FIELDS else 1.),source_api=api,source_field=source,source_record_id=record_id,source_unit=('percent' if api=='fina_indicator' and source in PERCENT_FIELDS else 'ratio' if source in RATIO_FIELDS else 'CNY/share' if source in PER_SHARE_FIELDS else 'shares' if source=='total_share' else 'CNY'),multiplier=(.01 if api=='fina_indicator' and source in PERCENT_FIELDS else 1.),period_semantics=('single' if source.startswith('q_') else 'point' if api=='balancesheet' or (api=='fina_indicator' and source not in INDICATOR_FLOW_FIELDS) or source in {'c_cash_equ_end_period','c_cash_equ_beg_period','end_bal_cash','end_bal_cash_equ'} else 'cumulative')))
  except (ValueError,TypeError) as exc:
   # Remove a partially normalized rejected record atomically.
   del observations[row_start:]
   quarantine.append({'row_index':index,'source_record_id':row.get('source_record_id'),'reason':str(exc),'source_api':api})
 out=pd.DataFrame(observations,columns=COLUMNS)
 if not out.empty and out.duplicated(['security_id','field','report_period','available_at']).any():raise ValueError('conflicting financial records at same effective time')
 return out,pd.DataFrame(quarantine)

def snapshot_quarters(observations,cutoff):
 """Resolve visible versions first, then derive quarter/TTM without year bridges."""
 f=observations.copy();cutoff=pd.Timestamp(cutoff)
 if f.empty:return pd.DataFrame()
 f['effective_at']=pd.to_datetime(f.revision_at).fillna(pd.to_datetime(f.available_at))
 f=f[f.effective_at<=cutoff].sort_values('effective_at').drop_duplicates(['security_id','field','report_period'],keep='last')
 output=[]
 for (security,field),group in f.groupby(['security_id','field']):
  group=group.sort_values('report_period');by={pd.Timestamp(r.report_period).to_period('Q'):r for r in group.itertuples()};quarters={}
  for period,row in by.items():
   if row.period_semantics=='single':single=row.value
   elif row.period_semantics=='point':single=np.nan
   elif period.quarter==1:single=row.value
   elif period-1 in by and (period-1).year==period.year:single=row.value-by[period-1].value
   else:single=np.nan
   quarters[period]=single
  for period,row in by.items():
   four=[quarters.get(period-i,np.nan) for i in range(4)]
   ttm=sum(four) if all(pd.notna(x) for x in four) else np.nan
   output.append({'security_id':security,'field':field,'report_period':row.report_period,'value':row.value,'single_quarter':quarters[period],'ttm':ttm,'asof':cutoff,'source_record_id':row.source_record_id})
 return pd.DataFrame(output)

def normalize_events(raw,api,calendar):
 source={'forecast':'ann_date','disclosure_date':'actual_date'}.get(api)
 if source is None:raise ValueError('unsupported event type')
 rows=[];bad=[]
 for i,r in raw.iterrows():
  try:
   first_value=r.get('first_ann_date')
   if api=='forecast' and pd.notna(first_value) and str(first_value).strip() not in {'','None','nan','0'}:
    first=_timestamp(first_value);announced=_timestamp(r.get(source))
    if pd.notna(first) and pd.notna(announced) and first>announced:
     raise ValueError('first forecast announcement follows announcement date')
   when=available_time(r.get(source),calendar)
   rows.append({'security_id':str(r.ts_code),'announcement_at':_timestamp(r.get(source)),'available_at':when,'report_period':_timestamp(r.get('end_date')),'event_type':'forecast' if api=='forecast' else 'financial_disclosure','source_api':api,'source_field':source,'source_announcement_date':r.get(source)})
  except ValueError as exc:bad.append({'row_index':i,'reason':str(exc)})
 out=pd.DataFrame(rows)
 if not out.empty:out=out.drop_duplicates(['security_id','announcement_at','report_period','event_type'])
 return out,pd.DataFrame(bad)

def normalize_report_predictions(raw,calendar):
 """Normalize verified annual analyst records, not an invented consensus series.

 Explicit forecast_year and report_period_kind='annual' avoid guessing quarter
 codes. Only eps/PE/ROE/op_rt/op_pr have supported direct semantics. np is not
 mapped to parent net profit. No PB/PEG/CFPS substitutions.
 """
 rows=[];bad=[];mapping={'eps':('eps',1.),'pe':('pe',1.),'roe':('roe',.01),'op_rt':('operating_revenue',10000.),'op_pr':('operating_profit',10000.)}
 for i,r in raw.iterrows():
  row_start=len(rows)
  try:
   if r.get('report_period_kind')!='annual':raise ValueError('annual prediction period not verified')
   year=int(r.get('forecast_year'));when=available_time(r.get('report_date'),calendar)
   if year<1900 or year>2200 or not r.get('org_name'):raise ValueError('invalid year/organization')
   if pd.notna(r.get('revision_published_at')):when=max(when,available_time(r.revision_published_at,calendar))
   if pd.notna(r.get('create_time')):when=max(when,available_time(r.create_time,calendar))
   if pd.notna(r.get('update_time')):when=max(when,available_time(r.update_time,calendar))
   if r.get('record_role')!='original' and all(pd.isna(r.get(t)) for t in ['revision_published_at','create_time','update_time']):raise ValueError('prediction revision provenance missing')
   for source,(field,multiplier) in mapping.items():
    if source=='roe' and source in r and pd.notna(r[source]) and r.get('roe_source_unit')!='percent':raise ValueError('forecast ROE unit requires explicit percent evidence')
    if source in r and pd.notna(r[source]) and not np.isfinite(float(r[source])):raise ValueError('nonfinite prediction')
    if source in r and pd.notna(r[source]):rows.append({'security_id':str(r.ts_code),'forecast_year':year,'report_year':year,'report_period':pd.Timestamp(year,12,31),'available_at':when,'field':field,'value':float(r[source])*multiplier,'organization':r.org_name,'analyst_id':'org:'+str(r.org_name),'aggregation_entity':'institution','report_date':_timestamp(r.report_date),'source_api':'report_rc','source_field':source,'source_report_date':r.report_date,'source_record_id':source_record_id(r,'report_rc'),'create_time':r.get('create_time'),'update_time':r.get('update_time')})
  except (ValueError,TypeError) as exc:
   del rows[row_start:]
   bad.append({'row_index':i,'reason':str(exc)})
 return pd.DataFrame(rows),pd.DataFrame(bad)

def prediction_snapshot(records,cutoff):
 """Latest record per organization/security/year/field at the given cutoff."""
 f=records[records.available_at<=pd.Timestamp(cutoff)].copy()
 keys=['security_id','forecast_year','organization','field']
 if f.duplicated(keys+['available_at']).any():raise ValueError('ambiguous same-time analyst predictions')
 return f.sort_values('available_at').drop_duplicates(keys,keep='last')

def prepare_for_engine(observations,spec):
 """Resolve consumer aliases, reject source collisions, preserve single-quarter tags.

 Fundamental formulas use literal Chinese field names; the statement family
 resolves aliases internally. No sum or last-row preference across source APIs.
 """
 from factor_engine.factors.financial_statement import FIELD_ALIASES
 f=observations.copy()
 if f.empty:return f
 collision=f.groupby(['security_id','field'])['source_api'].nunique()
 if collision.gt(1).any():raise ValueError('multiple source APIs for canonical financial field; select explicit source first')
 f['accounting_method']=np.where(f.period_semantics.eq('single'),'Q','A')
 f.loc[f.field.eq('rd_total'),'field']='rd_investment_total'
 if spec.implementation=='financial_statement':
  # Statement engine derives Q from cumulative records itself; do not provide
  # already-single fields under cumulative aliases or silently ignore them.
  if f.period_semantics.eq('single').any():raise ValueError('statement engine requires cumulative source, not precomputed single-quarter records')
  return f
 if spec.implementation not in {'fundamental_formula','delta_roe_rank_gauss'}:return f
 aliases={a:canonical for canonical,values in FIELD_ALIASES.items() for a in values}
 aliases.update({'单季度净利润':'net_profit','单季度营业收入':'operating_revenue','单季度净资产收益率':'roe_quarter','单季度ROA':'roa_quarter','单季度_总资产净利润':'roa_quarter','利息费用':'interest_expense','流动负债合计':'total_current_liabilities','商誉':'goodwill','速动比率':'quick_ratio','期末总股本':'total_shares'})
 requested=set()
 def visit(node):
  if isinstance(node,dict):
   if 'field' in node and node.get('source','financial')=='financial':requested.add(node['field'])
   for value in node.values():visit(value)
  elif isinstance(node,list):
   for value in node:visit(value)
 visit(spec.parameters)
 if spec.implementation=='delta_roe_rank_gauss':requested.update(spec.parameters.get(k,default) for k,default in [('asset_field','资产总计'),('liability_field','负债合计'),('profit_field','净利润(不含少数股东损益)')])
 parts=[]
 for name in requested:
  if 'TTM' in name:
   canonical=aliases.get(name,{'EBITDA TTM':'ebitda','扣非净利润TTM':'net_profit_parent_excl_nr','财务费用(TTM)':'financial_expense'}.get(name))
   if canonical is None:raise ValueError('TTM field requires explicit semantic formula '+name)
   parts.append(derive_ttm_observations(f,canonical,name));continue
  canonical=aliases.get(name,name)
  if name.startswith('单季度') and not canonical.endswith('_quarter'):
   parts.append(derive_ttm_observations(f,canonical,name,mode='single'));continue
  if canonical=='other_operating_cash_flow' and not f.field.eq(canonical).any():
   parts.append(derive_component_net(f,{'cash_received_other_operating':1.,'cash_paid_other_operating':-1.},name));continue
  part=f[f.field.eq(canonical)].copy()
  if part.empty:raise ValueError('missing consumer financial field '+name)
  part['field']=name;parts.append(part)
 return pd.concat(parts,ignore_index=True) if parts else f


def derive_ttm_observations(observations,canonical,output_field,*,mode='ttm'):
 """Materialize versioned four-quarter totals at every source availability event."""
 if mode not in {'ttm','single'}:raise ValueError('unsupported derived mode')
 source=observations[observations.field.eq(canonical)].copy()
 if source.empty:raise ValueError('missing cumulative component '+canonical)
 if not source.period_semantics.eq('cumulative').all():raise ValueError('TTM requires cumulative flow records')
 effective=pd.to_datetime(source.revision_at).fillna(pd.to_datetime(source.available_at))
 rows=[]
 for cutoff in sorted(effective.unique()):
  visible=source[effective<=cutoff];snapshot=snapshot_quarters(visible,cutoff)
  for row in snapshot.itertuples():
   lineage=visible[(visible.security_id==row.security_id)&(pd.to_datetime(visible.report_period)<=row.report_period)]
   rows.append({'security_id':row.security_id,'report_period':row.report_period,'available_at':pd.Timestamp(cutoff),'revision_at':pd.Timestamp(cutoff),'field':output_field,'value':row.ttm if mode=='ttm' else row.single_quarter,'source_api':'derived_'+mode+':'+str(source.source_api.iloc[0]),'source_field':canonical,'source_record_id':hashlib.sha256('|'.join(sorted(lineage.source_record_id.astype(str))).encode()).hexdigest(),'source_unit':'CNY','multiplier':1.,'period_semantics':mode,'accounting_method':mode.upper()})
 return pd.DataFrame(rows)


def derive_component_net(observations,components,output_field):
 """Signed same-asof, same-report-period component sum, no missing-to-zero."""
 source=observations[observations.field.isin(components)].copy()
 if not set(components)<=set(source.field):raise ValueError('missing net-flow components')
 effective=pd.to_datetime(source.revision_at).fillna(pd.to_datetime(source.available_at));rows=[]
 for cutoff in sorted(effective.unique()):
  visible=source[effective<=cutoff].assign(effective_at=effective[effective<=cutoff]).sort_values('effective_at').drop_duplicates(['security_id','report_period','field'],keep='last')
  for (security,period),group in visible.groupby(['security_id','report_period']):
   values=group.set_index('field').value
   value=sum(values[k]*weight for k,weight in components.items()) if set(components)<=set(values.index) else np.nan
   rows.append({'security_id':security,'report_period':period,'available_at':pd.Timestamp(cutoff),'revision_at':pd.Timestamp(cutoff),'field':output_field,'value':value,'source_api':'derived_component_net','source_field':'|'.join(components),'source_record_id':hashlib.sha256('|'.join(sorted(group.source_record_id.astype(str))).encode()).hexdigest(),'source_unit':'CNY','multiplier':1.,'period_semantics':'cumulative','accounting_method':'A'})
 return pd.DataFrame(rows)
