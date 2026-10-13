"""Documented daily flags and point-in-time valuation features.

No inference from adjusted limit prices; no latest financial backfill.
"""
import numpy as np
import pandas as pd


def limit_flags(status):
    numeric = pd.to_numeric(status, errors='raise')
    if not numeric.dropna().isin(range(7)).all():
        raise ValueError('unknown Tushare limit_status enumeration')
    known = numeric.notna()
    return pd.DataFrame({'is_limit_up': numeric.isin([2,3]).astype(float).where(known),
                         'is_limit_down': numeric.isin([5,6]).astype(float).where(known)})


def attach_book_equity(daily, observations, *, equity_field, calculation_time='15:00:00'):
    """Join only equity visible at each daily cutoff, resolving period then vintage.

    equity_field must be explicitly chosen from the reviewed mapping; this API
    does not equate Tushare PB with an arbitrary book-equity definition.
    """
    from factor_engine.data.pit import pit_snapshot
    required={'security_id','report_period','available_at','field','value'}
    if not required <= set(observations):
        raise ValueError('missing PIT financial columns')
    out=daily.copy()
    f=observations[observations.field.eq(equity_field)].copy()
    available=pd.to_datetime(f.available_at,errors='raise')
    revision=pd.to_datetime(f.get('revision_at',pd.Series(pd.NaT,index=f.index)),errors='raise')
    if available.isna().any() or (revision.notna() & revision.lt(available)).any():
        raise ValueError('missing availability or revision predates availability')
    effective=revision.fillna(available)
    if (pd.to_datetime(f.report_period)>effective.dt.normalize()).any():
        raise ValueError('financial period after effective date')
    dates=pd.DatetimeIndex(out.index.get_level_values('date')).unique().sort_values()
    snap=pit_snapshot(f,dates,fields=[equity_field],calculation_time=calculation_time)
    values=snap.set_index(['date','security_id'])['value'] if not snap.empty else pd.Series(dtype=float)
    out['book_equity']=values.reindex(out.index)
    if 'total_market_cap' not in out:raise ValueError('book-to-price requires total_market_cap')
    out['book_to_price']=out.book_equity/out.total_market_cap.where(out.total_market_cap>0)
    out['book_to_price']=out.book_to_price.where(out.book_to_price>0)
    return out
