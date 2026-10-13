"""Cash dividends from normalized, dated implementation events in CNY."""
import numpy as np
import pandas as pd

from factor_engine.catalog import FactorSpec
from factor_engine.data.base import FactorContext
from factor_engine.registry import registry
from factor_engine.universe import apply_universe

RULES = {
    "cal_div_pretax": (0, "pretax", False),
    "cal_div_ttm_pretax": (1, "pretax", False),
    "cal_div_ttm_afttax": (1, "aftertax", False),
    "cal_div_rate_ttm_afttax": (1, "aftertax", True),
    "div_rate_3y": (3, "pretax", True),
}
REPURCHASE_IDS = {'repurchase_amount_ttm', 'repurchase_rate_ttm', 'div_rep2freefloat_ttm'}


def repurchase_window(days, events, coverage_start='2011-01-01'):
    days = pd.DatetimeIndex(days)
    start = days - pd.DateOffset(years=1)
    result = pd.Series(0., index=days)
    if not events.empty:
        if events.available_at.isna().any():
            raise ValueError('Repurchase event requires a dated disclosure of uncertainty')
        events = events.sort_values('available_at')
        times = pd.DatetimeIndex(events.available_at)
        lo, hi = times.searchsorted(start, side='right'), times.searchsorted(days, side='right')
        valid = events.valid.eq(True) & np.isfinite(events.increment_cny)
        total = np.r_[0., np.cumsum(events.increment_cny.where(valid, 0.))]
        bad = np.r_[0, np.cumsum(~valid)]
        result.iloc[:] = total[hi] - total[lo]
        result.iloc[(bad[hi] - bad[lo]) > 0] = np.nan
    result.loc[start < pd.Timestamp(coverage_start)] = np.nan
    return result


def dividend_window(days, events, years, tax, coverage_start="2000-01-01"):
    days = pd.DatetimeIndex(days)
    starts = days - pd.DateOffset(years=years) if years else pd.DatetimeIndex([pd.Timestamp(d.year, 1, 1) - pd.Timedelta(days=1) for d in days])
    values = np.zeros(len(days), dtype=float)
    if not events.empty:
        ex_date = pd.to_datetime(events.ex_date, errors="raise")
        available = pd.to_datetime(events.available_at, errors="raise")
        known = available.copy()
        for field in ("imp_ann_date", "ann_date"):
            if field in events:
                known = known.fillna(pd.to_datetime(events[field], errors="raise"))
        if known.isna().any():
            raise ValueError("Cash dividend event requires dated availability or announcement")
        amount = pd.to_numeric(events[tax + "_cash_cny"], errors="raise").to_numpy(float)
        valid = events.valid.eq(True).to_numpy() & np.isfinite(amount) & (amount >= 0)
        # Matrix axes are requested dates and this issuer's original events.
        # Explicit ns resolution prevents NumPy comparing mixed datetime units.
        day_values = days.to_numpy(dtype="datetime64[ns]")[:, None]
        ex_values = ex_date.to_numpy(dtype="datetime64[ns]")[None, :]
        known_values = known.to_numpy(dtype="datetime64[ns]")[None, :]
        available_values = available.to_numpy(dtype="datetime64[ns]")[None, :]
        known_by_day = pd.isna(known_values) | (known_values <= day_values)
        unknown_date = pd.isna(ex_values) & known_by_day
        visible = (available_values <= day_values) | (pd.isna(available_values) & known_by_day)
        mask = (ex_values > starts.to_numpy(dtype="datetime64[ns]")[:, None]) & (ex_values <= day_values) & visible
        values = mask @ np.where(valid, amount, 0.)
        values[unknown_date.any(axis=1) | (mask & ~valid[None, :]).any(axis=1)] = np.nan
    values[starts < pd.Timestamp(coverage_start)] = np.nan
    return pd.Series(values, index=days)


@registry.register("corporate_dividend")
def compute_corporate_dividend(context: FactorContext, spec: FactorSpec) -> pd.Series:
    if spec.factor_id not in RULES and spec.factor_id not in REPURCHASE_IDS:
        raise NotImplementedError(spec.factor_id)
    needs_dividend = spec.factor_id in RULES or spec.factor_id == 'div_rep2freefloat_ttm'
    if needs_dividend and "cash_dividends" not in context.events:
        raise ValueError("Cash dividends require a queried cash_dividends event table; absence is not zero")
    events = context.events.get("cash_dividends", pd.DataFrame())
    years, tax, divide_cap = RULES.get(spec.factor_id, (1, 'pretax', spec.factor_id == 'repurchase_rate_ttm'))
    required = {"security_id", "ex_date", "available_at", "valid", tax + "_cash_cny"}
    if needs_dividend and not required.issubset(events.columns):
        raise ValueError("Cash dividend event schema is incomplete")
    rep = context.events.get('repurchases', pd.DataFrame())
    if spec.factor_id in REPURCHASE_IDS and not {'security_id', 'available_at', 'valid', 'increment_cny'}.issubset(rep.columns):
        raise ValueError('Repurchases require a queried normalized repurchases table; absence is not zero')
    if context.daily.index.duplicated().any():
        raise ValueError("Duplicate dividend stock/date input")
    if divide_cap and "total_market_cap" not in context.daily:
        raise ValueError("Dividend yield requires CNY total_market_cap")
    if spec.factor_id == 'div_rep2freefloat_ttm' and not {'raw_close', 'free_float_shares'}.issubset(context.daily.columns):
        raise ValueError('Cash payout yield requires raw_close and free_float_shares')
    targets = pd.DatetimeIndex(context.calculation_dates)
    groups = {code: frame for code, frame in events.groupby("security_id", sort=False)} if needs_dividend else {}
    rep_groups = {code: frame for code, frame in rep.groupby('security_id', sort=False)} if spec.factor_id in REPURCHASE_IDS else {}
    pieces = []
    for code, frame in context.daily.groupby(level="security_id", sort=False):
        frame = frame.droplevel("security_id").sort_index()
        frame = frame[frame.index.isin(targets)]
        values = dividend_window(frame.index, groups.get(code, events.iloc[:0]), years, tax, spec.parameters.get("source_coverage_start", "2000-01-01")) if needs_dividend else pd.Series(0., index=frame.index)
        if spec.factor_id in REPURCHASE_IDS:
            values += repurchase_window(frame.index, rep_groups.get(code, rep.iloc[:0]), spec.parameters.get('repurchase_coverage_start', '2011-01-01'))
        if spec.factor_id == 'div_rep2freefloat_ttm':
            freecap = frame.raw_close * frame.free_float_shares
            values /= freecap.where(frame.raw_close.gt(0) & frame.free_float_shares.gt(0))
        if divide_cap:
            values = values / frame.total_market_cap.where(frame.total_market_cap.gt(0))
            if years == 3:
                values /= 3.
        values.index = pd.MultiIndex.from_arrays([values.index, [code] * len(values)], names=["date", "security_id"])
        pieces.append(values)
    result = pd.concat(pieces) if pieces else pd.Series(dtype=float, index=pd.MultiIndex.from_arrays([[], []], names=["date", "security_id"]))
    if context.universe:
        result = apply_universe(result, dict(context.universe))
    result.name = spec.factor_id
    return result.replace([np.inf, -np.inf], np.nan).sort_index()
