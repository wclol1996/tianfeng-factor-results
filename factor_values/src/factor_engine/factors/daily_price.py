from __future__ import annotations

import re
from dataclasses import replace

import numpy as np
import pandas as pd

from factor_engine.catalog import FactorSpec
from factor_engine.data.base import FactorContext
from factor_engine.factors.common import (
    INDEX_NAMES,
    group_diff,
    group_pct_change,
    group_shift,
    group_transform,
    require_daily,
    restore_declared_market_dates,
    rolling,
    rolling_pair,
    safe_divide,
)
from factor_engine.registry import registry
from factor_engine.universe import apply_universe


HORIZON_SUFFIX = {"1m": 20, "3m": 60, "6m": 120, "1y": 240}

DECLARED_CALENDAR_FACTORS = frozenset({
    "STR_20d", "STR_60d", "Turn20", "Turn60",
    "UTR_20d", "UTR_60d", "UTR2_20d", "UTR2_60d",
    "shift_mom", "shift_mom_60d", "shift_mom_120d",
    "ddr_mom_20d", "ddr_mom_60d", "ddr_mom_120d",
    "cal_momentum_overnight_1m", "cal_momentum_overnight_3m",
    "cal_momentum_overnight_6m", "cal_momentum_overnight_1y",
    "cal_momentum_off_limit_1y", "cal_momentum_off_limit_prod_1y",
    "cal_momentum_off_limit_prod_6m", "cal_momentum_off_limit_prod_3m",
    "cal_momentum_range_1m", "pos_mom", "max_drawdown_duration",
    "mfle20", "mfle40", "mfle60", "mfle120",
    "amtprop1m_quantile1y", "winrate_weekly",
})

SUPPORTED_PATTERNS = tuple(map(re.compile, [
    r"^intraday_amp_vol_(1m|3m|6m|1y)$",
    r"^up_shadow_vol_(1m|3m|6m|1y)$",
    r"^w_down_shadow_vol_(1m|3m|6m|1y)$",
    r"^vol_(60d|90d|120d|240d|5y)$",
    r"^vol(60|120|240)_(up|down|jup|jdown|j|nj)$",
    r"^res_vol_(20d|60d|120d|240d|5y)$",
    r"^resvol(60|120|240)_(?:(up|down|jup|jdown|j|nj)_)?capm$",
    r"^mom(60|120|240)_20$",
    r"^shift_mom(?:_(60|120)d)?$",
    r"^cal_momentum_off_limit_prod_(1y|6m|3m)$",
    r"^Ret(60|120|240)_20_RemoveUpLimit$",
    r"^ddr_mom_(20|60|120)d$",
    r"^cal_momentum_overnight_(1m|3m|6m|1y)$",
    r"^weighted_skew_(1m|3m|6m|1y)$",
    r"^(csk|ckt)(?:_(up|down))?_120d$",
    r"^volume_price_corr_(1m|3m|6m|1y)$",
    r"^turnover(20|60|120|240)$",
    r"^Turn(20|60)$",
    r"^STR_(20|60)d$",
    r"^UTR_(20|60)d$",
    r"^UTR2_(20|60)d$",
    r"^Amihud_ILLIQ(20|60|120|240)$",
    r"^mfle(20|40|60|120)$",
]))
SUPPORTED_EXACT = frozenset({
    "resvol_j_capm", "high250", "reverse20",
    "cal_momentum_range_1m",
    "turnover_price_corr", "turnover_return_corr1", "turnoverChg_return_corr1",
    "turnoverChg_return_corr2", "amtprop1m_quantile1y", "winrate_weekly",
    "ideal_reversal", "ideal_swing",
    "cal_momentum_off_limit_1y",
    "max_drawdown_duration", "pos_mom",
})


def supports_factor_id(factor_id: str) -> bool:
    return factor_id in SUPPORTED_EXACT or any(pattern.fullmatch(factor_id) for pattern in SUPPORTED_PATTERNS)


def _column(context: FactorContext, name: str) -> pd.Series:
    require_daily(context.daily, [name])
    return context.daily[name].astype(float).sort_index()


def _market_return(context: FactorContext, benchmark_id: str | None) -> pd.Series:
    frame = context.index_daily.copy()
    if frame.empty:
        raise ValueError("index_daily is required for benchmark-relative factors")
    if isinstance(frame.index, pd.MultiIndex):
        frame = frame.reset_index()
    elif frame.index.name == "date":
        frame = frame.reset_index()
    required = {"date", "close"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"index_daily is missing columns: {sorted(missing)}")
    if "index_id" in frame and benchmark_id:
        frame = frame[frame["index_id"].astype(str).eq(benchmark_id)]
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    close = frame.sort_values("date").drop_duplicates("date", keep="last").set_index("date")["close"].astype(float)
    return close.pct_change(fill_method=None)


def _rolling_product(returns: pd.Series, window: int) -> pd.Series:
    return rolling(returns, window, lambda values: float(np.prod(1.0 + values) - 1.0))


def _fixed_calendar_window_complete(values: pd.Series, window: int) -> pd.Series:
    """Identify windows spanning exactly ``window`` market trading dates."""

    dates = pd.DatetimeIndex(values.index.get_level_values("date"))
    calendar = dates.unique().sort_values()
    positions = pd.Series(
        calendar.get_indexer(dates).astype(float),
        index=values.index,
        name="calendar_position",
    )
    span = positions - group_shift(positions, window - 1)
    return span.eq(window - 1)


def _fixed_calendar_rolling_sum(values: pd.Series, window: int) -> pd.Series:
    """Roll over exactly ``window`` market trading dates without compression."""

    result = rolling(values, window, "sum")
    return result.where(_fixed_calendar_window_complete(values, window))


def _fixed_calendar_shift(values: pd.Series, periods: int) -> pd.Series:
    """Look up an exact earlier market date without requiring intervening quotes."""
    if periods < 0:
        raise ValueError("future market-date shifts are not permitted")
    dates = pd.DatetimeIndex(values.index.get_level_values("date"))
    calendar = dates.unique().sort_values()
    if calendar.empty:
        return values.copy()
    positions = calendar.get_indexer(dates) - periods
    lookup = pd.MultiIndex.from_arrays(
        [calendar.take(np.maximum(positions, 0)), values.index.get_level_values("security_id")],
        names=INDEX_NAMES,
    )
    result = pd.Series(values.reindex(lookup).to_numpy(), index=values.index, name=values.name)
    return result.where(positions >= 0)


def _fixed_calendar_std(
    values: pd.Series, window: int, ddof: int = 1, *, fresh_window: bool = False,
) -> pd.Series:
    """Sample standard deviation; an exactly constant full window has zero variance."""
    if fresh_window:
        # A direct window reduction cannot inherit numerical error from old data.
        def fresh_std(sample: pd.Series) -> pd.Series:
            result = np.full(len(sample), np.nan)
            if len(sample) >= window:
                windows = np.lib.stride_tricks.sliding_window_view(sample.to_numpy(), window)
                result[window - 1:] = np.std(windows, axis=1, ddof=ddof)
            return pd.Series(result, index=sample.index)

        result = group_transform(values, fresh_std)
    else:
        result = (
            rolling(values, window, "std") if ddof == 1
            else rolling(values, window, lambda sample: float(np.std(sample, ddof=ddof)))
        )
    constant = rolling(values, window, "min").eq(rolling(values, window, "max"))
    return result.mask(constant, 0.0).where(_fixed_calendar_window_complete(values, window))


def _remove_up_limit_momentum(
    returns: pd.Series,
    limit_up_status: pd.Series,
    long_window: int,
    skip_window: int,
) -> pd.Series:
    """Sum returns over a fixed window, excluding limit-up days and the last month.

    A limit-up observation stays in the natural trading-date window but
    contributes zero to the return sum. Unknown status remains missing, so a
    window containing an unknown status cannot silently treat it as non-limit.
    """

    if long_window <= skip_window or skip_window < 1:
        raise ValueError(
            "remove-up-limit momentum requires long_window > skip_window >= 1"
        )
    numeric_status = pd.to_numeric(limit_up_status, errors="coerce")
    known = numeric_status.isin([0.0, 1.0])
    adjusted_returns = returns.where(known).mask(numeric_status.eq(1.0), 0.0)
    long_return = _fixed_calendar_rolling_sum(adjusted_returns, long_window)
    recent_return = _fixed_calendar_rolling_sum(adjusted_returns, skip_window)
    return long_return - recent_return


def _unconflicted_limit_status(status: pd.Series, context: FactorContext) -> pd.Series:
    if "limit_status_conflict" in context.daily:
        conflict = context.daily["limit_status_conflict"].reindex(status.index)
        return status.where(conflict.eq(False))
    return status


def _limit_event_status(context: FactorContext, spec: FactorSpec) -> pd.Series:
    """Return 1 for a Gildata-flagged limit event, 0 for a normal/no-limit day.

    Price-limit prices remain available for audit, but the factor deliberately
    does not infer status from adjusted prices. Unknown vendor status propagates
    as missing instead of being silently treated as a normal day.
    """

    up_field = str(spec.parameters.get("limit_up_field", "is_limit_up"))
    down_field = str(spec.parameters.get("limit_down_field", "is_limit_down"))
    up = pd.to_numeric(_column(context, up_field), errors="coerce")
    down = pd.to_numeric(_column(context, down_field), errors="coerce")
    known = up.isin([0.0, 1.0]) & down.isin([0.0, 1.0])
    status = (up.eq(1.0) | down.eq(1.0)).astype(float).where(known)

    no_limit_field = str(spec.parameters.get("no_price_limit_field", "is_no_price_limit"))
    if no_limit_field in context.daily:
        no_limit = pd.to_numeric(context.daily[no_limit_field], errors="coerce")
        status = status.mask(no_limit.eq(1.0), 0.0)
    elif "has_price_limit" in context.daily:
        has_limit = pd.to_numeric(context.daily["has_price_limit"], errors="coerce")
        status = status.mask(has_limit.eq(0.0), 0.0)
    return _unconflicted_limit_status(status, context)


def _mask_limit_event_returns(returns: pd.Series, status: pd.Series) -> pd.Series:
    known = status.isin([0.0, 1.0])
    return returns.where(known).mask(status.eq(1.0), 0.0)


def _fixed_calendar_rolling_product(values: pd.Series, window: int) -> pd.Series:
    result = _rolling_product(values, window)
    return result.where(_fixed_calendar_window_complete(values, window))


def _shift_momentum(
    returns: pd.Series,
    long_window: int,
    excluded_recent_days: int,
) -> pd.Series:
    """Long-window simple-return sum less the most recent return sum."""

    if long_window <= excluded_recent_days or excluded_recent_days < 1:
        raise ValueError(
            "shift momentum requires long_window > excluded_recent_days >= 1"
        )
    # The excluded recent returns cancel algebraically. Their missingness must
    # not invalidate an otherwise fully observed included interval. Shifting
    # by market date also avoids compressing a suspension into the window.
    included = _fixed_calendar_rolling_sum(
        returns, long_window - excluded_recent_days,
    )
    return _fixed_calendar_shift(included, excluded_recent_days)


def _range_adjusted_momentum(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    window: int,
    lower_quantile: float,
    upper_quantile: float,
) -> pd.Series:
    """High-amplitude close returns minus low-amplitude close returns."""

    if window < 2:
        raise ValueError("range-adjusted momentum window must be at least 2")
    if not 0.0 < lower_quantile < upper_quantile < 1.0:
        raise ValueError("range-adjusted momentum quantiles must satisfy 0 < lower < upper < 1")
    previous_close = _fixed_calendar_shift(close, 1)
    close_return = safe_divide(close, previous_close) - 1.0
    amplitude = safe_divide(high - low, previous_close)

    def calculate(return_values: np.ndarray, amplitude_values: np.ndarray) -> float:
        lower = float(np.quantile(amplitude_values, lower_quantile, method="linear"))
        upper = float(np.quantile(amplitude_values, upper_quantile, method="linear"))
        weights = np.select(
            [amplitude_values > upper, amplitude_values < lower],
            [1.0, -1.0],
            default=0.0,
        )
        return float(np.dot(return_values, weights))

    result = rolling_pair(close_return, amplitude, window, calculate)
    return result.where(_fixed_calendar_window_complete(close_return, window))


def _rolling_jump_variance(returns: pd.Series, window: int, selector: str) -> pd.Series:
    def calculate(values: np.ndarray) -> float:
        centered = values - values.mean()
        sigma = centered.std(ddof=1)
        if selector == "up":
            mask = values > 0
        elif selector == "down":
            mask = values < 0
        elif selector == "jup":
            mask = centered >= sigma
        elif selector == "jdown":
            mask = centered <= -sigma
        elif selector == "j":
            mask = np.abs(centered) >= sigma
        elif selector == "nj":
            mask = np.abs(centered) < sigma
        else:
            raise ValueError(f"unknown jump-variance selector: {selector}")
        return float(np.mean(np.square(centered[mask]))) if mask.any() else np.nan

    return rolling(returns, window, calculate)


def _rolling_capm(
    stock_returns: pd.Series,
    market_returns: pd.Series,
    window: int,
    statistic: str,
) -> pd.Series:
    joined = stock_returns.rename("stock").reset_index().merge(
        market_returns.rename("market").reset_index(), on="date", how="left",
    )
    output: list[pd.DataFrame] = []
    for _, group in joined.groupby("security_id", sort=False):
        group = group.sort_values("date").copy()
        result = np.full(len(group), np.nan)
        for position in range(window - 1, len(group)):
            sample = group.iloc[position - window + 1 : position + 1][["stock", "market"]].dropna()
            if len(sample) < window:
                continue
            design = np.column_stack([np.ones(len(sample)), sample["market"].to_numpy(dtype=float)])
            y = sample["stock"].to_numpy(dtype=float)
            coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
            residual = y - design @ coefficients
            if statistic == "std":
                value = residual.std(ddof=1)
            else:
                sigma = residual.std(ddof=1)
                if statistic == "all":
                    mask = np.ones(len(residual), dtype=bool)
                elif statistic == "up":
                    mask = residual > 0
                elif statistic == "down":
                    mask = residual < 0
                elif statistic == "jup":
                    mask = residual >= sigma
                elif statistic == "jdown":
                    mask = residual <= -sigma
                elif statistic == "j":
                    mask = np.abs(residual) >= sigma
                elif statistic == "nj":
                    mask = np.abs(residual) < sigma
                else:
                    raise ValueError(f"unknown CAPM residual statistic: {statistic}")
                value = np.square(residual[mask]).mean() if mask.any() else np.nan
            result[position] = value
        group["result"] = result
        output.append(group[["date", "security_id", "result"]])
    return pd.concat(output).set_index(INDEX_NAMES)["result"].sort_index()


def _rolling_weighted_skew(prices: pd.Series, weights: pd.Series, window: int) -> pd.Series:
    """Arithmetic price mean/sample variance; volume weights only the third moment."""

    if window < 2:
        raise ValueError("weighted skew window must be at least 2")
    frame = pd.concat([prices.rename("prices"), weights.rename("weights")], axis=1).reset_index()
    output: list[pd.DataFrame] = []
    for _, group in frame.groupby("security_id", sort=False):
        group = group.sort_values("date").copy()
        result = np.full(len(group), np.nan)
        p = group["prices"].to_numpy(dtype=float)
        w = group["weights"].to_numpy(dtype=float)
        for position in range(window - 1, len(group)):
            x = p[position - window + 1 : position + 1]
            weight = w[position - window + 1 : position + 1]
            valid = np.isfinite(x) & (x > 0.0) & np.isfinite(weight) & (weight >= 0.0)
            if valid.sum() < window or weight[valid].sum() <= 0:
                continue
            x, weight = x[valid], weight[valid]
            mean = x.mean()
            variance = np.var(x, ddof=1)
            if variance > 0:
                result[position] = np.average(np.power(x - mean, 3), weights=weight) / variance**1.5
        group["result"] = result
        output.append(group[["date", "security_id", "result"]])
    result = pd.concat(output).set_index(INDEX_NAMES)["result"].sort_index()
    return result.where(_fixed_calendar_window_complete(prices, window))


def _rolling_stock_market_comoment(
    stock_returns: pd.Series,
    market_returns: pd.Series,
    window: int,
    market_power: int,
    regime: str,
    minimum_regime_observations: int,
) -> pd.Series:
    """Population co-skewness/co-kurtosis against a market return series.

    Means are estimated from the complete rolling window before the optional
    up/down market mask is applied.  For a selected sample ``S`` the statistic
    is::

        E_S[xc * yc**p] / (sqrt(E_S[xc**2]) * E_S[yc**2]**(p/2))

    where ``p=2`` gives co-skewness and ``p=3`` gives co-kurtosis.  The
    unconditioned means in ``xc`` and ``yc`` match the source definition.
    """

    if window < 2:
        raise ValueError("stock-market comoment window must be at least 2")
    if market_power not in {2, 3}:
        raise ValueError("stock-market comoment power must be 2 or 3")
    if regime not in {"all", "up", "down"}:
        raise ValueError(f"unknown stock-market comoment regime: {regime}")
    if minimum_regime_observations < 2:
        raise ValueError("minimum regime observations must be at least 2")

    market = market_returns.copy()
    market.index = pd.DatetimeIndex(market.index).normalize()
    market = market[~market.index.duplicated(keep="last")].sort_index()
    market_calendar = pd.DatetimeIndex(market.index)
    market_positions = pd.Series(
        np.arange(len(market_calendar), dtype=int),
        index=market_calendar,
    )

    joined = stock_returns.rename("stock").reset_index()
    joined["date"] = pd.to_datetime(joined["date"]).dt.normalize()
    joined = joined.merge(
        market.rename_axis("date").rename("market").reset_index(),
        on="date",
        how="left",
    )
    joined["market_position"] = joined["date"].map(market_positions).fillna(-1).astype(int)

    output: list[pd.DataFrame] = []
    for _, group in joined.groupby("security_id", sort=False):
        group = group.sort_values("date").copy()
        stock_values = group["stock"].to_numpy(dtype=float)
        market_values = group["market"].to_numpy(dtype=float)
        positions = group["market_position"].to_numpy(dtype=int)
        values = np.full(len(group), np.nan)

        for position in range(window - 1, len(group)):
            begin = position - window + 1
            calendar_window = positions[begin : position + 1]
            if (
                calendar_window[0] < 0
                or not np.all(np.diff(calendar_window) == 1)
            ):
                continue
            x = stock_values[begin : position + 1]
            y = market_values[begin : position + 1]
            if not (np.isfinite(x).all() and np.isfinite(y).all()):
                continue

            x_centered = x - x.mean()
            y_centered = y - y.mean()
            # A return theoretically equal to the full-window market mean can
            # acquire an O(machine-epsilon) residual after prices are converted
            # back to returns. Keep that equality observation out of both
            # strict regimes instead of assigning it according to round-off.
            equality_tolerance = (
                np.finfo(float).eps
                * max(1.0, float(np.max(np.abs(y))), abs(float(y.mean())))
                * window
            )
            if regime == "up":
                selected = y_centered > equality_tolerance
            elif regime == "down":
                selected = y_centered < -equality_tolerance
            else:
                selected = np.ones(window, dtype=bool)
            if selected.sum() < minimum_regime_observations:
                continue

            selected_x = x_centered[selected]
            selected_y = y_centered[selected]
            stock_variance = float(np.mean(np.square(selected_x)))
            market_variance = float(np.mean(np.square(selected_y)))
            denominator = np.sqrt(stock_variance) * market_variance ** (market_power / 2.0)
            if not np.isfinite(denominator) or denominator <= 0.0:
                continue
            numerator = float(np.mean(selected_x * np.power(selected_y, market_power)))
            values[position] = numerator / denominator

        group["result"] = values
        output.append(group[["date", "security_id", "result"]])

    if not output:
        return pd.Series(
            dtype=float,
            index=pd.MultiIndex.from_arrays([[], []], names=INDEX_NAMES),
        )
    return pd.concat(output).set_index(INDEX_NAMES)["result"].sort_index()


def _cross_sectional_market_cap_residual(
    values: pd.Series,
    total_market_cap: pd.Series,
    universe: dict[pd.Timestamp, frozenset[str]] | None,
    market_cap_transform: str,
    minimum_samples: int,
) -> pd.Series:
    """Neutralize a daily cross-section against total market capitalization."""

    if market_cap_transform not in {"raw", "log"}:
        raise ValueError(
            "market_cap_transform must be 'raw' or 'log'",
        )
    if minimum_samples < 3:
        raise ValueError("market-cap neutralization minimum_samples must be at least 3")

    cap = total_market_cap.where(total_market_cap > 0.0)
    if market_cap_transform == "log":
        cap = np.log(cap)
    if universe:
        values = apply_universe(values, universe)
        cap = apply_universe(cap, universe)

    combined = pd.concat([
        values.rename("value"),
        cap.rename("total_market_cap"),
    ], axis=1)
    output: list[pd.Series] = []
    for _, group in combined.groupby(level="date", sort=False):
        residual = pd.Series(np.nan, index=group.index, dtype=float)
        clean = group.replace([np.inf, -np.inf], np.nan).dropna()
        if len(clean) >= minimum_samples:
            regressor = clean["total_market_cap"].to_numpy(dtype=float)
            # A scalar rescaling leaves an OLS-with-intercept residual
            # unchanged while avoiding poor conditioning for market caps near
            # 1e12 or larger.
            scale = float(np.median(np.abs(regressor)))
            if np.isfinite(scale) and scale > 0.0:
                design = np.column_stack([
                    np.ones(len(clean)),
                    regressor / scale,
                ])
                if np.linalg.matrix_rank(design) == 2:
                    target = clean["value"].to_numpy(dtype=float)
                    if np.all(target == target[0]):
                        # The intercept fits an exactly constant response;
                        # floating dust must not invent a downstream rank.
                        residual.loc[clean.index] = 0.0
                    else:
                        coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
                        residual.loc[clean.index] = target - design @ coefficients
        output.append(residual)
    if not output:
        return values * np.nan
    return pd.concat(output).sort_index()


def _cross_sectional_utr_score(
    turn: pd.Series,
    str_factor: pd.Series,
    stable_fraction: float,
    rank_method: str,
) -> pd.Series:
    """Combine market-cap-neutral TURN and STR using the report's rank sum."""

    if not 0.0 < stable_fraction < 1.0:
        raise ValueError("UTR stable_fraction must be between 0 and 1")
    if rank_method not in {"average", "min", "max", "first", "dense"}:
        raise ValueError("UTR rank_method is not supported")

    combined = pd.concat([
        turn.rename("turn"),
        str_factor.rename("str"),
    ], axis=1)
    output: list[pd.Series] = []
    for _, group in combined.groupby(level="date", sort=False):
        score = pd.Series(np.nan, index=group.index, dtype=float)
        clean = group.replace([np.inf, -np.inf], np.nan).dropna()
        if clean.empty:
            output.append(score)
            continue

        score1 = clean["str"].rank(method=rank_method, ascending=True)
        str_percentile = clean["str"].rank(
            method=rank_method,
            ascending=True,
            pct=True,
        )
        stable = str_percentile <= stable_fraction

        conditional_turn_score = pd.Series(np.nan, index=clean.index, dtype=float)
        conditional_turn_score.loc[stable] = clean.loc[stable, "turn"].rank(
            method=rank_method,
            ascending=False,
        )
        conditional_turn_score.loc[~stable] = clean.loc[~stable, "turn"].rank(
            method=rank_method,
            ascending=True,
        )
        score.loc[clean.index] = score1 + conditional_turn_score
        output.append(score)

    if not output:
        return turn * np.nan
    return pd.concat(output).sort_index()


def _rolling_percentile_of_last(series: pd.Series, window: int) -> pd.Series:
    def percentile(values: np.ndarray) -> float:
        valid = values[np.isfinite(values)]
        if len(valid) != window:
            return np.nan
        return float((valid <= valid[-1]).mean())

    return rolling(series, window, percentile)


def _rolling_ideal_amplitude(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    window: int,
    tail_fraction: float,
) -> pd.Series:
    """Mean amplitude of high-close days minus that of low-close days.

    The window is counted in valid stock trading observations, not merely rows
    in the market calendar. Equal closing prices at either tail cutoff stay
    in the same group; the groups can contain more than the nominal count.
    """

    if window < 2:
        raise ValueError("ideal_swing: window must be at least 2")
    if not 0 < tail_fraction <= 0.5:
        raise ValueError("ideal_swing: tail_fraction must be in (0, 0.5]")
    tail_count = int(np.floor(window * tail_fraction))
    if tail_count < 1:
        raise ValueError("ideal_swing: tail_fraction selects no observations")

    frame = pd.concat([
        close.rename("close"),
        high.rename("high"),
        low.rename("low"),
    ], axis=1).reset_index()
    output: list[pd.DataFrame] = []
    for _, group in frame.groupby("security_id", sort=False):
        group = group.sort_values("date").copy()
        close_values = group["close"].to_numpy(dtype=float)
        high_values = group["high"].to_numpy(dtype=float)
        low_values = group["low"].to_numpy(dtype=float)
        amplitudes = high_values / low_values - 1.0
        valid = (
            np.isfinite(close_values)
            & np.isfinite(high_values)
            & np.isfinite(low_values)
            & (close_values > 0)
            & (low_values > 0)
            & (high_values >= low_values)
        )
        values = np.full(len(group), np.nan)
        valid_positions: list[int] = []
        for position in range(len(group)):
            if not valid[position]:
                continue
            valid_positions.append(position)
            if len(valid_positions) < window:
                continue
            sample_positions = np.asarray(valid_positions[-window:], dtype=int)
            prices = close_values[sample_positions]
            ordered = np.partition(prices, [tail_count - 1, window - tail_count])
            low_positions = sample_positions[prices <= ordered[tail_count - 1]]
            high_positions = sample_positions[prices >= ordered[window - tail_count]]
            values[position] = float(
                amplitudes[high_positions].mean()
                - amplitudes[low_positions].mean()
            )
        group["result"] = values
        output.append(group[["date", "security_id", "result"]])
    if not output:
        return pd.Series(
            dtype=float,
            index=pd.MultiIndex.from_arrays([[], []], names=INDEX_NAMES),
        )
    return pd.concat(output).set_index(INDEX_NAMES)["result"].sort_index()


def _rolling_ideal_reversal(
    close: pd.Series,
    amount: pd.Series,
    trade_count: pd.Series,
    window: int,
    selection_count: int,
) -> pd.Series:
    """Return high-ticket-day returns less low-ticket-day returns.

    Daily average ticket size is ``amount / trade_count``.  Inside each fixed
    market-trading-date window, the largest and smallest exact groups are
    selected with date-stable tie breaking, and their close-to-close returns
    are summed without compounding.
    """

    if window < 2:
        raise ValueError("ideal_reversal: window must be at least 2")
    if selection_count < 1 or 2 * selection_count > window:
        raise ValueError(
            "ideal_reversal: selection_count must form two non-overlapping groups"
        )

    previous_close = group_shift(close, 1)
    returns = safe_divide(close, previous_close) - 1.0
    average_trade_amount = safe_divide(
        amount.where(amount >= 0.0),
        trade_count.where(trade_count > 0.0),
    )

    def calculate(return_values: np.ndarray, ticket_values: np.ndarray) -> float:
        # rolling_pair supplies observations in ascending date order.  The
        # positional secondary key makes equal ticket sizes deterministic and
        # preserves exactly ``selection_count`` observations in each group.
        positions = np.arange(len(ticket_values))
        order = np.lexsort((positions, ticket_values))
        low_positions = order[:selection_count]
        high_positions = order[-selection_count:]
        return float(
            return_values[high_positions].sum()
            - return_values[low_positions].sum()
        )

    result = rolling_pair(
        returns,
        average_trade_amount,
        window,
        calculate,
        min_periods=window,
    )
    return result.where(_fixed_calendar_window_complete(returns, window))


def _rolling_max_drawdown_duration(
    prices: pd.Series,
    window: int,
) -> pd.Series:
    """Longest consecutive underwater spell in a trailing price window.

    ``window`` counts adjusted closing-price observations.  Therefore the
    current 230-day implementation window is ``t-229,...,t``; the source
    workbook does not independently specify this window length. A close equal to or above the
    running peak is a recovery and resets the duration to zero.
    """

    if window < 2:
        raise ValueError("max_drawdown_duration: window must be at least two observations")

    def calculate(values: np.ndarray) -> float:
        if not np.isfinite(values).all() or (values <= 0.0).any():
            return np.nan
        running_peak = np.maximum.accumulate(values)
        underwater = values < running_peak
        longest = 0
        current = 0
        for is_underwater in underwater:
            current = current + 1 if is_underwater else 0
            longest = max(longest, current)
        return float(longest)

    result = rolling(prices, window, calculate)
    return result.where(_fixed_calendar_window_complete(prices, window))


def _rolling_ols_slope(
    dependent: pd.Series,
    independent: pd.Series,
    window: int,
) -> pd.Series:
    """Per-security rolling OLS slope with an explicit intercept."""

    if window < 3:
        raise ValueError("rolling OLS window must be at least three observations")

    def calculate(y: np.ndarray, x: np.ndarray) -> float:
        # Center before fitting: an intercept makes the slope translation
        # invariant, whereas rank([1, native-CNY x]) depends on its units.
        centered_x = x - x[0]
        centered_x -= centered_x.mean()
        scale = np.max(np.abs(centered_x))
        if scale == 0:
            return np.nan
        scaled_x = centered_x / scale
        centered_y = y - y[0]
        centered_y -= centered_y.mean()
        return float(np.dot(scaled_x, centered_y) / np.dot(scaled_x, scaled_x) / scale)

    result = rolling_pair(
        dependent,
        independent,
        window,
        calculate,
        min_periods=window,
    )
    return result.where(_fixed_calendar_window_complete(dependent, window))


def _confirmed_turnover_context(context: FactorContext, spec: FactorSpec) -> FactorContext:
    """Use the same inputs and construction population as the frozen history."""
    policy = spec.parameters.get("history_policy")
    if policy not in {"confirmed_suspension_log_cap_v1", "confirmed_suspension_turnover_ratio_v1"}:
        return context
    if spec.parameters.get("historical_st_policy") and not context.extras.get("requires_st_exclusion"):
        raise ValueError("Current turnover policy requires historical ST status for construction and output populations")
    master = context.extras.get("security_master")
    suspensions = context.extras.get("confirmed_full_day_suspensions")
    if master is None or suspensions is None:
        raise ValueError("Canonical turnover requires security_master and confirmed_full_day_suspensions inputs")
    master = master[["security_id", "list_date", "delist_date"]].drop_duplicates().copy()
    if master.security_id.duplicated().any():
        raise ValueError("Conflicting security lifecycle records")
    code = master.security_id.astype(str)
    master = master[code.str.match(r"^(?:[03]\d{5}\.SZ|6\d{5}\.SH)$")]
    master = master.set_index("security_id")
    daily = context.daily.reset_index()
    if daily.duplicated(INDEX_NAMES).any():
        raise ValueError("Duplicate turnover source stock/date")
    dates = pd.DatetimeIndex(daily.date.unique()).sort_values()
    columns = pd.Index(sorted(set(daily.security_id) & set(master.index)), name="security_id")
    turn = daily.pivot(index="date", columns="security_id", values="turnover_rate").reindex(index=dates, columns=columns)
    turn = turn.where(turn.ge(0)) * float(spec.parameters["turnover_input_multiplier"])
    cap = (
        daily.pivot(index="date", columns="security_id", values="total_market_cap").reindex(index=dates, columns=columns)
        if policy == "confirmed_suspension_log_cap_v1"
        else pd.DataFrame(np.nan, index=dates, columns=columns)
    )
    required = {"date", "security_id"}
    if not required.issubset(suspensions.columns):
        raise ValueError("Confirmed suspension input needs normalized date and security_id")
    ri = dates.get_indexer(pd.to_datetime(suspensions.date))
    ci = columns.get_indexer(suspensions.security_id)
    valid = (ri >= 0) & (ci >= 0)
    mask = np.zeros(turn.shape, dtype=bool)
    mask[ri[valid], ci[valid]] = True
    if "volume" in daily:
        volume = daily.pivot(index="date", columns="security_id", values="volume").reindex(index=dates, columns=columns)
        # A traded quote contradicts an all-day halt. It cannot authorize
        # substituting zero turnover or carrying market capitalization.
        mask &= ~volume.gt(0).to_numpy()
    values = turn.to_numpy(copy=True)
    values[mask & ~np.isfinite(values)] = 0.0
    turn = pd.DataFrame(values, index=dates, columns=columns)
    values = cap.to_numpy(copy=True)
    missing_cap = mask & ~np.isfinite(values)
    values[missing_cap] = cap.ffill().to_numpy()[missing_cap]
    cap = pd.DataFrame(values, index=dates, columns=columns)
    turn.index.name = cap.index.name = "date"
    listed = pd.to_datetime(master.reindex(columns).list_date).to_numpy("datetime64[ns]")
    delisted = pd.to_datetime(master.reindex(columns).delist_date).to_numpy("datetime64[ns]")
    population = {
        day: frozenset(columns[(listed <= day.to_datetime64()) & (np.isnat(delisted) | (delisted > day.to_datetime64()))])
        for day in dates
    }
    if context.extras.get("requires_st_exclusion"):
        from factor_engine.universe import historical_st_eligible
        population = {day: historical_st_eligible(context, day, codes) for day, codes in population.items()}
    # pandas2.x stack defaults drop an all-unknown market date; explicitly
    # restore the input axis so unknown sessions cannot shorten rolling windows.
    prepared = pd.concat([turn.stack().rename("turnover_rate"), cap.stack().rename("total_market_cap")], axis=1).reindex(
        pd.MultiIndex.from_product([dates, columns], names=INDEX_NAMES),
    ).sort_index()
    return replace(context, daily=prepared, eligible_universe=population)


@registry.register("daily_price")
def compute_daily_price(context: FactorContext, spec: FactorSpec) -> pd.Series:
    factor_id = spec.factor_id
    if not supports_factor_id(factor_id):
        raise NotImplementedError(f"daily_price does not implement {factor_id}")
    if factor_id in DECLARED_CALENDAR_FACTORS:
        context = replace(context, daily=restore_declared_market_dates(context.daily, context.calculation_dates))
    context = _confirmed_turnover_context(context, spec)
    # Several turnover/amount families are deliberately price independent.
    # Do not make their runtime contract wider than the catalog by eagerly
    # reading close or calculating returns before branch selection.
    price_independent = bool(re.fullmatch(
        r"(?:turnover(?:20|60|120|240)|Turn(?:20|60)|STR_(?:20|60)d|"
        r"UTR2?_(?:20|60)d|mfle(?:20|40|60|120)|amtprop1m_quantile1y)",
        factor_id,
    ))
    if price_independent:
        close = None
        returns = None
    else:
        close = _column(context, "close")
        # A return requires adjacent market sessions, including the close just
        # before the rolling window. Missing quote rows must not bridge a gap.
        returns = group_pct_change(close).where(_fixed_calendar_window_complete(close, 2))
    result: pd.Series

    match = re.fullmatch(r"intraday_amp_vol_(1m|3m|6m|1y)", factor_id)
    if match:
        high, low = _column(context, "high"), _column(context, "low")
        amplitude = safe_divide(high - low, group_shift(close, 1))
        amplitude = amplitude.where(_fixed_calendar_window_complete(close, 2))
        window = HORIZON_SUFFIX[match.group(1)]
        result = _fixed_calendar_std(amplitude, window)
    elif (match := re.fullmatch(r"up_shadow_vol_(1m|3m|6m|1y)", factor_id)):
        high, open_price = _column(context, "high"), _column(context, "open")
        body_top = pd.concat([open_price, close], axis=1).max(axis=1)
        window = HORIZON_SUFFIX[match.group(1)]
        shadow = safe_divide(high - body_top, body_top)
        result = _fixed_calendar_std(shadow, window)
    elif (match := re.fullmatch(r"w_down_shadow_vol_(1m|3m|6m|1y)", factor_id)):
        low = _column(context, "low")
        window = HORIZON_SUFFIX[match.group(1)]
        shadow = safe_divide(close - low, low)
        result = _fixed_calendar_std(shadow, window)
    elif (match := re.fullmatch(r"vol_(60d|90d|120d|240d|5y)", factor_id)):
        window = {"60d": 60, "90d": 90, "120d": 120, "240d": 240, "5y": 1200}[match.group(1)]
        result = rolling(returns, window, "std").where(
            _fixed_calendar_window_complete(returns, window),
        )
    elif (match := re.fullmatch(r"vol(60|120|240)_(up|down|jup|jdown|j|nj)", factor_id)):
        result = _rolling_jump_variance(returns, int(match.group(1)), match.group(2))
    elif (match := re.fullmatch(r"res_vol_(20d|60d|120d|240d|5y)", factor_id)):
        window = {"20d": 20, "60d": 60, "120d": 120, "240d": 240, "5y": 1200}[match.group(1)]
        result = _rolling_capm(returns, _market_return(context, spec.benchmark_id), window, "std")
    elif factor_id == "resvol_j_capm":
        result = _rolling_capm(returns, _market_return(context, spec.benchmark_id), 60, "j")
    elif (match := re.fullmatch(r"resvol(60|120|240)_(?:(up|down|jup|jdown|j|nj)_)?capm", factor_id)):
        result = _rolling_capm(
            returns,
            _market_return(context, spec.benchmark_id),
            int(match.group(1)),
            match.group(2) or "all",
        )
    elif (match := re.fullmatch(r"mom(60|120|240)_20", factor_id)):
        long_window = int(match.group(1))
        result = safe_divide(_fixed_calendar_shift(close, 20), _fixed_calendar_shift(close, long_window)) - 1.0
    elif (match := re.fullmatch(r"shift_mom(?:_(60|120)d)?", factor_id)):
        default_window = int(match.group(1)) if match.group(1) else 252
        result = _shift_momentum(
            returns,
            int(spec.parameters.get("long_window", default_window)),
            int(spec.parameters.get("excluded_recent_days", 21)),
        )
    elif factor_id == "cal_momentum_off_limit_1y":
        intraday_return = safe_divide(close, _column(context, "open")) - 1.0
        adjusted = _mask_limit_event_returns(intraday_return, _limit_event_status(context, spec))
        result = _fixed_calendar_rolling_sum(
            adjusted,
            int(spec.parameters.get("window", 240)),
        )
    elif (match := re.fullmatch(r"cal_momentum_off_limit_prod_(1y|6m|3m)", factor_id)):
        default_window = {"1y": 240, "6m": 120, "3m": 60}[match.group(1)]
        adjusted = _mask_limit_event_returns(returns, _limit_event_status(context, spec))
        result = _fixed_calendar_rolling_product(
            adjusted,
            int(spec.parameters.get("window", default_window)),
        )
    elif (match := re.fullmatch(r"Ret(60|120|240)_20_RemoveUpLimit", factor_id)):
        long_window = int(spec.parameters.get("long_window", match.group(1)))
        skip_window = int(spec.parameters.get("skip_window", 20))
        limit_up_field = str(spec.parameters.get("limit_up_field", "is_limit_up"))
        result = _remove_up_limit_momentum(
            returns,
            _unconflicted_limit_status(_column(context, limit_up_field), context),
            long_window,
            skip_window,
        )
    elif factor_id == "cal_momentum_range_1m":
        result = _range_adjusted_momentum(
            close,
            _column(context, "high"),
            _column(context, "low"),
            int(spec.parameters.get("return_window", 20)),
            float(spec.parameters.get("lower_quantile", 0.2)),
            float(spec.parameters.get("upper_quantile", 0.8)),
        )
    elif (match := re.fullmatch(r"ddr_mom_(20|60|120)d", factor_id)):
        window = int(match.group(1))
        result = rolling(
            close,
            window,
            lambda values: float((values[-1] - values[0]) / np.abs(np.diff(values)).sum())
            if np.abs(np.diff(values)).sum() > 0 else np.nan,
        )
        result = result.where(_fixed_calendar_window_complete(close, window))
    elif (match := re.fullmatch(r"cal_momentum_overnight_(1m|3m|6m|1y)", factor_id)):
        open_price = _column(context, "open")
        ratio = safe_divide(open_price.where(open_price > 0), group_shift(close.where(close > 0), 1))
        ratio = ratio.where(_fixed_calendar_window_complete(close, 2))
        overnight_log = np.log(ratio.where(ratio > 0))
        window = HORIZON_SUFFIX[match.group(1)]
        result = rolling(overnight_log, window, "sum", min_periods=window)
        result = result.where(_fixed_calendar_window_complete(overnight_log, window))
    elif factor_id == "high250":
        peak = rolling(close, 250, "max")
        result = safe_divide(close - peak, peak).where(
            _fixed_calendar_window_complete(close, 250),
        )
    elif factor_id == "pos_mom":
        high = _column(context, "high")
        low = _column(context, "low")
        valid = (
            np.isfinite(close) & np.isfinite(high) & np.isfinite(low)
            & low.gt(0.0) & high.ge(low)
            & close.ge(low - 1e-5) & close.le(high + 1e-5)
        )
        historical_low = group_transform(
            low.where(valid),
            lambda values: values.expanding(min_periods=1).min(),
        )
        historical_high = group_transform(
            high.where(valid),
            lambda values: values.expanding(min_periods=1).max(),
        )
        result = safe_divide(close - historical_low, historical_high - historical_low).where(valid)
    elif factor_id == "max_drawdown_duration":
        result = _rolling_max_drawdown_duration(
            close,
            int(spec.parameters.get("window", 230)),
        )
    elif factor_id == "reverse20":
        result = safe_divide(close, _fixed_calendar_shift(close, 20)) - 1.0
    elif factor_id == "ideal_reversal":
        result = _rolling_ideal_reversal(
            close,
            _column(context, "amount"),
            _column(context, "trade_count"),
            int(spec.parameters.get("window", 20)),
            int(spec.parameters.get("selection_count", 10)),
        )
    elif factor_id == "ideal_swing":
        result = _rolling_ideal_amplitude(
            close,
            _column(context, "high"),
            _column(context, "low"),
            int(spec.parameters.get("window", 20)),
            float(spec.parameters.get("tail_fraction", 0.25)),
        )
    elif (match := re.fullmatch(r"weighted_skew_(1m|3m|6m|1y)", factor_id)):
        result = _rolling_weighted_skew(
            close,
            _column(context, "volume"),
            int(spec.parameters.get("window", 21 if match.group(1) == "1m" else HORIZON_SUFFIX[match.group(1)])),
        )
    elif (match := re.fullmatch(r"(csk|ckt)(?:_(up|down))?_120d", factor_id)):
        comoment_returns = group_pct_change(close.where(close > 0.0))
        result = _rolling_stock_market_comoment(
            comoment_returns,
            _market_return(context, spec.benchmark_id),
            int(spec.parameters.get("window", 120)),
            2 if match.group(1) == "csk" else 3,
            match.group(2) or "all",
            int(spec.parameters.get("minimum_regime_observations", 2)),
        )
    elif factor_id == "turnover_price_corr":
        result = rolling_pair(
            close, _column(context, "turnover_rate"), 20, _correlation,
        ).where(_fixed_calendar_window_complete(close, 20))
    elif factor_id == "turnover_return_corr1":
        result = rolling_pair(
            returns, group_shift(_column(context, "turnover_rate"), 1), 20, _correlation,
        ).where(_fixed_calendar_window_complete(returns, 20))
    elif factor_id == "turnoverChg_return_corr1":
        result = rolling_pair(
            returns, group_diff(_column(context, "turnover_rate")), 20, _correlation,
        ).where(_fixed_calendar_window_complete(returns, 20))
    elif factor_id == "turnoverChg_return_corr2":
        result = rolling_pair(
            group_shift(returns, 1), group_diff(_column(context, "turnover_rate")),
            20, _correlation,
        ).where(_fixed_calendar_window_complete(returns, 20))
    elif (match := re.fullmatch(r"volume_price_corr_(1m|3m|6m|1y)", factor_id)):
        window = int(spec.parameters.get("window", 21 if match.group(1) == "1m" else HORIZON_SUFFIX[match.group(1)]))
        def rank_correlation(prices: np.ndarray, volumes: np.ndarray) -> float:
            return _correlation(
                pd.Series(prices).rank(method="average").to_numpy(),
                pd.Series(volumes).rank(method="average").to_numpy(),
            )
        result = rolling_pair(close, _column(context, "volume"), window, rank_correlation).where(
            _fixed_calendar_window_complete(close, window),
        )
    elif (match := re.fullmatch(r"turnover(20|60|120|240)", factor_id)):
        window = int(match.group(1))
        turnover = _column(context, "turnover_rate")
        result = rolling(turnover, window, "mean").where(
            _fixed_calendar_window_complete(turnover, window),
        )
        output_unit = spec.parameters.get("output_unit", "ratio")
        if output_unit == "percent points":
            result = result * 100.0
        elif output_unit != "ratio":
            raise ValueError(f"{factor_id}: unknown output_unit {output_unit!r}")
    elif (match := re.fullmatch(r"Turn(20|60)", factor_id)):
        window = int(spec.parameters.get("window", match.group(1)))
        turnover = _column(context, "turnover_rate")
        turnover = turnover.where(turnover >= 0.0)
        turnover_mean = rolling(turnover, window, "mean").where(
            _fixed_calendar_window_complete(turnover, window),
        )
        construction_universe = context.eligible_universe or context.universe
        result = _cross_sectional_market_cap_residual(
            turnover_mean,
            _column(context, "total_market_cap"),
            dict(construction_universe) if construction_universe else None,
            str(spec.parameters.get("market_cap_transform", "raw")),
            int(spec.parameters.get("minimum_regression_samples", 3)),
        )
    elif (match := re.fullmatch(r"STR_(20|60)d", factor_id)):
        turnover = _column(context, "turnover_rate")
        turnover = turnover.where(turnover >= 0.0)
        window = int(spec.parameters.get("window", match.group(1)))
        ddof = int(spec.parameters.get("rolling_std_ddof", 1))
        if ddof not in {0, 1}:
            raise ValueError("STR rolling_std_ddof must be 0 or 1")
        turnover_std = _fixed_calendar_std(turnover, window, ddof=ddof, fresh_window=True)
        # The standalone workbook STR is mean / std. UTR's internal STR
        # residual is a separate component and is calculated in its own branch.
        turnover_mean = rolling(turnover, window, "mean").where(
            _fixed_calendar_window_complete(turnover, window),
        )
        result = safe_divide(turnover_mean, turnover_std.where(turnover_std.gt(0)))
    elif (match := re.fullmatch(r"UTR_(20|60)d", factor_id)):
        window = int(spec.parameters.get("window", match.group(1)))
        ddof = int(spec.parameters.get("rolling_std_ddof", 1))
        if ddof not in {0, 1}:
            raise ValueError("UTR rolling_std_ddof must be 0 or 1")
        turnover = _column(context, "turnover_rate")
        turnover = turnover.where(turnover >= 0.0)
        turnover_mean = rolling(turnover, window, "mean").where(
            _fixed_calendar_window_complete(turnover, window),
        )
        turnover_std = _fixed_calendar_std(turnover, window, ddof=ddof, fresh_window=True)
        construction_universe = context.eligible_universe or context.universe
        universe = dict(construction_universe) if construction_universe else None
        total_market_cap = _column(context, "total_market_cap")
        market_cap_transform = str(spec.parameters.get("market_cap_transform", "raw"))
        minimum_samples = int(spec.parameters.get("minimum_regression_samples", 3))
        turn = _cross_sectional_market_cap_residual(
            turnover_mean,
            total_market_cap,
            universe,
            market_cap_transform,
            minimum_samples,
        )
        str_factor = _cross_sectional_market_cap_residual(
            turnover_std,
            total_market_cap,
            universe,
            market_cap_transform,
            minimum_samples,
        )
        result = _cross_sectional_utr_score(
            turn,
            str_factor,
            float(spec.parameters.get("stable_fraction", 0.5)),
            str(spec.parameters.get("rank_method", "average")),
        )
    elif (match := re.fullmatch(r"UTR2_(20|60)d", factor_id)):
        window = int(spec.parameters.get("window", match.group(1)))
        ddof = int(spec.parameters.get("rolling_std_ddof", 1))
        if ddof not in {0, 1}:
            raise ValueError("UTR2 rolling_std_ddof must be 0 or 1")
        turnover = _column(context, "turnover_rate")
        turnover = turnover.where(turnover >= 0.0)
        turnover_mean = rolling(turnover, window, "mean").where(
            _fixed_calendar_window_complete(turnover, window),
        )
        turnover_std = _fixed_calendar_std(turnover, window, ddof=ddof, fresh_window=True)
        construction_universe = context.eligible_universe or context.universe
        universe = dict(construction_universe) if construction_universe else None
        total_market_cap = _column(context, "total_market_cap")
        market_cap_transform = str(spec.parameters.get("market_cap_transform", "raw"))
        minimum_samples = int(spec.parameters.get("minimum_regression_samples", 3))
        turn = _cross_sectional_market_cap_residual(
            turnover_mean,
            total_market_cap,
            universe,
            market_cap_transform,
            minimum_samples,
        )
        str_factor = _cross_sectional_market_cap_residual(
            turnover_std,
            total_market_cap,
            universe,
            market_cap_transform,
            minimum_samples,
        )
        result = str_factor + str_factor / (1.0 + str_factor.abs()) * turn
    elif (match := re.fullmatch(r"Amihud_ILLIQ(20|60|120|240)", factor_id)):
        illiquidity = safe_divide(returns.abs(), _column(context, "amount"))
        result = rolling(illiquidity, int(match.group(1)), "mean")
    elif (match := re.fullmatch(r"mfle(20|40|60|120)", factor_id)):
        window = int(spec.parameters.get("window", match.group(1)))
        float_market_cap = _column(context, "float_market_cap")
        float_market_cap_change = group_diff(float_market_cap).where(
            _fixed_calendar_window_complete(float_market_cap, 2),
        )
        main_fund_flow = (
            _column(context, "large_active_amount_diff")
            + _column(context, "institution_active_amount_diff")
        )
        result = _rolling_ols_slope(
            float_market_cap_change,
            main_fund_flow,
            window,
        )
    elif factor_id == "amtprop1m_quantile1y":
        amount = _column(context, "amount")
        # The catalog denominator is the full market's daily amount.  The
        # output/construction eligibility filter must not redefine that market
        # aggregate by excluding young, ST, suspended, or delisting stocks.
        market_total = amount.groupby(level="date").sum(min_count=1).reindex(
            amount.index.get_level_values("date"),
        )
        market_total.index = amount.index
        proportion = safe_divide(amount, market_total)
        monthly_proportion = rolling(proportion, 20, "mean").where(
            _fixed_calendar_window_complete(proportion, 20),
        )
        result = _rolling_percentile_of_last(monthly_proportion, 240).where(
            _fixed_calendar_window_complete(monthly_proportion, 240),
        )
    elif factor_id == "winrate_weekly":
        # Missing stock rows must not pull observations from before the five
        # market-session window into either count. Keep missing returns out of
        # the denominator, including the return immediately after a quote gap.
        calendar = pd.DatetimeIndex(returns.index.get_level_values("date")).unique()
        calendar = calendar.union(pd.DatetimeIndex(context.calculation_dates)).sort_values()
        panel = returns.unstack("security_id").reindex(calendar)
        positive_days = panel.gt(0).rolling(5, min_periods=1).sum()
        valid_days = panel.notna().rolling(5, min_periods=1).sum()
        result = (positive_days / valid_days.where(valid_days.gt(0))).stack().reindex(returns.index)
    else:
        raise NotImplementedError(f"daily_price does not implement {factor_id}")

    result.name = factor_id
    return result.replace([np.inf, -np.inf], np.nan).sort_index()


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or left.std(ddof=1) == 0 or right.std(ddof=1) == 0:
        return np.nan
    return float(np.corrcoef(left, right)[0, 1])
