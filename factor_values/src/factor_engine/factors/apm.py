from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from factor_engine.catalog import FactorSpec
from factor_engine.data.base import FactorContext
from factor_engine.factors.common import INDEX_NAMES
from factor_engine.registry import registry


SUPPORTED_EXACT = frozenset({"apm"})
SESSION_BOUNDS = {
    "morning": ("09:30", "11:30"),
    "afternoon": ("13:00", "15:00"),
}


def supports_factor_id(factor_id: str) -> bool:
    return factor_id in SUPPORTED_EXACT


def _empty_series(name: str = "value") -> pd.Series:
    index = pd.MultiIndex.from_arrays([[], []], names=INDEX_NAMES)
    return pd.Series(dtype=float, index=index, name=name)


def _infer_bar_minutes(frame: pd.DataFrame, id_column: str) -> int:
    if "bar_minutes" in frame:
        stated = pd.to_numeric(frame["bar_minutes"], errors="coerce").dropna().astype(int).unique()
        if len(stated) == 1 and stated[0] in {30, 60}:
            return int(stated[0])
        if len(stated) > 1:
            raise ValueError("APM intraday data mixes multiple bar_minutes values")

    sample = frame[[id_column, "date", "timestamp"]].copy()
    clock = sample["timestamp"].dt.hour * 60 + sample["timestamp"].dt.minute
    sample["session"] = np.select(
        [(clock >= 9 * 60 + 30) & (clock <= 11 * 60 + 30),
         (clock >= 13 * 60) & (clock <= 15 * 60)],
        ["morning", "afternoon"],
        default="outside",
    )
    sample = sample[sample["session"].ne("outside")]
    differences = (
        sample.sort_values([id_column, "date", "session", "timestamp"])
        .groupby([id_column, "date", "session"], sort=False)["timestamp"]
        .diff()
        .dt.total_seconds()
        .div(60)
    )
    candidates = differences[differences.isin([30.0, 60.0])]
    if candidates.empty:
        raise ValueError(
            "cannot infer whether APM input uses 30-minute or 60-minute bars; "
            "provide a canonical bar_minutes column or source_bar_minutes parameter"
        )
    return int(candidates.mode().iloc[0])


def _expected_clocks(session: str, bar_minutes: int, timestamp_convention: str) -> tuple[str, ...]:
    if timestamp_convention not in {"start", "end"}:
        raise ValueError("bar_timestamp_convention must be 'start' or 'end'")
    start_text, end_text = SESSION_BOUNDS[session]
    start = pd.Timestamp(f"2000-01-01 {start_text}")
    end = pd.Timestamp(f"2000-01-01 {end_text}")
    if timestamp_convention == "end":
        points = pd.date_range(start + pd.Timedelta(minutes=bar_minutes), end, freq=f"{bar_minutes}min")
    else:
        points = pd.date_range(start, end - pd.Timedelta(minutes=bar_minutes), freq=f"{bar_minutes}min")
    return tuple(points.strftime("%H:%M"))


def aggregate_adjusted_bars_to_120m(
    frame: pd.DataFrame,
    *,
    id_column: str,
    source_bar_minutes: int | None = None,
    timestamp_convention: str = "end",
) -> pd.DataFrame:
    """Aggregate complete 30/60-minute adjusted OHLC bars into session 120m bars.

    Morning and afternoon are aggregated independently so the lunch break can
    never be treated as part of a continuous 120-minute bar.  A session is
    emitted only when every expected source bar is present.
    """

    required = {"timestamp", id_column, "open", "high", "low", "close"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"APM intraday data is missing canonical columns: {sorted(missing)}")
    prepared = frame.copy()
    prepared["timestamp"] = pd.to_datetime(prepared["timestamp"])
    prepared["date"] = prepared["timestamp"].dt.normalize()
    prepared[id_column] = prepared[id_column].astype(str)
    prepared = prepared.sort_values([id_column, "timestamp"]).drop_duplicates(
        [id_column, "timestamp"], keep="last",
    )
    minutes = source_bar_minutes or _infer_bar_minutes(prepared, id_column)
    if minutes not in {30, 60}:
        raise ValueError("APM source_bar_minutes must be 30 or 60")

    prepared["clock"] = prepared["timestamp"].dt.strftime("%H:%M")
    output: list[dict[str, object]] = []
    for session in ("morning", "afternoon"):
        expected = _expected_clocks(session, minutes, timestamp_convention)
        sample = prepared[prepared["clock"].isin(expected)]
        for (identifier, trade_date), group in sample.groupby([id_column, "date"], sort=False):
            group = group.sort_values("timestamp")
            if tuple(group["clock"]) != expected:
                continue
            prices = group[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
            if prices.isna().any().any():
                continue
            row: dict[str, object] = {
                id_column: str(identifier),
                "date": pd.Timestamp(trade_date).normalize(),
                "session": session,
                "open": float(prices["open"].iloc[0]),
                "high": float(prices["high"].max()),
                "low": float(prices["low"].min()),
                "close": float(prices["close"].iloc[-1]),
                "source_bar_minutes": minutes,
                "source_bar_count": len(group),
            }
            for additive in ("volume", "amount"):
                if additive in group:
                    values = pd.to_numeric(group[additive], errors="coerce")
                    row[additive] = float(values.sum(min_count=len(group)))
            output.append(row)
    columns = [
        id_column, "date", "session", "open", "high", "low", "close",
        "source_bar_minutes", "source_bar_count",
    ]
    if not output:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(output).sort_values([id_column, "date", "session"]).reset_index(drop=True)


def _session_prices(bars: pd.DataFrame, id_column: str) -> pd.DataFrame:
    if bars.empty:
        return pd.DataFrame()
    morning = (
        bars[bars["session"].eq("morning")]
        .set_index(["date", id_column])[["open"]]
        .rename(columns={"open": "morning_open"})
    )
    afternoon = (
        bars[bars["session"].eq("afternoon")]
        .set_index(["date", id_column])[["open", "close"]]
        .rename(columns={"open": "afternoon_open", "close": "afternoon_close"})
    )
    joined = morning.join(afternoon, how="outer").sort_index()
    joined["previous_close"] = joined.groupby(level=id_column, sort=False)["afternoon_close"].shift(1)
    joined["overnight_return"] = joined["morning_open"].divide(joined["previous_close"]) - 1.0
    joined["afternoon_return"] = joined["afternoon_close"].divide(joined["afternoon_open"]) - 1.0
    return joined.replace([np.inf, -np.inf], np.nan)


def _index_intraday(context: FactorContext, benchmark_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    stock_frame = context.intraday.copy()
    index_frame = context.index_intraday.copy()
    if not index_frame.empty:
        id_column = "index_id" if "index_id" in index_frame else "security_id"
        if id_column not in index_frame:
            raise ValueError("index_intraday must contain index_id or security_id")
        index_frame = index_frame[index_frame[id_column].astype(str).eq(benchmark_id)].copy()
        if id_column != "index_id":
            index_frame = index_frame.rename(columns={id_column: "index_id"})
        return stock_frame, index_frame

    if "security_id" not in stock_frame:
        raise ValueError("intraday data must contain security_id")
    benchmark_mask = stock_frame["security_id"].astype(str).eq(benchmark_id)
    if not benchmark_mask.any():
        raise ValueError(
            f"APM requires intraday benchmark OHLC for {benchmark_id}; provide "
            "FactorContext.index_intraday or include the benchmark in intraday"
        )
    index_frame = stock_frame.loc[benchmark_mask].rename(columns={"security_id": "index_id"})
    return stock_frame.loc[~benchmark_mask].copy(), index_frame


def _rolling_statistic(
    stock_prices: pd.DataFrame,
    benchmark_prices: pd.DataFrame,
    window: int,
    ddof: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stock_overnight = stock_prices["overnight_return"].unstack("security_id")
    stock_afternoon = stock_prices["afternoon_return"].unstack("security_id")
    stock_close = stock_prices["afternoon_close"].unstack("security_id")

    benchmark = benchmark_prices.reset_index("index_id", drop=True)
    if benchmark.index.has_duplicates:
        raise ValueError("APM benchmark has duplicate 120-minute sessions for a date")
    dates = benchmark.index.sort_values().unique()
    stock_overnight = stock_overnight.reindex(dates)
    stock_afternoon = stock_afternoon.reindex(dates)
    stock_close = stock_close.reindex(dates)
    market_overnight = benchmark["overnight_return"].reindex(dates)
    market_afternoon = benchmark["afternoon_return"].reindex(dates)

    rolling_kwargs = {"window": window, "min_periods": window}
    observation_count = 2.0 * window
    sum_x = (market_overnight + market_afternoon).rolling(**rolling_kwargs).sum()
    sum_x2 = (market_overnight.pow(2) + market_afternoon.pow(2)).rolling(**rolling_kwargs).sum()
    sum_y = (stock_overnight + stock_afternoon).rolling(**rolling_kwargs).sum()
    sum_xy = (
        stock_overnight.mul(market_overnight, axis=0)
        + stock_afternoon.mul(market_afternoon, axis=0)
    ).rolling(**rolling_kwargs).sum()
    denominator = sum_x2 - sum_x.pow(2).div(observation_count)
    beta = (
        sum_xy - sum_y.mul(sum_x.div(observation_count), axis=0)
    ).divide(denominator, axis=0)
    beta = beta.where(denominator.abs().gt(np.finfo(float).eps), np.nan)

    raw_difference = stock_overnight - stock_afternoon
    market_difference = market_overnight - market_afternoon
    sum_d = raw_difference.rolling(**rolling_kwargs).sum()
    sum_d2 = raw_difference.pow(2).rolling(**rolling_kwargs).sum()
    sum_dx = market_difference.rolling(**rolling_kwargs).sum()
    sum_dx2 = market_difference.pow(2).rolling(**rolling_kwargs).sum()
    sum_d_dx = raw_difference.mul(market_difference, axis=0).rolling(**rolling_kwargs).sum()

    mean_delta = (sum_d - beta.mul(sum_dx, axis=0)).div(window)
    squared_sum = (
        sum_d2
        - 2.0 * beta * sum_d_dx
        + beta.pow(2).mul(sum_dx2, axis=0)
        - window * mean_delta.pow(2)
    ).clip(lower=0.0)
    divisor = window - ddof
    if divisor <= 0:
        raise ValueError("APM stat_ddof must be smaller than the rolling window")
    delta_std = np.sqrt(squared_sum / divisor)
    statistic = mean_delta.divide(delta_std / np.sqrt(window))
    complete = (
        stock_overnight.notna() & stock_afternoon.notna()
    ).rolling(**rolling_kwargs).sum().eq(window)
    statistic = statistic.where(complete & delta_std.gt(0.0))
    ret20 = stock_close.divide(stock_close.shift(window)) - 1.0
    return statistic.replace([np.inf, -np.inf], np.nan), ret20.replace([np.inf, -np.inf], np.nan)


def _allowed_for_date(
    universe: Mapping[pd.Timestamp, frozenset[str]],
    trade_date: pd.Timestamp,
    available: pd.Index,
) -> pd.Index:
    if not universe:
        return available
    normalized = pd.Timestamp(trade_date).normalize()
    allowed = universe.get(normalized, frozenset())
    return available.intersection(pd.Index(sorted(allowed)))


def _momentum_residual(
    statistic: pd.DataFrame,
    ret20: pd.DataFrame,
    context: FactorContext,
    *,
    include_intercept: bool,
    min_cross_section: int,
) -> pd.Series:
    requested = pd.DatetimeIndex(context.calculation_dates).normalize()
    dates = statistic.index if requested.empty else statistic.index.intersection(requested)
    output: list[pd.Series] = []
    universe = context.eligible_universe or context.universe
    for trade_date in dates:
        pair = pd.concat(
            [statistic.loc[trade_date].rename("stat"), ret20.loc[trade_date].rename("ret20")],
            axis=1,
        )
        allowed = _allowed_for_date(universe, trade_date, pair.index)
        pair = pair.loc[allowed].dropna()
        parameter_count = 2 if include_intercept else 1
        if len(pair) < max(min_cross_section, parameter_count + 1):
            continue
        x = pair["ret20"].to_numpy(dtype=float)
        y = pair["stat"].to_numpy(dtype=float)
        design = np.column_stack([np.ones(len(pair)), x]) if include_intercept else x[:, None]
        if np.linalg.matrix_rank(design) < design.shape[1]:
            continue
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        residual = pd.Series(y - design @ coefficients, index=pair.index, dtype=float)
        residual.index = pd.MultiIndex.from_product(
            [[pd.Timestamp(trade_date).normalize()], residual.index], names=INDEX_NAMES,
        )
        output.append(residual)
    return pd.concat(output).sort_index() if output else _empty_series()


@registry.register("apm")
def compute_apm(context: FactorContext, spec: FactorSpec) -> pd.Series:
    """Compute the overnight-versus-afternoon APM_NEW factor without look-ahead."""

    if not supports_factor_id(spec.factor_id):
        raise NotImplementedError(f"apm does not implement {spec.factor_id}")
    if spec.parameters.get("variant", "apm_new") != "apm_new":
        raise ValueError("the catalog APM implementation currently supports only apm_new")
    benchmark_id = spec.benchmark_id
    if not benchmark_id:
        raise ValueError("APM requires benchmark_id")

    configured_minutes = spec.parameters.get("source_bar_minutes", "auto")
    source_minutes = None if configured_minutes == "auto" else int(configured_minutes)
    timestamp_convention = str(spec.parameters.get("bar_timestamp_convention", "end"))
    window = int(spec.parameters.get("window", 20))
    ddof = int(spec.parameters.get("stat_ddof", 1))
    stock_intraday, index_intraday = _index_intraday(context, benchmark_id)
    stock_bars = aggregate_adjusted_bars_to_120m(
        stock_intraday,
        id_column="security_id",
        source_bar_minutes=source_minutes,
        timestamp_convention=timestamp_convention,
    )
    index_bars = aggregate_adjusted_bars_to_120m(
        index_intraday,
        id_column="index_id",
        source_bar_minutes=source_minutes,
        timestamp_convention=timestamp_convention,
    )
    stock_prices = _session_prices(stock_bars, "security_id")
    index_prices = _session_prices(index_bars, "index_id")
    if stock_prices.empty or index_prices.empty:
        return _empty_series(spec.factor_id)

    statistic, ret20 = _rolling_statistic(stock_prices, index_prices, window, ddof)
    result = _momentum_residual(
        statistic,
        ret20,
        context,
        include_intercept=bool(spec.parameters.get("cross_section_intercept", False)),
        min_cross_section=int(spec.parameters.get("min_cross_section", 2)),
    )
    result.name = spec.factor_id
    return result.replace([np.inf, -np.inf], np.nan).sort_index()
