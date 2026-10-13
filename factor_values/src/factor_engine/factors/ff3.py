from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import re

import numpy as np
import pandas as pd

from factor_engine.catalog import FactorSpec
from factor_engine.config import FF3Config
from factor_engine.data.base import FactorContext
from factor_engine.factors.common import INDEX_NAMES, group_pct_change, group_shift, require_daily, safe_divide
from factor_engine.registry import registry


RESIDUAL_PATTERN = re.compile(
    r"^resvol(60|120|240)(?:_(up|down|jup|jdown|j|nj|jup_ratio|jdown_ratio))?_ff3$",
)
CONDITIONED_PATTERN = re.compile(r"^resvol60_ff3_(lir|tvr|amp|iv)$")
PORTFOLIOS = ("SL", "SM", "SH", "BL", "BM", "BH")


def supports_factor_id(factor_id: str) -> bool:
    return bool(RESIDUAL_PATTERN.fullmatch(factor_id) or CONDITIONED_PATTERN.fullmatch(factor_id))


def _canonical_halt_context(context: FactorContext, spec: FactorSpec) -> FactorContext:
    if spec.parameters.get("history_policy") != "confirmed_full_day_halt_ff3_v1":
        return context
    key = "canonical_ff3_halt_context"
    cached = context.cache.get(key)
    if isinstance(cached, FactorContext):
        return cached
    if context.cache.get("canonical_ff3_halt_prepared") is True:
        return context
    master = context.extras.get("security_master")
    halts = context.extras.get("confirmed_full_day_suspensions")
    if master is None or halts is None:
        raise ValueError("Canonical FF3 requires security_master and confirmed_full_day_suspensions")
    require_daily(context.daily, ["close", "volume"])
    if context.daily.index.duplicated().any():
        raise ValueError("Duplicate FF3 stock/date observations")
    market = context.index_daily.reset_index() if "date" not in context.index_daily else context.index_daily
    if "date" not in market:
        raise ValueError("Canonical FF3 requires a dated benchmark calendar")
    if "index_id" in market:
        market = market[market.index_id.eq(spec.benchmark_id or context.ff3_config.market_index_id)]
    observed_dates = context.daily.index.get_level_values("date")
    dates = pd.DatetimeIndex(pd.to_datetime(market.date).unique()).sort_values()
    dates = dates[(dates >= observed_dates.min()) & (dates <= observed_dates.max())]
    master = master[master.security_id.astype(str).str.match(r"^(?:[03]\d{5}\.SZ|6\d{5}\.SH)$")].copy()
    codes = sorted(set(context.daily.index.get_level_values("security_id")) & set(master.security_id))
    daily = context.daily.reindex(pd.MultiIndex.from_product([dates, codes], names=INDEX_NAMES)).copy()
    from factor_engine.factors.event_price import confirmed_halt_historical_close
    normalized_halts = halts[["date", "security_id"]].copy()
    normalized_halts["date"] = pd.to_datetime(normalized_halts.date).dt.normalize()
    daily["close"], _ = confirmed_halt_historical_close(daily, normalized_halts, master)
    halt_index = pd.MultiIndex.from_frame(normalized_halts.drop_duplicates())
    halt = pd.Series(daily.index.isin(halt_index), index=daily.index) & ~daily.volume.gt(0)
    lifecycle = master.set_index("security_id").reindex(daily.index.get_level_values("security_id"))
    listed = pd.to_datetime(lifecycle.list_date).to_numpy()
    delisted = pd.to_datetime(lifecycle.delist_date).to_numpy()
    day_values = daily.index.get_level_values("date").to_numpy()
    halt &= (day_values >= listed) & (np.isnat(delisted) | (day_values < delisted))
    if "turnover_rate" in daily:
        daily["turnover_rate"] = daily.turnover_rate.mask(halt & daily.turnover_rate.isna(), 0.)
    if {"high", "low"}.issubset(daily.columns):
        amplitude = safe_divide(daily.high - daily.low, group_shift(daily.close, 1))
        daily["ff3_amplitude_signal"] = amplitude.mask(halt & amplitude.isna(), 0.)
    prepared = replace(context, daily=daily, cache={"canonical_ff3_halt_prepared": True})
    context.cache[key] = prepared
    return prepared


def _empty_series(name: str | None = None) -> pd.Series:
    index = pd.MultiIndex.from_arrays([[], []], names=INDEX_NAMES)
    return pd.Series(index=index, dtype=float, name=name)


def _eligible_mask(
    index: pd.MultiIndex,
    universe: Mapping[pd.Timestamp, frozenset[str]],
) -> np.ndarray:
    if not universe:
        return np.ones(len(index), dtype=bool)
    dates = index.get_level_values("date").normalize()
    securities = index.get_level_values("security_id").astype(str)
    return np.fromiter(
        (security in universe.get(date, frozenset()) for date, security in zip(dates, securities)),
        dtype=bool,
        count=len(index),
    )


def _market_return(context: FactorContext, benchmark_id: str, risk_free_daily: float) -> pd.DataFrame:
    frame = context.index_daily.copy()
    if frame.empty:
        raise ValueError("index_daily is required to construct the FF3 market factor")
    if isinstance(frame.index, pd.MultiIndex) or frame.index.name == "date":
        frame = frame.reset_index()
    missing = {"date", "close"} - set(frame.columns)
    if missing:
        raise ValueError(f"index_daily is missing columns: {sorted(missing)}")
    if "index_id" in frame.columns:
        frame = frame[frame["index_id"].astype(str).eq(benchmark_id)]
    if frame.empty:
        raise ValueError(f"index_daily has no rows for FF3 market index {benchmark_id}")
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    close = (
        frame.sort_values("date")
        .drop_duplicates("date", keep="last")
        .set_index("date")["close"]
        .astype(float)
    )
    market = close.pct_change(fill_method=None)
    return pd.DataFrame({"mkt": market - risk_free_daily, "rf": risk_free_daily}, index=market.index)


def _size_characteristic(daily: pd.DataFrame) -> pd.Series:
    if "size" in daily.columns:
        return daily["size"].astype(float)
    if "total_market_cap" not in daily.columns:
        raise ValueError("FF3 requires canonical 'size' or 'total_market_cap'")
    market_cap = daily["total_market_cap"].astype(float)
    return np.log(market_cap.where(market_cap > 0))


def _book_to_price_characteristic(daily: pd.DataFrame) -> pd.Series:
    if "book_to_price" in daily.columns:
        result = daily["book_to_price"].astype(float)
    elif "pb_lf" in daily.columns:
        result = safe_divide(pd.Series(1.0, index=daily.index), daily["pb_lf"])
    elif "pb" in daily.columns:
        result = safe_divide(pd.Series(1.0, index=daily.index), daily["pb"])
    elif {"book_equity", "total_market_cap"} <= set(daily.columns):
        result = safe_divide(daily["book_equity"], daily["total_market_cap"])
    else:
        raise ValueError(
            "FF3 requires canonical 'book_to_price', 'pb_lf', 'pb', or "
            "both 'book_equity' and 'total_market_cap'",
        )
    return result.where(result > 0)


def _weight_series(daily: pd.DataFrame, weighting: str) -> pd.Series:
    if weighting in {"total_market_cap", "float_market_cap"}:
        if weighting not in daily.columns:
            raise ValueError(f"FF3 {weighting} weighting requires canonical column {weighting!r}")
        result = daily[weighting].astype(float)
    elif weighting == "free_float_market_cap":
        if weighting in daily.columns:
            result = daily[weighting].astype(float)
        elif {"raw_close", "free_float_shares"} <= set(daily.columns):
            result = daily["raw_close"].astype(float) * daily["free_float_shares"].astype(float)
        else:
            raise ValueError(
                "FF3 free-float weighting requires 'free_float_market_cap' or "
                "both 'raw_close' and 'free_float_shares'",
            )
    else:
        raise ValueError(f"unsupported FF3 weighting: {weighting}")
    return result.where(result > 0)


def _formation_dates(dates: pd.DatetimeIndex, rebalance: str) -> pd.DatetimeIndex:
    unique = pd.DatetimeIndex(dates).normalize().sort_values().unique()
    if rebalance == "annual_june":
        june = unique[unique.month == 6]
        if june.empty:
            return june
        series = pd.Series(june, index=june.year)
        return pd.DatetimeIndex(series.groupby(level=0).max().to_numpy())
    if rebalance == "monthly":
        series = pd.Series(unique, index=unique.to_period("M"))
        return pd.DatetimeIndex(series.groupby(level=0).max().to_numpy())
    raise ValueError(f"unsupported FF3 rebalance: {rebalance}")


def _formation_labels(
    daily: pd.DataFrame,
    formation_dates: pd.DatetimeIndex,
    size: pd.Series,
    book_to_price: pd.Series,
    weights: pd.Series,
    universe: Mapping[pd.Timestamp, frozenset[str]],
    config: FF3Config,
) -> dict[pd.Timestamp, pd.Series]:
    characteristics = pd.concat(
        [size.rename("size"), book_to_price.rename("book_to_price"), weights.rename("weight")],
        axis=1,
    )
    labels: dict[pd.Timestamp, pd.Series] = {}
    for formation_date in formation_dates:
        try:
            snapshot = characteristics.xs(formation_date, level="date").copy()
        except KeyError:
            continue
        if universe:
            eligible = universe.get(formation_date, frozenset())
            snapshot = snapshot[snapshot.index.astype(str).isin(eligible)]
        snapshot = snapshot.replace([np.inf, -np.inf], np.nan).dropna()
        snapshot = snapshot[(snapshot["weight"] > 0) & (snapshot["book_to_price"] > 0)]
        if snapshot.empty:
            continue
        size_cut = snapshot["size"].quantile(config.size_quantile)
        value_low, value_high = snapshot["book_to_price"].quantile(
            list(config.book_to_price_quantiles),
        ).to_numpy(dtype=float)
        size_label = np.where(snapshot["size"] <= size_cut, "S", "B")
        value_label = np.where(
            snapshot["book_to_price"] <= value_low,
            "L",
            np.where(snapshot["book_to_price"] >= value_high, "H", "M"),
        )
        labels[formation_date] = pd.Series(
            np.char.add(size_label.astype(str), value_label.astype(str)),
            index=snapshot.index.astype(str),
            dtype="object",
        )
    return labels


def _held_membership(
    index: pd.MultiIndex,
    labels: dict[pd.Timestamp, pd.Series],
) -> pd.Series:
    result = pd.Series(index=index, dtype="object", name="portfolio")
    if not labels:
        return result
    formation_dates = pd.DatetimeIndex(sorted(labels))
    row_dates = index.get_level_values("date").normalize()
    securities = index.get_level_values("security_id").astype(str)
    # A close-of-day formation is effective on the next trading day, never on
    # the date whose closing characteristics created the sort.
    positions = np.searchsorted(formation_dates.to_numpy(), row_dates.to_numpy(), side="left") - 1
    for position in np.unique(positions[positions >= 0]):
        mask = positions == position
        mapping = labels[formation_dates[position]]
        result.iloc[np.flatnonzero(mask)] = mapping.reindex(securities[mask]).to_numpy()
    return result


def build_ff3_factors(
    context: FactorContext,
    benchmark_id: str | None = None,
    *,
    weighting: str | None = None,
    rebalance: str | None = None,
) -> pd.DataFrame:
    """Build MKT, SMB and HML from eligible A shares without future data.

    Formation uses the last trading day of June or each month and becomes
    effective on the next trading day. Portfolio returns use prior-day market
    value weights. ``context.eligible_universe`` must contain point-in-time
    all-A-share eligibility snapshots when strict filtering is required.
    """

    config = context.ff3_config
    weighting = weighting or config.default_weighting
    rebalance = rebalance or config.default_rebalance
    benchmark_id = benchmark_id or config.market_index_id
    cache_key = f"ff3:{benchmark_id}:{weighting}:{rebalance}:{config.risk_free_daily}"
    cached = context.cache.get(cache_key)
    if isinstance(cached, pd.DataFrame):
        return cached.copy()

    require_daily(context.daily, ["close"])
    daily = context.daily.sort_index()
    close = daily["close"].astype(float)
    returns = group_pct_change(close)
    size = _size_characteristic(daily)
    book_to_price = _book_to_price_characteristic(daily)
    weights = _weight_series(daily, weighting)
    formation_dates = _formation_dates(daily.index.get_level_values("date"), rebalance)
    model_universe = context.eligible_universe or context.universe
    labels = _formation_labels(
        daily, formation_dates, size, book_to_price, weights, model_universe, config,
    )
    membership = _held_membership(daily.index, labels)
    lagged_weight = group_shift(weights, 1)
    frame = pd.concat(
        [returns.rename("return"), lagged_weight.rename("weight"), membership],
        axis=1,
    ).reset_index()
    if model_universe:
        eligible_today = _eligible_mask(daily.index, model_universe)
        frame = frame[eligible_today]
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["return", "weight", "portfolio"],
    )
    frame = frame[(frame["weight"] > 0) & frame["portfolio"].isin(PORTFOLIOS)]
    if frame.empty:
        portfolio_returns = pd.DataFrame(columns=PORTFOLIOS, dtype=float)
    else:
        frame["weighted_return"] = frame["return"] * frame["weight"]
        grouped = frame.groupby(["date", "portfolio"], sort=True)
        portfolio_returns = (
            grouped["weighted_return"].sum().divide(grouped["weight"].sum()).unstack("portfolio")
        ).reindex(columns=PORTFOLIOS)

    small = portfolio_returns[["SL", "SM", "SH"]].mean(axis=1, skipna=False)
    big = portfolio_returns[["BL", "BM", "BH"]].mean(axis=1, skipna=False)
    high = portfolio_returns[["SH", "BH"]].mean(axis=1, skipna=False)
    low = portfolio_returns[["SL", "BL"]].mean(axis=1, skipna=False)
    style_factors = pd.DataFrame({"smb": small - big, "hml": high - low})
    factors = _market_return(context, benchmark_id, config.risk_free_daily).join(style_factors, how="outer")
    factors.index = pd.DatetimeIndex(factors.index).normalize()
    factors.index.name = "date"
    factors = factors[["mkt", "smb", "hml", "rf"]].sort_index()
    context.cache[cache_key] = factors
    return factors.copy()


def build_ff3_variants(
    context: FactorContext,
    benchmark_id: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Return every configured weighting and rebalance combination."""

    output: dict[str, pd.DataFrame] = {}
    for rebalance in context.ff3_config.rebalance_schemes:
        for weighting in context.ff3_config.weighting_schemes:
            key = f"{rebalance}__{weighting}"
            output[key] = build_ff3_factors(
                context, benchmark_id, weighting=weighting, rebalance=rebalance,
            )
    return output


def _residual_statistics(residual: np.ndarray) -> dict[str, float]:
    squared = np.square(residual)
    sigma = residual.std(ddof=1)
    masks = {
        "all": np.ones(len(residual), dtype=bool),
        "up": residual > 0,
        "down": residual < 0,
        "jup": residual >= sigma,
        "jdown": residual <= -sigma,
        "j": np.abs(residual) >= sigma,
        "nj": np.abs(residual) < sigma,
    }
    result = {
        name: float(squared[mask].mean()) if mask.any() else np.nan
        for name, mask in masks.items()
    }
    denominator = squared.sum()
    result["jup_ratio"] = float(squared[masks["jup"]].sum() / denominator) if denominator > 0 else np.nan
    result["jdown_ratio"] = float(squared[masks["jdown"]].sum() / denominator) if denominator > 0 else np.nan
    return result


def _rolling_statistics(
    context: FactorContext,
    benchmark_id: str,
    window: int,
) -> pd.DataFrame:
    config = context.ff3_config
    cache_key = (
        f"ff3_residual_stats:{benchmark_id}:{config.default_weighting}:"
        f"{config.default_rebalance}:{window}"
    )
    cached = context.cache.get(cache_key)
    if isinstance(cached, pd.DataFrame):
        return cached.copy()

    returns = group_pct_change(context.daily["close"].astype(float).sort_index())
    factors = build_ff3_factors(context, benchmark_id)
    joined = returns.rename("stock").reset_index().merge(
        factors.reset_index(), on="date", how="left",
    )
    target_dates = set(pd.DatetimeIndex(context.calculation_dates).normalize())
    output: list[dict[str, object]] = []
    for security_id, group in joined.groupby("security_id", sort=False):
        group = group.sort_values("date").reset_index(drop=True)
        target_positions = np.flatnonzero(group["date"].isin(target_dates).to_numpy())
        for position in target_positions:
            if position + 1 < window:
                continue
            sample = group.iloc[position - window + 1 : position + 1]
            columns = ["stock", "mkt", "smb", "hml", "rf"]
            if len(sample) != window or not np.isfinite(sample[columns].to_numpy(dtype=float)).all():
                continue
            design = np.column_stack([
                np.ones(window),
                sample[["mkt", "smb", "hml"]].to_numpy(dtype=float),
            ])
            y = sample["stock"].to_numpy(dtype=float) - sample["rf"].to_numpy(dtype=float)
            coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
            residual = y - design @ coefficients
            row: dict[str, object] = {
                "date": group.loc[position, "date"],
                "security_id": str(security_id),
            }
            row.update(_residual_statistics(residual))
            output.append(row)
    if output:
        result = pd.DataFrame(output).set_index(INDEX_NAMES).sort_index()
    else:
        result = pd.DataFrame(
            columns=["all", "up", "down", "jup", "jdown", "j", "nj", "jup_ratio", "jdown_ratio"],
            index=pd.MultiIndex.from_arrays([[], []], names=INDEX_NAMES),
            dtype=float,
        )
    context.cache[cache_key] = result
    return result.copy()


def _condition_signal(context: FactorContext, selector: str) -> pd.Series:
    daily = context.daily
    if selector == "lir":
        require_daily(daily, ["large_order_inflow_rate"])
        return daily["large_order_inflow_rate"].astype(float)
    if selector == "tvr":
        require_daily(daily, ["turnover_rate"])
        return daily["turnover_rate"].astype(float)
    if selector == "amp":
        if "ff3_amplitude_signal" in daily:
            return daily["ff3_amplitude_signal"].astype(float)
        require_daily(daily, ["close", "high", "low"])
        previous_close = group_shift(daily["close"].astype(float), 1)
        return safe_divide(daily["high"] - daily["low"], previous_close)
    if selector == "iv":
        require_daily(daily, ["intraday_volatility"])
        return daily["intraday_volatility"].astype(float)
    raise ValueError(f"unknown FF3 conditioning selector: {selector}")


def _rolling_conditioned_variance(
    context: FactorContext,
    benchmark_id: str,
    selector: str,
    window: int = 60,
) -> pd.Series:
    returns = group_pct_change(context.daily["close"].astype(float).sort_index())
    signal = _condition_signal(context, selector)
    factors = build_ff3_factors(context, benchmark_id)
    joined = pd.concat([returns.rename("stock"), signal.rename("signal")], axis=1).reset_index().merge(
        factors.reset_index(), on="date", how="left",
    )
    target_dates = set(pd.DatetimeIndex(context.calculation_dates).normalize())
    output: list[dict[str, object]] = []
    for security_id, group in joined.groupby("security_id", sort=False):
        group = group.sort_values("date").reset_index(drop=True)
        target_positions = np.flatnonzero(group["date"].isin(target_dates).to_numpy())
        for position in target_positions:
            if position + 1 < window:
                continue
            sample = group.iloc[position - window + 1 : position + 1]
            columns = ["stock", "signal", "mkt", "smb", "hml", "rf"]
            if len(sample) != window or not np.isfinite(sample[columns].to_numpy(dtype=float)).all():
                continue
            design = np.column_stack([
                np.ones(window),
                sample[["mkt", "smb", "hml"]].to_numpy(dtype=float),
            ])
            y = sample["stock"].to_numpy(dtype=float) - sample["rf"].to_numpy(dtype=float)
            coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
            residual = y - design @ coefficients
            percentile = sample["signal"].rank(method="average", pct=True).to_numpy(dtype=float)
            selected = percentile > (2.0 / 3.0)
            value = float(np.square(residual[selected]).mean()) if selected.any() else np.nan
            output.append({
                "date": group.loc[position, "date"],
                "security_id": str(security_id),
                "value": value,
            })
    if not output:
        return _empty_series()
    return pd.DataFrame(output).set_index(INDEX_NAMES)["value"].sort_index()


@registry.register("ff3_residual")
def compute_ff3_residual(context: FactorContext, spec: FactorSpec) -> pd.Series:
    if not supports_factor_id(spec.factor_id):
        raise NotImplementedError(f"ff3_residual does not implement {spec.factor_id}")
    context = _canonical_halt_context(context, spec)
    benchmark_id = spec.benchmark_id or context.ff3_config.market_index_id
    residual_match = RESIDUAL_PATTERN.fullmatch(spec.factor_id)
    if residual_match:
        window = int(residual_match.group(1))
        statistic = residual_match.group(2) or "all"
        statistics = _rolling_statistics(context, benchmark_id, window)
        result = statistics[statistic] if statistic in statistics else _empty_series()
    else:
        conditioned_match = CONDITIONED_PATTERN.fullmatch(spec.factor_id)
        if conditioned_match is None:  # pragma: no cover - guarded by supports_factor_id
            raise NotImplementedError(spec.factor_id)
        result = _rolling_conditioned_variance(context, benchmark_id, conditioned_match.group(1))
    result = result.replace([np.inf, -np.inf], np.nan).sort_index()
    result.name = spec.factor_id
    return result
