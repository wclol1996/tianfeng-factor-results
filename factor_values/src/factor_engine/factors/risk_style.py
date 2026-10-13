from __future__ import annotations

import numpy as np
import pandas as pd

from factor_engine.catalog import FactorSpec
from factor_engine.data.base import FactorContext
from factor_engine.factors.common import (
    INDEX_NAMES,
    cross_sectional_residual,
    group_pct_change,
    require_daily,
    safe_divide,
)
from factor_engine.registry import registry
from factor_engine.universe import apply_universe


SUPPORTED_FACTOR_IDS = frozenset({"beta", "lncap", "lncap_float", "lncap_freefloat", "midcap"})


def supports_factor_id(factor_id: str) -> bool:
    return factor_id in SUPPORTED_FACTOR_IDS


def _benchmark_returns(context: FactorContext, benchmark_id: str) -> pd.Series:
    frame = context.index_daily.copy()
    if frame.empty:
        raise ValueError("beta requires index_daily benchmark prices")
    if isinstance(frame.index, pd.MultiIndex) or frame.index.name == "date":
        frame = frame.reset_index()
    required = {"date", "close"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"index_daily is missing columns: {sorted(missing)}")
    if "index_id" in frame:
        frame = frame[frame["index_id"].astype(str).eq(benchmark_id)]
    if frame.empty:
        raise ValueError(f"beta benchmark {benchmark_id!r} has no observations")
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    close = (
        frame.sort_values("date")
        .drop_duplicates("date", keep="last")
        .set_index("date")["close"]
        .astype(float)
    )
    return close.where(close > 0.0).pct_change(fill_method=None)


def _rolling_ols_beta(
    stock_returns: pd.Series,
    market_returns: pd.Series,
    window: int,
    minimum_observations: int,
) -> pd.Series:
    """Estimate a stock's rolling market beta with an intercept.

    The calendar window always spans exactly ``window`` benchmark trading
    dates. ``minimum_observations`` can relax pairwise completeness without
    allowing a missing market date to be compressed out of the window.
    """

    if window < 2:
        raise ValueError("beta window must be at least 2")
    if not 2 <= minimum_observations <= window:
        raise ValueError("beta minimum_observations must be between 2 and window")

    market = market_returns.copy()
    market.index = pd.DatetimeIndex(market.index).normalize()
    market = market[~market.index.duplicated(keep="last")].sort_index()
    market_positions = pd.Series(
        np.arange(len(market), dtype=int),
        index=market.index,
    )
    joined = stock_returns.rename("stock").reset_index()
    joined["date"] = pd.to_datetime(joined["date"]).dt.normalize()
    joined = joined.merge(
        market.rename_axis("date").rename("market").reset_index(),
        on="date",
        how="left",
    )
    joined["market_position"] = (
        joined["date"].map(market_positions).fillna(-1).astype(int)
    )

    output: list[pd.DataFrame] = []
    for _, group in joined.groupby("security_id", sort=False):
        group = group.sort_values("date").copy()
        group_positions = group["market_position"].to_numpy(dtype=int)
        consecutive_return = np.r_[
            False,
            np.diff(group_positions) == 1,
        ]
        group.loc[~consecutive_return, "stock"] = np.nan
        values = np.full(len(group), np.nan)
        for position in range(window - 1, len(group)):
            sample = group.iloc[position - window + 1 : position + 1]
            calendar_positions = sample["market_position"].to_numpy(dtype=int)
            if (
                calendar_positions[0] < 0
                or not np.all(np.diff(calendar_positions) == 1)
            ):
                continue
            stock = sample["stock"].to_numpy(dtype=float)
            market_sample = sample["market"].to_numpy(dtype=float)
            valid = np.isfinite(stock) & np.isfinite(market_sample)
            if valid.sum() < minimum_observations:
                continue
            design = np.column_stack([
                np.ones(valid.sum()),
                market_sample[valid],
            ])
            if np.linalg.matrix_rank(design) < 2:
                continue
            coefficients, *_ = np.linalg.lstsq(
                design,
                stock[valid],
                rcond=None,
            )
            values[position] = coefficients[1]
        group["result"] = values
        output.append(group[["date", "security_id", "result"]])

    if not output:
        return pd.Series(
            dtype=float,
            index=pd.MultiIndex.from_arrays([[], []], names=INDEX_NAMES),
        )
    return pd.concat(output).set_index(INDEX_NAMES)["result"].sort_index()


@registry.register("risk_style")
def compute_risk_style(context: FactorContext, spec: FactorSpec) -> pd.Series:
    factor_id = spec.factor_id
    if not supports_factor_id(factor_id):
        raise NotImplementedError(f"risk_style does not implement {factor_id}")
    if factor_id == "beta":
        require_daily(context.daily, ["close"])
        close = context.daily["close"].astype(float).sort_index()
        stock_returns = group_pct_change(close.where(close > 0.0))
        window = int(spec.parameters.get("window", 60))
        result = _rolling_ols_beta(
            stock_returns,
            _benchmark_returns(context, spec.benchmark_id or "000985.CSI"),
            window,
            int(spec.parameters.get("minimum_observations", window)),
        )
        construction_universe = context.eligible_universe or context.universe
        if construction_universe:
            result = apply_universe(result, dict(construction_universe))
    elif factor_id == "lncap":
        require_daily(context.daily, ["total_market_cap"])
        result = np.log(context.daily["total_market_cap"].where(context.daily["total_market_cap"] > 0))
    elif factor_id == "lncap_float":
        require_daily(context.daily, ["float_market_cap"])
        result = np.log(context.daily["float_market_cap"].where(context.daily["float_market_cap"] > 0))
    elif factor_id == "lncap_freefloat":
        require_daily(context.daily, ["raw_close", "free_float_shares"])
        market_value = context.daily["raw_close"] * context.daily["free_float_shares"]
        result = np.log(market_value.where(market_value > 0))
    elif factor_id == "midcap":
        require_daily(context.daily, ["total_market_cap"])
        size = np.log(context.daily["total_market_cap"].where(context.daily["total_market_cap"] > 0))
        if context.universe:
            size = apply_universe(size, dict(context.universe))
        mean = size.groupby(level="date").transform("mean")
        standard_deviation = size.groupby(level="date").transform("std")
        standardized = safe_divide(size - mean, standard_deviation)
        nonlinear = standardized.pow(3)
        result = cross_sectional_residual(nonlinear, standardized.rename("size").to_frame())
    else:
        raise NotImplementedError(f"risk_style does not implement {factor_id}")
    result.name = factor_id
    return result.replace([np.inf, -np.inf], np.nan).sort_index()
