from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from factor_engine.catalog import FactorSpec
from factor_engine.data.base import FactorContext
from factor_engine.factors.common import (
    group_shift,
    require_daily,
    restore_declared_market_dates,
    rolling,
    safe_divide,
)
from factor_engine.registry import registry


SUPPORTED_EXACT = frozenset({
    "large_order_dplaTodtan",
    "small_order_dplaTodtan",
    "large_order_inflow_open_dplaTodtan",
    "net_buy_act_pct",
    "large_open_amount_prop_20d",
    "large_open_amount_prop_60d",
})


def supports_factor_id(factor_id: str) -> bool:
    return factor_id in SUPPORTED_EXACT


def _field(context: FactorContext, spec: FactorSpec, canonical_name: str) -> pd.Series:
    """Read a canonical money-flow variable through the future adapter map."""

    field_map = spec.parameters.get("field_map", {})
    source_name = str(field_map.get(canonical_name, canonical_name))
    require_daily(context.daily, [source_name])
    return (
        pd.to_numeric(context.daily[source_name], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .sort_index()
    )


def _complete_market_date_window(values: pd.Series, window: int) -> pd.Series:
    """Require observations on consecutive market dates, without compression."""

    dates = pd.DatetimeIndex(values.index.get_level_values("date")).normalize()
    calendar = dates.unique().sort_values()
    positions = pd.Series(
        calendar.get_indexer(dates).astype(float),
        index=values.index,
    )
    return (positions - group_shift(positions, window - 1)).eq(window - 1)


def _fixed_rolling_sum(values: pd.Series, window: int) -> pd.Series:
    return rolling(values, window, "sum").where(
        _complete_market_date_window(values, window),
    )


def _fixed_rolling_mean(values: pd.Series, window: int) -> pd.Series:
    return rolling(values, window, "mean").where(
        _complete_market_date_window(values, window),
    )


def _displacement_to_path_ratio(
    displacement: pd.Series,
    path_increment: pd.Series,
    window: int,
) -> pd.Series:
    displacement_sum = _fixed_rolling_sum(displacement, window)
    path_sum = _fixed_rolling_sum(path_increment.abs(), window)
    return safe_divide(displacement_sum, path_sum.where(path_sum > 0.0))


@registry.register("money_flow")
def compute_money_flow(context: FactorContext, spec: FactorSpec) -> pd.Series:
    """Formula-ready money-flow factors.

    Small-order source-native amounts have an explicit reviewed mapping.
    The other five formulas remain disabled pending their distinct source
    semantics; amount and volume are never active/passive classifications.
    """

    factor_id = spec.factor_id
    if not supports_factor_id(factor_id):
        raise NotImplementedError(f"money_flow does not implement {factor_id}")
    if factor_id == "small_order_dplaTodtan":
        context = replace(context, daily=restore_declared_market_dates(context.daily, context.calculation_dates))

    window = int(spec.parameters.get("window", 21 if factor_id == "net_buy_act_pct" else 20))
    if window < 1:
        raise ValueError("money-flow rolling window must be positive")

    if factor_id == "large_order_dplaTodtan":
        displacement = (
            _field(context, spec, "extra_large_order_buy_amount")
            + _field(context, spec, "large_order_buy_amount")
            - _field(context, spec, "extra_large_order_sell_amount")
            - _field(context, spec, "large_order_sell_amount")
        )
        active_net_buy = (
            _field(context, spec, "active_extra_large_order_buy_amount")
            + _field(context, spec, "active_large_order_buy_amount")
            - _field(context, spec, "active_extra_large_order_sell_amount")
            - _field(context, spec, "active_large_order_sell_amount")
        )
        result = _displacement_to_path_ratio(displacement, active_net_buy, window)
    elif factor_id == "small_order_dplaTodtan":
        active_small_net_buy = (
            _field(context, spec, "active_small_order_buy_amount")
            - _field(context, spec, "active_small_order_sell_amount")
        )
        result = _displacement_to_path_ratio(
            active_small_net_buy,
            active_small_net_buy,
            window,
        )
    elif factor_id == "large_order_inflow_open_dplaTodtan":
        open_large_order_net_inflow = _field(
            context,
            spec,
            "open_large_order_net_inflow_amount",
        )
        result = _displacement_to_path_ratio(
            open_large_order_net_inflow,
            open_large_order_net_inflow,
            window,
        )
    elif factor_id == "net_buy_act_pct":
        result = _fixed_rolling_mean(
            _field(context, spec, "moneyflow_pct_value"),
            window,
        )
    elif factor_id in {"large_open_amount_prop_20d", "large_open_amount_prop_60d"}:
        default_window = 20 if factor_id.endswith("20d") else 60
        window = int(spec.parameters.get("window", default_window))
        result = _fixed_rolling_mean(
            _field(context, spec, "open_moneyflow_pct_value_l"),
            window,
        )
    else:  # pragma: no cover - guarded by supports_factor_id
        raise NotImplementedError(f"money_flow does not implement {factor_id}")

    result.name = factor_id
    return result.replace([np.inf, -np.inf], np.nan).sort_index()
