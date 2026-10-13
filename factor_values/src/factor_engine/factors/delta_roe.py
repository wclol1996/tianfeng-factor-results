from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from factor_engine.catalog import FactorSpec
from factor_engine.data.base import FactorContext
from factor_engine.pipelines.delta_roe import _inverse_normal_cdf
from factor_engine.registry import registry


def _latest_delta_roe(
    reports: Mapping[pd.Timestamp, Mapping[str, float]],
    asset_field: str,
    liability_field: str,
    profit_field: str,
) -> float:
    quarterly: dict[int, tuple[pd.Timestamp, float, float]] = {}
    latest_ordinal: int | None = None
    for raw_period, values in reports.items():
        period = pd.Timestamp(raw_period).normalize()
        if period.strftime("%m-%d") not in {"03-31", "06-30", "09-30", "12-31"}:
            continue
        ordinal = period.year * 4 + period.quarter - 1
        latest_ordinal = ordinal if latest_ordinal is None else max(latest_ordinal, ordinal)
        try:
            assets = float(values[asset_field])
            liabilities = float(values[liability_field])
            profit_ytd = float(values[profit_field])
        except (KeyError, TypeError, ValueError):
            continue
        if not np.isfinite([assets, liabilities, profit_ytd]).all():
            continue
        quarterly[ordinal] = (period, assets - liabilities, profit_ytd)

    roe: dict[int, float] = {}
    for ordinal in sorted(quarterly):
        period, end_equity, profit_ytd = quarterly[ordinal]
        previous = quarterly.get(ordinal - 1)
        if previous is None:
            continue
        _, begin_equity, previous_ytd = previous
        single_quarter_profit = profit_ytd if period.quarter == 1 else profit_ytd - previous_ytd
        denominator = begin_equity + end_equity
        if denominator != 0 and np.isfinite(denominator):
            roe[ordinal] = 2.0 * single_quarter_profit / denominator

    delta = {
        ordinal: value - roe[ordinal - 4]
        for ordinal, value in roe.items()
        if ordinal - 4 in roe and np.isfinite(value - roe[ordinal - 4])
    }
    # A newly disclosed incomplete report invalidates the current value.
    return delta.get(latest_ordinal, np.nan)


def _rank_gauss(values: pd.Series) -> pd.Series:
    clean = values.dropna()
    if clean.empty:
        return clean
    ranks = clean.rank(method="average")
    probability = ranks.to_numpy(dtype=float) / (len(clean) + 1.0)
    return pd.Series(_inverse_normal_cdf(probability), index=clean.index)


@registry.register("delta_roe_rank_gauss")
def compute_delta_roe_rank_gauss(context: FactorContext, spec: FactorSpec) -> pd.Series:
    """Compute PIT daily DeltaROE ranks from cumulative quarterly statements."""

    observations = context.financial_observations.copy()
    required = {"security_id", "report_period", "available_at", "field", "value"}
    missing = required - set(observations.columns)
    if missing:
        raise ValueError(f"DeltaROE financial observations are missing columns: {sorted(missing)}")

    asset_field = str(spec.parameters.get("asset_field", "资产总计"))
    liability_field = str(spec.parameters.get("liability_field", "负债合计"))
    profit_field = str(spec.parameters.get("profit_field", "净利润(不含少数股东损益)"))
    fields = {asset_field, liability_field, profit_field}
    observations = observations[observations["field"].isin(fields)].copy()
    present = set(observations["field"].unique())
    if fields - present:
        raise ValueError(f"DeltaROE financial observations are missing fields: {sorted(fields - present)}")

    observations["security_id"] = observations["security_id"].astype(str)
    observations["report_period"] = pd.to_datetime(observations["report_period"]).dt.normalize()
    observations["available_at"] = pd.to_datetime(observations["available_at"])
    if "revision_at" in observations:
        observations["revision_at"] = pd.to_datetime(observations["revision_at"])
        observations["effective_at"] = observations["revision_at"].fillna(observations["available_at"])
    else:
        observations["effective_at"] = observations["available_at"]
    if observations["effective_at"].isna().any():
        raise ValueError("DeltaROE observations cannot have blank availability timestamps")
    observations = observations.sort_values("effective_at").drop_duplicates(
        ["security_id", "report_period", "field", "effective_at"], keep="last"
    ).reset_index(drop=True)

    reports: dict[str, dict[pd.Timestamp, dict[str, float]]] = {}
    current: dict[str, float] = {}
    results: list[pd.Series] = []
    cursor = 0
    dates = pd.DatetimeIndex(context.calculation_dates).normalize().sort_values().unique()
    for calculation_date in dates:
        if context.pit_allow_release_day:
            cutoff = calculation_date + pd.to_timedelta(context.pit_calculation_time)
        else:
            cutoff = calculation_date
        affected: set[str] = set()
        while cursor < len(observations) and observations.at[cursor, "effective_at"] <= cutoff:
            row = observations.iloc[cursor]
            security = row["security_id"]
            period = row["report_period"]
            reports.setdefault(security, {}).setdefault(period, {})[row["field"]] = row["value"]
            affected.add(security)
            cursor += 1
        for security in affected:
            current[security] = _latest_delta_roe(
                reports[security], asset_field, liability_field, profit_field
            )

        cross_section = pd.Series(current, dtype=float).replace([np.inf, -np.inf], np.nan)
        if context.universe:
            allowed = context.universe.get(calculation_date, frozenset())
            cross_section = cross_section[cross_section.index.isin(allowed)]
        else:
            cross_section = cross_section.iloc[0:0]
        transformed = _rank_gauss(cross_section)
        transformed.index = pd.MultiIndex.from_arrays(
            [[calculation_date] * len(transformed), transformed.index],
            names=["date", "security_id"],
        )
        results.append(transformed)

    if not results:
        index = pd.MultiIndex.from_arrays([[], []], names=["date", "security_id"])
        return pd.Series(dtype=float, index=index, name=spec.factor_id)
    output = pd.concat(results).sort_index()
    output.name = spec.factor_id
    return output
