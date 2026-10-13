from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from factor_engine.catalog import FactorSpec
from factor_engine.data.base import FactorContext
from factor_engine.factors.common import (
    INDEX_NAMES,
    cross_sectional_rank_gauss,
    cross_sectional_residual,
    cross_sectional_zscore,
    group_pct_change,
    safe_divide,
)
from factor_engine.registry import registry


# ``forecast_growth`` is the signed FY2/FY1 EPS growth rate.  ``con_np_adj``
# is the one-month (20 trading day) revision of rolling consensus net profit;
# neither formula is an expected-valuation factor.  Every catalog factor is
# reconstructed from primitive consensus fields; provider-supplied final factor
# values are deliberately ignored so a reference result can never enter the
# production formula path.
FORMULA_IDS = frozenset({
    "forecast_growth",
    "con_np_adj",
    "con_ep_roll_rg",
    "con_pe_roll_rg",
    "con_np_chg_20d",
    "con_np_chg_60d",
    "con_roe_fy1",
    "con_pe_fy1",
    "con_peg_fy1",
    "con_pb_fy1",
    "con_gagr_2y",  # Keep the catalog's historical spelling.
    "con_np_yoy_fy1",
    "con_np_yoy_fy2",
    "con_opr_yoy_fy1",
    "con_opr_yoy_fy2",
    "con_np2opr_fy1",
    "con_cfo2opr_fy1",
    "con_opp2opr_fy1",
})

RANK_GAUSS_IDS = frozenset({"con_ep_roll_rg", "con_pe_roll_rg"})
CHANGE_WINDOWS = {
    "con_np_adj": 20,
    "con_np_chg_20d": 20,
    "con_np_chg_60d": 60,
}
FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "net_profit": ("net_profit", "NET_PROFIT", "NetProfit", "一致预期净利润", "净利润"),
    "net_profit_parent": (
        "net_profit_parent",
        "NetProfitParentCompanyOwners", "NPParentCompanyOwners",
        "归属母公司净利润", "归母净利润", "预测归母净利润",
    ),
    "operating_revenue": (
        "operating_revenue", "revenue", "OperatingRevenue", "营业收入",
        "预测营业收入", "一致预期营业收入",
    ),
    "operating_profit": (
        "operating_profit", "OperatingProfit", "营业利润", "预测营业利润",
        "一致预期营业利润",
    ),
    "eps": ("eps", "EPS", "EarningsPerShare", "每股收益", "一致预期EPS"),
    "roe": ("roe", "ROE", "ReturnOnEquity", "一致预期ROE"),
    "pe": ("pe", "PE", "PriceEarningsRatio", "一致预期PE"),
    "pb": ("pb", "PB", "PriceBookRatio", "一致预期PB"),
    "peg": ("peg", "PEG", "一致预期PEG"),
    "cfps": (
        "cfps", "CFPS", "CashFlowPerShare", "每股现金流", "一致预期每股现金流",
    ),
    "total_shares": ("total_shares", "TotalShares", "总股本", "一致预期总股本"),
}


def supports_factor_id(factor_id: str) -> bool:
    return factor_id in FORMULA_IDS


def _empty_series() -> pd.Series:
    return pd.Series(dtype=float, index=pd.Index([], name="security_id"))


def _canonical_field(value: object) -> str:
    text = str(value).strip()
    for canonical, aliases in FIELD_ALIASES.items():
        if text in aliases:
            return canonical
    return text


def _find_column(frame: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    return next((name for name in names if name in frame), None)


def _normalize_forecast_type(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().upper().replace(" ", "").replace("_", "")
    aliases = {
        "1": "FY1", "FY01": "FY1", "F1": "FY1", "预测一年": "FY1",
        "2": "FY2", "FY02": "FY2", "F2": "FY2", "预测二年": "FY2",
        "3": "FY3", "FY03": "FY3", "F3": "FY3", "预测三年": "FY3",
    }
    return aliases.get(text, text)


def _reject_same_time_value_conflicts(frame: pd.DataFrame, keys: list[str], label: str) -> None:
    # Unknown values are versions too; do not choose finite over explicit null.
    counts = frame.groupby(keys, dropna=False)["value"].nunique(dropna=False)
    if counts.gt(1).any():
        raise ValueError(f"same-time {label} value conflict")


def _normalize_observations(observations: pd.DataFrame) -> pd.DataFrame:
    required = {"security_id", "available_at", "field", "value"}
    missing = required - set(observations.columns)
    if missing:
        raise ValueError(f"consensus observations are missing columns: {sorted(missing)}")

    frame = observations.copy()
    frame["security_id"] = frame["security_id"].astype(str)
    frame["available_at"] = pd.to_datetime(frame["available_at"], errors="coerce")
    if "revision_at" in frame:
        frame["revision_at"] = pd.to_datetime(frame["revision_at"], errors="coerce")
        if frame["revision_at"].lt(frame["available_at"]).any():
            raise ValueError("consensus revision predates availability")
        frame["effective_at"] = frame["revision_at"].fillna(frame["available_at"])
    else:
        frame["revision_at"] = pd.NaT
        frame["effective_at"] = frame["available_at"]
    if frame["effective_at"].isna().any():
        raise ValueError("consensus observations cannot have blank availability timestamps")

    year_column = _find_column(
        frame, ("forecast_year", "target_year", "ForecastYear", "TargetYear", "预测年度"),
    )
    type_column = _find_column(
        frame, ("forecast_type", "prediction_type", "ForecastType", "预测类型"),
    )
    frame["forecast_year"] = (
        pd.to_numeric(frame[year_column], errors="coerce").astype("Int64")
        if year_column is not None else pd.Series(pd.NA, index=frame.index, dtype="Int64")
    )
    frame["forecast_type"] = (
        frame[type_column].map(_normalize_forecast_type)
        if type_column is not None else pd.Series(None, index=frame.index, dtype="object")
    )
    frame["field"] = frame["field"].map(_canonical_field)
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")

    report_column = _find_column(
        frame, ("report_date", "write_date", "forecast_date", "ReportDate", "撰写日期", "预测日期"),
    )
    frame["report_date"] = (
        pd.to_datetime(frame[report_column], errors="coerce")
        if report_column is not None else frame["available_at"]
    )
    analyst_column = _find_column(
        frame, ("analyst_id", "forecaster_id", "author_id", "AnalystID", "分析师ID"),
    )
    frame["analyst_id"] = (
        frame[analyst_column].astype("string")
        if analyst_column is not None else pd.Series(pd.NA, index=frame.index, dtype="string")
    )
    if "analyst_weight" in frame:
        frame["analyst_weight"] = pd.to_numeric(frame["analyst_weight"], errors="coerce")
    else:
        frame["analyst_weight"] = 1.0
    frame["analyst_weight"] = frame["analyst_weight"].where(frame["analyst_weight"] > 0, 1.0).fillna(1.0)
    _reject_same_time_value_conflicts(
        frame, ["security_id", "forecast_year", "forecast_type", "field", "analyst_id", "effective_at"],
        "consensus",
    )
    frame["_input_order"] = np.arange(len(frame))
    return frame.sort_values(["effective_at", "_input_order"]).reset_index(drop=True)


def _locator_columns(frame: pd.DataFrame) -> list[str]:
    return ["security_id", "forecast_year", "forecast_type", "field"]


def _weighted_mean(group: pd.DataFrame, month_weights: Mapping[pd.Period, float]) -> float:
    weights = (
        group["report_month"].map(month_weights).astype(float)
        * group["analyst_weight"].astype(float)
    )
    valid = group["value"].notna() & weights.notna() & weights.gt(0)
    if not valid.any():
        return np.nan
    return float(np.average(group.loc[valid, "value"], weights=weights[valid]))


def _aggregate_analyst_detail(visible: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Apply the report's 3/6-month selection and 1:2:4 month weighting.

    Analyst accuracy weights can be supplied as ``analyst_weight`` after being
    estimated from a strictly prior-year training window.  This function never
    estimates them with current/future realized results.
    """

    detail = visible[visible["analyst_id"].notna()].copy()
    aggregate = visible[visible["analyst_id"].isna()].copy()
    if detail.empty:
        return aggregate.drop_duplicates(_locator_columns(aggregate), keep="last")

    detail = detail[detail["report_date"] <= cutoff].copy()
    detail["age_days"] = (cutoff.normalize() - detail["report_date"].dt.normalize()).dt.days
    detail = detail[detail["age_days"].between(0, 183)].copy()
    detail = detail.drop_duplicates(
        ["security_id", "forecast_year", "forecast_type", "field", "analyst_id"], keep="last",
    )
    rows: list[dict[str, object]] = []
    for keys, group in detail.groupby(_locator_columns(detail), dropna=False, sort=False):
        recent = group[group["age_days"] <= 92]
        chosen = recent if not recent.empty else group[group["age_days"] > 92]
        if chosen.empty:
            continue
        chosen = chosen.copy()
        chosen["report_month"] = chosen["report_date"].dt.to_period("M")
        months = sorted(chosen["report_month"].dropna().unique())[-3:]
        chosen = chosen[chosen["report_month"].isin(months)]
        # With fewer than three represented months, keep the latest month at
        # weight 4 and preserve the report's geometric recency pattern.
        scale = [1.0, 2.0, 4.0][-len(months):]
        month_weights = dict(zip(months, scale, strict=True))
        rows.append({
            "security_id": keys[0],
            "forecast_year": keys[1],
            "forecast_type": keys[2],
            "field": keys[3],
            "value": _weighted_mean(chosen, month_weights),
        })
    detailed = pd.DataFrame(rows)
    if aggregate.empty:
        return detailed
    aggregate = aggregate.drop_duplicates(_locator_columns(aggregate), keep="last")
    if detailed.empty:
        return aggregate[_locator_columns(aggregate) + ["value"]]
    # Explicit provider aggregates win if both kinds are supplied.
    combined = pd.concat([
        detailed.assign(_priority=0),
        aggregate[_locator_columns(aggregate) + ["value"]].assign(_priority=1),
    ], ignore_index=True)
    return combined.sort_values("_priority").drop_duplicates(_locator_columns(combined), keep="last")


def _visible_snapshot(observations: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    visible = observations[observations["effective_at"] <= cutoff]
    if visible.empty:
        return pd.DataFrame(columns=[*_locator_columns(observations), "value"])
    if visible["analyst_id"].notna().any():
        return _aggregate_analyst_detail(visible, cutoff)
    return visible.drop_duplicates(_locator_columns(visible), keep="last")


def _annual_disclosures(context: FactorContext) -> pd.DataFrame:
    source = context.extras.get("annual_report_disclosures")
    if source is None or source.empty:
        return pd.DataFrame(columns=["security_id", "report_year", "effective_at"])
    frame = source.copy()
    security_column = _find_column(frame, ("security_id", "wind_code", "stock_code"))
    year_column = _find_column(frame, ("report_year", "year", "报告年度"))
    available_column = _find_column(frame, ("available_at", "announcement_at", "公告时间", "公告日期"))
    if security_column is None or year_column is None or available_column is None:
        raise ValueError(
            "annual_report_disclosures requires security_id, report_year, and available_at"
        )
    normalized = pd.DataFrame({
        "security_id": frame[security_column].astype(str),
        "report_year": pd.to_numeric(frame[year_column], errors="coerce").astype("Int64"),
        "effective_at": pd.to_datetime(frame[available_column], errors="coerce"),
    })
    return normalized.dropna().sort_values("effective_at")


def _normalize_actuals(observations: pd.DataFrame) -> pd.DataFrame:
    if observations.empty:
        return pd.DataFrame(columns=[
            "security_id", "report_year", "field", "value", "effective_at",
        ])
    required = {"security_id", "report_period", "available_at", "field", "value"}
    if required - set(observations.columns):
        return pd.DataFrame(columns=[
            "security_id", "report_year", "field", "value", "effective_at",
        ])
    frame = observations.copy()
    frame["report_period"] = pd.to_datetime(frame["report_period"], errors="coerce")
    frame = frame[frame["report_period"].dt.strftime("%m-%d").eq("12-31")].copy()
    frame["security_id"] = frame["security_id"].astype(str)
    frame["report_year"] = frame["report_period"].dt.year.astype("Int64")
    frame["field"] = frame["field"].map(_canonical_field)
    frame = frame[frame["field"].isin({"net_profit", "operating_revenue"})]
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame["available_at"] = pd.to_datetime(frame["available_at"], errors="coerce")
    if "revision_at" in frame:
        revision = pd.to_datetime(frame["revision_at"], errors="coerce")
        if revision.lt(frame["available_at"]).any():
            raise ValueError("financial revision predates availability")
        frame["effective_at"] = revision.fillna(frame["available_at"])
    else:
        frame["effective_at"] = frame["available_at"]
    _reject_same_time_value_conflicts(
        frame, ["security_id", "report_year", "field", "effective_at"], "annual actual",
    )
    frame["_input_order"] = np.arange(len(frame))
    return frame.dropna(subset=["effective_at"]).sort_values(["effective_at", "_input_order"])


def _fy0_years(
    date: pd.Timestamp,
    cutoff: pd.Timestamp,
    securities: pd.Index,
    disclosures: pd.DataFrame,
    actuals: pd.DataFrame,
) -> pd.Series:
    # Before the new annual report is disclosed, the last completed fiscal year
    # is the previous calendar year.  A visible company-level disclosure moves
    # FY0/FY1 immediately; April 30 is never hard-coded.
    result = pd.Series(date.year - 1, index=securities.astype(str), dtype="int64")
    visible_actuals = actuals[actuals["effective_at"] <= cutoff]
    if not visible_actuals.empty:
        latest_year = visible_actuals.groupby("security_id")["report_year"].max().astype(int)
        result.update(latest_year)
    visible = disclosures[disclosures["effective_at"] <= cutoff]
    if not visible.empty:
        latest = visible.drop_duplicates("security_id", keep="last").set_index("security_id")
        result.update(latest["report_year"].astype(int))
    return result


def _snapshot_series(snapshot: pd.DataFrame, field: str) -> pd.Series:
    rows = snapshot[snapshot["field"].eq(field)]
    if rows.empty:
        return _empty_series()
    return rows.drop_duplicates("security_id", keep="last").set_index("security_id")["value"].astype(float)


def _forecast_horizon(
    snapshot: pd.DataFrame,
    field: str,
    horizon: int,
    fy0: pd.Series,
) -> pd.Series:
    rows = snapshot[snapshot["field"].eq(field)]
    if rows.empty:
        return _empty_series()
    label = f"FY{horizon}"
    typed = rows[rows["forecast_type"].eq(label)]
    typed_values = (
        typed.drop_duplicates("security_id", keep="last").set_index("security_id")["value"].astype(float)
        if not typed.empty else _empty_series()
    )
    annual = rows[rows["forecast_year"].notna()].copy()
    if annual.empty:
        return typed_values
    annual["target_year"] = annual["security_id"].map(fy0) + horizon
    annual = annual[annual["forecast_year"].astype(float).eq(annual["target_year"])]
    annual_values = (
        annual.drop_duplicates("security_id", keep="last").set_index("security_id")["value"].astype(float)
        if not annual.empty else _empty_series()
    )
    return typed_values.combine_first(annual_values)


def _forecast_calendar_year(
    snapshot: pd.DataFrame,
    field: str,
    target_year: int,
    fy0: pd.Series,
) -> pd.Series:
    rows = snapshot[snapshot["field"].eq(field)]
    if rows.empty:
        return _empty_series()
    annual = rows[rows["forecast_year"].eq(target_year)]
    annual_values = (
        annual.drop_duplicates("security_id", keep="last").set_index("security_id")["value"].astype(float)
        if not annual.empty else _empty_series()
    )
    typed_rows = rows[rows["forecast_type"].astype("string").str.match(r"^FY[1-9]$").fillna(False)].copy()
    if typed_rows.empty:
        return annual_values
    typed_rows["horizon"] = typed_rows["forecast_type"].str.removeprefix("FY").astype(int)
    typed_rows["mapped_year"] = typed_rows["security_id"].map(fy0) + typed_rows["horizon"]
    typed_rows = typed_rows[typed_rows["mapped_year"].eq(target_year)]
    typed_values = (
        typed_rows.drop_duplicates("security_id", keep="last").set_index("security_id")["value"].astype(float)
        if not typed_rows.empty else _empty_series()
    )
    return annual_values.combine_first(typed_values)


def _actual_fy0(actuals: pd.DataFrame, field: str, fy0: pd.Series, cutoff: pd.Timestamp) -> pd.Series:
    rows = actuals[(actuals["effective_at"] <= cutoff) & actuals["field"].eq(field)].copy()
    if rows.empty:
        return _empty_series()
    rows["target_year"] = rows["security_id"].map(fy0)
    rows = rows[rows["report_year"].astype(float).eq(rows["target_year"])]
    if rows.empty:
        return _empty_series()
    return rows.drop_duplicates(["security_id", "report_year", "field"], keep="last").set_index("security_id")["value"].astype(float)


def _daily_series(context: FactorContext, date: pd.Timestamp, names: tuple[str, ...]) -> pd.Series:
    if context.daily.empty:
        return _empty_series()
    field = next((name for name in names if name in context.daily), None)
    if field is None:
        return _empty_series()
    dates = context.daily.index.get_level_values("date").normalize()
    rows = context.daily.loc[dates == date, field]
    if rows.empty:
        return _empty_series()
    rows.index = rows.index.get_level_values("security_id").astype(str)
    rows.index.name = "security_id"
    return pd.to_numeric(rows, errors="coerce").astype(float)


def _allowed_universe(context: FactorContext, date: pd.Timestamp) -> frozenset[str] | None:
    if date in context.eligible_universe:
        return context.eligible_universe[date]
    if date in context.universe:
        return context.universe[date]
    return None


def _filter_allowed(values: pd.Series, context: FactorContext, date: pd.Timestamp) -> pd.Series:
    allowed = _allowed_universe(context, date)
    return values if allowed is None else values[values.index.astype(str).isin(allowed)]


def _rolling_consensus_profit(
    snapshot: pd.DataFrame,
    date: pd.Timestamp,
    fy0: pd.Series,
) -> pd.Series:
    current = _forecast_calendar_year(snapshot, "net_profit", date.year, fy0)
    following = _forecast_calendar_year(snapshot, "net_profit", date.year + 1, fy0)
    days = 366.0 if date.is_leap_year else 365.0
    following_weight = (date.dayofyear - 1.0) / days
    return current * (1.0 - following_weight) + following * following_weight


def _economic_growth(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return safe_divide(numerator, denominator) - 1.0


def _formula_for_date(
    context: FactorContext,
    factor_id: str,
    date: pd.Timestamp,
    cutoff: pd.Timestamp,
    snapshot: pd.DataFrame,
    fy0: pd.Series,
    actuals: pd.DataFrame,
) -> tuple[pd.Series, pd.Series | None]:
    np_fy1 = _forecast_horizon(snapshot, "net_profit", 1, fy0)
    np_fy2 = _forecast_horizon(snapshot, "net_profit", 2, fy0)
    revenue_fy1 = _forecast_horizon(snapshot, "operating_revenue", 1, fy0)
    revenue_fy2 = _forecast_horizon(snapshot, "operating_revenue", 2, fy0)

    if factor_id == "forecast_growth":
        eps_fy1 = _forecast_horizon(snapshot, "eps", 1, fy0)
        eps_fy2 = _forecast_horizon(snapshot, "eps", 2, fy0)
        return _economic_growth(eps_fy2, eps_fy1), None

    if factor_id in RANK_GAUSS_IDS or factor_id in CHANGE_WINDOWS:
        rolling_profit = _rolling_consensus_profit(snapshot, date, fy0)
        if factor_id in CHANGE_WINDOWS:
            return rolling_profit, None
        market_cap = _daily_series(context, date, ("total_market_cap", "market_cap"))
        raw = (
            safe_divide(rolling_profit, market_cap)
            if factor_id == "con_ep_roll_rg" else safe_divide(market_cap, rolling_profit)
        )
        return raw, None

    if factor_id == "con_roe_fy1":
        return _forecast_horizon(snapshot, "roe", 1, fy0), None
    if factor_id == "con_pe_fy1":
        provider = _forecast_horizon(snapshot, "pe", 1, fy0)
        market_cap = _daily_series(context, date, ("total_market_cap", "market_cap"))
        return provider.combine_first(safe_divide(market_cap, np_fy1)), None
    if factor_id == "con_pb_fy1":
        return _forecast_horizon(snapshot, "pb", 1, fy0), None

    np_fy0 = _actual_fy0(actuals, "net_profit", fy0, cutoff)
    revenue_fy0 = _actual_fy0(actuals, "operating_revenue", fy0, cutoff)
    np_growth_1 = _economic_growth(np_fy1, np_fy0)
    np_growth_2 = _economic_growth(np_fy2, np_fy1)
    revenue_growth_1 = _economic_growth(revenue_fy1, revenue_fy0)
    revenue_growth_2 = _economic_growth(revenue_fy2, revenue_fy1)

    if factor_id == "con_gagr_2y":
        ratio = safe_divide(np_fy2, np_fy0)
        return np.sqrt(ratio.where(ratio >= 0)) - 1.0, None
    if factor_id == "con_peg_fy1":
        provider = _forecast_horizon(snapshot, "peg", 1, fy0)
        pe = _forecast_horizon(snapshot, "pe", 1, fy0)
        market_cap = _daily_series(context, date, ("total_market_cap", "market_cap"))
        pe = pe.combine_first(safe_divide(market_cap, np_fy1))
        cagr = np.sqrt(safe_divide(np_fy2, np_fy0).where(lambda value: value >= 0)) - 1.0
        return provider.combine_first(safe_divide(pe, 100.0 * cagr)), None
    if factor_id == "con_np_yoy_fy1":
        return np_growth_1, np_growth_1
    if factor_id == "con_np_yoy_fy2":
        return np_growth_2, np_growth_2
    if factor_id == "con_opr_yoy_fy1":
        return revenue_growth_1, revenue_growth_1
    if factor_id == "con_opr_yoy_fy2":
        return revenue_growth_2, revenue_growth_2
    if factor_id == "con_np2opr_fy1":
        return safe_divide(np_fy1, revenue_fy1), None
    if factor_id == "con_opp2opr_fy1":
        operating_profit = _forecast_horizon(snapshot, "operating_profit", 1, fy0)
        return safe_divide(operating_profit, revenue_fy1), None
    if factor_id == "con_cfo2opr_fy1":
        cfps = _forecast_horizon(snapshot, "cfps", 1, fy0)
        shares = _forecast_horizon(snapshot, "total_shares", 1, fy0)
        shares = shares.combine_first(_daily_series(context, date, ("total_shares", "shares")))
        return safe_divide(cfps * shares, revenue_fy1), None
    raise KeyError(f"unsupported consensus factor: {factor_id}")


def industry_size_residual(values: pd.Series, context: FactorContext) -> pd.Series:
    """Optional derived layer; never applied to the 18 catalog reference outputs."""

    if values.empty or context.daily.empty:
        return values * np.nan
    joined = context.daily.reindex(values.index)
    industry_column = next((
        name for name in (
            "citic_industry_l1", "citic_level1", "中信一级行业",
            "industry_l1", "industry", "industry_code",
        ) if name in joined
    ), None)
    cap_column = next(
        (name for name in ("total_market_cap", "market_cap") if name in joined), None,
    )
    if industry_column is None or cap_column is None:
        return values * np.nan
    size = np.log(pd.to_numeric(joined[cap_column], errors="coerce").where(lambda value: value > 0))
    size = cross_sectional_zscore(size)
    industries = pd.get_dummies(joined[industry_column], prefix="industry", dtype=float)
    regressors = pd.concat([size.rename("size"), industries], axis=1)
    return cross_sectional_residual(cross_sectional_zscore(values), regressors)


def _calculation_grid(context: FactorContext, factor_id: str) -> pd.DatetimeIndex:
    targets = pd.DatetimeIndex(context.calculation_dates).normalize().sort_values().unique()
    if targets.empty or context.daily.empty or factor_id not in CHANGE_WINDOWS:
        return targets
    daily_dates = context.daily.index.get_level_values("date").normalize().sort_values().unique()
    daily_dates = daily_dates[daily_dates <= targets.max()]
    first_position = int(daily_dates.searchsorted(targets.min(), side="left"))
    history_start = max(0, first_position - CHANGE_WINDOWS[factor_id])
    return daily_dates[history_start:]


@registry.register("consensus_expectation")
def compute_consensus_expectation(context: FactorContext, spec: FactorSpec) -> pd.Series:
    factor_id = str(spec.parameters.get("formula_id", spec.factor_id))
    if factor_id not in FORMULA_IDS:
        raise ValueError(f"{spec.factor_id}: unsupported consensus formula_id {factor_id!r}")
    observations = context.cache.get("consensus:normalized_observations")
    if observations is None:
        observations = _normalize_observations(context.consensus_observations)
        context.cache["consensus:normalized_observations"] = observations
    actuals = context.cache.get("consensus:normalized_actuals")
    if actuals is None:
        actuals = _normalize_actuals(context.financial_observations)
        context.cache["consensus:normalized_actuals"] = actuals
    disclosures = context.cache.get("consensus:annual_disclosures")
    if disclosures is None:
        disclosures = _annual_disclosures(context)
        context.cache["consensus:annual_disclosures"] = disclosures
    snapshots = context.cache.setdefault("consensus:pit_snapshots", {})
    fy0_cache = context.cache.setdefault("consensus:fy0_years", {})
    dates = _calculation_grid(context, factor_id)
    results: list[pd.Series] = []

    for date in dates:
        cutoff = (
            date + pd.to_timedelta(context.pit_calculation_time)
            if context.pit_allow_release_day else date
        )
        snapshot = snapshots.get(cutoff)
        if snapshot is None:
            snapshot = _visible_snapshot(observations, cutoff)
            snapshots[cutoff] = snapshot
        if snapshot.empty:
            continue
        securities = pd.Index(snapshot["security_id"].astype(str).unique(), name="security_id")
        fy0 = fy0_cache.get(cutoff)
        if fy0 is None:
            fy0 = _fy0_years(date, cutoff, securities, disclosures, actuals)
            fy0_cache[cutoff] = fy0
        raw, _ = _formula_for_date(
            context, factor_id, date, cutoff, snapshot, fy0, actuals,
        )
        raw = _filter_allowed(raw.replace([np.inf, -np.inf], np.nan), context, date)
        raw.index = pd.MultiIndex.from_arrays(
            [[date] * len(raw), raw.index.astype(str)], names=INDEX_NAMES,
        )
        results.append(raw.astype(float))

    if not results:
        index = pd.MultiIndex.from_arrays([[], []], names=INDEX_NAMES)
        return pd.Series(dtype=float, index=index, name=spec.factor_id)

    raw_output = pd.concat(results).sort_index()
    context.cache[f"consensus:{factor_id}:raw"] = raw_output.copy()
    if factor_id in CHANGE_WINDOWS:
        # A missing forecast does not remove a market session from the lag.
        # Use the same calendar contract as the full-history reproduction.
        calendar = pd.DatetimeIndex(dates).sort_values().unique()
        wide = raw_output.unstack("security_id").reindex(calendar)
        changed = wide.pct_change(CHANGE_WINDOWS[factor_id], fill_method=None)
        output = changed.rename_axis(index="date", columns="security_id").stack().reindex(raw_output.index)
        context.cache["consensus:rolling_net_profit"] = raw_output.copy()
    elif factor_id in RANK_GAUSS_IDS:
        output = cross_sectional_rank_gauss(raw_output)
    else:
        output = raw_output

    postprocess = str(spec.parameters.get("postprocess", "none"))
    if postprocess == "industry_size_residual":
        output = industry_size_residual(output, context)
    elif postprocess not in {"none", "rank_gauss"}:
        raise ValueError(f"unsupported consensus postprocess: {postprocess!r}")
    if postprocess == "rank_gauss" and factor_id not in RANK_GAUSS_IDS:
        output = cross_sectional_rank_gauss(output)
    if factor_id == "con_roe_fy1" and spec.parameters.get("output_unit") == "percentage_points":
        output = output * 100.0  # Primitive report_rc ROE is normalized to a fraction.
    output.name = spec.factor_id
    return output.replace([np.inf, -np.inf], np.nan)
