from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from factor_engine.catalog import FactorSpec
from factor_engine.data.base import FactorContext
from factor_engine.factors.common import INDEX_NAMES, safe_divide
from factor_engine.registry import registry


SUPPORTED_EXACT = frozenset({"pb_int"})

# Rates disclosed in figure 6 of the supplied Guosheng Securities report.
# Level-2 rates take precedence over level-1 rates.  The report only printed a
# subset of industries, so unknown industries deliberately remain missing until
# the adapter supplies a PIT ``rd_depreciation_rate`` field.
REPORT_RD_DEPRECIATION_L1 = {
    "计算机": 0.5630,
    "医药": 0.3761,
    "电子": 0.7388,
    "国防军工": 0.5795,
    "通信": 0.4349,
    "汽车": 0.3215,
    "传媒": 0.4957,
}

REPORT_RD_DEPRECIATION_L2 = {
    "计算机设备": 0.5969,
    "计算机软件": 0.5982,
    "化学制药": 0.2911,
    "中药生产": 0.3396,
    "生物医药II": 0.5493,
    "其他医药医疗": 0.9999,
    "半导体": 0.9999,
    "元器件": 0.7249,
    "航空航天": 0.2190,
    "其他军工II": 0.6870,
    "通信设备制造": 0.3965,
    "增值服务II": 0.6538,
    "乘用车II": 0.6799,
    "商用车": 0.4858,
    "汽车零部件II": 0.2335,
    "媒体": 0.4957,
}

FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "rd_investment_total": (
        "rd_investment_total", "研发投入总额", "研究开发支出",
        "AShareRDexpenditure:1900015003",
    ),
    "rd_expensed": (
        "rd_expensed", "费用化研究支出", "费用化研发支出",
        "AShareRDexpenditure:1900015001",
    ),
    "rd_capitalized": (
        "rd_capitalized", "资本化研究支出", "资本化研发支出",
        "AShareRDexpenditure:1900015002",
    ),
    "management_rd_expense": (
        "management_rd_expense", "管理费用中的研发费用", "管理费用研发费",
    ),
    "sales_advertising_expense": (
        "sales_advertising_expense", "广告宣传推广费(销售费用)",
        "广告宣传推广费（销售费用）",
    ),
    "sales_employee_compensation": (
        "sales_employee_compensation", "工资薪酬(销售费用)",
        "工资薪酬（销售费用）",
    ),
    "management_employee_compensation": (
        "management_employee_compensation", "工资薪酬(管理费用)",
        "工资薪酬（管理费用）",
    ),
    "goodwill": ("goodwill", "商誉", "Goodwill"),
    "rd_depreciation_rate": (
        "rd_depreciation_rate", "研发资产折旧率", "研发支出折旧率",
    ),
}

DAILY_ALIASES: dict[str, tuple[str, ...]] = {
    "total_market_cap": ("total_market_cap", "market_cap", "总市值"),
    "industry_l1": (
        "citic_industry_l1", "citic_level1", "中信一级行业", "industry_l1",
    ),
    "industry_l2": (
        "citic_industry_l2", "citic_level2", "中信二级行业", "industry_l2",
    ),
}


def supports_factor_id(factor_id: str) -> bool:
    return factor_id in SUPPORTED_EXACT


def _canonical_field(value: object) -> str:
    text = str(value).strip()
    for canonical, aliases in FIELD_ALIASES.items():
        if text in aliases:
            return canonical
    return text


def _normalize_observations(observations: pd.DataFrame) -> pd.DataFrame:
    required = {"security_id", "report_period", "available_at", "field", "value"}
    missing = required - set(observations.columns)
    if missing:
        raise ValueError(f"pb_int financial observations are missing columns: {sorted(missing)}")
    frame = observations.copy()
    frame["security_id"] = frame["security_id"].astype(str)
    frame["report_period"] = pd.to_datetime(frame["report_period"]).dt.normalize()
    frame["available_at"] = pd.to_datetime(frame["available_at"])
    frame["revision_at"] = (
        pd.to_datetime(frame["revision_at"])
        if "revision_at" in frame else pd.NaT
    )
    frame["effective_at"] = frame["revision_at"].fillna(frame["available_at"])
    if frame["effective_at"].isna().any():
        raise ValueError("pb_int financial observations cannot have a blank effective timestamp")

    method_column = next(
        (name for name in ("accounting_method", "accountingMethod", "核算方式") if name in frame),
        None,
    )
    if method_column is not None:
        method = frame[method_column].astype(str).str.strip().str.upper()
        frame = frame[method.isin({"A", "累计值", "CUMULATIVE", "YTD", ""})].copy()

    frame["field"] = frame["field"].map(_canonical_field)
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame[frame["report_period"].dt.strftime("%m-%d").eq("12-31")].copy()
    frame["_input_order"] = np.arange(len(frame))
    return frame.sort_values(["effective_at", "_input_order"]).reset_index(drop=True)


def _visible_annual_frame(observations: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    visible = observations[observations["effective_at"] <= cutoff]
    if visible.empty:
        index = pd.MultiIndex.from_arrays([[], []], names=["security_id", "report_period"])
        return pd.DataFrame(index=index)
    latest = visible.drop_duplicates(
        ["security_id", "report_period", "field"], keep="last",
    )
    wide = latest.pivot(
        index=["security_id", "report_period"], columns="field", values="value",
    ).sort_index()
    wide.columns.name = None
    return wide


def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce")


def _daily_cross_section(context: FactorContext, date: pd.Timestamp) -> pd.DataFrame:
    if context.daily.empty:
        return pd.DataFrame()
    dates = context.daily.index.get_level_values("date").normalize()
    if date not in dates:
        return pd.DataFrame()
    current = context.daily.xs(date, level="date").copy()
    current.index = current.index.astype(str)
    output = pd.DataFrame(index=current.index)
    for canonical, aliases in DAILY_ALIASES.items():
        source = next((alias for alias in aliases if alias in current), None)
        if source is not None:
            output[canonical] = current[source]
    return output


def _establishment_dates(context: FactorContext) -> pd.Series:
    master = context.extras.get("security_master")
    if master is None or master.empty or "security_id" not in master:
        return pd.Series(dtype="datetime64[ns]")
    source = next(
        (
            name for name in (
                "establishment_date", "company_establishment_date", "成立日期",
            )
            if name in master
        ),
        None,
    )
    if source is None:
        return pd.Series(dtype="datetime64[ns]")
    normalized = master[["security_id", source]].copy()
    normalized["security_id"] = normalized["security_id"].astype(str)
    normalized[source] = pd.to_datetime(normalized[source], errors="coerce").dt.normalize()
    return normalized.drop_duplicates("security_id", keep="last").set_index("security_id")[source]


def _rd_investment(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    direct = _column(frame, "rd_investment_total")
    expensed = _column(frame, "rd_expensed")
    capitalized = _column(frame, "rd_capitalized")
    reconstructed = (expensed + capitalized).where(expensed.notna() & capitalized.notna())
    management = _column(frame, "management_rd_expense")
    values = direct.combine_first(reconstructed).combine_first(expensed).combine_first(management)
    source = pd.Series("missing", index=frame.index, dtype="string")
    source.loc[management.notna()] = "management_rd_expense"
    source.loc[expensed.notna()] = "rd_expensed"
    source.loc[reconstructed.notna()] = "rd_expensed_plus_capitalized"
    source.loc[direct.notna()] = "rd_investment_total"
    return values, source


def _sga_investment(frame: pd.DataFrame) -> pd.Series:
    components = pd.concat(
        [
            _column(frame, "sales_advertising_expense"),
            _column(frame, "sales_employee_compensation"),
            _column(frame, "management_employee_compensation"),
        ],
        axis=1,
    )
    return components.sum(axis=1, min_count=3)


def _report_rate(
    industry_l1: object,
    industry_l2: object,
    parameters: Mapping[str, object],
    calculation_date: pd.Timestamp,
) -> float:
    effective_from = pd.Timestamp(
        parameters.get("report_rate_effective_from", "2020-06-01"),
    ).normalize()
    if (
        calculation_date < effective_from
        and not bool(parameters.get("allow_prepublication_report_rates", False))
    ):
        return np.nan
    level1 = {
        **REPORT_RD_DEPRECIATION_L1,
        **dict(parameters.get("rd_depreciation_rate_l1_overrides", {})),
    }
    level2 = {
        **REPORT_RD_DEPRECIATION_L2,
        **dict(parameters.get("rd_depreciation_rate_l2_overrides", {})),
    }
    if pd.notna(industry_l2) and str(industry_l2) in level2:
        return float(level2[str(industry_l2)])
    if pd.notna(industry_l1) and str(industry_l1) in level1:
        return float(level1[str(industry_l1)])
    fallback = parameters.get("rd_depreciation_rate_fallback")
    return float(fallback) if fallback is not None else np.nan


def _initial_capital(investment: float, retention: float, growth: float, years: int) -> float:
    if years < 0 or not np.isfinite(investment):
        return np.nan
    ratio = retention / (1.0 + growth)
    return float(investment * np.power(ratio, np.arange(years + 1)).sum())


def _capitalized_annual_assets(
    frame: pd.DataFrame,
    daily: pd.DataFrame,
    establishment_dates: pd.Series,
    parameters: Mapping[str, object],
    calculation_date: pd.Timestamp,
) -> pd.DataFrame:
    rd, rd_source = _rd_investment(frame)
    sga = _sga_investment(frame)
    goodwill = _column(frame, "goodwill")
    direct_rate = _column(frame, "rd_depreciation_rate")
    growth = float(parameters.get("pre_disclosure_growth_rate", 0.15))
    sga_capitalization = float(parameters.get("sga_capitalization_rate", 0.15))
    sga_depreciation = float(parameters.get("sga_depreciation_rate", 0.20))
    if growth <= -1.0:
        raise ValueError("pb_int pre_disclosure_growth_rate must be greater than -1")
    if not 0.0 <= sga_capitalization <= 1.0:
        raise ValueError("pb_int sga_capitalization_rate must be in [0, 1]")
    if not 0.0 <= sga_depreciation <= 1.0:
        raise ValueError("pb_int sga_depreciation_rate must be in [0, 1]")

    output = pd.DataFrame(index=frame.index)
    output["rd_investment"] = rd
    output["rd_source"] = rd_source
    output["sga_investment"] = sga
    output["goodwill"] = goodwill
    output["rd_depreciation_rate"] = np.nan
    output["rd_capital"] = np.nan
    output["sga_capital"] = np.nan
    output["intangible_capital"] = np.nan

    for security_id, group in frame.groupby(level="security_id", sort=False):
        group = group.sort_index(level="report_period")
        establishment = establishment_dates.get(str(security_id), pd.NaT)
        industry_l1 = daily.at[str(security_id), "industry_l1"] if (
            str(security_id) in daily.index and "industry_l1" in daily
        ) else pd.NA
        industry_l2 = daily.at[str(security_id), "industry_l2"] if (
            str(security_id) in daily.index and "industry_l2" in daily
        ) else pd.NA
        mapped_rate = _report_rate(industry_l1, industry_l2, parameters, calculation_date)

        rd_capital = np.nan
        sga_capital = np.nan
        previous_year: int | None = None
        for index in group.index:
            report_period = pd.Timestamp(index[1])
            year = int(report_period.year)
            annual_rd = float(rd.loc[index]) if pd.notna(rd.loc[index]) else np.nan
            annual_sga = float(sga.loc[index]) if pd.notna(sga.loc[index]) else np.nan
            rate = float(direct_rate.loc[index]) if pd.notna(direct_rate.loc[index]) else mapped_rate
            valid_rate = np.isfinite(rate) and 0.0 < rate < 1.0
            output.at[index, "rd_depreciation_rate"] = rate if valid_rate else np.nan

            if previous_year is None:
                years = year - int(establishment.year) if pd.notna(establishment) else -1
                rd_capital = _initial_capital(annual_rd, 1.0 - rate, growth, years) if valid_rate else np.nan
                sga_capital = _initial_capital(
                    sga_capitalization * annual_sga,
                    1.0 - sga_depreciation,
                    growth,
                    years,
                )
            elif year == previous_year + 1:
                rd_capital = (
                    (1.0 - rate) * rd_capital + annual_rd
                    if valid_rate and np.isfinite(rd_capital) and np.isfinite(annual_rd)
                    else np.nan
                )
                sga_capital = (
                    (1.0 - sga_depreciation) * sga_capital
                    + sga_capitalization * annual_sga
                    if np.isfinite(sga_capital) and np.isfinite(annual_sga)
                    else np.nan
                )
            else:
                # A missing annual observation cannot be silently interpreted
                # as zero investment; invalidate the later recursive state.
                rd_capital = np.nan
                sga_capital = np.nan
            previous_year = year

            output.at[index, "rd_capital"] = rd_capital
            output.at[index, "sga_capital"] = sga_capital
            if np.isfinite(rd_capital) and np.isfinite(sga_capital) and pd.notna(goodwill.loc[index]):
                output.at[index, "intangible_capital"] = rd_capital + sga_capital - float(goodwill.loc[index])
    return output


def _quality_reasons(row: pd.Series, market_cap: float, establishment: object) -> str:
    reasons: list[str] = []
    if pd.isna(establishment):
        reasons.append("missing_establishment_date")
    if pd.isna(row["rd_investment"]):
        reasons.append("missing_rd_investment")
    if pd.isna(row["sga_investment"]):
        reasons.append("missing_sga_components")
    if pd.isna(row["goodwill"]):
        reasons.append("missing_goodwill")
    if pd.isna(row["rd_depreciation_rate"]):
        reasons.append("missing_pit_rd_depreciation_rate")
    if not np.isfinite(market_cap) or market_cap <= 0.0:
        reasons.append("nonpositive_or_missing_total_market_cap")
    if pd.isna(row["intangible_capital"]):
        reasons.append("incomplete_annual_recursion")
    return "|".join(dict.fromkeys(reasons)) if reasons else "ok"


@registry.register("intangible_value")
def compute_intangible_value(context: FactorContext, spec: FactorSpec) -> pd.Series:
    if spec.factor_id != "pb_int":
        raise NotImplementedError(f"intangible_value does not implement {spec.factor_id}")
    observations = _normalize_observations(context.financial_observations)
    establishment_dates = _establishment_dates(context)
    results: list[pd.Series] = []
    quality: list[pd.DataFrame] = []

    for calculation_date in pd.DatetimeIndex(context.calculation_dates).normalize().sort_values().unique():
        cutoff = (
            calculation_date + pd.to_timedelta(context.pit_calculation_time)
            if context.pit_allow_release_day else calculation_date
        )
        annual = _visible_annual_frame(observations, cutoff)
        daily = _daily_cross_section(context, calculation_date)
        if annual.empty or daily.empty or "total_market_cap" not in daily:
            continue
        assets = _capitalized_annual_assets(
            annual,
            daily,
            establishment_dates,
            spec.parameters,
            calculation_date,
        )
        latest_index = assets.reset_index().groupby("security_id")["report_period"].idxmax()
        latest = assets.reset_index().loc[latest_index].set_index("security_id")
        latest.index = latest.index.astype(str)
        market_cap = pd.to_numeric(daily["total_market_cap"], errors="coerce").reindex(latest.index)
        factor = safe_divide(
            latest["intangible_capital"],
            market_cap.where(market_cap > 0.0),
        )

        allowed = context.universe.get(calculation_date)
        if allowed is None:
            allowed = context.eligible_universe.get(calculation_date)
        if allowed is not None:
            keep = latest.index.isin(allowed)
            latest = latest.loc[keep]
            market_cap = market_cap.loc[keep]
            factor = factor.loc[keep]

        flags = latest.copy()
        flags["total_market_cap"] = market_cap
        flags["establishment_date"] = establishment_dates.reindex(flags.index)
        flags["factor_value"] = factor
        flags["reason_codes"] = [
            _quality_reasons(flags.loc[security_id], float(market_cap.loc[security_id]), flags.at[security_id, "establishment_date"])
            for security_id in flags.index
        ]
        flags["valid_for_factor"] = flags["reason_codes"].eq("ok")
        flags.index = pd.MultiIndex.from_arrays(
            [[calculation_date] * len(flags), flags.index], names=INDEX_NAMES,
        )
        quality.append(flags)

        factor.index = pd.MultiIndex.from_arrays(
            [[calculation_date] * len(factor), factor.index], names=INDEX_NAMES,
        )
        results.append(factor.astype(float))

    context.cache["quality_flags:pb_int"] = (
        pd.concat(quality).sort_index()
        if quality
        else pd.DataFrame(index=pd.MultiIndex.from_arrays([[], []], names=INDEX_NAMES))
    )
    if not results:
        return pd.Series(
            dtype=float,
            index=pd.MultiIndex.from_arrays([[], []], names=INDEX_NAMES),
            name=spec.factor_id,
        )
    output = pd.concat(results).sort_index().replace([np.inf, -np.inf], np.nan)
    output.name = spec.factor_id
    return output
