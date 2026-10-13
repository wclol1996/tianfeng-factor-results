from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


QUARTER_ENDS = frozenset({"03-31", "06-30", "09-30", "12-31"})


@dataclass(frozen=True)
class SourceAudit:
    file_name: str
    input_rows: int
    type_a_rows: int
    january_1_rows_removed: int
    non_quarter_rows_removed: int
    exact_duplicates_removed: int
    b_share_rows_removed: int
    output_rows: int
    securities: int
    min_report_period: str | None
    max_report_period: str | None


def _security_id(code: pd.Series) -> pd.Series:
    raw = code.astype("string").str.strip().str.upper().str.replace(r"\.0$", "", regex=True)
    existing = raw.str.fullmatch(r"\d{6}\.(SZ|SH|BJ)", na=False)
    normalized = raw.str.extract(r"^(\d+)", expand=False).str.zfill(6)
    suffix = np.where(
        normalized.str.startswith(("4", "8", "92"), na=False),
        ".BJ",
        np.where(normalized.str.startswith(("6", "9"), na=False), ".SH", ".SZ"),
    )
    inferred = normalized + suffix
    return inferred.where(~existing, raw)


def read_csmar_statement(
    path: str | Path,
    value_columns: Iterable[str],
    include_b_shares: bool = False,
) -> tuple[pd.DataFrame, SourceAudit]:
    """Read a CSMAR statement export and retain consolidated quarter-end rows.

    The first two data rows in these exports are Chinese labels and units. January
    1 rows are comparative/opening figures from a later filing, rather than a new
    report period, and are deliberately excluded to avoid retrospective leakage.
    """

    source = Path(path)
    values = list(value_columns)
    columns = ["Stkcd", "ShortName", "Accper", "Typrep", *values]
    frame = pd.read_excel(
        source,
        sheet_name="sheet1",
        header=0,
        skiprows=[1, 2],
        usecols=columns,
        dtype={"Stkcd": "string", "Typrep": "string"},
    )
    input_rows = len(frame)
    frame["Typrep"] = frame["Typrep"].str.strip()
    frame = frame[frame["Typrep"].eq("A")].copy()
    type_a_rows = len(frame)
    frame["report_period"] = pd.to_datetime(frame["Accper"], errors="coerce").dt.normalize()
    if frame["report_period"].isna().any():
        raise ValueError(f"{source.name}: invalid report dates: {int(frame['report_period'].isna().sum())}")

    month_day = frame["report_period"].dt.strftime("%m-%d")
    january_rows = int(month_day.eq("01-01").sum())
    non_quarter_rows = int((~month_day.isin(QUARTER_ENDS) & ~month_day.eq("01-01")).sum())
    frame = frame[month_day.isin(QUARTER_ENDS)].copy()
    frame["security_id"] = _security_id(frame["Stkcd"])
    is_b_share = frame["security_id"].str.startswith(("200", "900"), na=False)
    b_share_rows = int(is_b_share.sum())
    if not include_b_shares:
        frame = frame[~is_b_share].copy()
    for column in values:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    exact_duplicates = int(frame.duplicated().sum())
    frame = frame.drop_duplicates()
    key = ["security_id", "report_period"]
    conflicts = frame.duplicated(key, keep=False)
    if conflicts.any():
        sample = frame.loc[conflicts, key + values].sort_values(key).head(20)
        raise ValueError(
            f"{source.name}: conflicting duplicate security/report rows cannot be resolved without "
            f"an availability timestamp:\n{sample.to_string(index=False)}"
        )

    output = frame[["security_id", "ShortName", "report_period", *values]].rename(
        columns={"ShortName": "short_name"}
    ).sort_values(["security_id", "report_period"])
    audit = SourceAudit(
        file_name=source.name,
        input_rows=input_rows,
        type_a_rows=type_a_rows,
        january_1_rows_removed=january_rows,
        non_quarter_rows_removed=non_quarter_rows,
        exact_duplicates_removed=exact_duplicates,
        b_share_rows_removed=0 if include_b_shares else b_share_rows,
        output_rows=len(output),
        securities=output["security_id"].nunique(),
        min_report_period=_iso_or_none(output["report_period"].min()),
        max_report_period=_iso_or_none(output["report_period"].max()),
    )
    return output.reset_index(drop=True), audit


def _iso_or_none(value: object) -> str | None:
    return None if pd.isna(value) else pd.Timestamp(value).date().isoformat()


def build_quarterly_delta_roe(
    balance_sheet: pd.DataFrame,
    income_statement: pd.DataFrame,
    start: str | pd.Timestamp = "2010-01-01",
) -> pd.DataFrame:
    """Build single-quarter ROE and its same-quarter year-over-year change.

    Income statement values are treated as year-to-date cumulative values. Q1 is
    used directly; Q2/Q3/Q4 are differences from the preceding report in the same
    fiscal year. ROE uses average beginning/end net assets, where net assets follow
    the user-specified total-assets-minus-total-liabilities definition.
    """

    merged = balance_sheet.merge(
        income_statement.drop(columns=["short_name"], errors="ignore"),
        on=["security_id", "report_period"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    merged["net_assets"] = merged["A001000000"] - merged["A002000000"]
    merged["year"] = merged["report_period"].dt.year
    merged["quarter"] = merged["report_period"].dt.quarter
    merged["quarter_ordinal"] = merged["year"] * 4 + merged["quarter"] - 1
    merged = merged.sort_values(["security_id", "quarter_ordinal"])

    previous = merged[["security_id", "quarter_ordinal", "B002000101", "net_assets"]].copy()
    previous["quarter_ordinal"] += 1
    previous = previous.rename(
        columns={"B002000101": "previous_ytd_profit", "net_assets": "begin_net_assets"}
    )
    merged = merged.merge(previous, on=["security_id", "quarter_ordinal"], how="left", validate="one_to_one")

    merged["single_quarter_parent_profit"] = np.where(
        merged["quarter"].eq(1),
        merged["B002000101"],
        merged["B002000101"] - merged["previous_ytd_profit"],
    )
    average_assets_sum = merged["begin_net_assets"] + merged["net_assets"]
    merged["quarterly_roe"] = (
        2.0 * merged["single_quarter_parent_profit"] / average_assets_sum.where(average_assets_sum.ne(0))
    )

    prior_year = merged[["security_id", "quarter_ordinal", "quarterly_roe"]].copy()
    prior_year["quarter_ordinal"] += 4
    prior_year = prior_year.rename(columns={"quarterly_roe": "quarterly_roe_prior_year"})
    merged = merged.merge(prior_year, on=["security_id", "quarter_ordinal"], how="left", validate="one_to_one")
    merged["delta_roe"] = merged["quarterly_roe"] - merged["quarterly_roe_prior_year"]
    merged["availability_proxy"] = merged["report_period"].map(statutory_deadline_proxy)
    merged["source_match"] = merged["_merge"].astype("string")

    columns = [
        "security_id",
        "short_name",
        "report_period",
        "availability_proxy",
        "year",
        "quarter",
        "A001000000",
        "A002000000",
        "net_assets",
        "begin_net_assets",
        "B002000101",
        "previous_ytd_profit",
        "single_quarter_parent_profit",
        "quarterly_roe",
        "quarterly_roe_prior_year",
        "delta_roe",
        "source_match",
    ]
    return merged.loc[merged["report_period"].ge(pd.Timestamp(start)), columns].sort_values(
        ["report_period", "security_id"]
    ).reset_index(drop=True)


def statutory_deadline_proxy(report_period: pd.Timestamp) -> pd.Timestamp:
    """Conservative normal-filing deadline used only when actual announcement dates are absent."""

    report_period = pd.Timestamp(report_period).normalize()
    year, quarter = report_period.year, report_period.quarter
    if quarter == 1:
        return pd.Timestamp(year=year, month=4, day=30)
    if quarter == 2:
        return pd.Timestamp(year=year, month=8, day=31)
    if quarter == 3:
        return pd.Timestamp(year=year, month=10, day=31)
    return pd.Timestamp(year=year + 1, month=4, day=30)


def load_announcement_dates(path: str | Path) -> pd.DataFrame:
    """Load a future PIT adapter file with security/report/availability columns."""

    source = Path(path)
    if source.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(source)
    elif source.suffix.lower() == ".pkl":
        frame = pd.read_pickle(source)
    else:
        frame = pd.read_csv(source)
    aliases = {
        "Stkcd": "security_id",
        "wind_code": "security_id",
        "Accper": "report_period",
        "Annodt": "available_at",
        "announcement_date": "available_at",
    }
    frame = frame.rename(columns={key: value for key, value in aliases.items() if key in frame.columns})
    required = {"security_id", "report_period", "available_at"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"announcement file is missing columns: {sorted(missing)}")
    if not frame["security_id"].astype("string").str.contains(r"\.", regex=True).all():
        frame["security_id"] = _security_id(frame["security_id"])
    frame["report_period"] = pd.to_datetime(frame["report_period"]).dt.normalize()
    frame["available_at"] = pd.to_datetime(frame["available_at"])
    frame = frame.sort_values("available_at").drop_duplicates(
        ["security_id", "report_period"], keep="first"
    )
    return frame[["security_id", "report_period", "available_at"]]


def fetch_sse_trading_calendar(start: str | date, end: str | date) -> pd.DatetimeIndex:
    """Use Shanghai Composite observations as an exchange-open calendar."""

    start_ts, end_ts = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    query = urlencode(
        {
            "secid": "1.000001",
            "klt": "101",
            "fqt": "0",
            "lmt": "10000",
            "end": end_ts.strftime("%Y%m%d"),
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52",
        }
    )
    url = f"https://push2his.eastmoney.com/api/qt/stock/kline/get?{query}"
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    klines = ((payload.get("data") or {}).get("klines") or [])
    calendar = pd.DatetimeIndex(pd.to_datetime([item.split(",", 1)[0] for item in klines])).normalize()
    calendar = calendar[(calendar >= start_ts) & (calendar <= end_ts)].sort_values().unique()
    if len(calendar) < 2 or calendar.has_duplicates or not calendar.is_monotonic_increasing:
        raise ValueError("downloaded trading calendar failed validation")
    if (calendar.dayofweek >= 5).any():
        raise ValueError("downloaded trading calendar unexpectedly contains weekend dates")
    return calendar


def load_trading_calendar(path: str | Path, start: str | date, end: str | date) -> pd.DatetimeIndex:
    source = Path(path)
    if source.suffix.lower() == ".pkl":
        obj = pd.read_pickle(source)
        dates = obj.index if hasattr(obj, "index") else obj
    else:
        frame = pd.read_csv(source)
        date_column = "date" if "date" in frame.columns else frame.columns[0]
        dates = frame[date_column]
    calendar = pd.DatetimeIndex(pd.to_datetime(dates)).normalize().sort_values().unique()
    return calendar[(calendar >= pd.Timestamp(start)) & (calendar <= pd.Timestamp(end))]


def _next_trading_days(available: pd.Series, calendar: pd.DatetimeIndex) -> pd.Series:
    normalized = pd.to_datetime(available).dt.normalize().to_numpy(dtype="datetime64[ns]")
    positions = calendar.values.searchsorted(normalized, side="right")
    result = np.full(len(positions), np.datetime64("NaT"), dtype="datetime64[ns]")
    valid = positions < len(calendar)
    result[valid] = calendar.values[positions[valid]]
    return pd.Series(pd.to_datetime(result), index=available.index)


def _inverse_normal_cdf(probability: np.ndarray) -> np.ndarray:
    """Vectorized Acklam approximation for the standard-normal quantile."""

    p = np.asarray(probability, dtype=float)
    result = np.full(p.shape, np.nan, dtype=float)
    valid = (p > 0) & (p < 1)
    if not valid.any():
        return result
    a = np.array([-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
                  1.383577518672690e2, -3.066479806614716e1, 2.506628277459239])
    b = np.array([-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
                  6.680131188771972e1, -1.328068155288572e1])
    c = np.array([-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838,
                  -2.549732539343734, 4.374664141464968, 2.938163982698783])
    d = np.array([7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996,
                  3.754408661907416])
    low, high = 0.02425, 1 - 0.02425
    lower = valid & (p < low)
    upper = valid & (p > high)
    central = valid & ~(lower | upper)
    if lower.any():
        q = np.sqrt(-2 * np.log(p[lower]))
        result[lower] = np.polyval(c, q) / np.polyval(np.r_[d, 1.0], q)
    if upper.any():
        q = np.sqrt(-2 * np.log(1 - p[upper]))
        result[upper] = -np.polyval(c, q) / np.polyval(np.r_[d, 1.0], q)
    if central.any():
        q = p[central] - 0.5
        r = q * q
        result[central] = np.polyval(a, r) * q / np.polyval(np.r_[b, 1.0], r)
    return result


def rank_gauss_cross_section(frame: pd.DataFrame) -> pd.DataFrame:
    """Map average cross-sectional ranks to Phi^-1(rank / (N + 1))."""

    ranks = frame.rank(axis=1, method="average", na_option="keep")
    counts = frame.notna().sum(axis=1)
    probability = ranks.div(counts.add(1), axis=0)
    values = _inverse_normal_cdf(probability.to_numpy(dtype=float))
    return pd.DataFrame(values, index=frame.index, columns=frame.columns)


def load_eligibility_mask(path: str | Path, calendar: pd.DatetimeIndex, columns: pd.Index) -> pd.DataFrame:
    """Load optional date/security/is_eligible data for future status filtering."""

    source = Path(path)
    frame = pd.read_pickle(source) if source.suffix.lower() == ".pkl" else pd.read_csv(source)
    if isinstance(frame.index, pd.DatetimeIndex) and not {
        "date", "security_id", "is_eligible"
    }.issubset(frame.columns):
        return frame.reindex(index=calendar, columns=columns).fillna(False).astype(bool)
    required = {"date", "security_id", "is_eligible"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"eligibility file is missing columns: {sorted(missing)}")
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    return frame.pivot(index="date", columns="security_id", values="is_eligible").reindex(
        index=calendar, columns=columns
    ).fillna(False).astype(bool)


def expand_to_daily(
    quarterly: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    announcement_dates: pd.DataFrame | None = None,
    eligibility_mask: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create daily raw DeltaROE and daily RankGauss matrices.

    Values become effective on the first trading day strictly after the supplied
    availability date. This is deliberately conservative when only a date (not a
    timestamp) is known.
    """

    events = quarterly.dropna(subset=["delta_roe"]).copy()
    if announcement_dates is None:
        events["available_at"] = events["availability_proxy"]
    else:
        events = events.merge(
            announcement_dates,
            on=["security_id", "report_period"],
            how="left",
            validate="one_to_one",
        )
        events = events.dropna(subset=["available_at"])
    events["effective_date"] = _next_trading_days(events["available_at"], calendar)
    events = events.dropna(subset=["effective_date"])
    events = events.sort_values(["effective_date", "security_id", "report_period"]).drop_duplicates(
        ["effective_date", "security_id"], keep="last"
    )
    event_matrix = events.pivot(index="effective_date", columns="security_id", values="delta_roe").sort_index()
    event_state = event_matrix.ffill()

    raw_daily = event_state.reindex(calendar).ffill()
    raw_daily.index.name = "date"
    if eligibility_mask is not None:
        raw_daily = raw_daily.where(eligibility_mask.reindex_like(raw_daily).fillna(False))
        rank_daily = rank_gauss_cross_section(raw_daily)
    else:
        rank_event_state = rank_gauss_cross_section(event_state)
        rank_daily = rank_event_state.reindex(calendar).ffill()
    rank_daily.index.name = "date"
    return raw_daily.astype(float), rank_daily.astype(float), events


def quality_summary(
    quarterly: pd.DataFrame,
    raw_daily: pd.DataFrame,
    rank_daily: pd.DataFrame,
    events: pd.DataFrame,
    source_audits: Iterable[SourceAudit],
    availability_mode: str,
    eligibility_applied: bool,
    missing_status_to_nan: bool = False,
) -> dict[str, object]:
    counts = raw_daily.notna().sum(axis=1)
    rank_mean = rank_daily.mean(axis=1, skipna=True).abs()
    report_counts = quarterly.groupby(quarterly["report_period"].dt.strftime("%Y-%m-%d"))["delta_roe"].count()
    warnings = []
    if availability_mode != "actual":
        warnings.append(
            "Source statements do not contain actual announcement/revision timestamps; "
            "deadline-proxy output is not exact PIT."
        )
    if missing_status_to_nan:
        warnings.append(
            "No stock-status eligibility file was supplied; all formal daily factor values "
            "were set to missing by policy."
        )
    elif not eligibility_applied:
        warnings.append(
            "No stock-status eligibility file was applied; "
            "delisted/ST/suspended/listing-age filtering is not represented."
        )
    return {
        "pit_status": "exact_announcement_dates" if availability_mode == "actual" else "PROVISIONAL_deadline_proxy",
        "availability_mode": availability_mode,
        "availability_rule": "first trading day strictly after available_at",
        "eligibility_filter_applied": eligibility_applied,
        "missing_status_to_nan": missing_status_to_nan,
        "source_audits": [audit.__dict__ for audit in source_audits],
        "quarterly_rows": int(len(quarterly)),
        "quarterly_non_null_delta_roe": int(quarterly["delta_roe"].notna().sum()),
        "quarterly_report_periods": int(quarterly["report_period"].nunique()),
        "daily_start": _iso_or_none(raw_daily.index.min()),
        "daily_end": _iso_or_none(raw_daily.index.max()),
        "daily_rows": int(len(raw_daily)),
        "daily_columns": int(raw_daily.shape[1]),
        "first_non_null_daily_count": int(counts[counts.gt(0)].iloc[0]) if counts.gt(0).any() else 0,
        "last_non_null_daily_count": int(counts.iloc[-1]) if len(counts) else 0,
        "max_abs_rank_gauss_daily_mean": float(rank_mean.max()) if len(rank_mean) else None,
        "effective_event_rows": int(len(events)),
        "effective_event_dates": int(events["effective_date"].nunique()),
        "delta_roe_counts_by_report_period": {key: int(value) for key, value in report_counts.items()},
        "warnings": warnings,
    }
