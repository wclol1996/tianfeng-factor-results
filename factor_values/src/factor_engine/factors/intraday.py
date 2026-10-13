from __future__ import annotations

import re

import numpy as np
import pandas as pd

from factor_engine.catalog import FactorSpec
from factor_engine.data.base import FactorContext
from factor_engine.factors.common import INDEX_NAMES, rolling, safe_divide, suffix_window
from factor_engine.registry import registry
from factor_engine.universe import apply_universe


SUPPORTED_PATTERNS = tuple(map(re.compile, [
    r"^minute_vol_(20|60)d$",
    r"^real_(var|skew|kurtosis|upvar|downvar)_(d|w|m|q)$",
    r"^ratio_real(up|down)var_(d|w|m)$",
    r"^trendratio_(d|w|m)$",
    r"^intraday_maxdrawdown_(d|w|m)$",
    r"^ratio_volumeH[1-8]_(d|w|m)$",
    r"^corr_(VP|VR|VRlag|VRlead)_(d|w|m)$",
    r"^Amihud_illiq_(d|w|m)$",
    r"^ret_(H1|H8)_(d|w|m)$",
    r"^ret_close2H1_(d|w|m)$",
    r"^corr_(VP|VR|VRlag|VRlead)H(1|8)_(d|w|m)$",
    r"^corr_VRH(1|8)(lag|lead)_(d|w|m)$",
    r"^real_(var|skew|kurtosis)H(1|8)_(d|w|m)$",
    r"^real_varlarge_(d|w|m|q)$",
    r"^real_(skew|kurtosis)large_(d|w|m)$",
    r"^ratio_realvar_(d|w|m|q)$",
    r"^corr_(VP|VR|VRlag|VRlead)large_(d|w|m)$",
]))
SUPPORTED_EXACT = frozenset({
    "vol_intraday", "vol_intraday_5min", "UID", "UID_60d", "UTD",
    "smart_money",
})
LARGE_VOLUME_FACTOR_IDS = frozenset(
    {f"real_varlarge_{suffix}" for suffix in "dwmq"}
    | {f"real_{statistic}large_{suffix}" for statistic in ("skew", "kurtosis") for suffix in "dwm"}
    | {f"ratio_realvar_{suffix}" for suffix in "dwmq"}
    | {
        f"corr_{kind}large_{suffix}"
        for kind in ("VP", "VR", "VRlag", "VRlead")
        for suffix in "dwm"
    }
)


def supports_factor_id(factor_id: str) -> bool:
    return factor_id in SUPPORTED_EXACT or any(pattern.fullmatch(factor_id) for pattern in SUPPORTED_PATTERNS)


def _prepare(context: FactorContext, require_close: bool = True) -> pd.DataFrame:
    frame = context.intraday.copy()
    required = {"timestamp", "date", "security_id"}
    if require_close:
        required.add("close")
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"intraday data is missing canonical columns: {sorted(missing)}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame = frame.sort_values(["security_id", "date", "timestamp"]).drop_duplicates(
        ["security_id", "timestamp"], keep="last",
    )
    group = frame.groupby(["security_id", "date"], sort=False)
    if "close" in frame:
        frame["minute_return"] = group["close"].pct_change(fill_method=None)
    else:
        frame["minute_return"] = np.nan
    frame["bar_number"] = group.cumcount()
    frame["bar_count"] = group["timestamp"].transform("size")
    frame["bucket"] = np.minimum((frame["bar_number"] * 8 // frame["bar_count"]) + 1, 8).astype(int)
    return frame


def _to_series(values: pd.DataFrame, name: str = "value") -> pd.Series:
    return values.set_index(INDEX_NAMES)[name].sort_index()


def _aggregate_daily(series: pd.Series, suffix: str) -> pd.Series:
    window = suffix_window(suffix)
    return series if window == 1 else rolling(series, window, "mean")


def _daily_stat(frame: pd.DataFrame, statistic: str, bucket: int | None = None) -> pd.Series:
    sample = frame if bucket is None else frame[frame["bucket"].eq(bucket)].copy()
    grouped = sample.groupby(["date", "security_id"], sort=False)
    if statistic == "variance":
        result = grouped["minute_return"].var(ddof=1)
    elif statistic == "up_variance":
        result = grouped["minute_return"].apply(lambda values: values[values > 0].var(ddof=1))
    elif statistic == "down_variance":
        result = grouped["minute_return"].apply(lambda values: values[values < 0].var(ddof=1))
    elif statistic == "skew":
        result = grouped["minute_return"].skew()
    elif statistic == "kurtosis":
        result = grouped["minute_return"].apply(lambda values: values.kurt())
    elif statistic == "volatility":
        result = grouped["minute_return"].std(ddof=1)
    elif statistic == "trend_ratio":
        result = grouped["close"].apply(_trend_ratio)
    elif statistic == "max_drawdown":
        result = grouped["close"].apply(_max_drawdown)
    else:
        raise ValueError(f"unknown intraday statistic: {statistic}")
    result.index = result.index.set_names(INDEX_NAMES)
    return result.sort_index().astype(float)


def _large_volume_sample(frame: pd.DataFrame) -> pd.DataFrame:
    """Select exactly floor(N/3) valid-volume bars per security and day.

    Bars are ordered by descending minute volume.  Timestamp ascending is the
    deterministic boundary tie-break specified for this reproduction.
    """

    if "volume" not in frame:
        raise ValueError("intraday volume is required for large-volume factors")
    sample = frame.copy()
    sample["volume"] = pd.to_numeric(sample["volume"], errors="coerce")
    sample = sample[sample["volume"].notna()].copy()
    if sample.empty:
        return sample
    group_keys = ["security_id", "date"]
    sample["large_volume_count"] = sample.groupby(group_keys, sort=False)["volume"].transform("size") // 3
    sample = sample.sort_values(
        ["security_id", "date", "volume", "timestamp"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )
    sample["large_volume_rank"] = sample.groupby(group_keys, sort=False).cumcount() + 1
    sample = sample[sample["large_volume_rank"] <= sample["large_volume_count"]]
    return sample.sort_values(["security_id", "date", "timestamp"])


def _trend_ratio(values: pd.Series) -> float:
    clean = values.dropna().to_numpy(dtype=float)
    if len(clean) < 2:
        return np.nan
    path = np.abs(np.diff(clean)).sum()
    return float(abs(clean[-1] - clean[0]) / path) if path > 0 else np.nan


def _max_drawdown(values: pd.Series) -> float:
    clean = values.dropna().to_numpy(dtype=float)
    if len(clean) < 2:
        return np.nan
    running_peak = np.maximum.accumulate(clean)
    drawdown = clean / running_peak - 1.0
    return float(-drawdown.min())


def _daily_return(frame: pd.DataFrame, bucket: int) -> pd.Series:
    sample = frame[frame["bucket"].eq(bucket)]
    result = sample.groupby(["date", "security_id"], sort=False)["close"].apply(
        lambda values: values.iloc[-1] / values.iloc[0] - 1.0 if len(values) >= 2 and values.iloc[0] != 0 else np.nan,
    )
    result.index = result.index.set_names(INDEX_NAMES)
    return result.sort_index()


def _close_after_h1_return(frame: pd.DataFrame) -> pd.Series:
    day_close = frame.groupby(["date", "security_id"], sort=False)["close"].last()
    h1_close = frame[frame["bucket"].eq(1)].groupby(["date", "security_id"], sort=False)["close"].last()
    result = safe_divide(day_close, h1_close) - 1.0
    result.index = result.index.set_names(INDEX_NAMES)
    return result.sort_index()


def _volume_ratio(frame: pd.DataFrame, bucket: int) -> pd.Series:
    if "volume" not in frame:
        raise ValueError("intraday volume is required")
    total = frame.groupby(["date", "security_id"], sort=False)["volume"].sum()
    segment = frame[frame["bucket"].eq(bucket)].groupby(["date", "security_id"], sort=False)["volume"].sum()
    result = safe_divide(segment, total)
    result.index = result.index.set_names(INDEX_NAMES)
    return result.sort_index()


def _daily_correlation(frame: pd.DataFrame, kind: str, bucket: int | None = None) -> pd.Series:
    if "volume" not in frame:
        raise ValueError("intraday volume is required for correlation factors")
    sample = frame if bucket is None else frame[frame["bucket"].eq(bucket)].copy()
    # Keep the return attached to the minute itself.  Recomputing pct_change
    # after slicing H8 would incorrectly discard the first return in the final
    # half-hour.  H1 still excludes the overnight observation because
    # ``_prepare`` resets pct_change at each trading-date boundary.
    sample["local_return"] = sample["minute_return"]
    if kind == "VP":
        sample["target"] = sample["close"]
    elif kind == "VR":
        sample["target"] = sample["local_return"]
    elif kind == "VRlag":
        sample["target"] = sample.groupby(["security_id", "date"], sort=False)["local_return"].shift(1)
    elif kind == "VRlead":
        sample["target"] = sample.groupby(["security_id", "date"], sort=False)["local_return"].shift(-1)
    else:
        raise ValueError(f"unknown correlation kind: {kind}")

    def correlation(group: pd.DataFrame) -> float:
        clean = group[["volume", "target"]].dropna()
        if len(clean) < 3 or clean["volume"].std() == 0 or clean["target"].std() == 0:
            return np.nan
        return float(clean["volume"].corr(clean["target"]))

    result = sample.groupby(["date", "security_id"], sort=False).apply(correlation, include_groups=False)
    result.index = result.index.set_names(INDEX_NAMES)
    return result.sort_index()


def _daily_large_correlation(frame: pd.DataFrame, kind: str) -> pd.Series:
    if "volume" not in frame:
        raise ValueError("intraday volume is required for large-volume correlation factors")
    prepared = frame.copy()
    group = prepared.groupby(["security_id", "date"], sort=False)
    if kind == "VP":
        prepared["target"] = prepared["close"]
    elif kind == "VR":
        prepared["target"] = prepared["minute_return"]
    elif kind == "VRlag":
        prepared["target"] = group["minute_return"].shift(1)
    elif kind == "VRlead":
        prepared["target"] = group["minute_return"].shift(-1)
    else:
        raise ValueError(f"unknown large-volume correlation kind: {kind}")
    sample = _large_volume_sample(prepared)

    def correlation(values: pd.DataFrame) -> float:
        clean = values[["volume", "target"]].dropna()
        if len(clean) < 3 or clean["volume"].std() == 0 or clean["target"].std() == 0:
            return np.nan
        return float(clean["volume"].corr(clean["target"]))

    result = sample.groupby(["date", "security_id"], sort=False).apply(
        correlation, include_groups=False,
    )
    result.index = result.index.set_names(INDEX_NAMES)
    return result.sort_index()


def _amihud(frame: pd.DataFrame) -> pd.Series:
    if "amount" not in frame:
        raise ValueError("intraday amount is required for Amihud factors")
    sample = frame.copy()
    sample["illiquidity"] = np.where(
        sample["amount"].astype(float) > 0,
        sample["minute_return"].abs() / sample["amount"].astype(float),
        np.nan,
    )
    result = sample.groupby(["date", "security_id"], sort=False)["illiquidity"].mean()
    result.index = result.index.set_names(INDEX_NAMES)
    return result.sort_index()


def _five_minute_volatility(frame: pd.DataFrame) -> pd.Series:
    output: list[tuple[pd.Timestamp, str, float]] = []
    for (security_id, trade_date), group in frame.groupby(["security_id", "date"], sort=False):
        prices = group.set_index("timestamp")["close"].resample("5min").last().dropna()
        output.append((trade_date, str(security_id), float(prices.pct_change(fill_method=None).std(ddof=1))))
    result = pd.DataFrame(output, columns=["date", "security_id", "value"])
    return _to_series(result)


def _uid_daily_volatility(
    frame: pd.DataFrame,
    ddof: int,
    minimum_intraday_returns: int,
) -> pd.Series:
    """Return each stock-day's standard deviation of intraday-only returns."""

    if ddof not in {0, 1}:
        raise ValueError("UID intraday_std_ddof must be 0 or 1")
    if minimum_intraday_returns < 2:
        raise ValueError("UID minimum_intraday_returns must be at least 2")

    def calculate(values: pd.Series) -> float:
        clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
        if len(clean) < minimum_intraday_returns or len(clean) <= ddof:
            return np.nan
        return float(np.std(clean, ddof=ddof))

    result = frame.groupby(["date", "security_id"], sort=False)["minute_return"].apply(
        calculate,
    )
    result.index = result.index.set_names(INDEX_NAMES)
    return result.sort_index().astype(float)


def _utd_daily_turnover_volatility(
    context: FactorContext,
    frame: pd.DataFrame,
    float_shares_field: str,
    ddof: int,
    minimum_intraday_observations: int,
) -> pd.Series:
    """Return the intraday dispersion of minute volume / float shares."""

    if "volume" not in frame:
        raise ValueError("UTD intraday volume is required")
    if ddof not in {0, 1}:
        raise ValueError("UTD intraday_std_ddof must be 0 or 1")
    if minimum_intraday_observations < 2:
        raise ValueError("UTD minimum_intraday_observations must be at least 2")
    if list(context.daily.index.names) != INDEX_NAMES:
        raise ValueError("UTD daily frame must use a ['date', 'security_id'] MultiIndex")
    if float_shares_field not in context.daily:
        raise ValueError(f"UTD daily data is missing {float_shares_field!r}")

    sample = frame.copy()
    sample["security_id"] = sample["security_id"].astype(str)
    shares = (
        context.daily[float_shares_field]
        .astype(float)
        .rename("float_shares")
        .reset_index()
    )
    shares["date"] = pd.to_datetime(shares["date"]).dt.normalize()
    shares["security_id"] = shares["security_id"].astype(str)
    sample = sample.merge(
        shares,
        on=["date", "security_id"],
        how="left",
        validate="many_to_one",
    )
    volume = pd.to_numeric(sample["volume"], errors="coerce")
    float_shares = pd.to_numeric(sample["float_shares"], errors="coerce")
    sample["minute_turnover"] = np.where(
        (volume >= 0.0) & (float_shares > 0.0),
        volume / float_shares,
        np.nan,
    )

    def calculate(values: pd.Series) -> float:
        clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
        if len(clean) < minimum_intraday_observations or len(clean) <= ddof:
            return np.nan
        return float(np.std(clean, ddof=ddof))

    result = sample.groupby(["date", "security_id"], sort=False)["minute_turnover"].apply(
        calculate,
    )
    result.index = result.index.set_names(INDEX_NAMES)
    return result.sort_index().astype(float)


def _rolling_smart_money(
    frame: pd.DataFrame,
    market_calendar: pd.DatetimeIndex,
    *,
    window: int,
    smart_volume_fraction: float,
    volume_power: float,
    minimum_minutes_per_day: int,
    include_first_minute_return: bool,
) -> pd.Series:
    """Compute Q = VWAP_smart / VWAP_all on a fixed ten-day minute window.

    By default the first bar of a day uses the previous market day's final
    adjusted close, because the supplied smart-money definition says every
    minute return and does not contain UID's explicit overnight exclusion.
    ``include_first_minute_return=False`` preserves a configurable intraday-only
    alternative.  The deterministic smart set includes the first S-ranked bar
    that crosses the target cumulative-volume share.
    """

    if window < 1:
        raise ValueError("smart_money window must be positive")
    if not 0.0 < smart_volume_fraction <= 1.0:
        raise ValueError("smart_money smart_volume_fraction must be in (0, 1]")
    if volume_power < 0.0:
        raise ValueError("smart_money volume_power must be non-negative")
    if minimum_minutes_per_day < 2:
        raise ValueError("smart_money minimum_minutes_per_day must be at least 2")
    if "volume" not in frame:
        raise ValueError("smart_money requires intraday volume")

    calendar = pd.DatetimeIndex(market_calendar).normalize().sort_values().unique()
    calendar_positions = pd.Series(np.arange(len(calendar), dtype=int), index=calendar)
    prepared = frame.copy()
    prepared["close"] = pd.to_numeric(prepared["close"], errors="coerce")
    prepared["volume"] = pd.to_numeric(prepared["volume"], errors="coerce")
    prepared["minute_return"] = pd.to_numeric(
        prepared["minute_return"], errors="coerce",
    )
    prepared["market_position"] = (
        prepared["date"].map(calendar_positions).fillna(-1).astype(int)
    )
    prepared["score_return"] = prepared["minute_return"]
    if include_first_minute_return:
        security_group = prepared.groupby("security_id", sort=False)
        continuous_return = security_group["close"].pct_change(fill_method=None)
        first_bar = prepared.groupby(
            ["security_id", "date"], sort=False,
        ).cumcount().eq(0)
        previous_market_position = security_group["market_position"].shift(1)
        has_previous_market_day = (
            prepared["market_position"] - previous_market_position
        ).eq(1)
        prepared.loc[first_bar & has_previous_market_day, "score_return"] = (
            continuous_return[first_bar & has_previous_market_day]
        )

    output: list[dict[str, object]] = []
    for security_id, security in prepared.groupby("security_id", sort=False):
        security = security.sort_values(["date", "timestamp"]).copy()
        by_date = {date: group for date, group in security.groupby("date", sort=False)}
        security_dates = pd.DatetimeIndex(sorted(by_date))
        positions = security_dates.map(calendar_positions).fillna(-1).to_numpy(dtype=int)
        for date_position in range(window - 1, len(security_dates)):
            window_dates = security_dates[date_position - window + 1 : date_position + 1]
            window_positions = positions[date_position - window + 1 : date_position + 1]
            if window_positions[0] < 0 or not np.all(np.diff(window_positions) == 1):
                continue
            parts = [by_date[date] for date in window_dates]
            if any(
                int((part["close"].gt(0.0) & part["volume"].gt(0.0)).sum())
                < minimum_minutes_per_day
                for part in parts
            ):
                continue
            sample = pd.concat(parts, ignore_index=True)
            all_valid = (
                sample["close"].gt(0.0)
                & sample["volume"].gt(0.0)
                & np.isfinite(sample["close"])
                & np.isfinite(sample["volume"])
            )
            all_trades = sample.loc[all_valid].copy()
            total_volume = float(all_trades["volume"].sum())
            if not np.isfinite(total_volume) or total_volume <= 0.0:
                continue
            vwap_all = float(
                np.average(
                    all_trades["close"].to_numpy(dtype=float),
                    weights=all_trades["volume"].to_numpy(dtype=float),
                )
            )

            candidates = all_trades[np.isfinite(all_trades["score_return"])].copy()
            if candidates.empty:
                continue
            candidates["smart_score"] = (
                candidates["score_return"].abs()
                / candidates["volume"].pow(volume_power)
            )
            candidates = candidates.sort_values(
                ["smart_score", "timestamp"],
                ascending=[False, True],
                kind="mergesort",
            )
            prior_cumulative_volume = candidates["volume"].cumsum() - candidates["volume"]
            smart = candidates[
                prior_cumulative_volume < smart_volume_fraction * total_volume
            ]
            if smart.empty:
                continue
            vwap_smart = float(
                np.average(
                    smart["close"].to_numpy(dtype=float),
                    weights=smart["volume"].to_numpy(dtype=float),
                )
            )
            if np.isfinite(vwap_smart) and np.isfinite(vwap_all) and vwap_all > 0.0:
                output.append({
                    "date": window_dates[-1],
                    "security_id": str(security_id),
                    "value": vwap_smart / vwap_all,
                })

    if not output:
        return pd.Series(
            dtype=float,
            index=pd.MultiIndex.from_arrays([[], []], names=INDEX_NAMES),
        )
    return pd.DataFrame(output).set_index(INDEX_NAMES)["value"].sort_index()


def _rolling_distribution_cv(
    daily_volatility: pd.Series,
    market_calendar: pd.DatetimeIndex,
    window: int,
    minimum_observations: int,
    ddof: int,
) -> pd.Series:
    """Calculate std(Vol_daily) / mean(Vol_daily) on a fixed date window."""

    if window < 2:
        raise ValueError("distribution CV window must be at least 2")
    if not 2 <= minimum_observations <= window:
        raise ValueError("distribution CV minimum_observations must be between 2 and window")
    if ddof not in {0, 1}:
        raise ValueError("distribution CV rolling_std_ddof must be 0 or 1")

    calendar = pd.DatetimeIndex(market_calendar).normalize().sort_values().unique()
    calendar_positions = pd.Series(np.arange(len(calendar), dtype=int), index=calendar)
    frame = daily_volatility.rename("vol_daily").reset_index()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["market_position"] = frame["date"].map(calendar_positions).fillna(-1).astype(int)

    output: list[pd.DataFrame] = []
    for _, group in frame.groupby("security_id", sort=False):
        group = group.sort_values("date").copy()
        result = np.full(len(group), np.nan)
        for position in range(window - 1, len(group)):
            sample = group.iloc[position - window + 1 : position + 1]
            positions = sample["market_position"].to_numpy(dtype=int)
            if positions[0] < 0 or not np.all(np.diff(positions) == 1):
                continue
            values = sample["vol_daily"].to_numpy(dtype=float)
            valid = np.isfinite(values)
            if valid.sum() < minimum_observations or valid.sum() <= ddof:
                continue
            mean = float(values[valid].mean())
            if not np.isfinite(mean) or mean <= 0.0:
                continue
            result[position] = float(np.std(values[valid], ddof=ddof) / mean)
        group["result"] = result
        output.append(group[["date", "security_id", "result"]])

    if not output:
        return pd.Series(
            dtype=float,
            index=pd.MultiIndex.from_arrays([[], []], names=INDEX_NAMES),
        )
    return pd.concat(output).set_index(INDEX_NAMES)["result"].sort_index()


def _intraday_factor_market_cap_residual(
    context: FactorContext,
    uid: pd.Series,
    market_cap_field: str,
    market_cap_transform: str,
    minimum_samples: int,
) -> pd.Series:
    """Neutralize an intraday-derived factor against market cap each day."""

    if market_cap_transform not in {"raw", "log"}:
        raise ValueError("intraday factor market_cap_transform must be 'raw' or 'log'")
    if minimum_samples < 3:
        raise ValueError("intraday factor minimum_regression_samples must be at least 3")
    if list(context.daily.index.names) != INDEX_NAMES:
        raise ValueError("daily frame must use a ['date', 'security_id'] MultiIndex")
    if market_cap_field not in context.daily:
        raise ValueError(f"daily data is missing {market_cap_field!r}")

    market_cap = context.daily[market_cap_field].astype(float).sort_index()
    market_cap = market_cap.where(market_cap > 0.0)
    if market_cap_transform == "log":
        market_cap = np.log(market_cap)
    construction_universe = context.eligible_universe or context.universe
    if construction_universe:
        universe = dict(construction_universe)
        uid = apply_universe(uid, universe)
        market_cap = apply_universe(market_cap, universe)

    combined = pd.concat([
        uid.rename("uid"),
        market_cap.rename("market_cap"),
    ], axis=1)
    output: list[pd.Series] = []
    for _, group in combined.groupby(level="date", sort=False):
        residual = pd.Series(np.nan, index=group.index, dtype=float)
        clean = group.replace([np.inf, -np.inf], np.nan).dropna()
        if len(clean) >= minimum_samples:
            regressor = clean["market_cap"].to_numpy(dtype=float)
            scale = float(np.median(np.abs(regressor)))
            if np.isfinite(scale) and scale > 0.0:
                design = np.column_stack([
                    np.ones(len(clean)),
                    regressor / scale,
                ])
                if np.linalg.matrix_rank(design) == 2:
                    target = clean["uid"].to_numpy(dtype=float)
                    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
                    residual.loc[clean.index] = target - design @ coefficients
        output.append(residual)
    if not output:
        return uid * np.nan
    return pd.concat(output).sort_index()


@registry.register("intraday_microstructure")
def compute_intraday(context: FactorContext, spec: FactorSpec) -> pd.Series:
    factor_id = spec.factor_id
    if not supports_factor_id(factor_id):
        raise NotImplementedError(f"intraday_microstructure does not implement {factor_id}")
    frame = _prepare(context, require_close=factor_id != "UTD")

    if factor_id == "vol_intraday":
        result = _daily_stat(frame, "volatility")
    elif factor_id == "vol_intraday_5min":
        result = _five_minute_volatility(frame)
    elif factor_id in {"UID", "UID_60d"}:
        default_window = 20 if factor_id == "UID" else 60
        window = int(spec.parameters.get("window", default_window))
        if not context.daily.empty:
            market_calendar = context.daily.index.get_level_values("date")
        else:
            market_calendar = frame["date"]
        raw_uid = _rolling_distribution_cv(
            _uid_daily_volatility(
                frame,
                int(spec.parameters.get("intraday_std_ddof", 1)),
                int(spec.parameters.get("minimum_intraday_returns", 2)),
            ),
            pd.DatetimeIndex(market_calendar),
            window,
            int(spec.parameters.get("minimum_daily_observations", window)),
            int(spec.parameters.get("rolling_std_ddof", 1)),
        )
        result = _intraday_factor_market_cap_residual(
            context,
            raw_uid,
            str(spec.parameters.get("market_cap_field", "total_market_cap")),
            str(spec.parameters.get("market_cap_transform", "log")),
            int(spec.parameters.get("minimum_regression_samples", 3)),
        )
    elif factor_id == "UTD":
        window = int(spec.parameters.get("window", 20))
        market_calendar = context.daily.index.get_level_values("date")
        raw_utd = _rolling_distribution_cv(
            _utd_daily_turnover_volatility(
                context,
                frame,
                str(spec.parameters.get("float_shares_field", "float_shares")),
                int(spec.parameters.get("intraday_std_ddof", 1)),
                int(spec.parameters.get("minimum_intraday_observations", 2)),
            ),
            pd.DatetimeIndex(market_calendar),
            window,
            int(spec.parameters.get("minimum_daily_observations", window)),
            int(spec.parameters.get("rolling_std_ddof", 1)),
        )
        result = _intraday_factor_market_cap_residual(
            context,
            raw_utd,
            str(spec.parameters.get("market_cap_field", "total_market_cap")),
            str(spec.parameters.get("market_cap_transform", "log")),
            int(spec.parameters.get("minimum_regression_samples", 3)),
        )
    elif factor_id == "smart_money":
        market_calendar = (
            context.daily.index.get_level_values("date")
            if not context.daily.empty else frame["date"]
        )
        result = _rolling_smart_money(
            frame,
            pd.DatetimeIndex(market_calendar),
            window=int(spec.parameters.get("window", 10)),
            smart_volume_fraction=float(
                spec.parameters.get("smart_volume_fraction", 0.20),
            ),
            volume_power=float(spec.parameters.get("volume_power", 0.25)),
            minimum_minutes_per_day=int(
                spec.parameters.get("minimum_minutes_per_day", 2),
            ),
            include_first_minute_return=bool(
                spec.parameters.get("include_first_minute_return", True),
            ),
        )
    elif (match := re.fullmatch(r"minute_vol_(20|60)d", factor_id)):
        result = rolling(_daily_stat(frame, "volatility"), int(match.group(1)), "mean")
    elif (match := re.fullmatch(r"real_(var|skew|kurtosis|upvar|downvar)_(d|w|m|q)", factor_id)):
        statistic = {
            "var": "variance", "skew": "skew", "kurtosis": "kurtosis",
            "upvar": "up_variance", "downvar": "down_variance",
        }[match.group(1)]
        result = _aggregate_daily(_daily_stat(frame, statistic), match.group(2))
    elif (match := re.fullmatch(r"ratio_real(up|down)var_(d|w|m)", factor_id)):
        side = _daily_stat(frame, f"{match.group(1)}_variance")
        total = _daily_stat(frame, "variance")
        result = _aggregate_daily(safe_divide(side, total), match.group(2))
    elif (match := re.fullmatch(r"trendratio_(d|w|m)", factor_id)):
        result = _aggregate_daily(_daily_stat(frame, "trend_ratio"), match.group(1))
    elif (match := re.fullmatch(r"intraday_maxdrawdown_(d|w|m)", factor_id)):
        result = _aggregate_daily(_daily_stat(frame, "max_drawdown"), match.group(1))
    elif (match := re.fullmatch(r"ratio_volumeH([1-8])_(d|w|m)", factor_id)):
        result = _aggregate_daily(_volume_ratio(frame, int(match.group(1))), match.group(2))
    elif (match := re.fullmatch(r"corr_(VP|VR|VRlag|VRlead)_(d|w|m)", factor_id)):
        result = _aggregate_daily(_daily_correlation(frame, match.group(1)), match.group(2))
    elif (match := re.fullmatch(r"Amihud_illiq_(d|w|m)", factor_id)):
        result = _aggregate_daily(_amihud(frame), match.group(1))
    elif (match := re.fullmatch(r"ret_(H1|H8)_(d|w|m)", factor_id)):
        bucket = 1 if match.group(1) == "H1" else 8
        result = _aggregate_daily(_daily_return(frame, bucket), match.group(2))
    elif (match := re.fullmatch(r"ret_close2H1_(d|w|m)", factor_id)):
        result = _aggregate_daily(_close_after_h1_return(frame), match.group(1))
    elif (match := re.fullmatch(r"corr_(VP|VR|VRlag|VRlead)H(1|8)_(d|w|m)", factor_id)):
        result = _aggregate_daily(_daily_correlation(frame, match.group(1), int(match.group(2))), match.group(3))
    elif (match := re.fullmatch(r"corr_VRH(1|8)(lag|lead)_(d|w|m)", factor_id)):
        result = _aggregate_daily(
            _daily_correlation(frame, f"VR{match.group(2)}", int(match.group(1))),
            match.group(3),
        )
    elif (match := re.fullmatch(r"real_(var|skew|kurtosis)H(1|8)_(d|w|m)", factor_id)):
        statistic = {"var": "variance", "skew": "skew", "kurtosis": "kurtosis"}[match.group(1)]
        result = _aggregate_daily(_daily_stat(frame, statistic, int(match.group(2))), match.group(3))
    elif (match := re.fullmatch(r"real_varlarge_(d|w|m|q)", factor_id)):
        result = _aggregate_daily(
            _daily_stat(_large_volume_sample(frame), "variance"), match.group(1),
        )
    elif (match := re.fullmatch(r"real_(skew|kurtosis)large_(d|w|m)", factor_id)):
        result = _aggregate_daily(
            _daily_stat(_large_volume_sample(frame), match.group(1)), match.group(2),
        )
    elif (match := re.fullmatch(r"ratio_realvar_(d|w|m|q)", factor_id)):
        large = _daily_stat(_large_volume_sample(frame), "variance")
        total = _daily_stat(frame, "variance")
        result = _aggregate_daily(safe_divide(large, total), match.group(1))
    elif (match := re.fullmatch(r"corr_(VP|VR|VRlag|VRlead)large_(d|w|m)", factor_id)):
        result = _aggregate_daily(_daily_large_correlation(frame, match.group(1)), match.group(2))
    else:
        raise NotImplementedError(f"intraday_microstructure does not implement {factor_id}")

    result.name = factor_id
    return result.replace([np.inf, -np.inf], np.nan).sort_index()
