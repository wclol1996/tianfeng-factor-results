from __future__ import annotations

from collections.abc import Callable
from statistics import NormalDist

import numpy as np
import pandas as pd


INDEX_NAMES = ["date", "security_id"]


def restore_declared_market_dates(frame: pd.DataFrame, calculation_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Keep known warm-up dates and explicitly declared market sessions.

    Callers supply market-calendar calculation dates, not arbitrary civil dates.
    Missing quotes stay missing; absent warm-up sessions cannot be inferred.
    """
    require_daily(frame, [])
    dates = pd.DatetimeIndex(frame.index.get_level_values("date")).union(
        pd.DatetimeIndex(calculation_dates),
    ).unique().sort_values()
    codes = frame.index.get_level_values("security_id").unique()
    grid = pd.MultiIndex.from_product([dates, codes], names=INDEX_NAMES)
    return frame if frame.index.equals(grid) else frame.reindex(grid)


def require_daily(frame: pd.DataFrame, fields: list[str]) -> None:
    if list(frame.index.names) != INDEX_NAMES:
        raise ValueError("daily frame must use a ['date', 'security_id'] MultiIndex")
    missing = set(fields) - set(frame.columns)
    if missing:
        raise ValueError(f"daily data is missing canonical columns: {sorted(missing)}")


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator.astype(float).divide(denominator.astype(float))
    return result.replace([np.inf, -np.inf], np.nan)


def group_transform(series: pd.Series, function: Callable[[pd.Series], pd.Series]) -> pd.Series:
    frame = series.rename("value").reset_index()
    frame["result"] = frame.groupby("security_id", sort=False, group_keys=False)["value"].transform(function)
    return frame.set_index(INDEX_NAMES)["result"].sort_index()


def group_shift(series: pd.Series, periods: int) -> pd.Series:
    return group_transform(series, lambda values: values.shift(periods))


def group_diff(series: pd.Series, periods: int = 1) -> pd.Series:
    return group_transform(series, lambda values: values.diff(periods))


def group_pct_change(series: pd.Series, periods: int = 1) -> pd.Series:
    return group_transform(series, lambda values: values.pct_change(periods=periods, fill_method=None))


def rolling(
    series: pd.Series,
    window: int,
    operation: str | Callable[[np.ndarray], float],
    min_periods: int | None = None,
) -> pd.Series:
    minimum = window if min_periods is None else min_periods

    def calculate(values: pd.Series) -> pd.Series:
        roll = values.rolling(window, min_periods=minimum)
        if isinstance(operation, str):
            return getattr(roll, operation)()
        return roll.apply(operation, raw=True)

    return group_transform(series, calculate)


def rolling_pair(
    left: pd.Series,
    right: pd.Series,
    window: int,
    function: Callable[[np.ndarray, np.ndarray], float],
    min_periods: int | None = None,
) -> pd.Series:
    minimum = window if min_periods is None else min_periods
    aligned = pd.concat([left.rename("left"), right.rename("right")], axis=1).reset_index()
    output: list[pd.DataFrame] = []
    for _, group in aligned.groupby("security_id", sort=False):
        group = group.sort_values("date").copy()
        x = group["left"].to_numpy(dtype=float)
        y = group["right"].to_numpy(dtype=float)
        values = np.full(len(group), np.nan)
        for index in range(len(group)):
            begin = max(0, index - window + 1)
            a, b = x[begin : index + 1], y[begin : index + 1]
            valid = np.isfinite(a) & np.isfinite(b)
            if valid.sum() >= minimum:
                values[index] = function(a[valid], b[valid])
        group["result"] = values
        output.append(group[["date", "security_id", "result"]])
    if not output:
        return pd.Series(dtype=float, index=pd.MultiIndex.from_arrays([[], []], names=INDEX_NAMES))
    return pd.concat(output).set_index(INDEX_NAMES)["result"].sort_index()


def cross_sectional_rank_gauss(values: pd.Series) -> pd.Series:
    def transform(group: pd.Series) -> pd.Series:
        clean_count = group.notna().sum()
        if clean_count == 0:
            return group * np.nan
        percentile = group.rank(method="average") / (clean_count + 1.0)
        normal = NormalDist()
        return percentile.map(normal.inv_cdf)

    return values.groupby(level="date", group_keys=False).apply(transform).sort_index()


def cross_sectional_zscore(values: pd.Series) -> pd.Series:
    def transform(group: pd.Series) -> pd.Series:
        standard_deviation = group.std(ddof=1)
        if not np.isfinite(standard_deviation) or standard_deviation == 0:
            return group * np.nan
        return (group - group.mean()) / standard_deviation

    return values.groupby(level="date", group_keys=False).apply(transform).sort_index()


def cross_sectional_residual(y: pd.Series, regressors: pd.DataFrame) -> pd.Series:
    combined = pd.concat([y.rename("y"), regressors], axis=1)
    results: list[pd.Series] = []
    for _, group in combined.groupby(level="date", sort=False):
        clean = group.dropna()
        residual = pd.Series(np.nan, index=group.index, dtype=float)
        if len(clean) > len(regressors.columns) + 1:
            design = np.column_stack([np.ones(len(clean)), clean[regressors.columns].to_numpy(dtype=float)])
            coefficients, *_ = np.linalg.lstsq(design, clean["y"].to_numpy(dtype=float), rcond=None)
            residual.loc[clean.index] = clean["y"].to_numpy(dtype=float) - design @ coefficients
        results.append(residual)
    return pd.concat(results).sort_index() if results else y * np.nan


def suffix_window(suffix: str) -> int:
    mapping = {"d": 1, "w": 5, "m": 20, "q": 60}
    try:
        return mapping[suffix]
    except KeyError as error:
        raise ValueError(f"unsupported aggregation suffix: {suffix}") from error
