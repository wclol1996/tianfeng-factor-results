from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


REQUIRED_COLUMNS = {
    "security_id",
    "report_period",
    "available_at",
    "field",
    "value",
}


def _normalize(observations: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(observations.columns)
    if missing:
        raise ValueError(f"PIT observations are missing columns: {sorted(missing)}")
    frame = observations.copy()
    frame["report_period"] = pd.to_datetime(frame["report_period"]).dt.normalize()
    frame["available_at"] = pd.to_datetime(frame["available_at"])
    if "revision_at" not in frame:
        frame["revision_at"] = pd.NaT
    else:
        frame["revision_at"] = pd.to_datetime(frame["revision_at"])
    frame["effective_at"] = frame["revision_at"].fillna(frame["available_at"])
    if frame["effective_at"].isna().any():
        raise ValueError("PIT observations cannot have a blank effective timestamp")
    return frame


def pit_history(
    observations: pd.DataFrame,
    calculation_dates: Iterable,
    fields: Iterable[str] | None = None,
    depth: int = 1,
    allow_release_day: bool = True,
    calculation_time: str = "23:59:59.999999",
) -> pd.DataFrame:
    """Return the latest visible report history for every calculation date.

    Revisions are resolved inside each report period first. Only then are report
    periods ranked, preventing a late revision of an old report from replacing a
    newer report. ``report_lag=0`` is the latest report visible on that date.
    """

    if depth < 1:
        raise ValueError("depth must be at least 1")
    frame = _normalize(observations)
    if fields is not None:
        requested = set(fields)
        frame = frame[frame["field"].isin(requested)]
    dates = pd.DatetimeIndex(calculation_dates).normalize().sort_values().unique()
    result: list[pd.DataFrame] = []
    for calculation_date in dates:
        cutoff = calculation_date + pd.to_timedelta(calculation_time) if allow_release_day else calculation_date
        eligible = frame[frame["effective_at"] <= cutoff]
        if eligible.empty:
            continue
        latest_revision = (
            eligible.sort_values("effective_at")
            .drop_duplicates(["security_id", "field", "report_period"], keep="last")
            .sort_values(["security_id", "field", "report_period"], ascending=[True, True, False])
        )
        latest_revision["report_lag"] = latest_revision.groupby(["security_id", "field"]).cumcount()
        selected = latest_revision[latest_revision["report_lag"] < depth].copy()
        selected["date"] = calculation_date
        result.append(selected[[
            "date", "security_id", "field", "report_lag", "value",
            "report_period", "effective_at",
        ]])
    if not result:
        return pd.DataFrame(columns=[
            "date", "security_id", "field", "report_lag", "value",
            "report_period", "effective_at",
        ])
    return pd.concat(result, ignore_index=True)


def pit_snapshot(
    observations: pd.DataFrame,
    calculation_dates: Iterable,
    fields: Iterable[str] | None = None,
    allow_release_day: bool = True,
    calculation_time: str = "23:59:59.999999",
) -> pd.DataFrame:
    history = pit_history(
        observations,
        calculation_dates,
        fields=fields,
        depth=1,
        allow_release_day=allow_release_day,
        calculation_time=calculation_time,
    )
    return history.drop(columns=["report_lag"])
