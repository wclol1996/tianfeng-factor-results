from __future__ import annotations

from datetime import date

import pandas as pd


def scheduled_dates(
    calendar: pd.DatetimeIndex,
    start: date,
    end: date,
    frequency: str,
    last_calculated_date: date | None = None,
    allow_partial_month_end: bool = False,
) -> pd.DatetimeIndex:
    calendar = pd.DatetimeIndex(calendar).normalize().sort_values().unique()
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    eligible = calendar[(calendar >= start_ts) & (calendar <= end_ts)]
    if last_calculated_date is not None:
        eligible = eligible[eligible > pd.Timestamp(last_calculated_date)]
    if frequency == "daily":
        return eligible
    if frequency != "monthly":
        raise ValueError(f"unsupported factor frequency: {frequency}")

    month_ends: list[pd.Timestamp] = []
    positions = {value: index for index, value in enumerate(calendar)}
    for current in eligible:
        position = positions[current]
        if position + 1 < len(calendar):
            following = calendar[position + 1]
            if following.month != current.month or following.year != current.year:
                month_ends.append(current)
        elif allow_partial_month_end:
            month_ends.append(current)
    return pd.DatetimeIndex(month_ends)


def history_start(calendar: pd.DatetimeIndex, first_target: pd.Timestamp, lookback: int) -> pd.Timestamp:
    calendar = pd.DatetimeIndex(calendar).normalize().sort_values().unique()
    before = calendar[calendar < first_target]
    if lookback == 0:
        return first_target
    if len(before) < lookback:
        return calendar[0]
    return before[-lookback]

