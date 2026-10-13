from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from factor_engine.config import PriceAdjustment


PRICE_FIELDS = ("open", "high", "low", "close")


def normalize_security_status_frame(
    raw: pd.DataFrame,
    column_map: Mapping[str, str],
) -> pd.DataFrame:
    """Normalize Gildata point-in-time stock and price-limit status fields.

    ``is_limit_up``/``is_limit_down`` must be vendor status flags.  The engine
    keeps limit prices for audit but deliberately does not compare them with an
    adjusted price to infer a limit event.  ``is_no_price_limit`` or the inverse
    ``has_price_limit`` identifies IPO and other no-limit sessions.
    """

    inverse = {vendor: canonical for canonical, vendor in column_map.items()}
    missing_vendor = sorted(set(inverse) - set(raw.columns))
    if missing_vendor:
        raise ValueError(f"raw security status is missing vendor columns: {missing_vendor}")
    frame = raw[list(inverse)].rename(columns=inverse).copy()
    required = {"date", "security_id", "is_st", "is_suspended"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"normalized security status is missing columns: {sorted(missing)}")
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["security_id"] = frame["security_id"].astype(str)
    return (
        frame.drop_duplicates(["date", "security_id"], keep="last")
        .sort_values(["date", "security_id"])
        .reset_index(drop=True)
    )


def normalize_daily_frame(
    raw: pd.DataFrame,
    column_map: Mapping[str, str],
    adjustment: PriceAdjustment,
    price_column_map: Mapping[str, Mapping[str, str]] | None = None,
) -> pd.DataFrame:
    """Rename an unknown vendor schema into the engine's canonical daily schema.

    ``column_map`` is canonical-name -> vendor-name. If separate vendor columns
    exist for adjusted prices, ``price_column_map`` maps ``backward``, ``forward``
    and ``raw`` to their canonical OHLC vendor columns. This is the only place
    that should encode whether a vendor calls a series qfq/hfq or front/back.
    """

    selected = dict(column_map)
    if price_column_map:
        try:
            selected.update(price_column_map[adjustment.value])
        except KeyError as error:
            raise ValueError(f"price mapping does not define {adjustment.value!r}") from error
    missing_vendor = sorted({vendor for vendor in selected.values() if vendor not in raw.columns})
    if missing_vendor:
        raise ValueError(f"raw daily frame is missing vendor columns: {missing_vendor}")
    inverse = {vendor: canonical for canonical, vendor in selected.items()}
    frame = raw[list(inverse)].rename(columns=inverse).copy()
    required = {"date", "security_id", "close"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"normalized daily frame is missing columns: {sorted(missing)}")
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["security_id"] = frame["security_id"].astype(str)
    frame = frame.drop_duplicates(["date", "security_id"], keep="last")
    return frame.set_index(["date", "security_id"]).sort_index()


def normalize_intraday_frame(raw: pd.DataFrame, column_map: Mapping[str, str]) -> pd.DataFrame:
    inverse = {vendor: canonical for canonical, vendor in column_map.items()}
    missing_vendor = sorted(set(inverse) - set(raw.columns))
    if missing_vendor:
        raise ValueError(f"raw intraday frame is missing vendor columns: {missing_vendor}")
    frame = raw[list(inverse)].rename(columns=inverse).copy()
    required = {"timestamp", "security_id", "close"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"normalized intraday frame is missing columns: {sorted(missing)}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame["date"] = frame["timestamp"].dt.normalize()
    frame["security_id"] = frame["security_id"].astype(str)
    return frame.sort_values(["security_id", "timestamp"]).drop_duplicates(
        ["security_id", "timestamp"], keep="last",
    )


def normalize_index_intraday_frame(raw: pd.DataFrame, column_map: Mapping[str, str]) -> pd.DataFrame:
    """Normalize vendor 30/60-minute index OHLC for intraday benchmark factors."""

    inverse = {vendor: canonical for canonical, vendor in column_map.items()}
    missing_vendor = sorted(set(inverse) - set(raw.columns))
    if missing_vendor:
        raise ValueError(f"raw index intraday frame is missing vendor columns: {missing_vendor}")
    frame = raw[list(inverse)].rename(columns=inverse).copy()
    required = {"timestamp", "index_id", "open", "high", "low", "close"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"normalized index intraday frame is missing columns: {sorted(missing)}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame["date"] = frame["timestamp"].dt.normalize()
    frame["index_id"] = frame["index_id"].astype(str)
    return frame.sort_values(["index_id", "timestamp"]).drop_duplicates(
        ["index_id", "timestamp"], keep="last",
    )


def normalize_corporate_event_frame(
    raw: pd.DataFrame,
    column_map: Mapping[str, str],
    *,
    event_type: str,
) -> pd.DataFrame:
    """Normalize a disclosure/forecast event table without guessing event time."""

    inverse = {vendor: canonical for canonical, vendor in column_map.items()}
    missing_vendor = sorted(set(inverse) - set(raw.columns))
    if missing_vendor:
        raise ValueError(f"raw corporate event frame is missing vendor columns: {missing_vendor}")
    frame = raw[list(inverse)].rename(columns=inverse).copy()
    required = {"security_id", "announcement_at"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"normalized corporate event frame is missing columns: {sorted(missing)}")
    frame["security_id"] = frame["security_id"].astype(str)
    frame["announcement_at"] = pd.to_datetime(frame["announcement_at"], errors="coerce")
    frame["event_type"] = str(event_type)
    return (
        frame.dropna(subset=["announcement_at"])
        .sort_values(["security_id", "announcement_at"])
        .drop_duplicates(["security_id", "announcement_at"], keep="last")
        .reset_index(drop=True)
    )
