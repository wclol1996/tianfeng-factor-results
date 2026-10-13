from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from factor_engine.catalog import FactorSpec
from factor_engine.data.base import FactorContext
from factor_engine.factors.common import INDEX_NAMES, require_daily, restore_declared_market_dates, rolling
from factor_engine.factors.daily_price import _fixed_calendar_window_complete
from factor_engine.registry import registry


SUPPORTED_EXACT = frozenset({"lf_mom", "high250_ann", "aog", "jor", "ann_drawdown"})
DEFAULT_EVENT_KEYS = ("financial_report_disclosures", "earnings_forecasts")


def supports_factor_id(factor_id: str) -> bool:
    return factor_id in SUPPORTED_EXACT


def confirmed_halt_historical_close(daily: pd.DataFrame, halts: pd.DataFrame, master: pd.DataFrame) -> tuple[pd.Series, int]:
    """Carry only the immediately preceding known valuation on evidenced halts."""
    close = daily.close.unstack("security_id")
    volume = daily.volume.unstack("security_id").reindex_like(close)
    marked = pd.DataFrame(False, index=close.index, columns=close.columns)
    eligible_halts = halts[halts.date.isin(close.index) & halts.security_id.isin(close.columns)]
    for day, code in eligible_halts[["date", "security_id"]].drop_duplicates().itertuples(index=False, name=None):
        marked.at[day, code] = True
    marked &= ~volume.gt(0)
    lifecycle = master.set_index("security_id")
    if lifecycle.index.duplicated().any():
        raise ValueError("Conflicting security lifecycle records")
    for code in close:
        info = lifecycle.loc[code]
        outside = close.index < pd.Timestamp(info.list_date)
        if pd.notna(info.delist_date):
            outside |= close.index >= pd.Timestamp(info.delist_date)
        close.loc[outside, code] = np.nan
        marked.loc[outside, code] = False
    values = close.to_numpy(copy=True)
    marks = marked.to_numpy()
    count = 0
    for position in range(1, len(values)):
        fill = np.isnan(values[position]) & marks[position] & np.isfinite(values[position - 1])
        values[position, fill] = values[position - 1, fill]
        count += int(fill.sum())
    historical = pd.DataFrame(values, index=close.index, columns=close.columns)
    return historical.rename_axis(index="date", columns="security_id").stack().reindex(daily.index), count


def _empty_series(name: str = "value") -> pd.Series:
    index = pd.MultiIndex.from_arrays([[], []], names=INDEX_NAMES)
    return pd.Series(dtype=float, index=index, name=name)


def _announcement_column(frame: pd.DataFrame) -> str:
    for candidate in ("announcement_at", "available_at", "event_at", "announcement_date"):
        if candidate in frame:
            return candidate
    raise ValueError(
        "corporate event data must contain canonical announcement_at "
        "(available_at/event_at/announcement_date are accepted adapter fallbacks)"
    )


def _canonical_event_frame(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["security_id", "announcement_at", "event_source"])
    if "security_id" not in frame:
        raise ValueError(f"{source} event data is missing security_id")
    announcement_column = _announcement_column(frame)
    result = frame[["security_id", announcement_column]].copy()
    result.columns = ["security_id", "announcement_at"]
    result["security_id"] = result["security_id"].astype(str)
    result["announcement_at"] = pd.to_datetime(result["announcement_at"], errors="coerce")
    if "available_at" in frame:
        result["available_at"] = pd.to_datetime(frame["available_at"], errors="raise")
        if result["available_at"].isna().any():
            raise ValueError("event availability cannot be missing")
    else:
        # Preserve announcement-only adapters when combined with a source that
        # supplies availability. Event windows still enforce next-session use.
        result["available_at"] = result["announcement_at"]
    result["event_source"] = source
    return result.dropna(subset=["announcement_at"])


def _financial_disclosure_fallback(context: FactorContext) -> pd.DataFrame:
    observations = context.financial_observations
    if observations.empty or "security_id" not in observations or "available_at" not in observations:
        return pd.DataFrame()
    columns = ["security_id", "available_at"]
    has_announcement = "announcement_at" in observations
    if has_announcement:
        columns.append("announcement_at")
    if "report_period" in observations:
        columns.append("report_period")
    disclosures = observations[columns].drop_duplicates().copy()
    # Collapse repeated fields sharing the retained report/timestamp keys.
    # This does not certify revision identity or historical version completeness
    # when a legacy input supplies availability without the real announcement.
    if has_announcement:
        # Economic price windows follow the actual announcement; source
        # availability only delays activation and must not move that window.
        return disclosures
    # Legacy announcement-only adapters stored the release in available_at.
    # Keep this compatibility assumption separate from certified PIT inputs.
    return disclosures.rename(columns={"available_at": "announcement_at"})


def _collect_events(context: FactorContext, spec: FactorSpec) -> pd.DataFrame:
    missing_sources = [
        key for key in spec.parameters.get("required_event_keys", [])
        if key not in context.events or context.events[key] is None
    ]
    if missing_sources:
        raise ValueError(f"{spec.factor_id}: missing required event tables {missing_sources}")
    event_keys = tuple(spec.parameters.get("event_keys", DEFAULT_EVENT_KEYS))
    frames: list[pd.DataFrame] = []
    explicit_financial_disclosure = False
    for key in event_keys:
        source = context.events.get(str(key))
        if source is None or source.empty:
            continue
        frames.append(_canonical_event_frame(source, str(key)))
        if str(key) == "financial_report_disclosures":
            explicit_financial_disclosure = True
    if "financial_report_disclosures" in event_keys and not explicit_financial_disclosure:
        fallback = _financial_disclosure_fallback(context)
        if not fallback.empty:
            frames.append(_canonical_event_frame(fallback, "financial_report_disclosures"))
    if not frames:
        raise ValueError(
            f"{spec.factor_id} requires events['financial_report_disclosures'] and/or "
            "events['earnings_forecasts']; actual disclosures may fall back to "
            "financial_observations.available_at"
        )
    events = pd.concat(frames, ignore_index=True)
    events["event_date"] = events["announcement_at"].dt.normalize()
    # A forecast and an actual report disclosed on the same calendar date form
    # one price event, not two observations with an identical window.
    return (
        # A later-arriving duplicate must not erase an already public event.
        # These rows share the same natural-date price window, so the first
        # source that was actually available is sufficient to establish it.
        events.sort_values(["security_id", "event_date", "available_at", "announcement_at", "event_source"])
        .drop_duplicates(["security_id", "event_date"], keep="first")
        .reset_index(drop=True)
    )


def _respect_event_availability(frame: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Never activate an event before its explicit source availability.

    Preserve the economic announcement window. Late records become usable at
    the first market close at/after availability, without moving event prices.
    """
    if "available_at" not in frame or frame.empty:
        return frame
    result = frame.copy()
    closes = pd.DatetimeIndex(calendar).normalize() + pd.Timedelta(hours=15)
    availability = pd.DatetimeIndex(pd.to_datetime(result["available_at"], errors="raise"))
    positions = closes.searchsorted(availability, side="left")
    valid = positions < len(closes)
    result = result.loc[valid].copy()
    eligible = pd.DatetimeIndex(calendar)[positions[valid]].normalize()
    result["effective_date"] = np.maximum(
        pd.to_datetime(result["effective_date"]).to_numpy(), eligible.to_numpy(),
    )
    return result


def _event_windows(events: pd.DataFrame, calendar: pd.DatetimeIndex, close: pd.Series) -> pd.DataFrame:
    if calendar.empty:
        return events.iloc[0:0].assign(
            before_date=pd.NaT, after_date=pd.NaT, effective_date=pd.NaT, event_return=np.nan,
        )
    normalized_calendar = pd.DatetimeIndex(calendar).normalize().sort_values().unique()
    event_dates = pd.DatetimeIndex(events["event_date"])
    before_positions = normalized_calendar.searchsorted(event_dates, side="left") - 1
    after_positions = normalized_calendar.searchsorted(event_dates, side="right")
    complete = (before_positions >= 0) & (after_positions < len(normalized_calendar))
    windows = events.loc[complete].copy()
    if windows.empty:
        return windows.assign(
            before_date=pd.NaT, after_date=pd.NaT, effective_date=pd.NaT, event_return=np.nan,
        )
    windows["before_date"] = normalized_calendar[before_positions[complete]].to_numpy()
    windows["after_date"] = normalized_calendar[after_positions[complete]].to_numpy()
    windows["effective_date"] = windows["after_date"]
    windows = _respect_event_availability(windows, normalized_calendar)

    before_index = pd.MultiIndex.from_arrays(
        [windows["before_date"], windows["security_id"]], names=INDEX_NAMES,
    )
    after_index = pd.MultiIndex.from_arrays(
        [windows["after_date"], windows["security_id"]], names=INDEX_NAMES,
    )
    before_price = close.reindex(before_index).to_numpy(dtype=float)
    after_price = close.reindex(after_index).to_numpy(dtype=float)
    windows["event_return"] = np.where(
        np.isfinite(before_price) & np.isfinite(after_price) & (before_price != 0.0),
        after_price / before_price - 1.0,
        np.nan,
    )
    # Several weekend announcements may share one next trading day. The latest
    # announcement is the event that becomes current on that effective date.
    return (
        windows.sort_values(["security_id", "effective_date", "announcement_at"])
        .drop_duplicates(["security_id", "effective_date"], keep="last")
        .reset_index(drop=True)
    )


def _next_day_event_pairs(
    events: pd.DataFrame,
    calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Map every announcement to t and t+1 without using its clock time.

    ``reference_date`` is the last market trading day not later than the
    announcement's natural date. ``event_trade_date`` is the first market
    trading day strictly later than that natural date. Therefore a pre-market,
    intraday, and after-close announcement on the same date map identically.
    """

    if calendar.empty:
        return events.iloc[0:0].assign(
            reference_date=pd.NaT, history_reference_date=pd.NaT,
            event_trade_date=pd.NaT, effective_date=pd.NaT,
        )
    normalized_calendar = pd.DatetimeIndex(calendar).normalize().sort_values().unique()
    event_dates = pd.DatetimeIndex(events["event_date"])
    reference_positions = normalized_calendar.searchsorted(event_dates, side="right") - 1
    event_positions = normalized_calendar.searchsorted(event_dates, side="right")
    complete = (reference_positions >= 0) & (event_positions < len(normalized_calendar))
    pairs = events.loc[complete].copy()
    if pairs.empty:
        return pairs.assign(
            reference_date=pd.NaT, history_reference_date=pd.NaT,
            event_trade_date=pd.NaT, effective_date=pd.NaT,
        )
    pairs["reference_date"] = normalized_calendar[reference_positions[complete]].to_numpy()
    # Price gaps use the announcement-day close. Historical pre-announcement
    # windows instead exclude that natural date, including for intraday news.
    history_positions = normalized_calendar.searchsorted(event_dates, side="left") - 1
    pairs["history_reference_date"] = pd.NaT
    history_valid = history_positions[complete] >= 0
    pairs.loc[history_valid, "history_reference_date"] = normalized_calendar[history_positions[complete][history_valid]].to_numpy()
    pairs["event_trade_date"] = normalized_calendar[event_positions[complete]].to_numpy()
    pairs["effective_date"] = pairs["event_trade_date"]
    pairs = _respect_event_availability(pairs, normalized_calendar)
    return (
        pairs.sort_values(["security_id", "effective_date", "announcement_at"])
        .drop_duplicates(["security_id", "effective_date"], keep="last")
        .reset_index(drop=True)
    )


def _series_at(
    values: pd.Series,
    dates: pd.Series,
    securities: pd.Series,
) -> np.ndarray:
    index = pd.MultiIndex.from_arrays(
        [pd.to_datetime(dates).dt.normalize(), securities.astype(str)], names=INDEX_NAMES,
    )
    return values.reindex(index).to_numpy(dtype=float)


def _index_price(context: FactorContext, benchmark_id: str | None, field: str = "close") -> pd.Series:
    frame = context.index_daily.copy()
    if frame.empty:
        raise ValueError("index_daily is required for announcement gap factors")
    if isinstance(frame.index, pd.MultiIndex) or frame.index.name == "date":
        frame = frame.reset_index()
    required = {"date", field}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"index_daily is missing columns: {sorted(missing)}")
    if "index_id" in frame:
        ids = frame["index_id"].astype(str)
        target = benchmark_id or (ids.iloc[0] if ids.nunique() == 1 else None)
        if target is None:
            raise ValueError("benchmark_id is required when index_daily contains multiple indices")
        frame = frame[ids.eq(str(target))]
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    return (
        frame.sort_values("date")
        .drop_duplicates("date", keep="last")
        .set_index("date")[field]
        .astype(float)
    )


def _market_adjusted_gap(
    pairs: pd.DataFrame,
    stock_event_price: pd.Series,
    stock_close: pd.Series,
    index_close: pd.Series,
    index_event_price: pd.Series,
) -> np.ndarray:
    stock_reference = _series_at(
        stock_close, pairs["reference_date"], pairs["security_id"],
    )
    stock_event = _series_at(
        stock_event_price, pairs["event_trade_date"], pairs["security_id"],
    )
    index_reference = index_close.reindex(pairs["reference_date"]).to_numpy(dtype=float)
    index_event = index_event_price.reindex(pairs["event_trade_date"]).to_numpy(dtype=float)
    stock_gap = np.divide(
        stock_event, stock_reference,
        out=np.full(len(pairs), np.nan),
        where=np.isfinite(stock_event) & np.isfinite(stock_reference) & (stock_reference != 0.0),
    ) - 1.0
    market_return = np.divide(
        index_event, index_reference,
        out=np.full(len(pairs), np.nan),
        where=np.isfinite(index_event) & np.isfinite(index_reference) & (index_reference != 0.0),
    ) - 1.0
    return stock_gap - market_return


def _maximum_drawdown_magnitude(values: np.ndarray) -> float:
    if len(values) == 0 or not np.isfinite(values).all() or (values <= 0.0).any():
        return np.nan
    running_peak = np.maximum.accumulate(values)
    return float(np.max(1.0 - values / running_peak))


def _materialize_latest_event(
    windows: pd.DataFrame,
    calculation_dates: pd.DatetimeIndex,
    securities: pd.Index,
    value_column: str = "event_return",
) -> pd.Series:
    targets = pd.DatetimeIndex(calculation_dates).normalize().sort_values().unique()
    if targets.empty:
        return _empty_series()
    # Arrival order does not make an older announcement the latest event.
    # A correction announced later still has a later announcement timestamp
    # and can update the state after its own availability, without backfilling.
    ordered = windows.sort_values(["security_id", "effective_date", "announcement_at"])
    latest_announcement = ordered.groupby("security_id")["announcement_at"].cummax()
    ordered = ordered.loc[ordered["announcement_at"].eq(latest_announcement)]
    by_security = {str(key): group for key, group in ordered.groupby("security_id", sort=False)}
    output: list[pd.DataFrame] = []
    for security_id in securities.astype(str).unique():
        values = np.full(len(targets), np.nan, dtype=float)
        group = by_security.get(str(security_id))
        if group is not None and not group.empty:
            effective = pd.DatetimeIndex(group["effective_date"])
            positions = effective.searchsorted(targets, side="right") - 1
            has_event = positions >= 0
            # Indexing the event row directly, rather than forward-filling its
            # return, preserves NaN when the latest event window lacks a price.
            values[has_event] = group[value_column].to_numpy(dtype=float)[positions[has_event]]
        output.append(pd.DataFrame({
            "date": targets,
            "security_id": str(security_id),
            "value": values,
        }))
    if not output:
        return _empty_series()
    return pd.concat(output, ignore_index=True).set_index(INDEX_NAMES)["value"].sort_index()


@registry.register("event_price")
def compute_event_price(context: FactorContext, spec: FactorSpec) -> pd.Series:
    if not supports_factor_id(spec.factor_id):
        raise NotImplementedError(f"event_price does not implement {spec.factor_id}")
    context = replace(context, daily=restore_declared_market_dates(context.daily, context.calculation_dates))
    required_fields = {
        "lf_mom": ["close"],
        "high250_ann": ["close"],
        "aog": ["close", "open"],
        "jor": ["close", "low"],
        "ann_drawdown": ["close"],
    }[spec.factor_id]
    require_daily(context.daily, required_fields)
    close = context.daily["close"].astype(float).sort_index()
    historical_close = close
    if spec.factor_id in {"high250_ann", "ann_drawdown"} and spec.parameters.get("history_policy") == "confirmed_halt_historical_valuation_v1":
        if "historical_valuation_close" not in context.daily:
            raise ValueError("confirmed halt event history requires historical_valuation_close from the evidenced halt adapter")
        historical_close = context.daily["historical_valuation_close"].astype(float).sort_index()
    calendar = pd.DatetimeIndex(close.index.get_level_values("date")).normalize().sort_values().unique()
    events = _collect_events(context, spec)
    value_column = "event_return"
    if spec.factor_id == "lf_mom":
        windows = _event_windows(events, calendar, close)
    else:
        windows = _next_day_event_pairs(events, calendar)
        value_column = "event_value"
        if spec.factor_id == "high250_ann":
            window = int(spec.parameters.get("history_window", 250))
            historical_high = rolling(historical_close, window, "max").where(
                _fixed_calendar_window_complete(historical_close, window),
            )
            reference_high = _series_at(
                historical_high, windows["history_reference_date"], windows["security_id"],
            )
            event_close = _series_at(
                close, windows["event_trade_date"], windows["security_id"],
            )
            windows[value_column] = np.divide(
                event_close, reference_high,
                out=np.full(len(windows), np.nan),
                where=np.isfinite(event_close) & np.isfinite(reference_high) & (reference_high != 0.0),
            ) - 1.0
        elif spec.factor_id in {"aog", "jor"}:
            event_field = "open" if spec.factor_id == "aog" else "low"
            event_price = context.daily[event_field].astype(float).sort_index()
            windows[value_column] = _market_adjusted_gap(
                windows,
                event_price,
                close,
                _index_price(context, spec.benchmark_id, "close"),
                _index_price(context, spec.benchmark_id, event_field),
            )
        elif spec.factor_id == "ann_drawdown":
            window = int(spec.parameters.get("window", 252))
            if window < 2:
                raise ValueError("ann_drawdown window must contain at least two trading days")
            drawdown = rolling(historical_close, window, _maximum_drawdown_magnitude).where(
                _fixed_calendar_window_complete(historical_close, window),
            )
            windows[value_column] = _series_at(
                drawdown, windows["history_reference_date"], windows["security_id"],
            )
        else:  # pragma: no cover - protected by supports_factor_id
            raise NotImplementedError(spec.factor_id)
    context.cache[f"{spec.factor_id}:event_windows"] = windows.copy()
    targets = pd.DatetimeIndex(context.calculation_dates)
    if targets.empty:
        targets = calendar
    result = _materialize_latest_event(
        windows,
        targets,
        close.index.get_level_values("security_id").unique(),
        value_column=value_column,
    )
    result.name = spec.factor_id
    return result.replace([np.inf, -np.inf], np.nan).sort_index()
