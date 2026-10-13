from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from typing import Mapping

import pandas as pd

from factor_engine.catalog import FactorSpec
from factor_engine.config import FF3Config, PriceAdjustment


@dataclass
class FactorContext:
    """Normalized data supplied to factor implementations.

    Daily data uses a MultiIndex named ``date, security_id``. Intraday data is a
    long table. Financial observations remain point-in-time long records and are
    converted to snapshots only for requested calculation dates.
    """

    calculation_dates: pd.DatetimeIndex
    daily: pd.DataFrame = field(default_factory=pd.DataFrame)
    intraday: pd.DataFrame = field(default_factory=pd.DataFrame)
    index_daily: pd.DataFrame = field(default_factory=pd.DataFrame)
    index_intraday: pd.DataFrame = field(default_factory=pd.DataFrame)
    financial_observations: pd.DataFrame = field(default_factory=pd.DataFrame)
    consensus_observations: pd.DataFrame = field(default_factory=pd.DataFrame)
    universe: Mapping[pd.Timestamp, frozenset[str]] = field(default_factory=dict)
    eligible_universe: Mapping[pd.Timestamp, frozenset[str]] = field(default_factory=dict)
    ff3_config: FF3Config = field(default_factory=FF3Config)
    pit_allow_release_day: bool = True
    pit_calculation_time: str = "15:00:00"
    events: Mapping[str, pd.DataFrame] = field(default_factory=dict)
    extras: Mapping[str, pd.DataFrame] = field(default_factory=dict)
    cache: dict[str, object] = field(default_factory=dict, repr=False)

    def validate(self) -> None:
        if not self.daily.empty:
            if list(self.daily.index.names) != ["date", "security_id"]:
                raise ValueError("daily frame index must be named ['date', 'security_id']")
            if not self.daily.index.is_monotonic_increasing:
                self.daily.sort_index(inplace=True)
        if not self.intraday.empty:
            required = {"timestamp", "date", "security_id"}
            missing = required - set(self.intraday.columns)
            if missing:
                raise ValueError(f"intraday frame is missing columns: {sorted(missing)}")
        if not self.index_intraday.empty:
            required = {"timestamp", "date", "index_id"}
            missing = required - set(self.index_intraday.columns)
            if missing:
                raise ValueError(f"index_intraday frame is missing columns: {sorted(missing)}")


class DataProvider(ABC):
    """Adapter boundary for Wind, database, files, or future real-time APIs."""

    def available_history_start(self, spec: FactorSpec) -> date:
        """Earliest bound source date, required for unbounded carry state."""
        raise ValueError("Provider has no approved available-history boundary for cap carry")

    @abstractmethod
    def trading_calendar(self, start: date, end: date) -> pd.DatetimeIndex:
        raise NotImplementedError

    @abstractmethod
    def load_factor_context(
        self,
        spec: FactorSpec,
        start: date,
        end: date,
        calculation_dates: pd.DatetimeIndex,
        price_adjustment: PriceAdjustment,
        benchmark_id: str,
    ) -> FactorContext:
        """Load normalized data without applying a current-universe filter."""
        raise NotImplementedError

    @abstractmethod
    def security_master(self, start: date, end: date) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def security_status(self, start: date, end: date) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def index_constituents(self, index_id: str, start: date, end: date) -> pd.DataFrame:
        raise NotImplementedError


class InMemoryProvider(DataProvider):
    """Small deterministic provider for tests and adapter prototyping."""

    def __init__(
        self,
        calendar: pd.DatetimeIndex,
        context: FactorContext,
        security_master: pd.DataFrame,
        security_status: pd.DataFrame,
        constituents: pd.DataFrame | None = None,
    ) -> None:
        self._calendar = pd.DatetimeIndex(calendar).normalize().sort_values().unique()
        self._context = context
        self._master = security_master.copy()
        self._status = security_status.copy()
        self._constituents = pd.DataFrame() if constituents is None else constituents.copy()
        self.adjustment_requests: list[PriceAdjustment] = []

    def trading_calendar(self, start: date, end: date) -> pd.DatetimeIndex:
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        return self._calendar[(self._calendar >= start_ts) & (self._calendar <= end_ts)]

    def available_history_start(self, spec: FactorSpec) -> date:
        if self._calendar.empty or self._context.daily.empty:
            raise ValueError("Empty available-history inputs")
        first = pd.Timestamp(self._context.daily.index.get_level_values("date").min())
        if first != self._calendar[0]:
            raise ValueError("Daily history and declared calendar have different starts")
        return first.date()

    def load_factor_context(
        self,
        spec: FactorSpec,
        start: date,
        end: date,
        calculation_dates: pd.DatetimeIndex,
        price_adjustment: PriceAdjustment,
        benchmark_id: str,
    ) -> FactorContext:
        self.adjustment_requests.append(price_adjustment)
        self._context.calculation_dates = pd.DatetimeIndex(calculation_dates)
        self._context.validate()
        return self._context

    def security_master(self, start: date, end: date) -> pd.DataFrame:
        return self._master.copy()

    def security_status(self, start: date, end: date) -> pd.DataFrame:
        return self._status.copy()

    def index_constituents(self, index_id: str, start: date, end: date) -> pd.DataFrame:
        if self._constituents.empty:
            return self._constituents.copy()
        return self._constituents[self._constituents["index_id"].eq(index_id)].copy()
