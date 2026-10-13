from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, timedelta
from typing import Iterable

import pandas as pd

from factor_engine.catalog import FactorCatalog, FactorSpec
from factor_engine.config import EngineConfig
from factor_engine.data.base import DataProvider, FactorContext
from factor_engine.registry import load_builtins, registry
from factor_engine.scheduler import history_start, scheduled_dates
from factor_engine.storage import CsvFactorStore, FactorStore, RunState
from factor_engine.universe import UniverseBuilder, apply_universe


STATUS_COLUMNS_FOR_FACTORS = (
    "is_limit_up",
    "is_limit_down",
    "is_no_price_limit",
    "has_price_limit",
    "limit_up_price",
    "limit_down_price",
)


def attach_security_status(context: FactorContext, security_status: pd.DataFrame) -> None:
    """Expose point-in-time vendor status fields to factor implementations."""

    if context.daily.empty or security_status.empty:
        return
    status = security_status.copy()
    required = {"date", "security_id"}
    missing = required - set(status.columns)
    if missing:
        raise ValueError(f"security status is missing columns: {sorted(missing)}")
    status["date"] = pd.to_datetime(status["date"]).dt.normalize()
    if status[["date", "security_id"]].isna().any().any():
        raise ValueError("security status contains missing date/security keys")
    status["security_id"] = status["security_id"].astype(str)
    if status["security_id"].str.strip().eq("").any():
        raise ValueError("security status contains blank security keys")
    if status.duplicated(["date", "security_id"]).any():
        raise ValueError("security status contains duplicate date/security keys; resolve PIT versions upstream")
    available = [column for column in STATUS_COLUMNS_FOR_FACTORS if column in status]
    if not available:
        context.extras = {**context.extras, "security_status": status.copy()}
        return
    normalized = (
        status.set_index(["date", "security_id"])[available]
        .sort_index()
    )
    for column in available:
        incoming = normalized[column].reindex(context.daily.index)
        if column in context.daily:
            existing = context.daily[column]
            conflict = existing.notna() & incoming.notna() & existing.ne(incoming)
            if conflict.any():
                raise ValueError(f"daily data conflicts with security_status for {column}")
            context.daily[column] = existing.combine_first(incoming)
        else:
            context.daily[column] = incoming
    context.extras = {**context.extras, "security_status": status.copy()}


@dataclass(frozen=True)
class FactorRunResult:
    factor_id: str
    status: str
    first_date: date | None = None
    last_date: date | None = None
    rows: int = 0
    message: str = ""


class FactorEngine:
    def __init__(
        self,
        config: EngineConfig,
        provider: DataProvider,
        catalog: FactorCatalog | None = None,
        store: FactorStore | None = None,
    ) -> None:
        self.config = config
        self.provider = provider
        self.catalog = catalog or FactorCatalog.from_csv(config.catalog_path)
        self.store = store or CsvFactorStore(config.output_dir)
        self.state = RunState.load(config.state_path)
        self.universe_builder = UniverseBuilder(config.universe)
        load_builtins()

    def run(
        self,
        start: date,
        end: date,
        factor_ids: Iterable[str] | None = None,
        strict_pending: bool | None = None,
    ) -> list[FactorRunResult]:
        if start > end:
            raise ValueError("start date must not be after end date")
        strict = self.config.strict_pending if strict_pending is None else strict_pending
        results: list[FactorRunResult] = []
        for spec in self.catalog.select(factor_ids):
            if spec.status == "Pending":
                if strict:
                    raise NotImplementedError(f"{spec.factor_id}: {spec.pending_reason}")
                results.append(FactorRunResult(spec.factor_id, "Pending", message=spec.pending_reason))
                continue
            results.append(self._run_one(spec, start, end))
        return results

    def _run_one(self, spec: FactorSpec, start: date, end: date) -> FactorRunResult:
        needs_state_history = (
            spec.parameters.get("history_policy") == "confirmed_suspension_log_cap_v1"
            or (spec.factor_id in {"high250_ann", "ann_drawdown"}
                and spec.parameters.get("history_policy") == "confirmed_halt_historical_valuation_v1")
        )
        calendar_start = start - timedelta(days=max(60, spec.lookback_trading_days * 2 + 30))
        calendar_end = end + timedelta(days=45)
        full_calendar = self.provider.trading_calendar(calendar_start, calendar_end)
        targets = scheduled_dates(
            full_calendar,
            start,
            end,
            spec.schedule_frequency,
            last_calculated_date=self.state.last_date(spec.factor_id),
            allow_partial_month_end=self.config.allow_partial_month_end,
        )
        if targets.empty:
            return FactorRunResult(spec.factor_id, "UpToDate", message="no new scheduled trading date")
        if needs_state_history:
            calendar_start = self.provider.available_history_start(spec)
            if calendar_start > targets[0].date():
                raise ValueError("Available state history starts after requested calculation")
            full_calendar = self.provider.trading_calendar(calendar_start, calendar_end)
        load_start = full_calendar[0] if needs_state_history else history_start(full_calendar, targets[0], spec.lookback_trading_days)
        benchmark_id = spec.benchmark_id or self.config.benchmark_id
        context = self.provider.load_factor_context(
            spec,
            load_start.date(),
            targets[-1].date(),
            targets,
            self.config.price_adjustment,
            benchmark_id,
        )
        context.validate()
        context.pit_allow_release_day = self.config.pit.allow_release_day
        context.pit_calculation_time = self.config.pit.calculation_time
        context.ff3_config = self.config.ff3
        history_dates = full_calendar[(full_calendar >= load_start) & (full_calendar <= targets[-1])]
        master = self.provider.security_master(load_start.date(), targets[-1].date())
        status = self.provider.security_status(load_start.date(), targets[-1].date())
        attach_security_status(context, status)
        constituents = self.provider.index_constituents(benchmark_id, load_start.date(), targets[-1].date())
        context.extras = {**context.extras, "security_master": master.copy()}
        universe = self.universe_builder.build(history_dates, master, status, constituents)
        all_a_share_config = replace(self.config.universe, restrict_to_benchmark_members=False)
        context.eligible_universe = UniverseBuilder(all_a_share_config).build(
            history_dates, master, status, constituents=None,
        )
        context.universe = universe
        values = registry.get(spec.implementation)(context, spec)
        value_dates = pd.DatetimeIndex(values.index.get_level_values("date")).normalize()
        values = values[value_dates.isin(targets)]
        values = apply_universe(values, universe)
        last_date = self.store.append(spec.factor_id, values)
        self.state.update(spec.factor_id, last_date, spec.schedule_frequency)
        return FactorRunResult(
            factor_id=spec.factor_id,
            status="Appended",
            first_date=values.index.get_level_values("date").min().date(),
            last_date=last_date,
            rows=len(values),
        )
