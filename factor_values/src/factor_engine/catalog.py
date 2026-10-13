from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable


VALID_STATUSES = {"Implemented", "Pending"}
VALID_FREQUENCIES = {"daily", "monthly"}


@dataclass(frozen=True)
class FactorSpec:
    factor_id: str
    factor_name: str
    factor_class: str
    style: str
    description: str
    required_fields: tuple[str, ...]
    frequency: str
    status: str
    implementation: str
    parameters: dict
    lookback_trading_days: int
    supports_price_adjustment: bool
    benchmark_id: str | None
    source_rows: tuple[int, ...]
    pending_reason: str

    @property
    def schedule_frequency(self) -> str:
        """Physical output schedule, separate from the catalog's native label."""

        return "daily" if self.parameters.get("daily_state", False) else self.frequency

    @classmethod
    def from_row(cls, row: dict[str, str]) -> "FactorSpec":
        required = tuple(value for value in row.get("required_fields", "").split("|") if value)
        source_rows = tuple(int(value) for value in row.get("source_rows", "").split("|") if value)
        parameters = json.loads(row.get("parameters_json") or "{}")
        spec = cls(
            factor_id=row["factor_id"].strip(),
            factor_name=row.get("factor_name", "").strip(),
            factor_class=row.get("factor_class", "").strip(),
            style=row.get("style", "").strip(),
            description=row.get("description", "").strip(),
            required_fields=required,
            frequency=row.get("frequency", "daily").strip().lower(),
            status=row.get("status", "Pending").strip(),
            implementation=row.get("implementation", "").strip(),
            parameters=parameters,
            lookback_trading_days=int(row.get("lookback_trading_days") or 0),
            supports_price_adjustment=row.get("supports_price_adjustment", "false").strip().lower() == "true",
            benchmark_id=(row.get("benchmark_id") or "").strip() or None,
            source_rows=source_rows,
            pending_reason=row.get("pending_reason", "").strip(),
        )
        spec.validate()
        return spec

    def validate(self) -> None:
        if not self.factor_id:
            raise ValueError("factor_id cannot be blank")
        if self.status not in VALID_STATUSES:
            raise ValueError(f"{self.factor_id}: invalid status {self.status!r}")
        if self.frequency not in VALID_FREQUENCIES:
            raise ValueError(f"{self.factor_id}: invalid frequency {self.frequency!r}")
        if self.status == "Implemented" and not self.implementation:
            raise ValueError(f"{self.factor_id}: Implemented factor has no implementation")
        if self.status == "Pending" and not self.pending_reason:
            raise ValueError(f"{self.factor_id}: Pending factor must explain why")
        if self.lookback_trading_days < 0:
            raise ValueError(f"{self.factor_id}: lookback cannot be negative")
        if "daily_state" in self.parameters and not isinstance(self.parameters["daily_state"], bool):
            raise ValueError(f"{self.factor_id}: daily_state must be a boolean")


class FactorCatalog:
    def __init__(self, specs: Iterable[FactorSpec]):
        self._specs = {spec.factor_id: spec for spec in specs}
        if not self._specs:
            raise ValueError("factor catalog is empty")

    @classmethod
    def from_csv(cls, path: str | Path) -> "FactorCatalog":
        with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        specs = [FactorSpec.from_row(row) for row in rows]
        ids = [spec.factor_id for spec in specs]
        duplicates = sorted({factor_id for factor_id in ids if ids.count(factor_id) > 1})
        if duplicates:
            raise ValueError(f"catalog contains duplicate factor IDs: {duplicates}")
        return cls(specs)

    def get(self, factor_id: str) -> FactorSpec:
        try:
            return self._specs[factor_id]
        except KeyError as error:
            raise KeyError(f"unknown factor_id: {factor_id}") from error

    def select(self, factor_ids: Iterable[str] | None = None) -> list[FactorSpec]:
        if factor_ids is None:
            return list(self._specs.values())
        return [self.get(factor_id) for factor_id in factor_ids]

    @property
    def specs(self) -> tuple[FactorSpec, ...]:
        return tuple(self._specs.values())

    def summary(self) -> dict[str, int]:
        implemented = sum(spec.status == "Implemented" for spec in self._specs.values())
        return {
            "total": len(self._specs),
            "implemented": implemented,
            "pending": len(self._specs) - implemented,
            "daily": sum(spec.frequency == "daily" for spec in self._specs.values()),
            "monthly": sum(spec.frequency == "monthly" for spec in self._specs.values()),
        }
