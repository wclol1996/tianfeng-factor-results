from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime, timezone
import json
from numbers import Number
import os
from pathlib import Path

import pandas as pd


class FactorStore(ABC):
    @abstractmethod
    def append(self, factor_id: str, values: pd.Series) -> date:
        raise NotImplementedError


class CsvFactorStore(FactorStore):
    """Atomic, strictly append-only local store with one file per factor."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)

    def append(self, factor_id: str, values: pd.Series) -> date:
        if any(character in factor_id for character in "/\\"):
            raise ValueError(f"unsafe factor_id for local path: {factor_id!r}")
        if values.empty:
            raise ValueError(f"{factor_id}: refusing to advance state with no factor values")
        if list(values.index.names) != ["date", "security_id"]:
            raise ValueError("factor values must use a ['date', 'security_id'] MultiIndex")
        frame = values.rename("value").reset_index()
        if frame["date"].map(lambda value: isinstance(value, Number)).any():
            raise ValueError("factor dates must be explicit dates, not numeric epoch offsets")
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        if frame[["date", "security_id"]].isna().any().any():
            raise ValueError("factor values contain missing date/security keys")
        frame["security_id"] = frame["security_id"].astype(str)
        if frame["security_id"].str.strip().eq("").any():
            raise ValueError("factor values contain blank security keys")
        if frame.duplicated(["date", "security_id"]).any():
            raise ValueError("factor values contain duplicate date/security keys")
        frame["value"] = pd.to_numeric(frame["value"], errors="raise")
        if frame["value"].isin([float("inf"), float("-inf")]).any():
            raise ValueError("factor values contain infinite values")
        frame["factor_id"] = factor_id
        frame = frame[["date", "security_id", "factor_id", "value"]].dropna(subset=["value"])
        frame = frame.sort_values(["date", "security_id"])
        if frame.empty:
            raise ValueError(f"{factor_id}: all computed values are missing")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"{factor_id}.csv"
        if path.exists():
            existing = pd.read_csv(path, parse_dates=["date"], dtype={"security_id": str, "factor_id": str})
            existing_max = existing["date"].max()
            new_min = frame["date"].min()
            if new_min <= existing_max:
                raise ValueError(
                    f"{factor_id}: append-only violation; new minimum {new_min.date()} "
                    f"is not after existing maximum {existing_max.date()}",
                )
            combined = pd.concat([existing, frame], ignore_index=True)
        else:
            combined = frame
        temporary = path.with_suffix(".csv.tmp")
        combined.to_csv(temporary, index=False, date_format="%Y-%m-%d", encoding="utf-8")
        os.replace(temporary, path)
        return frame["date"].max().date()


@dataclass
class RunState:
    path: Path
    values: dict

    @classmethod
    def load(cls, path: str | Path) -> "RunState":
        state_path = Path(path)
        if state_path.exists():
            values = json.loads(state_path.read_text(encoding="utf-8"))
        else:
            values = {"factors": {}}
        values.setdefault("factors", {})
        return cls(state_path, values)

    def last_date(self, factor_id: str) -> date | None:
        raw = self.values["factors"].get(factor_id, {}).get("last_calculated_date")
        return date.fromisoformat(raw) if raw else None

    def update(self, factor_id: str, last_date: date, frequency: str) -> None:
        self.values["factors"][factor_id] = {
            "last_calculated_date": last_date.isoformat(),
            "frequency": frequency,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.values, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, self.path)
