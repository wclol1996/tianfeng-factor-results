from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import tomllib


class PriceAdjustment(str, Enum):
    """Vendor-independent price-adjustment choice."""

    BACKWARD = "backward"
    FORWARD = "forward"
    RAW = "raw"


@dataclass(frozen=True)
class UniverseConfig:
    min_listing_days: int = 365
    exclude_st: bool = True
    exclude_suspended: bool = True
    exclude_after_delisting: bool = True
    missing_status_policy: str = "exclude"
    restrict_to_benchmark_members: bool = False


@dataclass(frozen=True)
class PitConfig:
    allow_release_day: bool = True
    calculation_time: str = "15:00:00"
    availability_column: str = "available_at"
    revision_column: str = "revision_at"
    report_period_column: str = "report_period"


@dataclass(frozen=True)
class FF3Config:
    """Construction choices for the locally built A-share FF3 series."""

    market_index_id: str = "000985.CSI"
    default_weighting: str = "float_market_cap"
    weighting_schemes: tuple[str, ...] = (
        "total_market_cap",
        "float_market_cap",
        "free_float_market_cap",
    )
    default_rebalance: str = "annual_june"
    rebalance_schemes: tuple[str, ...] = ("annual_june", "monthly")
    risk_free_daily: float = 0.0
    size_quantile: float = 0.5
    book_to_price_quantiles: tuple[float, float] = (0.3, 0.7)

    def __post_init__(self) -> None:
        valid_weights = {"total_market_cap", "float_market_cap", "free_float_market_cap"}
        valid_rebalances = {"annual_june", "monthly"}
        if self.default_weighting not in valid_weights:
            raise ValueError(f"unsupported FF3 weighting: {self.default_weighting}")
        if not self.weighting_schemes or not set(self.weighting_schemes) <= valid_weights:
            raise ValueError("FF3 weighting_schemes contains an unsupported value")
        if self.default_weighting not in self.weighting_schemes:
            raise ValueError("FF3 default_weighting must be included in weighting_schemes")
        if self.default_rebalance not in valid_rebalances:
            raise ValueError(f"unsupported FF3 rebalance: {self.default_rebalance}")
        if not self.rebalance_schemes or not set(self.rebalance_schemes) <= valid_rebalances:
            raise ValueError("FF3 rebalance_schemes contains an unsupported value")
        if self.default_rebalance not in self.rebalance_schemes:
            raise ValueError("FF3 default_rebalance must be included in rebalance_schemes")
        low, high = self.book_to_price_quantiles
        if not 0 < self.size_quantile < 1:
            raise ValueError("FF3 size_quantile must be between zero and one")
        if not 0 < low < high < 1:
            raise ValueError("FF3 book_to_price_quantiles must satisfy 0 < low < high < 1")


@dataclass(frozen=True)
class EngineConfig:
    catalog_path: Path
    output_dir: Path
    state_path: Path
    price_adjustment: PriceAdjustment = PriceAdjustment.BACKWARD
    benchmark_id: str = "000985.CSI"
    strict_pending: bool = False
    allow_partial_month_end: bool = False
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    pit: PitConfig = field(default_factory=PitConfig)
    ff3: FF3Config = field(default_factory=FF3Config)
    provider_class_path: str = ""
    raw: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_toml(cls, path: str | Path) -> "EngineConfig":
        config_path = Path(path).resolve()
        with config_path.open("rb") as handle:
            raw = tomllib.load(handle)
        base = config_path.parent
        engine = raw.get("engine", {})
        universe = raw.get("universe", {})
        pit = raw.get("pit", {})
        ff3 = raw.get("ff3", {})
        provider = raw.get("provider", {})

        def resolve(value: str, default: str) -> Path:
            candidate = Path(value or default)
            return candidate if candidate.is_absolute() else (base / candidate).resolve()

        return cls(
            catalog_path=resolve(engine.get("catalog_path", ""), "factor_catalog.csv"),
            output_dir=resolve(engine.get("output_dir", ""), "data/factors"),
            state_path=resolve(engine.get("state_path", ""), "state/last_run.json"),
            price_adjustment=PriceAdjustment(engine.get("price_adjustment", "backward")),
            benchmark_id=str(engine.get("benchmark_id", "000985.CSI")),
            strict_pending=bool(engine.get("strict_pending", False)),
            allow_partial_month_end=bool(engine.get("allow_partial_month_end", False)),
            universe=UniverseConfig(**universe),
            pit=PitConfig(**pit),
            ff3=FF3Config(
                market_index_id=str(ff3.get("market_index_id", "000985.CSI")),
                default_weighting=str(ff3.get("default_weighting", "float_market_cap")),
                weighting_schemes=tuple(ff3.get("weighting_schemes", (
                    "total_market_cap", "float_market_cap", "free_float_market_cap",
                ))),
                default_rebalance=str(ff3.get("default_rebalance", "annual_june")),
                rebalance_schemes=tuple(ff3.get("rebalance_schemes", ("annual_june", "monthly"))),
                risk_free_daily=float(ff3.get("risk_free_daily", 0.0)),
                size_quantile=float(ff3.get("size_quantile", 0.5)),
                book_to_price_quantiles=tuple(ff3.get("book_to_price_quantiles", (0.3, 0.7))),
            ),
            provider_class_path=str(provider.get("class_path", "")),
            raw=raw,
        )
