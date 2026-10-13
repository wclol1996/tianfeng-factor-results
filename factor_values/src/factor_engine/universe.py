from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from factor_engine.config import UniverseConfig


def historical_st_eligible(context, date: pd.Timestamp, securities):
    """Apply the same date-effective ST gate to every construction population."""
    if not context.extras.get("requires_st_exclusion"):
        return securities
    source = context.extras.get("historical_st")
    if source is None or securities is None:
        raise ValueError("Historical ST policy requires a known population and date-effective status source")
    codes = sorted(securities)
    index = pd.MultiIndex.from_arrays([[date] * len(codes), codes], names=["date", "security_id"])
    keep = source.flags(index).eq(False).fillna(False).to_numpy(dtype=bool)
    return frozenset(code for code, allowed in zip(codes, keep) if allowed)


@dataclass(frozen=True)
class UniverseSnapshot:
    date: pd.Timestamp
    securities: frozenset[str]


class UniverseBuilder:
    def __init__(self, config: UniverseConfig):
        self.config = config

    def build(
        self,
        calculation_dates: pd.DatetimeIndex,
        security_master: pd.DataFrame,
        security_status: pd.DataFrame,
        constituents: pd.DataFrame | None = None,
    ) -> dict[pd.Timestamp, frozenset[str]]:
        if security_status.attrs.get("requires_st_exclusion") and not self.config.exclude_st:
            raise ValueError("Current historical ST policy requires UniverseConfig(exclude_st=True); legacy ST-inclusive outputs need separate audit configuration")
        master = security_master.copy()
        required_master = {"security_id", "list_date"}
        missing = required_master - set(master.columns)
        if missing:
            raise ValueError(f"security master is missing columns: {sorted(missing)}")
        # Use declared exchange identities, never infer a venue from numeric
        # prefixes. Filtering this population leaves historical daily inputs
        # intact for rolling calculations and full-market denominators.
        codes = master["security_id"].astype("string").str.strip().str.upper()
        bse = codes.str.endswith(".BJ", na=False)
        if "exchange" in master:
            bse |= master["exchange"].astype("string").str.strip().str.upper().isin(
                {"BSE", "BJ", "BEIJING"},
            )
        master = master.loc[~bse.fillna(False)].copy()
        master["list_date"] = pd.to_datetime(master["list_date"]).dt.normalize()
        if "delist_date" not in master:
            master["delist_date"] = pd.NaT
        else:
            master["delist_date"] = pd.to_datetime(master["delist_date"]).dt.normalize()

        status = security_status.copy()
        if not status.empty:
            required_status = {"date", "security_id", "is_st", "is_suspended"}
            missing = required_status - set(status.columns)
            if missing:
                raise ValueError(f"security status is missing columns: {sorted(missing)}")
            status["date"] = pd.to_datetime(status["date"]).dt.normalize()

        member_frame = pd.DataFrame() if constituents is None else constituents.copy()
        if not member_frame.empty:
            required_members = {"security_id", "effective_from", "effective_to"}
            missing = required_members - set(member_frame.columns)
            if missing:
                raise ValueError(f"constituents are missing columns: {sorted(missing)}")
            member_frame["effective_from"] = pd.to_datetime(member_frame["effective_from"]).dt.normalize()
            member_frame["effective_to"] = pd.to_datetime(member_frame["effective_to"]).dt.normalize()

        snapshots: dict[pd.Timestamp, frozenset[str]] = {}
        for raw_date in pd.DatetimeIndex(calculation_dates).normalize():
            age = (raw_date - master["list_date"]).dt.days
            eligible = master[age >= self.config.min_listing_days]
            if self.config.exclude_after_delisting:
                eligible = eligible[eligible["delist_date"].isna() | (raw_date < eligible["delist_date"])]
            ids = set(eligible["security_id"].astype(str))

            day_status = status[status["date"].eq(raw_date)] if not status.empty else status
            if day_status.empty and self.config.missing_status_policy == "exclude":
                ids.clear()
            elif not day_status.empty:
                day_status = day_status.set_index("security_id")
                ids &= set(day_status.index.astype(str))
                if self.config.exclude_st:
                    ids -= set(day_status.index[day_status["is_st"].fillna(True).astype(bool)].astype(str))
                if self.config.exclude_suspended:
                    ids -= set(day_status.index[day_status["is_suspended"].fillna(True).astype(bool)].astype(str))
                if "is_trading" in day_status:
                    ids -= set(day_status.index[~day_status["is_trading"].fillna(False).astype(bool)].astype(str))

            if self.config.restrict_to_benchmark_members:
                if member_frame.empty:
                    ids.clear()
                else:
                    # New members are included on effective_from; old members are
                    # excluded on effective_to. This includes the rebalance day.
                    active = member_frame[
                        (member_frame["effective_from"] <= raw_date)
                        & (member_frame["effective_to"].isna() | (raw_date < member_frame["effective_to"]))
                    ]
                    ids &= set(active["security_id"].astype(str))
            snapshots[raw_date] = frozenset(ids)
        return snapshots


def apply_universe(values: pd.Series, universe: dict[pd.Timestamp, frozenset[str]]) -> pd.Series:
    if list(values.index.names) != ["date", "security_id"]:
        raise ValueError("factor values must use a ['date', 'security_id'] MultiIndex")
    dates = pd.DatetimeIndex(values.index.get_level_values("date")).normalize()
    securities = values.index.get_level_values("security_id").astype(str)
    mask = [security in universe.get(date, frozenset()) for date, security in zip(dates, securities)]
    return values[pd.Series(mask, index=values.index)]
