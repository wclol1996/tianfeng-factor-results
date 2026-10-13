"""Explicit immutable native-total-MV restoration for approved historical inputs."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from collections.abc import Callable

import numpy as np
import pandas as pd

POLICY = "native_total_mv_direct_v1"


def native_total_market_cap_adapter(
    gate_path: str | Path, gate_sha256: str,
) -> Callable[..., pd.DataFrame]:
    """Bind approved source evidence once; return a total-cap-only restoration."""
    def checked_bytes(path: str | Path, expected: str) -> bytes:
        content = Path(path).read_bytes()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError("Native total-MV source binding SHA mismatch")
        return content

    gate = json.loads(checked_bytes(gate_path, gate_sha256))
    if gate.get("status") != "SOURCE_GATE_READY_FOR_FINANCIAL33_NATIVE_TOTALCAP_RESTORATION":
        raise ValueError("Native total-MV source gate is not ready")
    mapping = gate["native_field_mapping"]
    if (mapping["api"], mapping["field"], mapping["multiplier"], mapping["canonical"], mapping["canonical_unit"]) != (
        "daily_basic", "total_mv", 10000, "total_market_cap", "CNY",
    ):
        raise ValueError("Native total-MV policy requires the approved direct field and units")
    for key in ("source_QA", "existing_delta_exact_binding"):
        evidence = gate[key]
        checked_bytes(evidence["path"], evidence["sha256"])
    binding = gate["existing_delta_exact_binding"]
    source_manifest = json.loads(checked_bytes(
        Path(binding["overlay_path"]).parent / "manifest.json", binding["overlay_manifest_sha256"],
    ))
    checked_bytes(binding["source_contract_path"], binding["source_contract_sha256"])
    parents = frozenset(row["sha256"] for row in source_manifest["parents"])
    for row in source_manifest["parents"]:
        checked_bytes(row["path"], row["sha256"])
    delta = pd.read_parquet(io.BytesIO(checked_bytes(binding["overlay_path"], binding["overlay_sha256"])),
                            columns=["date", "security_id", "total_market_cap"])
    if delta[["date", "security_id"]].isna().any().any() or delta.duplicated(["date", "security_id"]).any():
        raise ValueError("Native total-MV overlay has invalid identity keys")
    delta["date"] = pd.to_datetime(delta["date"])
    overlay = delta.set_index(["date", "security_id"])["total_market_cap"]
    native = overlay.dropna()
    if not np.isfinite(native).all() or not native.gt(0).all():
        raise ValueError("Native total-MV overlay requires positive finite known values")

    def restore(
        daily: pd.DataFrame, *, parent_manifest_sha256: str, policy: str,
    ) -> pd.DataFrame:
        if policy != POLICY or parent_manifest_sha256 not in parents:
            raise ValueError("Unbound native total-MV policy or source parent")
        if list(daily.index.names) != ["date", "security_id"] or daily.index.has_duplicates:
            raise ValueError("Native total-MV input requires unique canonical date/security keys")
        dates = daily.index.get_level_values("date")
        codes = daily.index.get_level_values("security_id")
        if not isinstance(dates, pd.DatetimeIndex) or dates.hasnans or codes.isna().any():
            raise ValueError("Native total-MV input has invalid date/security keys")
        if not dates.equals(dates.normalize()) or (codes.astype(str).str.strip() == "").any():
            raise ValueError("Native total-MV input requires midnight dates and nonblank security")
        old = daily["total_market_cap"]
        known = old.dropna()
        if not np.isfinite(known).all() or not known.gt(0).all():
            raise ValueError("Native total-MV input has invalid known capitalizations")
        replacement = overlay.reindex(daily.index)
        both = old.notna() & replacement.notna()
        if not old[both].eq(replacement[both]).all():
            raise ValueError("Native total-MV known value conflicts with immutable source")
        out = daily.copy()
        out["total_market_cap"] = old.where(old.notna(), replacement)
        return out

    return restore
