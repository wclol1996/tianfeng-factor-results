from __future__ import annotations

from collections.abc import Mapping
import math

import numpy as np
import pandas as pd

from factor_engine.catalog import FactorSpec
from factor_engine.data.base import FactorContext
from factor_engine.data.pit import pit_history
from factor_engine.factors.common import (
    INDEX_NAMES,
    cross_sectional_rank_gauss,
    cross_sectional_zscore,
    safe_divide,
)
from factor_engine.registry import registry
from factor_engine.universe import apply_universe


class FormulaEvaluator:
    def __init__(self, context: FactorContext, allow_release_day: bool = True, calculation_time: str = "15:00:00"):
        self.context = context
        self.allow_release_day = allow_release_day
        self.calculation_time = calculation_time
        self._financial_cache: dict[tuple[str, int], pd.Series] = {}

    def field(self, field: str, source: str = "financial", lag: int = 0) -> pd.Series:
        if source == "daily":
            if lag:
                raise ValueError("daily fields do not use report_lag")
            if field not in self.context.daily:
                raise ValueError(f"daily data is missing canonical column: {field}")
            return self.context.daily[field].astype(float).sort_index()
        if source != "financial":
            raise ValueError(f"unsupported formula source: {source}")
        key = (field, lag)
        if key not in self._financial_cache:
            history = pit_history(
                self.context.financial_observations,
                self.context.calculation_dates,
                fields=[field],
                depth=lag + 1,
                allow_release_day=self.allow_release_day,
                calculation_time=self.calculation_time,
            )
            # A report lag is a calendar-quarter lag, not the nth available
            # record. Missing quarters must not turn YoY into another horizon.
            if history.empty:
                selected = history
            else:
                quarters = pd.to_datetime(history["report_period"]).dt.to_period("Q").astype("int64")
                latest = quarters.groupby([history["date"], history["security_id"]]).transform("max")
                selected = history[(latest - quarters).eq(lag)]
            if selected.empty:
                series = pd.Series(
                    dtype=float,
                    index=pd.MultiIndex.from_arrays([[], []], names=INDEX_NAMES),
                )
            else:
                series = selected.set_index(INDEX_NAMES)["value"].astype(float).sort_index()
            self._financial_cache[key] = series
        return self._financial_cache[key]

    def evaluate(self, expression: Mapping) -> pd.Series:
        operation = expression["op"]
        if operation == "field":
            return self.field(
                expression["field"], expression.get("source", "financial"), int(expression.get("lag", 0)),
            )
        if operation == "constant":
            raise ValueError("constant-only expressions do not define a security index")
        if operation in {"add", "sub", "mul", "div"}:
            left, right = (self.evaluate(item) for item in expression["args"])
            if operation == "add":
                return left.add(right, fill_value=np.nan)
            if operation == "sub":
                return left.subtract(right, fill_value=np.nan)
            if operation == "mul":
                return left.multiply(right)
            return safe_divide(left, right)
        if operation == "sum":
            values = [self.evaluate(item) for item in expression["args"]]
            return pd.concat(values, axis=1).sum(axis=1, min_count=len(values))
        if operation == "log":
            value = self.evaluate(expression["arg"])
            return np.log(value.where(value > 0))
        if operation == "inverse":
            value = self.evaluate(expression["arg"])
            return safe_divide(pd.Series(1.0, index=value.index), value)
        if operation in {"diff", "pct_change", "cagr"}:
            field = expression["field"]
            lag = int(expression["lag"])
            source = expression.get("source", "financial")
            current, previous = self.field(field, source, 0), self.field(field, source, lag)
            if operation == "diff":
                return current - previous
            growth = safe_divide(current, previous)
            if operation == "pct_change":
                return growth - 1.0
            years = float(expression["years"])
            return growth.where(growth > 0).pow(1.0 / years) - 1.0
        if operation in {"history_mean", "history_std"}:
            field = expression["field"]
            depth = int(expression["depth"])
            source = expression.get("source", "financial")
            values = pd.concat(
                [self.field(field, source, lag).rename(str(lag)) for lag in range(depth)], axis=1,
            )
            return values.mean(axis=1, skipna=False) if operation == "history_mean" else values.std(axis=1, ddof=1, skipna=False)
        if operation == "scale":
            return self.evaluate(expression["arg"]) * float(expression["value"])
        if operation == "average":
            values = [self.evaluate(item) for item in expression["args"]]
            return pd.concat(values, axis=1).mean(axis=1, skipna=False)
        raise ValueError(f"unsupported fundamental formula operation: {operation}")


@registry.register("fundamental_formula")
def compute_fundamental(context: FactorContext, spec: FactorSpec) -> pd.Series:
    expression = spec.parameters.get("expression")
    if not isinstance(expression, dict):
        raise ValueError(f"{spec.factor_id}: missing expression in parameters_json")
    evaluator = FormulaEvaluator(
        context,
        allow_release_day=bool(spec.parameters.get("allow_release_day", context.pit_allow_release_day)),
        calculation_time=str(spec.parameters.get("calculation_time", context.pit_calculation_time)),
    )
    result = evaluator.evaluate(expression)
    transform = spec.parameters.get("transform")
    if transform not in {None, "none"} and context.universe:
        result = apply_universe(result, dict(context.universe))
    if transform == "rank_gauss":
        result = cross_sectional_rank_gauss(result)
    elif transform == "zscore":
        result = cross_sectional_zscore(result)
    elif transform not in {None, "none"}:
        raise ValueError(f"{spec.factor_id}: unsupported cross-sectional transform {transform!r}")
    result.name = spec.factor_id
    return result.replace([np.inf, -np.inf], np.nan).sort_index()
