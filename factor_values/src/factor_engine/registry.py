from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from factor_engine.catalog import FactorSpec
from factor_engine.data.base import FactorContext


FactorFunction = Callable[[FactorContext, FactorSpec], pd.Series]


class FactorRegistry:
    def __init__(self) -> None:
        self._functions: dict[str, FactorFunction] = {}

    def register(self, name: str) -> Callable[[FactorFunction], FactorFunction]:
        def decorator(function: FactorFunction) -> FactorFunction:
            if name in self._functions:
                raise ValueError(f"factor implementation already registered: {name}")
            self._functions[name] = function
            return function

        return decorator

    def get(self, name: str) -> FactorFunction:
        try:
            return self._functions[name]
        except KeyError as error:
            raise KeyError(f"factor implementation is not registered: {name}") from error

    @property
    def names(self) -> frozenset[str]:
        return frozenset(self._functions)


registry = FactorRegistry()


def load_builtins() -> None:
    # Imports are intentionally local so catalog validation can run without
    # importing heavy pandas factor modules until needed.
    from factor_engine.factors import (  # noqa: F401
        apm,
        consensus,
        corporate_dividend,
        daily_price,
        delta_roe,
        event_price,
        ff3,
        financial_statement,
        fundamental,
        intraday,
        intangible_value,
        money_flow,
        risk_style,
    )
