from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from factor_engine.catalog import FactorSpec
from factor_engine.data.base import FactorContext
from factor_engine.factors.common import (
    INDEX_NAMES,
    cross_sectional_rank_gauss,
    group_transform,
    safe_divide,
)
from factor_engine.registry import registry
from factor_engine.universe import historical_st_eligible as _historical_st_eligible


FORMULA_IDS = frozenset({
    "EPTTM", "CFPTTM", "SPTTM",
    "finexp_ratio_ttm", "money2mv", "PCFTTM", "dol_ttm1",
    "npttm_chg", "rev_growth_com_3y", "rev_growth_com_2y", "tot_assets_cash_recovery", "factor_op_roa", "gross_margin_ttm", "gross_margin_ttm_chg", "roeq_mean_div_std_3y", "roeq_mean_sub_std_3y", "acc_profit_ttm",
    "cfottm2curliab", "cfottm2curliab_chg", "cfo2tb_ttm", "liab2assets", "liab_to_assets_chg", "goodwill2na", "goodwill2ta",
    "sug", "sue", "sur", "sucfo", "sur_cf", "performances_trend", "factor_roe_deducted",
    "factor_eaa", "factor_eap", "factor_eav", "factor_egv",
    "epttm_rg_zscore", "delta_roeq", "delta_roaq", "OPR_Skew", "npyoy",
    "npqyoy", "npttm_qoq", "capex_da_ratio_ttm", "revq_yoy_diff4",
    "revq_yoy_diff1", "npq_yoy_diff4", "npq_yoy_diff1", "gpmq_diff4",
    "gpmq_diff1", "npm_diff4", "npm_diff1", "cfo2rev_ttm_diff",
    "npttm_yoy", "salettm_yoy", "dol_ttm2", "POP_Skew", "Rel_POP_Skew",
    "divp_ttm_rg", "pdiv_ttm_rg", "ebitq2ev", "epq", "epq_rg", "peq_rg",
    "epttm_rg", "pettm_rg", "value_bivariate_rg", "ps_ttm_rg", "sp_ttm_rg",
    "pcf_ttm_rg", "cfp_ttm_rg", "spq_rg", "psq_rg", "cfpq_rg", "pcfq_rg",
    "fcffp_ttm", "EPTTM_3y_quantile", "CFPTTM_3y_quantile", "cfo_ttm_yoy",
    "roe_ttm", "roeq_mean_3y", "roeq_std_3y", "roeq_mean_12q", "roeq_std_12q", "op2or_ttm_chg", "taxes_payable_ratio_zscore", "cfo_yoy", "fcff_ttm", "fcff_ratio", "fcff_ratio_qua", "fcffoe_ttm",
    "fcffoe_ttm_diff4", "fcffoe_ttm_diff1", "fcffoe_qua",
    "fcffoe_qua_delta", "cfooe_ttm", "cfooe_ttm_diff4", "cfooe_ttm_diff1",
    "cfooe_qua", "cfooe_qua_delta", "cfoe_ttm", "cfoe_ttm_diff4",
    "cfoe_ttm_diff1", "cfoe_qua", "cfoe_qua_delta", "roe_ttm_rg", "roeq_rg",
    "eyoev_ttm", "net_margin_ttm", "sales_ttm_per_share",
    "NET_PROFIT_INCL_MIN_INT_INC_TTM", "NET_PROFIT_EXCL_MIN_INT_INC_TTM",
    "NET_PROFIT_AFTER_DED_NR_LP_TTM", "NET_PROFIT_INCL_MIN_INT_INC_QUA",
    "NET_PROFIT_EXCL_MIN_INT_INC_QUA", "NET_PROFIT_AFTER_DED_NR_LP_QUA",
    "roic_ttm", "ROE", "quick_ratio", "assets_turnoverTTM_chg", "profit_quality",
    "acc_profit_qua", "cfo_cv", "cfo_cv_pit",
    "opex_ttm", "da2rev_ttm", "ar2or_ttm", "ar2or_ttm_diff1",
    "assets_turnover_ttm", "inventory_turnover_ttm", "receivable_turnover_ttm",
    "opex2ta_ttm", "EPS_Delta_Vol", "da2gross_ttm",
    "capex2np_10y", "np_qua_chg_acc", "SalesG", "ProfitG", "CFOG",
    "np_yoy", "sale_yoy", "cum_rd_exp", "fcff_yoy", "fcff_cv", "fcffG",
    "pb_int", "net_wc", "nwc2rev_ttm", "np_per_share", "up_per_share", "cfo2rev_ttm", "ebit_ttm", "ebitda_ttm", "ebit2rev_ttm", "ebitda2rev_ttm", "ebit2interest_ttm", "re_qoq", "re_yoy", "bp_rg", "pb_rg", "cp_rg", "pc_rg",
    "PB_3y_quantile", "pb_roe_quantile_diff", "pb_roe_rank_dfif",
    "factor_np_growth", "factor_or_growth", "factor_np_growth_rel",
    "factor_or_growth_rel",
    "pb_roe_reg", "lnpb_roe_reg", "pb_roe_reg_rg",
    "ocfa_res",
    "res_roa", "res_roa2",
    "ABCFO1", "ABCFO4", "ABCFO5",
    "sfli", "sfli2",
    "nol", "nol2",
})

GROWTH_TREND_IDS = frozenset({
    "factor_np_growth", "factor_or_growth", "factor_np_growth_rel",
    "factor_or_growth_rel",
})

GROWTH_TREND_SPECS: dict[str, tuple[str, bool]] = {
    "factor_np_growth": ("net_profit", False),
    "factor_or_growth": ("operating_revenue", False),
    "factor_np_growth_rel": ("net_profit", True),
    "factor_or_growth_rel": ("operating_revenue", True),
}

GROWTH_TREND_WINDOW = 12

PB_ROE_REGRESSION_IDS = frozenset({
    "pb_roe_reg", "lnpb_roe_reg", "pb_roe_reg_rg",
})

OCFA_RESIDUAL_IDS = frozenset({"ocfa_res"})

IDIOSYNCRATIC_PROFIT_IDS = frozenset({"res_roa", "res_roa2"})

ABNORMAL_CFO_IDS = frozenset({"ABCFO1", "ABCFO4", "ABCFO5"})

SHORT_LOAN_LONG_INVESTMENT_IDS = frozenset({"sfli", "sfli2"})
UNEXPECTED_PRIMITIVE_FIELDS = {"sur": "operating_revenue", "sucfo": "operating_cash_flow", "sur_cf": "cash_received_sales"}
CASHFLOW_EQUITY_SOURCE_FIELDS = ("operating_cash_flow", "capital_expenditure_cash_paid", "cash_net_increase", "equity_parent", "operating_revenue")
PER_SHARE_PRIMITIVE_FIELDS = {"np_per_share": "net_profit", "sales_ttm_per_share": "operating_revenue", "up_per_share": "undistributed_profit"}
CASHFLOW_HISTORY_SOURCE_FIELDS = ("operating_cash_flow", "capital_expenditure_cash_paid", "operating_revenue")
STATEMENT_EARNINGS_SOURCE_FIELDS = ("ebit", "ebitda", "operating_revenue", "interst_income")

NET_OPERATING_LIABILITY_IDS = frozenset({"nol", "nol2"})

DEFAULT_NOL_EXCLUDED_CITIC_L1 = (
    "银行", "非银行金融", "综合金融", "综合",
)

QUALITY_FLAG_IDS = (
    GROWTH_TREND_IDS
    | PB_ROE_REGRESSION_IDS
    | OCFA_RESIDUAL_IDS
    | IDIOSYNCRATIC_PROFIT_IDS
    | ABNORMAL_CFO_IDS
    | SHORT_LOAN_LONG_INVESTMENT_IDS
    | NET_OPERATING_LIABILITY_IDS
)


# The adapter may emit these canonical names directly or leave the documented
# Gildata/Chinese subject name in ``field``.  Keeping aliases at this boundary
# prevents provider-specific codes from leaking into factor formulas.
FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "total_assets": ("total_assets", "资产总计", "TotalAssets"),
    "total_liabilities": ("total_liabilities", "负债合计", "TotalLiability"),
    "equity_parent": (
        "equity_parent", "股东权益合计(不含少数股东权益)", "净资产", "归属母公司股东权益",
        "EquityParentCompanyOwners",
    ),
    "inventory": ("inventory", "存货", "Inventory"),
    "fixed_assets": ("fixed_assets", "固定资产", "FixedAssets"),
    "other_receivables": ("other_receivables", "其他应收款", "OtherReceivables"),
    "non_interest_current_liabilities": (
        "non_interest_current_liabilities", "无息流动负债(元)", "无息流动负债",
    ),
    "notes_receivable": ("notes_receivable", "应收票据", "NotesReceivable"),
    "accounts_receivable_combined": (
        "accounts_receivable_combined", "应收票据及应收账款", "NotesAndAccountsReceivable",
    ),
    "accounts_receivable": ("accounts_receivable", "应收账款", "AccountsReceivable"),
    "prepayments": ("prepayments", "预付款项", "Prepayments"),
    "contract_assets": ("contract_assets", "合同资产", "ContractAssets"),
    "other_current_assets": ("other_current_assets", "其他流动资产", "OtherCurrentAssets"),
    "notes_payable": ("notes_payable", "应付票据", "NotesPayable"),
    "accounts_payable_combined": (
        "accounts_payable_combined", "应付票据及应付账款", "NotesAndAccountsPayable",
    ),
    "accounts_payable": ("accounts_payable", "应付账款", "AccountsPayable"),
    "employee_compensation_payable": (
        "employee_compensation_payable", "应付职工薪酬", "EmployeeCompensationPayable",
    ),
    "long_term_employee_compensation_payable": (
        "long_term_employee_compensation_payable", "长期应付职工薪酬", "lt_payroll_payable",
    ),
    "taxes_payable": ("taxes_payable", "应交税费", "TaxesPayable"),
    "other_payables": ("other_payables", "其他应付款", "OtherPayables"),
    "undistributed_profit": (
        "undistributed_profit", "未分配利润", "UndistributedProfit",
    ),
    "surplus_reserve": ("surplus_reserve", "盈余公积金", "盈余公积", "SurplusReserve"),
    "contract_liabilities": ("contract_liabilities", "合同负债", "ContractLiability"),
    "advance_receipts": ("advance_receipts", "预收款项", "AdvancesFromCustomers"),
    "cash": ("cash", "货币资金", "MonetaryFunds"),
    "interest_bearing_debt": ("interest_bearing_debt", "带息债务", "InterestBearingDebt"),
    "minority_interest": ("minority_interest", "少数股东权益", "MinorityInterests"),
    "other_equity_instruments": (
        "other_equity_instruments", "其他权益工具", "OtherEquityInstruments",
    ),
    "total_shares": ("total_shares", "总股本", "TotalShares"),
    "paid_in_capital": ("paid_in_capital", "实收资本", "股本金额"),
    "capital_reserve": ("capital_reserve", "资本公积", "CapitalReserve"),
    "long_term_borrowing": (
        "long_term_borrowing", "长期借款", "LongTermBorrowing",
    ),
    "bonds_payable": ("bonds_payable", "应付债券", "BondsPayable"),
    "noncurrent_liabilities_due_within_one_year": (
        "noncurrent_liabilities_due_within_one_year", "一年内到期的非流动负债",
        "一年内到期非流动负债", "NonCurrentLiabilitiesDueWithinOneYear",
    ),
    "enterprise_value": ("enterprise_value", "企业价值", "EnterpriseValue"),
    "net_working_capital": ("net_working_capital", "净营运资本", "NetWorkingCapital"),
    "cash_end": (
        "cash_end", "期末现金及现金等价物余额", "CashAndCashEquivalentsAtEnd",
    ),
    "operating_revenue": ("operating_revenue", "营业收入", "营业收入TTM", "OperatingRevenue"),
    "total_revenue": ("total_revenue", "营业总收入", "营业总收入TTM"),
    "sales_revenue": ("sales_revenue", "主营业务收入", "SalesRevenue"),
    "operating_cost": ("operating_cost", "营业成本", "OperatingCost"),
    "operating_profit": ("operating_profit", "营业利润", "OperatingProfit"),
    "net_profit": (
        "net_profit", "净利润", "净利润（含少数股东损益）", "净利润(含少数股东损益)",
        "NetProfit",
    ),
    "net_profit_parent": (
        "net_profit_parent", "归属母公司净利润", "归属母公司净利润TTM",
        "净利润(不含少数股东损益)", "NPParentCompanyOwners",
    ),
    "net_profit_parent_excl_nr": (
        "net_profit_parent_excl_nr", "扣非归属母公司净利润", "单季度扣非归母净利润",
        "NetProfitParentExclNonRecurring",
    ),
    "eps": ("eps", "每股收益", "单季度EPS", "单季度_每股收益", "BasicEPS"),
    "ebit": ("ebit", "息税前利润", "EBIT TTM", "EBIT"),
    "management_expense": ("management_expense", "管理费用", "管理费用TTM", "ManagementExpense"),
    "selling_expense": ("selling_expense", "销售费用", "销售费用TTM", "SellingExpense"),
    "financial_expense": ("financial_expense", "财务费用", "财务费用TTM", "FinancialExpense"),
    "rd_expense": ("rd_expense", "研发费用", "ResearchAndDevelopmentExpense"),
    "depreciation_amortization": (
        "depreciation_amortization", "折旧与摊销", "DepreciationAndAmortization",
    ),
    "operating_cash_flow": (
        "operating_cash_flow", "经营活动产生的现金流量净额", "经营活动产生的现金流量净额TTM",
        "NetOperateCashFlow",
    ),
    "other_operating_cash_flow": (
        "other_operating_cash_flow", "其他与经营活动相关的现金流",
        "其他与经营活动有关的现金流量净额", "OtherOperatingCashFlow",
    ),
    "cash_received_other_operating": (
        "cash_received_other_operating", "收到其他与经营活动有关的现金",
        "CashReceivedFromOtherOperatingActivities",
    ),
    "cash_paid_other_operating": (
        "cash_paid_other_operating", "支付其他与经营活动有关的现金",
        "CashPaidForOtherOperatingActivities",
    ),
    "cash_paid_employees": (
        "cash_paid_employees", "支付给职工以及为职工支付的现金",
        "支付给职工的现金流", "CashPaidToEmployees",
    ),
    "cash_net_increase": (
        "cash_net_increase", "现金及现金等价物净增加额", "NetIncreaseInCashAndCashEquivalents",
    ),
    "capital_expenditure_cash_paid": (
        "capital_expenditure_cash_paid", "capex", "资本开支",
        "购建固定资产、无形资产和其他长期资产支付的现金",
        "购建固定资产无形资产和其他长期资产支付的现金",
        "CashPaidToAcquireLongTermAssets",
    ),
    "capex_fixed": ("capex_fixed", "购建固定资产", "CashPaidForFixedAssets"),
    "capex_intangible": (
        "capex_intangible", "无形资产和其他长期资产支付的现金",
        "CashPaidForIntangibleAndOtherLongTermAssets",
    ),
    "cash_received_sales": (
        "cash_received_sales", "提供劳务收到的现金", "销售商品", "销售商品、提供劳务收到的现金",
        "CashReceivedFromSalesAndServices",
    ),
    "cash_received_disposal_long_term_assets": (
        "cash_received_disposal_long_term_assets", "处置固定资产产生的现金流入",
        "处置固定资产、无形资产和其他长期资产收回的现金净额",
        "处置固定资产无形资产和其他长期资产收回的现金净额",
        "CashReceivedFromDisposalOfLongTermAssets",
    ),
    "cash_dividend": ("cash_dividend", "现金分红", "现金分红总额", "CashDividend"),
}

POINT_FIELDS = frozenset({
    "total_assets", "total_liabilities", "equity_parent", "inventory", "fixed_assets",
    "other_receivables", "non_interest_current_liabilities", "notes_receivable",
    "accounts_receivable_combined", "accounts_receivable", "prepayments",
    "contract_assets",
    "other_current_assets", "notes_payable", "accounts_payable_combined",
    "accounts_payable", "employee_compensation_payable", "taxes_payable",
    "long_term_employee_compensation_payable",
    "other_payables", "undistributed_profit", "surplus_reserve", "contract_liabilities",
    "advance_receipts", "cash", "interest_bearing_debt", "minority_interest",
    "other_equity_instruments", "total_shares", "paid_in_capital", "equity_financing_cny", "long_debt_cny", "enterprise_value", "net_working_capital",
    "cash_end", "capital_reserve", "long_term_borrowing", "bonds_payable",
    "noncurrent_liabilities_due_within_one_year",
})

FLOW_FIELDS = frozenset(set(FIELD_ALIASES) - set(POINT_FIELDS))

DAILY_ALIASES: dict[str, tuple[str, ...]] = {
    "total_market_cap": ("total_market_cap", "market_cap"),
    "raw_close": ("raw_close",),
    "total_shares": ("total_shares", "shares"),
    "industry": (
        "citic_industry_l1", "citic_level1", "中信一级行业",
        "industry_l1", "industry", "industry_code",
    ),
    "industry_l1": (
        "citic_industry_l1", "citic_level1", "中信一级行业",
        "industry_l1", "industry", "industry_code",
    ),
    "industry_l2": (
        "citic_industry_l2", "citic_level2", "中信二级行业", "industry_l2",
    ),
    "book_to_price": ("book_to_price", "bp"),
    "price_to_book": ("pb", "pb_lf", "price_to_book"),
}

RANK_GAUSS_IDS = frozenset({
    "divp_ttm_rg", "pdiv_ttm_rg", "epq_rg", "peq_rg", "epttm_rg", "pettm_rg",
    "ps_ttm_rg", "sp_ttm_rg", "pcf_ttm_rg", "cfp_ttm_rg", "spq_rg", "psq_rg",
    "cfpq_rg", "pcfq_rg", "roe_ttm_rg", "roeq_rg",
    "bp_rg", "pb_rg", "cp_rg", "pc_rg",
})

TIME_TRANSFORM_IDS = frozenset({
    "epttm_rg_zscore", "EPTTM_3y_quantile", "CFPTTM_3y_quantile", "PB_3y_quantile",
})


def supports_factor_id(factor_id: str) -> bool:
    return factor_id in FORMULA_IDS


def required_financial_fields() -> frozenset[str]:
    """Canonical raw cumulative/point-in-time fields used by this implementation."""

    return frozenset(FIELD_ALIASES)


def _empty_report_series(index: pd.MultiIndex) -> pd.Series:
    return pd.Series(np.nan, index=index, dtype=float)


def _canonical_field(value: object) -> str:
    text = str(value).strip()
    for canonical, aliases in FIELD_ALIASES.items():
        if text in aliases:
            return canonical
    return text


def _normalize_observations(observations: pd.DataFrame) -> pd.DataFrame:
    required = {"security_id", "report_period", "available_at", "field", "value"}
    missing = required - set(observations.columns)
    if missing:
        raise ValueError(f"financial observations are missing columns: {sorted(missing)}")
    frame = observations.copy()
    frame["security_id"] = frame["security_id"].astype(str)
    frame["report_period"] = pd.to_datetime(frame["report_period"]).dt.normalize()
    frame["available_at"] = pd.to_datetime(frame["available_at"])
    if "revision_at" not in frame:
        frame["revision_at"] = pd.NaT
    else:
        frame["revision_at"] = pd.to_datetime(frame["revision_at"])
    if frame["revision_at"].lt(frame["available_at"]).any():
        raise ValueError("financial revision predates availability")
    frame["effective_at"] = frame["revision_at"].fillna(frame["available_at"])
    if frame["effective_at"].isna().any():
        raise ValueError("financial observations cannot have a blank effective timestamp")

    # Gildata offers A (cumulative) and Q (single-quarter) records together.
    # The engine deliberately retains A only and derives Q itself so every
    # factor uses the same report version.
    method_column = next(
        (name for name in ("accounting_method", "accountingMethod", "核算方式") if name in frame),
        None,
    )
    if method_column is not None:
        method = frame[method_column].fillna("").astype(str).str.strip().str.upper()
        cumulative = method.isin({"A", "累计值", "CUMULATIVE", "YTD", ""})
        # Published single-quarter EPS is a per-share ratio, not an additive
        # cash-flow amount. Its explicitly named canonical field retains Q rows.
        published_eps = frame["field"].eq("eps_quarter")
        frame = frame[cumulative | published_eps].copy()

    frame["field"] = frame["field"].map(_canonical_field)
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    quarter_end = frame["report_period"].dt.strftime("%m-%d").isin({"03-31", "06-30", "09-30", "12-31"})
    frame = frame[quarter_end].copy()
    frame["_input_order"] = np.arange(len(frame))
    return frame.sort_values(["effective_at", "_input_order"]).reset_index(drop=True)


def _visible_quarterly_frame(observations: pd.DataFrame, cutoff: pd.Timestamp, *, expense_vintage: bool = False) -> pd.DataFrame:
    eligible = observations[observations["effective_at"] <= cutoff]
    if eligible.empty:
        index = pd.MultiIndex.from_arrays([[], []], names=["security_id", "report_period"])
        return pd.DataFrame(index=index)
    latest = eligible.drop_duplicates(
        ["security_id", "report_period", "field"], keep="last",
    )
    wide = latest.pivot(
        index=["security_id", "report_period"], columns="field", values="value",
    ).sort_index()
    wide.columns.name = None
    if expense_vintage:
        fields = ["operating_cost", "management_expense", "selling_expense", "rd_expense"]
        vintages = latest[latest.field.isin(fields)].pivot(
            index=["security_id", "report_period"], columns="field", values="effective_at",
        ).reindex(columns=fields)
        wide = wide.join(vintages.add_suffix("__effective_at"))
    return wide


@dataclass
class QuarterlyFinancials:
    frame: pd.DataFrame
    context: FactorContext
    calculation_date: pd.Timestamp
    _cache: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if list(self.frame.index.names) != ["security_id", "report_period"]:
            raise ValueError("quarterly frame must use security_id/report_period index")
        periods = self.frame.index.get_level_values("report_period")
        self.ordinal = pd.Series(
            periods.year * 4 + periods.quarter - 1, index=self.frame.index, dtype=float,
        )
        self.quarter = pd.Series(periods.quarter, index=self.frame.index, dtype=int)

    def raw(self, name: str) -> pd.Series:
        key = f"raw:{name}"
        if key not in self._cache:
            if name in self.frame:
                self._cache[key] = self.frame[name].astype(float)
            else:
                self._cache[key] = _empty_report_series(self.frame.index)
        return self._cache[key]

    def lag(self, values: pd.Series, periods: int) -> pd.Series:
        shifted = values.groupby(level="security_id", sort=False).shift(periods)
        prior_ordinal = self.ordinal.groupby(level="security_id", sort=False).shift(periods)
        return shifted.where((self.ordinal - prior_ordinal).eq(periods))

    def exact_quarter_lag(self, values: pd.Series, quarters: int) -> pd.Series:
        keys = pd.MultiIndex.from_arrays([
            values.index.get_level_values("security_id"),
            values.index.get_level_values("report_period") - pd.offsets.QuarterEnd(quarters),
        ], names=values.index.names)
        return pd.Series(values.reindex(keys).to_numpy(), index=values.index)

    def cumulative_ttm(self, name: str) -> pd.Series:
        current = self.raw(name)
        periods = current.index.get_level_values("report_period")
        previous_year_end = periods.to_period("Y").start_time - pd.Timedelta(days=1)
        keys = pd.MultiIndex.from_arrays([
            current.index.get_level_values("security_id"), previous_year_end,
        ], names=current.index.names)
        annual = pd.Series(current.reindex(keys).to_numpy(), index=current.index)
        result = current + annual - self.exact_quarter_lag(current, 4)
        return result.where(self.quarter.ne(4), current)

    def single(self, name: str) -> pd.Series:
        key = f"single:{name}"
        if key not in self._cache:
            cumulative = self.raw(name)
            previous = self.lag(cumulative, 1)
            self._cache[key] = cumulative.where(self.quarter.eq(1), cumulative - previous)
        return self._cache[key]

    def rolling(
        self,
        values: pd.Series,
        window: int,
        operation: str | Callable[[np.ndarray], float],
        min_periods: int | None = None,
    ) -> pd.Series:
        minimum = window if min_periods is None else min_periods

        def calculate(group: pd.Series) -> pd.Series:
            roll = group.rolling(window, min_periods=minimum)
            if isinstance(operation, str):
                return getattr(roll, operation)()
            return roll.apply(operation, raw=True)

        result = values.groupby(level="security_id", group_keys=False, sort=False).apply(calculate)
        if result.index.nlevels > values.index.nlevels:
            result = result.droplevel(0)
        return result.reindex(values.index)

    def ttm(self, name: str) -> pd.Series:
        key = f"ttm:{name}"
        if key not in self._cache:
            single = self.single(name)
            total = self.rolling(single, 4, "sum")
            fourth_ordinal = self.ordinal.groupby(level="security_id", sort=False).shift(3)
            total = total.where((self.ordinal - fourth_ordinal).eq(3))
            # Cumulative Q4 is already the full annual TTM; single Q4 still
            # needs Q3 for subtraction and remains missing when Q3 is absent.
            self._cache[key] = total.where(~self.quarter.eq(4), self.raw(name))
        return self._cache[key]

    def signed_growth(self, values: pd.Series, periods: int) -> pd.Series:
        previous = self.lag(values, periods)
        return safe_divide(values - previous, previous.where(previous.ne(0)))

    def diff(self, values: pd.Series, periods: int) -> pd.Series:
        return values - self.lag(values, periods)

    def mean(self, values: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
        return self.rolling(values, window, "mean", min_periods)

    def std(self, values: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
        return self.rolling(values, window, "std", min_periods)

    def skew(self, values: pd.Series, window: int = 8) -> pd.Series:
        return self.rolling(values, window, "skew")

    def total(self, values: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
        return self.rolling(values, window, "sum", min_periods)

    def cumulative_sum(self, values: pd.Series) -> pd.Series:
        return values.groupby(level="security_id", sort=False).cumsum()

    def daily_security(self, name: str) -> pd.Series:
        if self.context.daily.empty:
            return pd.Series(dtype=float)
        aliases = DAILY_ALIASES[name]
        column = next((alias for alias in aliases if alias in self.context.daily), None)
        if column is None:
            return pd.Series(dtype=float)
        dates = self.context.daily.index.get_level_values("date").normalize()
        available_dates = dates[dates <= self.calculation_date]
        if available_dates.empty:
            return pd.Series(dtype=float)
        selected_date = available_dates.max()
        values = self.context.daily.xs(selected_date, level="date")[column]
        values.index = values.index.astype(str)
        return values

    def daily(self, name: str) -> pd.Series:
        by_security = self.daily_security(name)
        securities = self.frame.index.get_level_values("security_id")
        return pd.Series(securities.map(by_security), index=self.frame.index, dtype=float)

    def daily_at_report(self, name: str) -> pd.Series:
        key = f"daily_at_report:{name}"
        if key in self._cache:
            return self._cache[key]
        if self.context.daily.empty:
            self._cache[key] = _empty_report_series(self.frame.index)
            return self._cache[key]
        aliases = DAILY_ALIASES[name]
        column = next((alias for alias in aliases if alias in self.context.daily), None)
        if column is None:
            self._cache[key] = _empty_report_series(self.frame.index)
            return self._cache[key]
        output = _empty_report_series(self.frame.index)
        periods = self.frame.index.get_level_values("report_period")
        for security in self.frame.index.get_level_values("security_id").unique():
            try:
                history = self.context.daily.xs(security, level="security_id")[column].sort_index()
            except KeyError:
                continue
            rows = self.frame.index.get_level_values("security_id") == security
            targets = periods[rows]
            positions = history.index.searchsorted(targets, side="right") - 1
            valid = positions >= 0
            values = np.full(len(targets), np.nan)
            values[valid] = history.iloc[positions[valid]].to_numpy(dtype=float)
            output.loc[rows] = values
        self._cache[key] = output
        return output

    def latest(self, values: pd.Series) -> pd.Series:
        clean = values.replace([np.inf, -np.inf], np.nan).dropna()
        if clean.empty:
            return pd.Series(dtype=float)
        frame = clean.rename("value").reset_index().sort_values(["security_id", "report_period"])
        latest = frame.drop_duplicates("security_id", keep="last").set_index("security_id")["value"]
        latest.index = latest.index.astype(str)
        return latest.astype(float)


def _equity(financials: QuarterlyFinancials) -> pd.Series:
    reported = financials.raw("equity_parent")
    derived = financials.raw("total_assets") - financials.raw("total_liabilities")
    return reported.combine_first(derived)


def _receivables(financials: QuarterlyFinancials) -> pd.Series:
    combined = financials.raw("accounts_receivable_combined")
    components = _sum_available([
        financials.raw("notes_receivable"), financials.raw("accounts_receivable"),
    ])
    return combined.combine_first(components)


def _sum_available(values: list[pd.Series]) -> pd.Series:
    return pd.concat(values, axis=1).sum(axis=1, min_count=1)


def _payables(financials: QuarterlyFinancials) -> pd.Series:
    combined = financials.raw("accounts_payable_combined")
    components = _sum_available([
        financials.raw("notes_payable"), financials.raw("accounts_payable"),
    ])
    return combined.combine_first(components)


def _strict_net_working_capital(financials: QuarterlyFinancials) -> pd.Series:
    receivables = financials.raw("accounts_receivable_combined").combine_first(
        financials.raw("notes_receivable") + financials.raw("accounts_receivable"))
    payables = financials.raw("accounts_payable_combined").combine_first(
        financials.raw("notes_payable") + financials.raw("accounts_payable"))
    assets = _strict_sum([receivables, *[financials.raw(name) for name in
        ("prepayments", "other_receivables", "inventory", "other_current_assets")]])
    liabilities = _strict_sum([payables, *[financials.raw(name) for name in
        ("advance_receipts", "employee_compensation_payable", "taxes_payable", "other_payables")]])
    return assets - liabilities


def _net_working_capital(financials: QuarterlyFinancials) -> pd.Series:
    operating_current_assets = _sum_available([
        _receivables(financials),
        financials.raw("prepayments"),
        financials.raw("other_receivables"),
        financials.raw("inventory"),
        financials.raw("other_current_assets"),
    ])
    operating_current_liabilities = _sum_available([
        _payables(financials),
        financials.raw("advance_receipts"),
        financials.raw("employee_compensation_payable"),
        financials.raw("taxes_payable"),
        financials.raw("other_payables"),
    ])
    return operating_current_assets - operating_current_liabilities


def _book_to_price(financials: QuarterlyFinancials) -> pd.Series:
    supplied = financials.daily("book_to_price")
    derived = safe_divide(_equity(financials), financials.daily("total_market_cap"))
    return supplied.combine_first(derived)


def _price_to_book(financials: QuarterlyFinancials) -> pd.Series:
    supplied = financials.daily("price_to_book")
    reciprocal = safe_divide(
        pd.Series(1.0, index=financials.frame.index),
        _book_to_price(financials),
    )
    return supplied.combine_first(reciprocal)


def _quadratic_trend_coefficient(values: np.ndarray) -> float:
    if len(values) != 8 or not np.isfinite(values).all():
        return np.nan
    time = np.arange(8, dtype=float) - 3.5
    design = np.column_stack([np.ones(8), time, time**2])
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    return float(coefficients[2])


def _ols_slope_with_intercept(values: np.ndarray) -> float:
    """OLS time coefficient for t=1..N with an explicit intercept."""

    time = np.arange(1, len(values) + 1, dtype=float)
    centered_time = time - time.mean()
    centered_values = values - values.mean()
    denominator = float(np.dot(centered_time, centered_time))
    if denominator == 0:
        return np.nan
    return float(np.dot(centered_time, centered_values) / denominator)


def _growth_trend_values_and_flags(
    financials: QuarterlyFinancials,
    factor_id: str,
) -> tuple[pd.Series, pd.DataFrame]:
    """Calculate a 12-quarter trend and retain report-level quality evidence."""

    flow_field, relative = GROWTH_TREND_SPECS[factor_id]
    flow = financials.single(flow_field)
    equity = (financials.raw("total_assets") - financials.raw("total_liabilities")) if relative else _equity(financials)
    output = _empty_report_series(financials.frame.index)
    flags = pd.DataFrame(index=financials.frame.index)
    flags["window_start_report_period"] = pd.NaT
    flags["window_end_report_period"] = pd.NaT
    flags["observed_quarter_count"] = 0
    flags["has_12_consecutive_quarters"] = False
    flags["complete_flow_window"] = False
    flags["complete_equity_window"] = False
    flags["has_negative_equity"] = False
    flags["has_zero_equity_observation"] = False
    flags["average_equity"] = np.nan
    flags["flow_standard_deviation_ddof0"] = np.nan
    flags["valid_for_factor"] = False
    flags["reason_codes"] = "insufficient_history"

    # These factors are exposed as the current PIT state, so only the newest
    # visible report window for each security needs to be evaluated.
    for _, security_frame in financials.frame.groupby(level="security_id", sort=False):
        security_index = security_frame.index.sort_values()
        current_index = security_index[-1]
        window_index = security_index[-GROWTH_TREND_WINDOW:]
        observed = len(window_index)
        ordinals = financials.ordinal.loc[window_index].to_numpy(dtype=float)
        is_consecutive = (
            observed == GROWTH_TREND_WINDOW
            and np.array_equal(np.diff(ordinals), np.ones(GROWTH_TREND_WINDOW - 1))
        )
        flow_values = flow.loc[window_index].to_numpy(dtype=float)
        equity_values = equity.loc[window_index].to_numpy(dtype=float)
        complete_flow = is_consecutive and np.isfinite(flow_values).all()
        complete_equity = is_consecutive and np.isfinite(equity_values).all()
        finite_equity = equity_values[np.isfinite(equity_values)]
        has_negative_equity = bool((finite_equity < 0).any())
        has_zero_equity = bool((finite_equity == 0).any())
        average_equity = float(equity_values.mean()) if complete_equity else np.nan
        flow_std = float(flow_values.std(ddof=0)) if complete_flow else np.nan

        report_periods = window_index.get_level_values("report_period")
        flags.at[current_index, "window_start_report_period"] = report_periods.min()
        flags.at[current_index, "window_end_report_period"] = report_periods.max()
        flags.at[current_index, "observed_quarter_count"] = observed
        flags.at[current_index, "has_12_consecutive_quarters"] = is_consecutive
        flags.at[current_index, "complete_flow_window"] = complete_flow
        flags.at[current_index, "complete_equity_window"] = complete_equity
        flags.at[current_index, "has_negative_equity"] = has_negative_equity
        flags.at[current_index, "has_zero_equity_observation"] = has_zero_equity
        flags.at[current_index, "average_equity"] = average_equity
        flags.at[current_index, "flow_standard_deviation_ddof0"] = flow_std

        reasons: list[str] = []
        if observed < GROWTH_TREND_WINDOW:
            reasons.append("insufficient_history")
        if observed >= 2 and not np.array_equal(
            np.diff(ordinals), np.ones(observed - 1),
        ):
            reasons.append("nonconsecutive_quarters")
        if is_consecutive and not complete_flow:
            reasons.append("missing_flow")
        if is_consecutive and not complete_equity:
            reasons.append("missing_equity")
        if has_negative_equity:
            reasons.append("negative_equity")
        if has_zero_equity:
            reasons.append("zero_equity_observation")

        valid = complete_flow
        if relative:
            valid = (
                valid
                and complete_equity
                and not has_negative_equity
                and np.isfinite(average_equity)
                and average_equity > 0
            )
            if complete_equity and average_equity <= 0:
                reasons.append("nonpositive_average_equity")
        elif complete_flow and flow_std == 0:
            valid = False
            reasons.append("zero_flow_variance")

        if valid:
            if relative:
                output.at[current_index] = (
                    _ols_slope_with_intercept(flow_values) / average_equity
                )
            else:
                standardized = (flow_values - flow_values.mean()) / flow_std
                output.at[current_index] = _ols_slope_with_intercept(standardized)
        flags.at[current_index, "valid_for_factor"] = valid
        flags.at[current_index, "reason_codes"] = "|".join(reasons) if reasons else "ok"

    return output, flags


def _latest_report_frame(values: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Select each security's newest visible report without dropping missing values."""

    frame = values.reset_index().sort_values(["security_id", "report_period"])
    latest = frame.drop_duplicates("security_id", keep="last").set_index("security_id")
    latest.index = latest.index.astype(str)
    if isinstance(values, pd.Series):
        result = latest[values.name or 0]
        result.name = values.name
        return result
    return latest.drop(columns="report_period")


def _all_a_share_universe(
    context: FactorContext,
    date: pd.Timestamp,
) -> frozenset[str] | None:
    """Return the PIT eligible A-share pool, independent of final output restrictions."""

    if date in context.eligible_universe:
        return _historical_st_eligible(context, date, context.eligible_universe[date])
    if date in context.universe:
        return _historical_st_eligible(context, date, context.universe[date])
    return _historical_st_eligible(context, date, None)


def _pb_roe_regression_cross_section(
    financials: QuarterlyFinancials,
    factor_id: str,
    context: FactorContext,
    parameters: dict[str, object],
) -> pd.Series:
    """Equal-weight cross-sectional PB-on-ROE residual with an intercept."""

    date = financials.calculation_date
    pb_source = str(parameters.get("pb_source", "database")).strip().lower()
    if pb_source == "database":
        pb = financials.daily_security("price_to_book")
    elif pb_source == "derived_native_parent":
        market_cap = financials.daily_security("total_market_cap")
        latest_equity = _latest_report_frame(financials.raw("equity_parent"))
        pb = safe_divide(market_cap.where(market_cap.gt(0)), latest_equity)
    elif pb_source == "manual":
        market_cap = financials.daily_security("total_market_cap")
        latest_equity = financials.latest(_equity(financials))
        pb = safe_divide(market_cap, latest_equity)
    else:
        raise ValueError(
            f"{factor_id}: pb_source must be 'database', 'manual', or 'derived_native_parent', got {pb_source!r}",
        )

    if parameters.get("history_policy") == "pb_roe_current_report_ols_v2":
        equity = financials.raw("equity_parent")
        average_equity = (equity + financials.exact_quarter_lag(equity, 4)) / 2.0
        roe = _latest_report_frame(safe_divide(financials.cumulative_ttm("net_profit_parent"), average_equity))
    else:
        roe = financials.latest(_return_on_stock(
            financials,
            financials.ttm("net_profit_parent"),
            _equity(financials),
            4,
        ))
    allowed = _all_a_share_universe(context, date)
    if allowed is None:
        securities = pd.Index(sorted(set(pb.index) | set(roe.index)), dtype="object")
    else:
        securities = pd.Index(sorted(str(value) for value in allowed), dtype="object")
    pb = pb.reindex(securities).astype(float).replace([np.inf, -np.inf], np.nan)
    roe = roe.reindex(securities).astype(float).replace([np.inf, -np.inf], np.nan)

    log_dependent = factor_id == "lnpb_roe_reg"
    dependent = np.log(pb.where(pb > 0)) if log_dependent else pb.copy()
    valid_input = dependent.notna() & roe.notna()
    sample_size = int(valid_input.sum())
    residual = pd.Series(np.nan, index=securities, dtype=float)
    intercept = np.nan
    roe_coefficient = np.nan
    full_rank = False
    if sample_size >= 3:
        sample_roe = roe.loc[valid_input].to_numpy(dtype=float)
        design = np.column_stack([np.ones(sample_size), sample_roe])
        full_rank = bool(np.linalg.matrix_rank(design) == design.shape[1])
        if full_rank:
            sample_dependent = dependent.loc[valid_input].to_numpy(dtype=float)
            coefficients, *_ = np.linalg.lstsq(design, sample_dependent, rcond=None)
            intercept = float(coefficients[0])
            roe_coefficient = float(coefficients[1])
            residual.loc[valid_input] = sample_dependent - design @ coefficients

    flags = pd.DataFrame(index=securities)
    flags["pb_source"] = pb_source
    flags["pb"] = pb
    flags["roe"] = roe
    flags["dependent_value"] = dependent
    flags["has_pb"] = pb.notna()
    flags["pb_positive"] = pb > 0
    flags["has_roe"] = roe.notna()
    flags["included_in_regression"] = valid_input
    flags["regression_sample_size"] = sample_size
    flags["regression_full_rank"] = full_rank
    flags["intercept"] = intercept
    flags["roe_coefficient"] = roe_coefficient
    flags["valid_for_factor"] = residual.notna()
    reason_codes: list[str] = []
    for security in securities:
        reasons: list[str] = []
        if pd.isna(pb.loc[security]):
            reasons.append("missing_pb")
        elif log_dependent and pb.loc[security] <= 0:
            reasons.append("nonpositive_pb")
        elif not log_dependent and pb.loc[security] <= 0:
            reasons.append("nonpositive_pb_retained")
        if pd.isna(roe.loc[security]):
            reasons.append("missing_roe")
        if valid_input.loc[security] and sample_size < 3:
            reasons.append("insufficient_regression_sample")
        if valid_input.loc[security] and sample_size >= 3 and not full_rank:
            reasons.append("constant_roe")
        reason_codes.append("|".join(reasons) if reasons else "ok")
    flags["reason_codes"] = reason_codes
    financials._cache[f"quality_flags:{factor_id}"] = flags
    financials._cache[f"raw_factor:{factor_id}"] = residual.copy()

    if factor_id == "pb_roe_reg_rg":
        return _rank_cross_section(residual, date)
    return residual


def _ocfa_residual_and_flags(
    financials: QuarterlyFinancials,
) -> tuple[pd.Series, pd.DataFrame]:
    """Latest residual from a 12-quarter time trend in fixed-asset turnover."""

    operating_cost_ttm = financials.ttm("operating_cost")
    fixed_assets = financials.raw("fixed_assets")
    ocfa = safe_divide(
        operating_cost_ttm,
        fixed_assets.where(fixed_assets.ne(0)),
    )
    output = _empty_report_series(financials.frame.index)
    flags = pd.DataFrame(index=financials.frame.index)
    flags["window_start_report_period"] = pd.NaT
    flags["window_end_report_period"] = pd.NaT
    flags["observed_quarter_count"] = 0
    flags["has_12_consecutive_quarters"] = False
    flags["complete_ocfa_window"] = False
    flags["has_nonpositive_fixed_assets"] = False
    flags["latest_ocfa"] = np.nan
    flags["fitted_latest_ocfa"] = np.nan
    flags["intercept"] = np.nan
    flags["time_coefficient"] = np.nan
    flags["valid_for_factor"] = False
    flags["reason_codes"] = "insufficient_history"

    for _, security_frame in financials.frame.groupby(level="security_id", sort=False):
        security_index = security_frame.index.sort_values()
        current_index = security_index[-1]
        window_index = security_index[-12:]
        observed = len(window_index)
        ordinals = financials.ordinal.loc[window_index].to_numpy(dtype=float)
        is_consecutive = (
            observed == 12 and np.array_equal(np.diff(ordinals), np.ones(11))
        )
        ocfa_values = ocfa.loc[window_index].to_numpy(dtype=float)
        fixed_asset_values = fixed_assets.loc[window_index].to_numpy(dtype=float)
        complete_ocfa = is_consecutive and np.isfinite(ocfa_values).all()
        finite_fixed_assets = fixed_asset_values[
            np.isfinite(fixed_asset_values)
        ]
        has_nonpositive_fixed_assets = bool(
            (finite_fixed_assets <= 0).any()
        )

        report_periods = window_index.get_level_values("report_period")
        flags.at[current_index, "window_start_report_period"] = report_periods.min()
        flags.at[current_index, "window_end_report_period"] = report_periods.max()
        flags.at[current_index, "observed_quarter_count"] = observed
        flags.at[current_index, "has_12_consecutive_quarters"] = is_consecutive
        flags.at[current_index, "complete_ocfa_window"] = complete_ocfa
        flags.at[
            current_index, "has_nonpositive_fixed_assets"
        ] = has_nonpositive_fixed_assets
        flags.at[current_index, "latest_ocfa"] = (
            float(ocfa_values[-1]) if np.isfinite(ocfa_values[-1]) else np.nan
        )

        reasons: list[str] = []
        if observed < 12:
            reasons.append("insufficient_history")
        if observed >= 2 and not np.array_equal(
            np.diff(ordinals), np.ones(observed - 1),
        ):
            reasons.append("nonconsecutive_quarters")
        if is_consecutive and not complete_ocfa:
            reasons.append("incomplete_ocfa_window")
        if has_nonpositive_fixed_assets:
            reasons.append("nonpositive_fixed_assets")

        if complete_ocfa:
            time = np.arange(1.0, 13.0)
            slope = _ols_slope_with_intercept(ocfa_values)
            intercept = float(ocfa_values.mean() - slope * time.mean())
            fitted_latest = float(intercept + slope * time[-1])
            residual = float(ocfa_values[-1] - fitted_latest)
            output.at[current_index] = residual
            flags.at[current_index, "fitted_latest_ocfa"] = fitted_latest
            flags.at[current_index, "intercept"] = intercept
            flags.at[current_index, "time_coefficient"] = slope
            flags.at[current_index, "valid_for_factor"] = True
        flags.at[current_index, "reason_codes"] = "|".join(reasons) if reasons else "ok"

    return output, flags


def _capex(financials: QuarterlyFinancials, mode: str) -> pd.Series:
    """Prefer the reported combined cash outflow; never add its components again.

    A missing combined observation may fall back to the strict sum of the two
    disjoint components. Neither a missing component nor a missing reporting
    period is interpreted as zero. Apply the same cumulative-to-quarter/TTM
    conversion to each alternative before selecting it.
    """
    reported = getattr(financials, mode)("capital_expenditure_cash_paid")
    fixed = getattr(financials, mode)("capex_fixed")
    intangible = getattr(financials, mode)("capex_intangible")
    return reported.combine_first(fixed + intangible)


def _free_cash_flow(financials: QuarterlyFinancials, mode: str) -> pd.Series:
    return getattr(financials, mode)("operating_cash_flow") - _capex(financials, mode)


def _ten_complete_fiscal_years_capex(financials: QuarterlyFinancials) -> pd.Series:
    """Match the frozen annual-history rule without requiring interim reports."""
    f = financials
    annual = pd.concat([
        f.raw("capital_expenditure_cash_paid").rename("capex"),
        f.raw("net_profit").rename("profit"),
    ], axis=1).loc[f.quarter.eq(4)]
    observations = f.context.financial_observations
    if not observations.empty:
        visible = _normalize_observations(observations)
        cutoff = f.calculation_date + pd.to_timedelta(f.context.pit_calculation_time) if f.context.pit_allow_release_day else f.calculation_date
        visible = visible[visible.effective_at.le(cutoff)]
        visible = visible[visible.field.isin(["capital_expenditure_cash_paid", "net_profit"])]
        # A disclosed null still counts as an available field and invalidates
        # the window; it must not silently select an older annual endpoint.
        seen = visible.groupby(["security_id", "report_period"]).field.nunique()
        annual = annual[seen.reindex(annual.index).eq(2)]
    else:
        annual = annual[annual.notna().all(axis=1)]
    output = _empty_report_series(f.frame.index)
    for security, group in annual.groupby(level="security_id", sort=False):
        dates = group.index.get_level_values("report_period")
        years = pd.Series(dates.year, index=group.index)
        sums = group.rolling(10, min_periods=10).sum()
        ratio = safe_divide(sums.capex, sums.profit)
        ratio = ratio.where(years.sub(years.shift(9)).eq(9))
        target = f.frame.index[f.frame.index.get_level_values("security_id") == security]
        positions = dates.searchsorted(target.get_level_values("report_period"), side="right") - 1
        valid = positions >= 0
        output.loc[target[valid]] = ratio.to_numpy()[positions[valid]]
    return output


def _enterprise_value(financials: QuarterlyFinancials) -> pd.Series:
    reported = financials.raw("enterprise_value")
    derived = (
        financials.daily("total_market_cap")
        + financials.raw("interest_bearing_debt").fillna(0.0)
        + financials.raw("minority_interest").fillna(0.0)
        + financials.raw("other_equity_instruments").fillna(0.0)
        - financials.raw("cash").fillna(0.0)
    )
    return reported.combine_first(derived)


def _average_stock(financials: QuarterlyFinancials, values: pd.Series, periods: int) -> pd.Series:
    return (values + financials.lag(values, periods)) / 2.0


def _return_on_stock(
    financials: QuarterlyFinancials,
    numerator: pd.Series,
    stock: pd.Series,
    periods: int,
) -> pd.Series:
    return safe_divide(numerator, _average_stock(financials, stock, periods))


def _return_on_stock_sum(
    financials: QuarterlyFinancials,
    numerator: pd.Series,
    stock: pd.Series,
    periods: int,
) -> pd.Series:
    return safe_divide(numerator, stock + financials.lag(stock, periods))


def _standardized_unexpected(financials: QuarterlyFinancials, values: pd.Series) -> pd.Series:
    increment = values - financials.lag(values, 4)
    prior_increment = financials.lag(increment, 1)
    mean = financials.mean(prior_increment, 8)
    standard_deviation = financials.std(prior_increment, 8)
    return safe_divide(increment - mean, standard_deviation.where(standard_deviation.ne(0)))


def _gross_profit_ttm(financials: QuarterlyFinancials) -> pd.Series:
    return financials.ttm("operating_revenue") - financials.ttm("operating_cost")


def _operating_expense_ttm(financials: QuarterlyFinancials) -> pd.Series:
    return (
        financials.ttm("management_expense")
        + financials.ttm("selling_expense")
        + financials.ttm("rd_expense")
        + financials.ttm("depreciation_amortization")
    )


def _strict_sum(series: list[pd.Series]) -> pd.Series:
    """Add required components without silently treating missing data as zero."""

    if not series:
        raise ValueError("at least one component is required")
    return pd.concat(series, axis=1).sum(axis=1, min_count=len(series))


def _sfli_components(
    financials: QuarterlyFinancials,
    parameters: dict[str, object],
) -> pd.DataFrame:
    """Build the quarterly inputs for the Zhong et al. funding-gap measure."""

    capex_direct = financials.single("capital_expenditure_cash_paid")
    capex_components = _strict_sum([
        financials.single("capex_fixed"),
        financials.single("capex_intangible"),
    ])
    capex = capex_direct.combine_first(capex_components)

    long_debt_mode = str(
        parameters.get("long_debt_mode", "component_change"),
    ).strip().lower()
    if long_debt_mode == "component_change":
        raw_components = parameters.get(
            "long_debt_components", ["long_term_borrowing", "bonds_payable"],
        )
        if not isinstance(raw_components, (list, tuple)) or not raw_components:
            raise ValueError("sfli: long_debt_components must be a non-empty list")
        long_debt_components = [str(value) for value in raw_components]
        unsupported = set(long_debt_components) - set(POINT_FIELDS)
        if unsupported:
            raise ValueError(
                f"sfli: unsupported long-debt components {sorted(unsupported)}",
            )
        long_debt = _strict_sum([
            financials.raw(name) for name in long_debt_components
        ])
        long_debt_increase = financials.diff(long_debt, 1)
    elif long_debt_mode == "zhong2016_reclassification":
        # The academic implementation treats debt reclassified as current as
        # part of current-period long-term borrowing: LT debt_t + due_1y_t - LT debt_(t-1).
        long_debt = _strict_sum([
            financials.raw("long_term_borrowing"),
            financials.raw("noncurrent_liabilities_due_within_one_year"),
        ])
        long_debt_increase = (
            long_debt - financials.lag(financials.raw("long_term_borrowing"), 1)
        )
    else:
        raise ValueError(
            "sfli: long_debt_mode must be component_change or "
            "zhong2016_reclassification",
        )

    raw_equity_components = parameters.get(
        "equity_financing_components", ["capital_reserve", "paid_in_capital"],
    )
    if not isinstance(raw_equity_components, (list, tuple)) or not raw_equity_components:
        raise ValueError("sfli: equity_financing_components must be a non-empty list")
    equity_components = [str(value) for value in raw_equity_components]
    if "total_shares" in equity_components:
        raise ValueError("sfli: equity financing requires monetary paid_in_capital, not a share count")
    unsupported_equity = set(equity_components) - set(POINT_FIELDS)
    if unsupported_equity:
        raise ValueError(
            f"sfli: unsupported equity-financing components {sorted(unsupported_equity)}",
        )
    equity_financing = _strict_sum([
        financials.raw(name) for name in equity_components
    ])
    equity_increase = financials.diff(equity_financing, 1)

    operating_cash_flow = financials.single("operating_cash_flow")
    disposal_cash_inflow = financials.single(
        "cash_received_disposal_long_term_assets",
    )
    total_assets = financials.raw("total_assets")
    funding_gap = capex - (
        long_debt_increase
        + equity_increase
        + operating_cash_flow
        + disposal_cash_inflow
    )
    sfli1 = safe_divide(funding_gap, total_assets.where(total_assets > 0))

    return pd.DataFrame({
        "capex_cash_paid": capex,
        "long_term_debt": long_debt,
        "long_term_debt_increase": long_debt_increase,
        "equity_financing": equity_financing,
        "equity_financing_increase": equity_increase,
        "operating_cash_flow": operating_cash_flow,
        "disposal_cash_inflow": disposal_cash_inflow,
        "total_assets": total_assets,
        "funding_gap": funding_gap,
        "sfli1": sfli1,
    }, index=financials.frame.index)


def _sfli_values_and_flags(
    financials: QuarterlyFinancials,
    factor_id: str,
    parameters: dict[str, object],
) -> tuple[pd.Series, pd.DataFrame]:
    """Calculate SFLI1 or its eight-quarter mean-to-volatility variant."""

    components = _sfli_components(financials, parameters)
    sfli1 = components["sfli1"]
    factor_value = sfli1.copy()
    flags = components.copy()
    flags["window_quarters"] = 1
    flags["window_observation_count"] = sfli1.notna().astype(int)
    flags["has_consecutive_window"] = sfli1.notna()
    flags["sfli1_window_mean"] = sfli1
    flags["sfli1_window_std"] = np.nan
    flags["std_ddof"] = np.nan

    if factor_id == "sfli2":
        window = int(parameters.get("window_quarters", 8))
        if window < 2:
            raise ValueError("sfli2: window_quarters must be at least 2")
        ddof = int(parameters.get("std_ddof", 1))
        if ddof < 0 or ddof >= window:
            raise ValueError("sfli2: std_ddof must satisfy 0 <= ddof < window")
        mean = financials.mean(sfli1, window)
        std = financials.rolling(
            sfli1,
            window,
            lambda values: float(np.std(values, ddof=ddof)),
        )
        observation_count = financials.rolling(
            sfli1.notna().astype(float), window, "sum", min_periods=1,
        )
        first_ordinal = financials.ordinal.groupby(
            level="security_id", sort=False,
        ).shift(window - 1)
        consecutive = (financials.ordinal - first_ordinal).eq(window - 1)
        factor_value = safe_divide(mean, std.where(std.ne(0))).where(consecutive)
        flags["window_quarters"] = window
        flags["window_observation_count"] = observation_count
        flags["has_consecutive_window"] = consecutive
        flags["sfli1_window_mean"] = mean
        flags["sfli1_window_std"] = std
        flags["std_ddof"] = ddof

    flags["factor_value"] = factor_value
    flags["valid_for_factor"] = factor_value.notna()
    # Only the newest visible report is exposed on this calculation date.
    # Keep quality rows at the same grain to avoid repeatedly materializing
    # reason strings for every historical quarter during daily PIT runs.
    latest_index = flags.groupby(level="security_id", sort=False).tail(1).index
    flags = flags.loc[latest_index].copy()
    reasons: list[str] = []
    component_columns = [
        "capex_cash_paid", "long_term_debt_increase",
        "equity_financing_increase", "operating_cash_flow",
        "disposal_cash_inflow", "total_assets",
    ]
    for index in flags.index:
        row_reasons: list[str] = []
        missing = [
            name for name in component_columns if pd.isna(flags.at[index, name])
        ]
        if missing:
            row_reasons.append("missing_components:" + ",".join(missing))
        if pd.notna(flags.at[index, "total_assets"]) and flags.at[index, "total_assets"] <= 0:
            row_reasons.append("nonpositive_total_assets")
        if factor_id == "sfli2":
            window = int(flags.at[index, "window_quarters"])
            if flags.at[index, "window_observation_count"] < window:
                row_reasons.append("insufficient_complete_sfli1_history")
            if not bool(flags.at[index, "has_consecutive_window"]):
                row_reasons.append("nonconsecutive_quarters")
            if flags.at[index, "window_observation_count"] >= window:
                window_std = flags.at[index, "sfli1_window_std"]
                if pd.notna(window_std) and window_std == 0:
                    row_reasons.append("zero_sfli1_window_std")
        reasons.append("|".join(row_reasons) if row_reasons else "ok")
    flags["reason_codes"] = reasons
    return factor_value, flags


def _net_operating_liability_features(
    financials: QuarterlyFinancials,
    parameters: dict[str, object],
) -> pd.DataFrame:
    """Build the report-period inputs for NOL1/NOL2 without future data."""

    if parameters.get("balance_total_fallback", "require_both_details") != "require_both_details":
        raise ValueError("NOL requires both details when the reported balance total is missing")
    # Match the historical NOL calculator: a reported total is authoritative;
    # otherwise both longstanding captions are required. One absent component
    # cannot be interpreted as a disclosed zero.
    payables = financials.raw("accounts_payable_combined").combine_first(
        financials.raw("notes_payable") + financials.raw("accounts_payable"),
    )
    advance_receipts = financials.raw("advance_receipts")
    contract_liabilities = financials.raw("contract_liabilities")
    # The two liability captions straddle the new revenue standard and can also
    # coexist.  Require at least one of them, then add every value actually
    # reported instead of forcing a missing legacy/new caption to zero globally.
    advance_and_contract_liabilities = _sum_available([
        advance_receipts, contract_liabilities,
    ])
    employee_compensation_payable = financials.raw(
        "employee_compensation_payable",
    )
    employee_scope = parameters.get("employee_compensation_scope", "current_only")
    if employee_scope == "current_and_long_term":
        # The source adapter must distinguish an explicitly unreported optional
        # caption (zero) from a missing record or a conflicting version (NaN).
        employee_compensation_payable = employee_compensation_payable + financials.raw(
            "long_term_employee_compensation_payable",
        )
    elif employee_scope != "current_only":
        raise ValueError("NOL employee_compensation_scope is not supported")
    operating_liabilities = _strict_sum([
        payables,
        advance_and_contract_liabilities,
        employee_compensation_payable,
    ])

    prepayments = financials.raw("prepayments")
    contract_assets = financials.raw("contract_assets")
    receivables = financials.raw("accounts_receivable_combined").combine_first(
        financials.raw("notes_receivable") + financials.raw("accounts_receivable"),
    )
    # Contract assets did not exist as a separate caption before the new
    # revenue standard.  It is an additive optional line; the two long-lived
    # base captions remain required.
    operating_assets = _strict_sum([prepayments, receivables]) + contract_assets.fillna(0.0)
    total_assets = financials.raw("total_assets")
    net_operating_liabilities = operating_liabilities - operating_assets
    nol = safe_divide(
        net_operating_liabilities,
        total_assets.where(total_assets > 0),
    )

    revenue_mode = str(
        parameters.get("revenue_mode", "single_quarter"),
    ).strip().lower()
    if revenue_mode == "single_quarter":
        revenue = financials.single("operating_revenue")
    elif revenue_mode == "cumulative":
        revenue = financials.raw("operating_revenue")
    else:
        raise ValueError(
            "nol2: revenue_mode must be 'single_quarter' or 'cumulative'",
        )
    revenue_lag_quarters = int(parameters.get("revenue_lag_quarters", 1))
    if revenue_lag_quarters < 1:
        raise ValueError("nol2: revenue_lag_quarters must be at least 1")
    lag_revenue = financials.exact_quarter_lag(revenue, revenue_lag_quarters)

    frame = pd.DataFrame(index=financials.frame.index)
    frame["accounts_and_notes_payable"] = payables
    frame["advance_receipts"] = advance_receipts
    frame["contract_liabilities"] = contract_liabilities
    frame["advance_and_contract_liabilities"] = advance_and_contract_liabilities
    frame["employee_compensation_payable"] = employee_compensation_payable
    frame["operating_liabilities"] = operating_liabilities
    frame["prepayments"] = prepayments
    frame["contract_assets"] = contract_assets
    frame["accounts_and_notes_receivable"] = receivables
    frame["operating_assets"] = operating_assets
    frame["total_assets"] = total_assets
    frame["net_operating_liabilities"] = net_operating_liabilities
    frame["nol"] = nol
    frame["operating_revenue"] = revenue
    frame["lag_operating_revenue"] = lag_revenue
    frame["revenue_scaled"] = safe_divide(revenue, total_assets.where(total_assets > 0))
    frame["lag_revenue_scaled"] = safe_divide(
        lag_revenue, total_assets.where(total_assets > 0),
    )
    frame["revenue_mode"] = revenue_mode
    frame["revenue_lag_quarters"] = revenue_lag_quarters
    frame["industry_l1"] = _categorical_daily_at_report(financials, "industry_l1")

    listing_dates = _security_listing_dates(financials.context)
    security_ids = frame.index.get_level_values("security_id").astype(str)
    mapped_listing_dates = pd.DatetimeIndex(security_ids.map(listing_dates))
    listing_age_days = pd.Series(
        (financials.calculation_date - mapped_listing_dates).days.astype(float),
        index=frame.index,
    ).where(lambda values: values >= 0)
    frame["list_date"] = mapped_listing_dates
    frame["listing_age_days"] = listing_age_days
    return frame.replace([np.inf, -np.inf], np.nan)


def _listed_a_share_model_universe(context: FactorContext, date: pd.Timestamp, min_listing_days: int) -> frozenset[str]:
    master = context.extras.get("security_master")
    if master is None or not {"security_id", "list_date"}.issubset(master.columns):
        raise ValueError("Listed-stock regression universe requires historical security master")
    codes = master.security_id.astype(str)
    listed = pd.to_datetime(master.list_date)
    delisted = pd.to_datetime(master.delist_date) if "delist_date" in master else pd.Series(pd.NaT, index=master.index)
    valid = (date - listed).dt.days.ge(min_listing_days) & (delisted.isna() | delisted.gt(date))
    # Synthetic test IDs are allowed; actual numeric securities must be SH/SZ A shares.
    numeric = codes.str.match(r"^\d{6}(?:\.[A-Z]+)?$")
    mainland = codes.str.match(r"^(?:[03]\d{5}(?:\.SZ)?|6\d{5}(?:\.SH)?)$")
    valid &= ~numeric | mainland
    return _historical_st_eligible(context, date, frozenset(codes[valid]))


def _report_regression_universe(context: FactorContext, date: pd.Timestamp, parameters: dict[str, object]):
    if parameters.get("model_universe_policy") == "historically_listed_a_shares_without_output_gates":
        return _listed_a_share_model_universe(context, date, int(parameters.get("min_listing_days", 365)))
    return _all_a_share_universe(context, date)


def _net_operating_liability_values_and_flags(
    financials: QuarterlyFinancials,
    factor_id: str,
    context: FactorContext,
    parameters: dict[str, object],
) -> tuple[pd.Series, pd.DataFrame]:
    """Calculate NOL1 or the industry-regression residual NOL2."""

    features = _net_operating_liability_features(financials, parameters)
    min_listing_days = int(parameters.get("min_listing_days", 365))
    if min_listing_days < 0:
        raise ValueError(f"{factor_id}: min_listing_days must be non-negative")
    raw_excluded = parameters.get(
        "excluded_citic_l1", list(DEFAULT_NOL_EXCLUDED_CITIC_L1),
    )
    if not isinstance(raw_excluded, (list, tuple, set, frozenset)):
        raise ValueError(f"{factor_id}: excluded_citic_l1 must be a list-like value")
    excluded_industries = frozenset(str(value).strip() for value in raw_excluded)

    date = financials.calculation_date
    allowed = _listed_a_share_model_universe(context, date, min_listing_days) if parameters.get("model_universe_policy") == "listed_A_shares_before_trading_gate" else _all_a_share_universe(context, date)
    security_ids = features.index.get_level_values("security_id").astype(str)
    eligible_a_share = pd.Series(True, index=features.index)
    if allowed is not None:
        eligible_a_share &= security_ids.isin(allowed)
    listing_age_eligible = features["listing_age_days"].ge(min_listing_days)
    industry_available = features["industry_l1"].notna()
    industry_excluded = features["industry_l1"].astype("string").isin(
        excluded_industries,
    )
    eligible = (
        eligible_a_share
        & listing_age_eligible
        & industry_available
        & ~industry_excluded
    )

    raw_residual = _empty_report_series(features.index)
    fitted_nol = _empty_report_series(features.index)
    sample_size = pd.Series(0, index=features.index, dtype=int)
    required_sample_size = pd.Series(0, index=features.index, dtype=int)
    full_rank = pd.Series(False, index=features.index, dtype=bool)
    intercept = _empty_report_series(features.index)
    current_revenue_coefficient = _empty_report_series(features.index)
    lag_revenue_coefficient = _empty_report_series(features.index)
    residual_std = _empty_report_series(features.index)

    if factor_id == "nol":
        factor_value = features["nol"].where(eligible)
    else:
        min_samples = int(parameters.get("min_regression_samples", 5))
        if min_samples < 4:
            raise ValueError("nol2: min_regression_samples must be at least 4")
        group_keys = pd.DataFrame({
            "_report_period": features.index.get_level_values("report_period"),
            "industry_l1": features["industry_l1"].astype("string"),
        }, index=features.index)
        groups = group_keys.loc[eligible].groupby(
            ["_report_period", "industry_l1"], sort=False,
        ).groups
        model_columns = ["nol", "revenue_scaled", "lag_revenue_scaled"]
        for _, raw_group_index in groups.items():
            group_index = pd.MultiIndex.from_tuples(
                raw_group_index, names=features.index.names,
            )
            model = features.loc[group_index, model_columns].dropna()
            required = max(min_samples, 4)
            sample_size.loc[group_index] = len(model)
            required_sample_size.loc[group_index] = required
            if len(model) < required:
                continue
            x = model[["revenue_scaled", "lag_revenue_scaled"]].to_numpy(dtype=float)
            y = model["nol"].to_numpy(dtype=float)
            design = np.column_stack([np.ones(len(model)), x])
            coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
            errors = y - design @ coefficients
            rank_is_full = bool(np.linalg.matrix_rank(design) == design.shape[1])
            raw_residual.loc[model.index] = errors
            fitted_nol.loc[model.index] = y - errors
            full_rank.loc[group_index] = rank_is_full
            intercept.loc[group_index] = float(coefficients[0])
            current_revenue_coefficient.loc[group_index] = float(coefficients[1])
            lag_revenue_coefficient.loc[group_index] = float(coefficients[2])

        residual_standardization = str(
            parameters.get("residual_standardization", "citic_l1_zscore"),
        ).strip().lower()
        if residual_standardization == "citic_l1_zscore":
            residual_ddof = int(parameters.get("residual_std_ddof", 1))
            if residual_ddof not in {0, 1}:
                raise ValueError("nol2: residual_std_ddof must be 0 or 1")
            residual_frame = features[["industry_l1"]].copy()
            residual_frame["raw_residual"] = raw_residual
            factor_value = _zscore_columns_by_report_industry(
                residual_frame,
                ["raw_residual"],
                eligible & raw_residual.notna(),
                ddof=residual_ddof,
            )["raw_residual"]
            keys = pd.DataFrame({
                "_report_period": features.index.get_level_values("report_period"),
                "industry_l1": features["industry_l1"].astype("string"),
            }, index=features.index)
            for _, raw_group_index in keys.loc[
                eligible & raw_residual.notna()
            ].groupby(["_report_period", "industry_l1"], sort=False).groups.items():
                group_index = pd.MultiIndex.from_tuples(
                    raw_group_index, names=features.index.names,
                )
                group_std = float(raw_residual.loc[group_index].std(ddof=residual_ddof))
                residual_std.loc[group_index] = group_std
        elif residual_standardization == "none":
            residual_ddof = np.nan
            factor_value = raw_residual.copy()
        else:
            raise ValueError(
                "nol2: residual_standardization must be "
                "'citic_l1_zscore' or 'none'",
            )

    flags = features.copy()
    flags["source_report_period"] = features.index.get_level_values("report_period")
    flags["eligible_a_share"] = eligible_a_share
    flags["listing_age_eligible"] = listing_age_eligible
    flags["industry_excluded"] = industry_excluded
    flags["included_in_factor_sample"] = eligible
    flags["min_listing_days"] = min_listing_days
    flags["regression_sample_size"] = sample_size
    flags["required_regression_sample_size"] = required_sample_size
    flags["regression_full_rank"] = full_rank
    flags["intercept"] = intercept
    flags["current_revenue_coefficient"] = current_revenue_coefficient
    flags["lag_revenue_coefficient"] = lag_revenue_coefficient
    flags["fitted_nol"] = fitted_nol
    flags["raw_residual"] = raw_residual
    flags["residual_industry_std"] = residual_std
    flags["factor_value"] = factor_value
    flags["valid_for_factor"] = factor_value.notna()
    flags["warning_codes"] = np.where(
        (factor_id == "nol2") & raw_residual.notna() & ~full_rank,
        "rank_deficient_minimum_norm_ols",
        "",
    )

    required_components = [
        "accounts_and_notes_payable",
        "advance_and_contract_liabilities",
        "employee_compensation_payable",
        "prepayments",
        "accounts_and_notes_receivable",
        "total_assets",
    ]
    reasons: list[str] = []
    for index in flags.index:
        row_reasons: list[str] = []
        if not bool(flags.at[index, "eligible_a_share"]):
            row_reasons.append("not_in_eligible_a_share_universe")
        if pd.isna(flags.at[index, "list_date"]):
            row_reasons.append("missing_list_date")
        elif not bool(flags.at[index, "listing_age_eligible"]):
            row_reasons.append(f"listed_less_than_{min_listing_days}_days")
        if pd.isna(flags.at[index, "industry_l1"]):
            row_reasons.append("missing_citic_industry_l1")
        elif bool(flags.at[index, "industry_excluded"]):
            row_reasons.append("excluded_citic_industry_l1")
        missing_components = [
            name for name in required_components if pd.isna(flags.at[index, name])
        ]
        if missing_components:
            row_reasons.append("missing_components:" + ",".join(missing_components))
        if pd.notna(flags.at[index, "total_assets"]) and flags.at[index, "total_assets"] <= 0:
            row_reasons.append("nonpositive_total_assets")
        if factor_id == "nol2":
            missing_regressors = [
                name for name in ("revenue_scaled", "lag_revenue_scaled")
                if pd.isna(flags.at[index, name])
            ]
            if missing_regressors:
                row_reasons.append("missing_regressors:" + ",".join(missing_regressors))
            required = int(flags.at[index, "required_regression_sample_size"])
            if (
                not row_reasons
                and required > 0
                and int(flags.at[index, "regression_sample_size"]) < required
            ):
                row_reasons.append("insufficient_regression_sample")
            if not row_reasons and raw_residual.notna().loc[index] and pd.isna(factor_value.loc[index]):
                row_reasons.append("invalid_residual_standardization")
        reasons.append("|".join(row_reasons) if row_reasons else "ok")
    flags["reason_codes"] = reasons
    return factor_value, flags


def _raw_formula(financials: QuarterlyFinancials, factor_id: str) -> pd.Series:
    f = financials
    equity = _equity(f)
    total_assets = f.raw("total_assets")
    market_cap = f.daily("total_market_cap")
    enterprise_value = _enterprise_value(f)

    if factor_id == "np_qua_chg_acc":
        return f.rolling(
            f.single("net_profit_parent_excl_nr"), 8, _quadratic_trend_coefficient,
        )
    if factor_id == "SalesG":
        return f.signed_growth(f.ttm("operating_revenue"), 4)
    if factor_id == "ProfitG":
        return f.signed_growth(f.ttm("net_profit_parent"), 4)
    if factor_id == "CFOG":
        return f.signed_growth(f.ttm("operating_cash_flow"), 4)
    if factor_id == "np_yoy":
        return f.signed_growth(f.raw("net_profit_parent"), 4)
    if factor_id == "sale_yoy":
        revenue = f.raw("sales_revenue").combine_first(f.raw("operating_revenue"))
        return f.signed_growth(revenue, 4)
    if factor_id == "cum_rd_exp":
        accumulated_rd = f.cumulative_sum(f.single("rd_expense"))
        return safe_divide(accumulated_rd, total_assets)
    if factor_id == "fcff_yoy":
        return f.signed_growth(_free_cash_flow(f, "raw"), 4)
    if factor_id == "fcff_cv":
        free_cash_flow = _free_cash_flow(f, "single")
        return safe_divide(f.mean(free_cash_flow, 8), f.std(free_cash_flow, 8))
    if factor_id == "fcffG":
        return f.signed_growth(_free_cash_flow(f, "ttm"), 4)
    if factor_id == "net_wc":
        return _net_working_capital(f)
    if factor_id in {"re_qoq", "re_yoy"}:
        retained_earnings = _sum_available([
            f.raw("undistributed_profit"), f.raw("surplus_reserve"),
        ])
        return f.signed_growth(retained_earnings, 1 if factor_id == "re_qoq" else 4)
    if factor_id == "sug":
        return _standardized_unexpected(f, f.single("net_profit_parent_excl_nr"))
    if factor_id == "sue":
        return _standardized_unexpected(f, f.single("net_profit_parent_excl_nr"))
    if factor_id == "sur":
        return _standardized_unexpected(f, f.single("operating_revenue"))
    if factor_id == "sucfo":
        return _standardized_unexpected(f, f.single("operating_cash_flow"))
    if factor_id == "sur_cf":
        return _standardized_unexpected(f, f.single("cash_received_sales"))
    if factor_id == "factor_eaa":
        growth = f.signed_growth(f.single("eps"), 4)
        return f.diff(growth, 1)
    if factor_id == "factor_eap":
        increment = f.single("eps") - f.lag(f.single("eps"), 4)
        scaled = safe_divide(increment, f.lag(f.daily_at_report("raw_close"), 1))
        return f.diff(scaled, 1)
    if factor_id in {"factor_eav", "factor_egv"}:
        eps = f.single("eps")
        increment = eps - f.lag(eps, 4)
        scaled = safe_divide(increment, f.std(eps, 8).where(f.std(eps, 8).ne(0)))
        return f.diff(scaled, 1) if factor_id == "factor_eav" else scaled
    if factor_id in {"epttm_rg_zscore", "epttm_rg", "EPTTM_3y_quantile"}:
        return safe_divide(f.ttm("net_profit_parent"), market_cap)
    if factor_id == "bp_rg":
        return _book_to_price(f)
    if factor_id in {"pb_rg", "PB_3y_quantile"}:
        return _price_to_book(f)
    if factor_id == "cp_rg":
        return safe_divide(f.raw("cash_end"), market_cap)
    if factor_id == "pc_rg":
        return safe_divide(market_cap, f.raw("cash_end"))
    if factor_id == "delta_roeq":
        roe = _return_on_stock(f, f.single("net_profit_parent"), equity, 1)
        return f.diff(roe, 4)
    if factor_id == "delta_roaq":
        roa = _return_on_stock(f, f.single("net_profit_parent"), total_assets, 1)
        return f.diff(roa, 4)
    if factor_id == "OPR_Skew":
        margin = safe_divide(f.ttm("operating_profit"), f.ttm("operating_revenue"))
        return f.skew(margin, 8)
    if factor_id == "npyoy":
        return f.signed_growth(f.raw("net_profit_parent"), 4)
    if factor_id == "npqyoy":
        return f.signed_growth(f.single("net_profit_parent"), 4)
    if factor_id == "npttm_qoq":
        return f.signed_growth(f.ttm("net_profit_parent"), 1)
    if factor_id == "capex_da_ratio_ttm":
        return safe_divide(_capex(f, "ttm"), f.ttm("depreciation_amortization"))
    if factor_id in {"revq_yoy_diff4", "revq_yoy_diff1"}:
        growth = f.signed_growth(f.single("operating_revenue"), 4)
        return f.diff(growth, 4 if factor_id.endswith("diff4") else 1)
    if factor_id in {"npq_yoy_diff4", "npq_yoy_diff1"}:
        growth = f.signed_growth(f.single("net_profit_parent"), 4)
        return f.diff(growth, 4 if factor_id.endswith("diff4") else 1)
    if factor_id in {"gpmq_diff4", "gpmq_diff1"}:
        revenue = f.single("operating_revenue")
        margin = safe_divide(revenue - f.single("operating_cost"), revenue)
        return f.diff(margin, 4 if factor_id.endswith("diff4") else 1)
    if factor_id in {"npm_diff4", "npm_diff1"}:
        margin = safe_divide(f.single("net_profit_parent"), f.single("operating_revenue"))
        return f.diff(margin, 4 if factor_id.endswith("diff4") else 1)
    if factor_id == "cfo2rev_ttm_diff":
        ratio = safe_divide(f.ttm("operating_cash_flow"), f.ttm("operating_revenue"))
        return f.diff(ratio, 1)
    if factor_id == "npttm_yoy":
        return f.signed_growth(f.ttm("net_profit_parent"), 4)
    if factor_id == "salettm_yoy":
        return f.signed_growth(f.ttm("sales_revenue").combine_first(f.ttm("operating_revenue")), 4)
    if factor_id == "dol_ttm2":
        fixed_cost = (
            f.ttm("operating_cost") + f.ttm("management_expense")
            + f.ttm("selling_expense") + f.ttm("rd_expense")
        )
        return safe_divide(fixed_cost, _average_stock(f, total_assets, 4))
    if factor_id in {"POP_Skew", "Rel_POP_Skew"}:
        period_market_cap = f.daily_at_report("total_market_cap")
        pop = safe_divide(f.ttm("operating_profit"), period_market_cap)
        return f.skew(pop, 8)
    if factor_id in {"divp_ttm_rg", "pdiv_ttm_rg"}:
        yield_value = safe_divide(f.ttm("cash_dividend"), market_cap)
        return safe_divide(pd.Series(1.0, index=yield_value.index), yield_value) if factor_id.startswith("pdiv") else yield_value
    if factor_id == "ebitq2ev":
        return safe_divide(f.single("ebit"), enterprise_value)
    if factor_id in {"epq", "epq_rg"}:
        return safe_divide(f.single("net_profit_parent"), market_cap)
    if factor_id == "peq_rg":
        return safe_divide(market_cap, f.single("net_profit_parent"))
    if factor_id == "pettm_rg":
        return safe_divide(market_cap, f.ttm("net_profit_parent"))
    if factor_id == "ps_ttm_rg":
        return safe_divide(market_cap, f.ttm("operating_revenue"))
    if factor_id == "sp_ttm_rg":
        return safe_divide(f.ttm("operating_revenue"), market_cap)
    if factor_id == "pcf_ttm_rg":
        return safe_divide(market_cap, f.ttm("operating_cash_flow"))
    if factor_id in {"cfp_ttm_rg", "CFPTTM_3y_quantile"}:
        return safe_divide(f.ttm("operating_cash_flow"), market_cap)
    if factor_id == "spq_rg":
        return safe_divide(f.single("operating_revenue"), market_cap)
    if factor_id == "psq_rg":
        return safe_divide(market_cap, f.single("operating_revenue"))
    if factor_id == "cfpq_rg":
        return safe_divide(f.single("operating_cash_flow"), market_cap)
    if factor_id == "pcfq_rg":
        return safe_divide(market_cap, f.single("operating_cash_flow"))
    if factor_id == "fcffp_ttm":
        return safe_divide(_free_cash_flow(f, "ttm"), market_cap)
    if factor_id == "cfo_ttm_yoy":
        return f.signed_growth(f.ttm("operating_cash_flow"), 4)
    if factor_id == "cfo_yoy":
        return f.signed_growth(f.raw("operating_cash_flow"), 4)
    if factor_id == "fcff_ttm":
        return _free_cash_flow(f, "ttm")
    if factor_id == "fcff_ratio":
        return safe_divide(_free_cash_flow(f, "ttm"), enterprise_value)
    if factor_id == "fcff_ratio_qua":
        return safe_divide(_free_cash_flow(f, "single"), enterprise_value)

    if factor_id.startswith("fcffoe_"):
        mode = "ttm" if "ttm" in factor_id else "single"
        base = _return_on_stock_sum(f, _free_cash_flow(f, mode), equity, 4 if mode == "ttm" else 1)
        if mode == "single":
            base = base.where(~f.quarter.eq(4))
        if factor_id.endswith("diff4") or factor_id.endswith("delta"):
            return f.diff(base, 4)
        if factor_id.endswith("diff1"):
            return f.diff(base, 1)
        return base
    if factor_id.startswith("cfooe_"):
        mode = "ttm" if "ttm" in factor_id else "single"
        cash_flow = getattr(f, mode)("operating_cash_flow")
        base = _return_on_stock_sum(f, cash_flow, equity, 4 if mode == "ttm" else 1)
        if mode == "single":
            base = base.where(~f.quarter.eq(4))
        if factor_id.endswith("diff4") or factor_id.endswith("delta"):
            return f.diff(base, 4)
        if factor_id.endswith("diff1"):
            return f.diff(base, 1)
        return base
    if factor_id.startswith("cfoe_"):
        mode = "ttm" if "ttm" in factor_id else "single"
        cash_flow = getattr(f, mode)("cash_net_increase")
        base = _return_on_stock_sum(f, cash_flow, equity, 4 if mode == "ttm" else 1)
        if mode == "single":
            base = base.where(~f.quarter.eq(4))
        if factor_id.endswith("diff4") or factor_id.endswith("delta"):
            return f.diff(base, 4)
        if factor_id.endswith("diff1"):
            return f.diff(base, 1)
        return base

    if factor_id == "roe_ttm_rg":
        return _return_on_stock(f, f.ttm("net_profit_parent"), equity, 4)
    if factor_id == "roeq_rg":
        return _return_on_stock(f, f.single("net_profit_parent"), equity, 1)
    if factor_id == "eyoev_ttm":
        return safe_divide(f.ttm("ebit"), enterprise_value)
    if factor_id == "net_margin_ttm":
        return safe_divide(f.ttm("net_profit_parent"), f.ttm("operating_revenue"))
    if factor_id == "sales_ttm_per_share":
        shares = f.raw("total_shares").combine_first(f.daily("total_shares"))
        return safe_divide(f.ttm("operating_revenue"), shares)
    if factor_id == "NET_PROFIT_INCL_MIN_INT_INC_TTM":
        return f.ttm("net_profit")
    if factor_id == "NET_PROFIT_EXCL_MIN_INT_INC_TTM":
        return f.ttm("net_profit_parent")
    if factor_id == "NET_PROFIT_AFTER_DED_NR_LP_TTM":
        return f.ttm("net_profit_parent_excl_nr")
    if factor_id == "NET_PROFIT_INCL_MIN_INT_INC_QUA":
        return f.single("net_profit")
    if factor_id == "NET_PROFIT_EXCL_MIN_INT_INC_QUA":
        return f.single("net_profit_parent")
    if factor_id == "NET_PROFIT_AFTER_DED_NR_LP_QUA":
        return f.single("net_profit_parent_excl_nr")
    if factor_id == "roic_ttm":
        invested_capital = (
            f.raw("inventory") + f.raw("fixed_assets") + _receivables(f)
            + f.raw("other_receivables") - f.raw("non_interest_current_liabilities")
        )
        return safe_divide(f.ttm("ebit"), invested_capital)
    if factor_id == "ROE":
        beginning_equity = _empty_report_series(f.frame.index)
        for quarter in range(1, 5):
            mask = f.quarter.eq(quarter)
            beginning_equity.loc[mask] = f.lag(equity, quarter).loc[mask]
        return safe_divide(2.0 * f.raw("net_profit_parent"), equity + beginning_equity)
    if factor_id in {"assets_turnoverTTM_chg", "assets_turnover_ttm"}:
        turnover = safe_divide(f.ttm("operating_revenue"), _average_stock(f, total_assets, 4))
        return f.diff(turnover, 1) if factor_id.endswith("chg") else turnover
    if factor_id == "profit_quality":
        quality = f.ttm("operating_cash_flow") - f.ttm("operating_profit")
        return safe_divide(quality, total_assets)
    if factor_id == "acc_profit_qua":
        return f.single("net_profit") - f.single("operating_cash_flow")
    if factor_id in {"cfo_cv", "cfo_cv_pit"}:
        # Source workbook E530: PIT mean/std of eight single-quarter CFOs.
        # Both IDs use the same PIT observations; equal values are intentional.
        cash_flow = f.single("operating_cash_flow")
        result = safe_divide(f.mean(cash_flow, 8), f.std(cash_flow, 8))
        first_ordinal = f.ordinal.groupby(level="security_id", sort=False).shift(7)
        return result.where((f.ordinal - first_ordinal).eq(7))
    if factor_id == "opex_ttm":
        return _operating_expense_ttm(f)
    if factor_id == "da2rev_ttm":
        return safe_divide(f.ttm("depreciation_amortization"), f.ttm("operating_revenue"))
    if factor_id in {"ar2or_ttm", "ar2or_ttm_diff1"}:
        numerator = _sum_available([
            f.raw("advance_receipts"), f.raw("contract_liabilities"),
        ])
        revenue = f.ttm("operating_revenue")
        ratio = safe_divide(numerator, revenue.where(revenue > 0))
        return f.diff(ratio, 1) if factor_id.endswith("diff1") else ratio
    if factor_id == "inventory_turnover_ttm":
        return safe_divide(f.ttm("operating_cost"), _average_stock(f, f.raw("inventory"), 4))
    if factor_id == "receivable_turnover_ttm":
        return safe_divide(f.ttm("operating_revenue"), _average_stock(f, _receivables(f), 4))
    if factor_id == "opex2ta_ttm":
        return safe_divide(_operating_expense_ttm(f), total_assets)
    if factor_id == "EPS_Delta_Vol":
        return f.std(f.signed_growth(f.single("eps"), 4), 8)
    if factor_id == "da2gross_ttm":
        return safe_divide(f.ttm("depreciation_amortization"), _gross_profit_ttm(f))
    if factor_id == "capex2np_10y":
        return _ten_complete_fiscal_years_capex(f)
    raise KeyError(f"unsupported financial-statement factor: {factor_id}")


def _allowed_universe(context: FactorContext, date: pd.Timestamp) -> frozenset[str] | None:
    if date in context.universe:
        return _historical_st_eligible(context, date, context.universe[date])
    if date in context.eligible_universe:
        return _historical_st_eligible(context, date, context.eligible_universe[date])
    return _historical_st_eligible(context, date, None)


def _filter_universe(values: pd.Series, context: FactorContext, date: pd.Timestamp) -> pd.Series:
    allowed = _allowed_universe(context, date)
    return values if allowed is None else values[values.index.isin(allowed)]


def _filter_cross_section_population(values: pd.Series, context: FactorContext, date: pd.Timestamp) -> pd.Series:
    """Rank and standardize against all eligible A shares before output selection."""
    allowed = _all_a_share_universe(context, date)
    return values if allowed is None else values[values.index.isin(allowed)]


def _industry_series(financials: QuarterlyFinancials) -> pd.Series:
    daily = financials.daily_security("industry")
    if not daily.empty:
        return daily.astype("string")
    source = financials.context.extras.get("industry")
    if source is None or source.empty:
        return pd.Series(dtype="string")
    frame = source.copy()
    if isinstance(frame.index, pd.MultiIndex) and list(frame.index.names) == INDEX_NAMES:
        frame = frame.reset_index()
    if "date" in frame:
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        frame = frame[frame["date"] <= financials.calculation_date]
        frame = frame.sort_values("date").drop_duplicates("security_id", keep="last")
    column = next((name for name in DAILY_ALIASES["industry"] if name in frame), None)
    if column is None:
        return pd.Series(dtype="string")
    return frame.set_index(frame["security_id"].astype(str))[column].astype("string")


def _categorical_daily_at_report(
    financials: QuarterlyFinancials,
    name: str,
) -> pd.Series:
    """Map a categorical daily field to each report date without using future labels."""

    output = pd.Series(pd.NA, index=financials.frame.index, dtype="object")
    if financials.context.daily.empty:
        return output
    aliases = DAILY_ALIASES[name]
    column = next((alias for alias in aliases if alias in financials.context.daily), None)
    if column is None:
        return output
    report_periods = financials.frame.index.get_level_values("report_period")
    for security in financials.frame.index.get_level_values("security_id").unique():
        try:
            history = financials.context.daily.xs(security, level="security_id")[column]
        except KeyError:
            continue
        history = history.sort_index()
        rows = financials.frame.index.get_level_values("security_id") == security
        targets = report_periods[rows]
        positions = history.index.searchsorted(targets, side="right") - 1
        valid = positions >= 0
        values = np.full(len(targets), None, dtype=object)
        values[valid] = history.iloc[positions[valid]].astype(object).to_numpy()
        output.loc[rows] = values
    return output


def _security_listing_dates(context: FactorContext) -> pd.Series:
    master = context.extras.get("security_master")
    if master is None or master.empty or not {"security_id", "list_date"}.issubset(master.columns):
        return pd.Series(dtype="datetime64[ns]")
    frame = master[["security_id", "list_date"]].copy()
    frame["security_id"] = frame["security_id"].astype(str)
    frame["list_date"] = pd.to_datetime(frame["list_date"], errors="coerce").dt.normalize()
    return frame.drop_duplicates("security_id", keep="last").set_index("security_id")["list_date"]


IDIOSYNCRATIC_PROFIT_VARIABLES = (
    "roa",
    "revenue_yoy",
    "leverage",
    "expense_rate",
    "equity_growth",
    "pb",
    "working_capital",
    "gross_margin",
)
IDIOSYNCRATIC_PROFIT_REGRESSORS = IDIOSYNCRATIC_PROFIT_VARIABLES[1:]


def _idiosyncratic_profit_features(
    financials: QuarterlyFinancials,
    parameters: dict[str, object],
) -> pd.DataFrame:
    revenue = financials.single("operating_revenue")
    total_assets = financials.raw("total_assets")
    equity = _equity(financials)
    expense_fields = ["selling_expense", "management_expense", "financial_expense"]
    if bool(parameters.get("expense_include_rd", False)):
        expense_fields.append("rd_expense")
    expenses = sum(
        (financials.single(name) for name in expense_fields),
        start=pd.Series(0.0, index=financials.frame.index),
    )
    working_capital = financials.raw("net_working_capital").combine_first(
        _strict_net_working_capital(financials)
        if parameters.get("strict_working_capital_components", False)
        else _net_working_capital(financials),
    )
    pb_source = str(parameters.get("pb_source", "database")).strip().lower()
    if pb_source == "database":
        pb = financials.daily_at_report("price_to_book")
    elif pb_source == "manual":
        pb = safe_divide(financials.daily_at_report("total_market_cap"), equity)
    else:
        raise ValueError(
            f"idiosyncratic profitability: unsupported pb_source {pb_source!r}",
        )

    frame = pd.DataFrame(index=financials.frame.index)
    frame["roa"] = safe_divide(
        2.0 * financials.single("net_profit"),
        total_assets + financials.lag(total_assets, 1),
    )
    frame["revenue_yoy"] = financials.signed_growth(revenue, 4)
    frame["leverage"] = safe_divide(financials.raw("total_liabilities"), total_assets)
    frame["expense_rate"] = safe_divide(expenses, revenue)
    frame["equity_growth"] = financials.signed_growth(equity, 4)
    frame["pb"] = pb
    frame["working_capital"] = working_capital
    frame["gross_margin"] = safe_divide(
        revenue - financials.single("operating_cost"), revenue,
    )
    frame["industry_l1"] = _categorical_daily_at_report(financials, "industry_l1")
    frame["industry_l2"] = _categorical_daily_at_report(financials, "industry_l2")

    listing_dates = _security_listing_dates(financials.context)
    securities = frame.index.get_level_values("security_id")
    report_periods = frame.index.get_level_values("report_period")
    mapped_listing_dates = pd.DatetimeIndex(securities.map(listing_dates))
    frame["list_date"] = mapped_listing_dates
    frame["listing_age_days"] = (report_periods - mapped_listing_dates).days.astype(float)
    return frame.replace([np.inf, -np.inf], np.nan)


def _standardize_within_report_industry(
    features: pd.DataFrame,
    eligible: pd.Series,
) -> pd.DataFrame:
    standardized = pd.DataFrame(
        np.nan, index=features.index, columns=IDIOSYNCRATIC_PROFIT_VARIABLES,
    )
    report_periods = features.index.get_level_values("report_period")
    keys = pd.DataFrame({
        "_report_period": report_periods,
        "industry_l1": features["industry_l1"].astype("string"),
    }, index=features.index)
    valid_groups = eligible & keys["industry_l1"].notna()
    for _, group_index in keys.loc[valid_groups].groupby(
        ["_report_period", "industry_l1"], sort=False,
    ).groups.items():
        group_index = pd.MultiIndex.from_tuples(group_index, names=features.index.names)
        for column in IDIOSYNCRATIC_PROFIT_VARIABLES:
            values = features.loc[group_index, column].astype(float)
            finite = values.notna()
            if not finite.any():
                continue
            mean = float(values.loc[finite].mean())
            std = float(values.loc[finite].std(ddof=0))
            standardized.loc[values.index[finite], column] = (
                0.0 if std == 0 else (values.loc[finite] - mean) / std
            )
    return standardized


def _ridge_residual(
    dependent: np.ndarray,
    regressors: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    design = np.column_stack([np.ones(len(dependent)), regressors])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    system = design.T @ design + penalty
    target = design.T @ dependent
    try:
        coefficients = np.linalg.solve(system, target)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.pinv(system) @ target
    return dependent - design @ coefficients, coefficients


def _idiosyncratic_profit_residual_and_flags(
    financials: QuarterlyFinancials,
    factor_id: str,
    context: FactorContext,
    parameters: dict[str, object],
) -> tuple[pd.Series, pd.DataFrame]:
    features = _idiosyncratic_profit_features(financials, parameters)
    alpha = float(parameters.get("ridge_alpha", 1.0))
    if alpha < 0:
        raise ValueError(f"{factor_id}: ridge_alpha must be non-negative")
    min_samples = int(parameters.get("min_regression_samples", 5))
    if min_samples < 3:
        raise ValueError(f"{factor_id}: min_regression_samples must be at least 3")
    min_listing_days = int(parameters.get("min_listing_days", 365))

    date = financials.calculation_date
    allowed = _report_regression_universe(context, date, parameters)
    security_ids = features.index.get_level_values("security_id").astype(str)
    eligible = pd.Series(True, index=features.index)
    if allowed is not None:
        eligible &= security_ids.isin(allowed)
    eligible &= features["listing_age_days"].ge(min_listing_days)
    standardized = _standardize_within_report_industry(features, eligible)
    residual = _empty_report_series(features.index)
    fitted = _empty_report_series(features.index)
    sample_size = pd.Series(0, index=features.index, dtype=int)

    if factor_id == "res_roa":
        group_keys = pd.DataFrame({
            "_report_period": features.index.get_level_values("report_period"),
            "industry_l1": features["industry_l1"].astype("string"),
        }, index=features.index)
        group_eligible = eligible & group_keys["industry_l1"].notna()
        groups = group_keys.loc[group_eligible].groupby(
            ["_report_period", "industry_l1"], sort=False,
        ).groups
        for _, raw_group_index in groups.items():
            group_index = pd.MultiIndex.from_tuples(raw_group_index, names=features.index.names)
            model = standardized.loc[
                group_index, ["roa", *IDIOSYNCRATIC_PROFIT_REGRESSORS]
            ].dropna()
            sample_size.loc[group_index] = len(model)
            if len(model) < min_samples:
                continue
            errors, _ = _ridge_residual(
                model["roa"].to_numpy(dtype=float),
                model[list(IDIOSYNCRATIC_PROFIT_REGRESSORS)].to_numpy(dtype=float),
                alpha,
            )
            residual.loc[model.index] = errors
            fitted.loc[model.index] = model["roa"].to_numpy(dtype=float) - errors
    else:
        report_periods = features.index.get_level_values("report_period")
        report_keys = pd.DataFrame(
            {"_report_period": report_periods}, index=features.index,
        )
        for _, raw_group_index in report_keys.loc[eligible].groupby(
            "_report_period", sort=False,
        ).groups.items():
            group_index = pd.MultiIndex.from_tuples(raw_group_index, names=features.index.names)
            base = standardized.loc[
                group_index, ["roa", *IDIOSYNCRATIC_PROFIT_REGRESSORS]
            ]
            industry_l2 = features.loc[group_index, "industry_l2"].astype("string")
            model = pd.concat([base, industry_l2.rename("industry_l2")], axis=1).dropna()
            sample_size.loc[group_index] = len(model)
            if len(model) < min_samples:
                continue
            dummies = pd.get_dummies(
                model["industry_l2"], prefix="industry_l2", drop_first=True, dtype=float,
            )
            x = pd.concat([
                model[list(IDIOSYNCRATIC_PROFIT_REGRESSORS)].astype(float), dummies,
            ], axis=1)
            errors, _ = _ridge_residual(
                model["roa"].to_numpy(dtype=float), x.to_numpy(dtype=float), alpha,
            )
            residual.loc[model.index] = errors
            fitted.loc[model.index] = model["roa"].to_numpy(dtype=float) - errors

    flags = features.copy()
    for column in IDIOSYNCRATIC_PROFIT_VARIABLES:
        flags[f"standardized_{column}"] = standardized[column]
    flags["source_report_period"] = features.index.get_level_values("report_period")
    flags["eligible_a_share"] = security_ids.isin(allowed) if allowed is not None else True
    flags["listing_age_eligible"] = features["listing_age_days"].ge(min_listing_days)
    flags["complete_standardized_model"] = standardized[
        list(IDIOSYNCRATIC_PROFIT_VARIABLES)
    ].notna().all(axis=1)
    flags["regression_sample_size"] = sample_size
    flags["ridge_alpha"] = alpha
    flags["fitted_standardized_roa"] = fitted
    flags["valid_for_factor"] = residual.notna()
    reasons: list[str] = []
    for index in flags.index:
        row_reasons: list[str] = []
        if not bool(flags.at[index, "eligible_a_share"]):
            row_reasons.append("not_in_eligible_a_share_universe")
        if pd.isna(flags.at[index, "list_date"]):
            row_reasons.append("missing_list_date")
        elif not bool(flags.at[index, "listing_age_eligible"]):
            row_reasons.append("listed_less_than_365_days")
        if pd.isna(flags.at[index, "industry_l1"]):
            row_reasons.append("missing_industry_l1")
        if factor_id == "res_roa2" and pd.isna(flags.at[index, "industry_l2"]):
            row_reasons.append("missing_industry_l2")
        missing_variables = [
            name for name in IDIOSYNCRATIC_PROFIT_VARIABLES
            if pd.isna(flags.at[index, name])
        ]
        if missing_variables:
            row_reasons.append("missing_variables:" + ",".join(missing_variables))
        if (
            not row_reasons
            and flags.at[index, "regression_sample_size"] < min_samples
        ):
            row_reasons.append("insufficient_regression_sample")
        reasons.append("|".join(row_reasons) if row_reasons else "ok")
    flags["reason_codes"] = reasons
    return residual, flags


ABNORMAL_CFO_BASE_REGRESSORS = (
    "inverse_lag_assets",
    "revenue_scaled",
    "revenue_change_scaled",
)


def _abnormal_cfo_features(financials: QuarterlyFinancials) -> pd.DataFrame:
    """Build single-quarter Roychowdhury-style CFO model inputs."""

    total_assets = financials.raw("total_assets")
    lag_assets_raw = financials.lag(total_assets, 1)
    lag_assets = lag_assets_raw.where(lag_assets_raw > 0)
    revenue = financials.single("operating_revenue")
    operating_cash_flow = financials.single("operating_cash_flow")
    other_direct = financials.single("other_operating_cash_flow")
    other_derived = (
        financials.single("cash_received_other_operating")
        - financials.single("cash_paid_other_operating")
    )
    other_operating_cash_flow = other_direct.combine_first(other_derived)
    employee_cash = financials.single("cash_paid_employees")

    frame = pd.DataFrame(index=financials.frame.index)
    frame["operating_cash_flow_scaled"] = safe_divide(
        operating_cash_flow, lag_assets,
    )
    frame["inverse_lag_assets"] = safe_divide(
        pd.Series(1.0, index=frame.index), lag_assets,
    )
    frame["revenue_scaled"] = safe_divide(revenue, lag_assets)
    frame["revenue_change_scaled"] = safe_divide(
        financials.diff(revenue, 1), lag_assets,
    )
    frame["other_operating_cash_flow_scaled"] = safe_divide(
        other_operating_cash_flow, lag_assets,
    )
    frame["employee_cash_scaled"] = safe_divide(employee_cash, lag_assets)
    frame["lag_total_assets"] = lag_assets_raw
    frame["industry_l1"] = _categorical_daily_at_report(financials, "industry_l1")
    frame["other_operating_cash_flow_source"] = np.select(
        [other_direct.notna(), other_derived.notna()],
        ["direct_net", "received_minus_paid"],
        default="missing",
    )

    listing_dates = _security_listing_dates(financials.context)
    securities = frame.index.get_level_values("security_id")
    report_periods = frame.index.get_level_values("report_period")
    mapped_listing_dates = pd.DatetimeIndex(securities.map(listing_dates))
    listing_age_days = pd.Series(
        (report_periods - mapped_listing_dates).days.astype(float), index=frame.index,
    ).where(lambda values: values >= 0)
    listing_age_years = listing_age_days / 365.25
    frame["list_date"] = mapped_listing_dates
    frame["listing_age_days"] = listing_age_days
    frame["listing_age_years"] = listing_age_years
    frame["listing_age_log1p_years"] = np.log1p(listing_age_years)
    frame["listing_age_cohort"] = pd.cut(
        listing_age_years,
        bins=[-np.inf, 3.0, 5.0, 10.0, 20.0, np.inf],
        labels=["lt3", "3to5", "5to10", "10to20", "ge20"],
        right=False,
    ).astype("object")
    return frame.replace([np.inf, -np.inf], np.nan)


def _zscore_columns_by_report_industry(
    frame: pd.DataFrame,
    columns: tuple[str, ...] | list[str],
    eligible: pd.Series,
    *,
    ddof: int = 0,
) -> pd.DataFrame:
    if ddof < 0:
        raise ValueError("industry z-score ddof must be non-negative")
    output = pd.DataFrame(np.nan, index=frame.index, columns=list(columns), dtype=float)
    keys = pd.DataFrame({
        "_report_period": frame.index.get_level_values("report_period"),
        "industry_l1": frame["industry_l1"].astype("string"),
    }, index=frame.index)
    valid_groups = eligible & keys["industry_l1"].notna()
    for _, raw_group_index in keys.loc[valid_groups].groupby(
        ["_report_period", "industry_l1"], sort=False,
    ).groups.items():
        group_index = pd.MultiIndex.from_tuples(raw_group_index, names=frame.index.names)
        for column in columns:
            values = pd.to_numeric(frame.loc[group_index, column], errors="coerce")
            finite = values.notna()
            if int(finite.sum()) <= ddof:
                continue
            mean = float(values.loc[finite].mean())
            std = float(values.loc[finite].std(ddof=ddof))
            output.loc[values.index[finite], column] = (
                0.0 if std == 0 else (values.loc[finite] - mean) / std
            )
    return output


def _abnormal_cfo_regressor_names(factor_id: str) -> list[str]:
    regressors = list(ABNORMAL_CFO_BASE_REGRESSORS)
    if factor_id in {"ABCFO4", "ABCFO5"}:
        regressors.extend([
            "other_operating_cash_flow_scaled",
            "employee_cash_scaled",
        ])
    return regressors


def _abnormal_cfo_residual_and_flags(
    financials: QuarterlyFinancials,
    factor_id: str,
    context: FactorContext,
    parameters: dict[str, object],
) -> tuple[pd.Series, pd.DataFrame]:
    """Quarter-by-industry abnormal operating-cash-flow residual."""

    features = _abnormal_cfo_features(financials)
    regression_mode = str(
        parameters.get("regression_mode", "citic_l1_separate_regressions"),
    ).strip().lower()
    allowed_modes = {
        "citic_l1_separate_regressions", "pooled", "citic_l1_dummies",
    }
    if regression_mode not in allowed_modes:
        raise ValueError(
            f"{factor_id}: regression_mode must be one of {sorted(allowed_modes)}, "
            f"got {regression_mode!r}",
        )
    standardization_stage = str(
        parameters.get("standardization_stage", "residual"),
    ).strip().lower()
    if standardization_stage not in {"none", "inputs", "residual", "both"}:
        raise ValueError(
            f"{factor_id}: standardization_stage must be none/inputs/residual/both",
        )
    listing_age_transform = str(
        parameters.get("listing_age_transform", "log1p_years"),
    ).strip().lower()
    if listing_age_transform not in {"log1p_years", "raw_years", "cohort_dummies"}:
        raise ValueError(
            f"{factor_id}: unsupported listing_age_transform "
            f"{listing_age_transform!r}",
        )
    min_samples = int(parameters.get("min_regression_samples", 10))
    if min_samples < 3:
        raise ValueError(f"{factor_id}: min_regression_samples must be at least 3")

    regressors = _abnormal_cfo_regressor_names(factor_id)
    listing_age_numeric: str | None = None
    if factor_id == "ABCFO5" and listing_age_transform != "cohort_dummies":
        listing_age_numeric = (
            "listing_age_log1p_years"
            if listing_age_transform == "log1p_years"
            else "listing_age_years"
        )
        regressors.append(listing_age_numeric)

    date = financials.calculation_date
    allowed = _report_regression_universe(context, date, parameters)
    security_ids = features.index.get_level_values("security_id").astype(str)
    eligible = pd.Series(True, index=features.index)
    if allowed is not None:
        eligible &= security_ids.isin(allowed)
    # CITIC L1 is also needed after pooled regressions because the confirmed
    # output standardization is performed within each industry.
    eligible &= features["industry_l1"].notna()

    dependent_name = "operating_cash_flow_scaled"
    numeric_columns = [dependent_name, *regressors]
    if standardization_stage in {"inputs", "both"}:
        standardized_inputs = _zscore_columns_by_report_industry(
            features, numeric_columns, eligible,
        )
        model_values = features.copy()
        model_values.loc[:, numeric_columns] = standardized_inputs[numeric_columns]
    else:
        standardized_inputs = pd.DataFrame(
            np.nan, index=features.index, columns=numeric_columns, dtype=float,
        )
        model_values = features

    raw_residual = _empty_report_series(features.index)
    fitted = _empty_report_series(features.index)
    sample_size = pd.Series(0, index=features.index, dtype=int)
    required_sample_size = pd.Series(0, index=features.index, dtype=int)
    full_rank = pd.Series(False, index=features.index, dtype=bool)
    report_keys = pd.DataFrame({
        "_report_period": features.index.get_level_values("report_period"),
        "industry_l1": features["industry_l1"].astype("string"),
    }, index=features.index)
    if regression_mode == "citic_l1_separate_regressions":
        grouped = report_keys.loc[eligible].groupby(
            ["_report_period", "industry_l1"], sort=False,
        ).groups
    else:
        grouped = report_keys.loc[eligible].groupby("_report_period", sort=False).groups

    for _, raw_group_index in grouped.items():
        group_index = pd.MultiIndex.from_tuples(raw_group_index, names=features.index.names)
        base_columns = [dependent_name, *regressors]
        model = model_values.loc[group_index, base_columns].copy()
        if factor_id == "ABCFO5" and listing_age_transform == "cohort_dummies":
            model["listing_age_cohort"] = features.loc[
                group_index, "listing_age_cohort"
            ].astype("string")
        if regression_mode == "citic_l1_dummies":
            model["industry_l1"] = features.loc[group_index, "industry_l1"].astype("string")
        model = model.dropna()

        x = model[regressors].astype(float)
        if factor_id == "ABCFO5" and listing_age_transform == "cohort_dummies":
            cohort_dummies = pd.get_dummies(
                model["listing_age_cohort"], prefix="listing_age", drop_first=True,
                dtype=float,
            )
            x = pd.concat([x, cohort_dummies], axis=1)
        if regression_mode == "citic_l1_dummies":
            industry_dummies = pd.get_dummies(
                model["industry_l1"], prefix="citic_l1", drop_first=True, dtype=float,
            )
            x = pd.concat([x, industry_dummies], axis=1)

        required = max(min_samples, x.shape[1] + 2)
        sample_size.loc[group_index] = len(model)
        required_sample_size.loc[group_index] = required
        if len(model) < required:
            continue
        y = model[dependent_name].to_numpy(dtype=float)
        design = np.column_stack([np.ones(len(model)), x.to_numpy(dtype=float)])
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        errors = y - design @ coefficients
        rank_is_full = bool(np.linalg.matrix_rank(design) == design.shape[1])
        raw_residual.loc[model.index] = errors
        fitted.loc[model.index] = y - errors
        full_rank.loc[group_index] = rank_is_full

    if standardization_stage in {"residual", "both"}:
        residual_frame = features[["industry_l1"]].copy()
        residual_frame["raw_residual"] = raw_residual
        factor_value = _zscore_columns_by_report_industry(
            residual_frame, ["raw_residual"], eligible & raw_residual.notna(),
        )["raw_residual"]
    else:
        factor_value = raw_residual.copy()

    flags = features.copy()
    for column in numeric_columns:
        flags[f"standardized_input_{column}"] = standardized_inputs[column]
    flags["source_report_period"] = features.index.get_level_values("report_period")
    flags["eligible_a_share"] = security_ids.isin(allowed) if allowed is not None else True
    flags["complete_model_inputs"] = model_values[numeric_columns].notna().all(axis=1)
    if factor_id == "ABCFO5" and listing_age_transform == "cohort_dummies":
        flags["complete_model_inputs"] &= features["listing_age_cohort"].notna()
    flags["regression_mode"] = regression_mode
    flags["standardization_stage"] = standardization_stage
    flags["listing_age_transform"] = listing_age_transform if factor_id == "ABCFO5" else "not_used"
    flags["regression_sample_size"] = sample_size
    flags["required_regression_sample_size"] = required_sample_size
    flags["regression_full_rank"] = full_rank
    flags["fitted_dependent"] = fitted
    flags["raw_residual"] = raw_residual
    flags["factor_value"] = factor_value
    flags["valid_for_factor"] = factor_value.notna()
    flags["warning_codes"] = np.where(
        raw_residual.notna() & ~full_rank, "rank_deficient_minimum_norm_ols", "",
    )

    reasons: list[str] = []
    required_variables = [dependent_name, *regressors]
    for index in flags.index:
        row_reasons: list[str] = []
        if not bool(flags.at[index, "eligible_a_share"]):
            row_reasons.append("not_in_eligible_a_share_universe")
        if pd.isna(flags.at[index, "lag_total_assets"]):
            row_reasons.append("missing_lag_total_assets")
        elif flags.at[index, "lag_total_assets"] <= 0:
            row_reasons.append("nonpositive_lag_total_assets")
        if pd.isna(flags.at[index, "industry_l1"]):
            row_reasons.append("missing_citic_industry_l1")
        if factor_id == "ABCFO5" and pd.isna(flags.at[index, "list_date"]):
            row_reasons.append("missing_list_date")
        missing_variables = [
            name for name in required_variables if pd.isna(flags.at[index, name])
        ]
        if factor_id == "ABCFO5" and listing_age_transform == "cohort_dummies":
            if pd.isna(flags.at[index, "listing_age_cohort"]):
                missing_variables.append("listing_age_cohort")
        if missing_variables:
            row_reasons.append("missing_variables:" + ",".join(missing_variables))
        required_count = int(flags.at[index, "required_regression_sample_size"])
        if not row_reasons and required_count > 0:
            if int(flags.at[index, "regression_sample_size"]) < required_count:
                row_reasons.append("insufficient_regression_sample")
        reasons.append("|".join(row_reasons) if row_reasons else "ok")
    flags["reason_codes"] = reasons
    return factor_value, flags


def _eight_quarter_skew(array: np.ndarray) -> float:
    if not np.isfinite(array).all():
        return np.nan
    centered = array - array.mean()
    second = np.mean(centered ** 2)
    return 0.0 if second == 0 else np.sqrt(56.0) / 6.0 * np.mean(centered ** 3) / second ** 1.5


def _tercile(values: pd.Series) -> pd.Series:
    if values.empty:
        return values
    percentile = values.rank(method="average", pct=True)
    return np.ceil(percentile * 3.0).clip(1, 3)


def _expense_bundle_ttm(financials: QuarterlyFinancials, fields: tuple[str, ...]) -> pd.Series:
    """Preserve pre-2018 reported expense totals without inventing an R&D value."""
    # Before the 2018 format change, expensed R&D is already inside
    # operating/management/selling costs. Never fill the R&D field itself.
    periods = financials.frame.index.get_level_values("report_period")
    legacy = pd.Series(periods <= pd.Timestamp("2018-03-31"), index=financials.frame.index)
    for name in fields:
        stamps = financials.frame.get(name + "__effective_at", pd.Series(pd.NaT, index=legacy.index))
        legacy &= pd.to_datetime(stamps).lt(pd.Timestamp("2018-06-15"))
    rd_stamp = financials.frame.get("rd_expense__effective_at", pd.Series(pd.NaT, index=legacy.index))
    legacy &= rd_stamp.isna() | pd.to_datetime(rd_stamp).lt(pd.Timestamp("2018-06-15"))
    rd = financials.raw("rd_expense")
    bundle = _strict_sum([financials.raw(name) for name in fields]) + rd.fillna(0.0)
    bundle = bundle.where(rd.notna() | legacy)
    prior_year_end = periods.to_period("Y").start_time - pd.Timedelta(days=1)
    keys = pd.MultiIndex.from_arrays([
        bundle.index.get_level_values("security_id"), prior_year_end,
    ], names=bundle.index.names)
    annual = pd.Series(bundle.reindex(keys).to_numpy(), index=bundle.index)
    return (bundle + annual - financials.exact_quarter_lag(bundle, 4)).where(financials.quarter.ne(4), bundle)


def _current_cross_section(
    financials: QuarterlyFinancials,
    factor_id: str,
    context: FactorContext,
    parameters: dict[str, object] | None = None,
) -> pd.Series:
    date = financials.calculation_date
    parameters = {} if parameters is None else parameters
    if parameters.get("history_policy") == "published_quarter_eps_clock_v4":
        if factor_id not in {"EPS_Delta_Vol", "factor_eaa", "factor_eap", "factor_eav", "factor_egv"}:
            raise ValueError("Published EPS policy is restricted to native quarter EPS factors")
        if "eps_quarter" not in financials.frame:
            raise ValueError("Published EPS policy requires eps_quarter; cumulative EPS is not a substitute")
        eps = financials.raw("eps_quarter")
        exact_lag = financials.exact_quarter_lag
        prior = exact_lag(eps, 4)
        growth = safe_divide(eps - prior, prior.abs())
        complete_eight = (financials.ordinal - financials.ordinal.groupby(level="security_id").shift(7)).eq(7)
        if factor_id == "EPS_Delta_Vol":
            raw = financials.std(growth, 8).where(complete_eight)
        elif factor_id == "factor_eaa":
            raw = growth - exact_lag(growth, 1)
        elif factor_id == "factor_eap":
            # Haitong factor study 83, p5: EGP uses the previous quarter's
            # closing price, while EPS growth still compares q with q-4.
            price = exact_lag(financials.daily_at_report("raw_close"), 1)
            scaled = safe_divide(eps - prior, price.where(price.gt(0)))
            raw = scaled - exact_lag(scaled, 1)
        else:
            deviation = financials.std(eps, 8).where(complete_eight)
            scaled = safe_divide(eps - prior, deviation.where(deviation.ne(0)))
            raw = scaled - exact_lag(scaled, 1) if factor_id == "factor_eav" else scaled
        return _filter_universe(_latest_report_frame(raw), context, date)
    policy = parameters.get("history_policy")
    if policy == "report_price_operating_profit_skew_v1":
        if factor_id not in {"POP_Skew", "Rel_POP_Skew"}:
            raise ValueError("Report-price skew policy does not cover this factor")
        cap = financials.daily_at_report("total_market_cap")
        ratio = safe_divide(financials.cumulative_ttm("operating_profit"), cap.where(cap.gt(0)))
        complete = (financials.ordinal - financials.ordinal.groupby(level="security_id").shift(7)).eq(7)
        # Recenter each eight-quarter window independently: incremental rolling
        # skew can retain cancellation error after an old extreme value exits.
        skew = financials.rolling(ratio, 8, lambda values: float(pd.Series(values).skew()))
        current = _filter_universe(_latest_report_frame(skew.where(complete)), context, date)
        if factor_id == "Rel_POP_Skew":
            industry = _industry_series(financials).reindex(current.index)
            valid = pd.concat([current.rename("value"), industry.rename("industry")], axis=1).dropna()
            return valid.value - valid.groupby("industry").value.transform("median")
        return current
    if policy == "tax_payable_total_equity_zscore_v2":
        if factor_id != "taxes_payable_ratio_zscore":
            raise ValueError("Tax payable policy does not cover this factor")
        # The definition says net assets, not parent-attributable net assets.
        # Use the same consolidated statement's assets less liabilities.
        equity = financials.raw("total_assets") - financials.raw("total_liabilities")
        ratio = safe_divide(financials.raw("taxes_payable"), equity)
        current = _filter_cross_section_population(_latest_report_frame(ratio), context, date)
        deviation = current.std(ddof=1)
        return (current - current.mean()) / deviation if pd.notna(deviation) and deviation != 0 else current * np.nan
    if policy == "valuation_720_trading_sessions_v1":
        if factor_id not in {"EPTTM_3y_quantile", "CFPTTM_3y_quantile"}:
            raise ValueError("Statement valuation quantile policy does not cover this factor")
        field = "net_profit" if factor_id == "EPTTM_3y_quantile" else "operating_cash_flow"
        cap = financials.daily("total_market_cap")
        # Output eligibility must not remove observations from the time window.
        return _latest_report_frame(safe_divide(financials.cumulative_ttm(field), cap.where(cap.gt(0))))
    if policy == "roe_ttm_current_report_rank_v1":
        if factor_id not in {"roe_ttm_rg", "pb_roe_quantile_diff", "pb_roe_rank_dfif"}:
            raise ValueError("Parent TTM ROE rank policy does not cover this factor")
        equity = financials.raw("equity_parent")
        profit_field = "net_profit" if factor_id == "roe_ttm_rg" else "net_profit_parent"
        roe = safe_divide(2.0 * financials.cumulative_ttm(profit_field), equity + financials.exact_quarter_lag(equity, 4))
        current = _filter_cross_section_population(_latest_report_frame(roe), context, date)
        if factor_id == "roe_ttm_rg":
            return _rank_cross_section(current, date)
        if parameters.get("pb_source") == "derived_native_parent":
            cap = financials.daily_security("total_market_cap")
            pb = safe_divide(cap.where(cap.gt(0)), _latest_report_frame(equity))
        else:
            pb = financials.daily_security("price_to_book")
        pair = pd.concat([pb.rename("pb"), current.rename("roe")], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
        pair = pair.loc[pair.index.intersection(current.index)]
        percentile = factor_id == "pb_roe_quantile_diff"
        return pair.pb.rank(method="average", pct=percentile) - pair.roe.rank(method="average", pct=percentile)
    if policy == "quarter_roe_parent_rank_v1":
        if factor_id != "roeq_rg":
            raise ValueError("Quarter ROE rank policy does not cover this factor")
        equity = financials.raw("equity_parent")
        roe = safe_divide(2.0 * financials.single("net_profit_parent"), equity + financials.exact_quarter_lag(equity, 1))
        current = _filter_cross_section_population(_latest_report_frame(roe), context, date)
        return _rank_cross_section(current, date)
    if policy == "capex_ten_complete_years_v2":
        if factor_id != "capex2np_10y":
            raise ValueError("Ten-year capex policy does not cover this factor")
        return _filter_universe(_latest_report_frame(_ten_complete_fiscal_years_capex(financials)), context, date)
    if policy == "fcff_ttm_strict_enterprise_v1":
        if factor_id != "fcff_ratio":
            raise ValueError("TTM cash flow yield policy does not cover this factor")
        numerator = financials.cumulative_ttm("operating_cash_flow") - financials.cumulative_ttm("capital_expenditure_cash_paid")
        cap = financials.daily("total_market_cap")
        adjustment = financials.raw("interest_bearing_debt") + financials.raw("minority_interest") + financials.raw("other_equity_instruments") - financials.raw("cash")
        return _filter_universe(_latest_report_frame(safe_divide(numerator, cap.where(cap.gt(0)) + adjustment)), context, date)
    if policy == "fcff_quarter_workbook_ev_v1":
        if factor_id != "fcff_ratio_qua":
            raise ValueError("Quarter cash flow yield policy does not cover this factor")
        numerator = financials.single("operating_cash_flow") - financials.single("capital_expenditure_cash_paid")
        cap = financials.daily("total_market_cap")
        denominator = cap.where(cap.gt(0)) + financials.raw("total_liabilities") - financials.raw("cash")
        return _filter_universe(_latest_report_frame(safe_divide(numerator, denominator)), context, date)
    if policy == "valuation_disclosed_rank_v1":
        fields = {"EPTTM": ("net_profit", "ttm"), "CFPTTM": ("operating_cash_flow", "ttm"), "SPTTM": ("operating_revenue", "ttm"),
                  "epq": ("net_profit_parent", "single"), "epq_rg": ("net_profit_parent", "single"), "peq_rg": ("net_profit_parent", "single"),
                  "epttm_rg": ("net_profit_parent", "ttm"), "pettm_rg": ("net_profit_parent", "ttm"), "value_bivariate_rg": ("net_profit_parent", "ttm"),
                  "bp_rg": ("equity_parent", "raw"), "pb_rg": ("equity_parent", "raw"),
                  "ps_ttm_rg": ("operating_revenue", "ttm"), "sp_ttm_rg": ("operating_revenue", "ttm"),
                  "pcf_ttm_rg": ("operating_cash_flow", "ttm"), "cfp_ttm_rg": ("operating_cash_flow", "ttm"),
                  "spq_rg": ("operating_revenue", "single"), "psq_rg": ("operating_revenue", "single"),
                  "cfpq_rg": ("operating_cash_flow", "single"), "pcfq_rg": ("operating_cash_flow", "single")}
        if factor_id not in fields:
            raise ValueError("Valuation rank policy does not cover this factor")
        field, period = fields[factor_id]
        method = financials.raw if period == "raw" else financials.single if period == "single" else financials.cumulative_ttm
        numerator = method(field)
        cap = financials.daily("total_market_cap")
        cap = cap.where(cap.gt(0))
        inverse = factor_id in {"peq_rg", "pettm_rg", "pb_rg", "ps_ttm_rg", "pcf_ttm_rg", "psq_rg", "pcfq_rg"}
        value = safe_divide(cap, numerator) if inverse else safe_divide(numerator, cap)
        current = (_filter_cross_section_population if factor_id.endswith("_rg") else _filter_universe)(_latest_report_frame(value), context, date).replace([np.inf, -np.inf], np.nan)
        if factor_id == "value_bivariate_rg":
            book_yield = _latest_report_frame(safe_divide(financials.raw("equity_parent"), cap))
            pair = pd.concat([current.rename("ep"), book_yield.reindex(current.index).rename("bp")], axis=1).dropna()
            current = (pair.ep.rank(method="average", pct=True) + pair.bp.rank(method="average", pct=True)) / 2.0
        return _rank_cross_section(current, date) if factor_id.endswith("_rg") else current
    if policy == "financial_extended_components_v1":
        if factor_id not in {"SalesG", "ProfitG", "re_qoq", "re_yoy", "finexp_ratio_ttm", "money2mv", "PCFTTM", "dol_ttm1", "cp_rg", "pc_rg"}:
            raise ValueError("Extended financial policy does not cover this factor")
        ttm = financials.cumulative_ttm
        if factor_id in {"SalesG", "ProfitG", "re_qoq", "re_yoy"}:
            if factor_id in {"SalesG", "ProfitG"}:
                current = ttm("operating_revenue" if factor_id == "SalesG" else "net_profit_parent")
            else:
                current = financials.raw("undistributed_profit") + financials.raw("surplus_reserve")
            prior = financials.exact_quarter_lag(current, 1 if factor_id == "re_qoq" else 4)
            value = safe_divide(current - prior, prior.abs())
        elif factor_id == "finexp_ratio_ttm":
            value = safe_divide(ttm("financial_expense"), ttm("operating_revenue"))
        elif factor_id == "dol_ttm1":
            value = safe_divide(ttm("ebit"), ttm("ebitda"))
        else:
            cap = financials.daily("total_market_cap")
            cap = cap.where(cap.gt(0))
            if factor_id in {"cp_rg", "pc_rg"}:
                cash = financials.raw("cash_end")
                value = safe_divide(cash, cap) if factor_id == "cp_rg" else safe_divide(cap, cash)
            else:
                value = safe_divide(financials.raw("cash"), cap) if factor_id == "money2mv" else safe_divide(cap, ttm("operating_cash_flow"))
        current = (_filter_cross_section_population if factor_id.endswith("_rg") else _filter_universe)(_latest_report_frame(value), context, date).replace([np.inf, -np.inf], np.nan)
        return _rank_cross_section(current, date) if factor_id in {"cp_rg", "pc_rg"} else current
    if policy == "financial_remaining_disclosed_history_v1" and factor_id not in GROWTH_TREND_IDS:
        allowed = {"roe_ttm", "npttm_chg", "rev_growth_com_3y", "rev_growth_com_2y", "sale_yoy", "tot_assets_cash_recovery", "factor_op_roa", "gross_margin_ttm", "gross_margin_ttm_chg", "roeq_mean_div_std_3y", "roeq_mean_sub_std_3y", "roeq_mean_3y", "roeq_std_3y", "roeq_mean_12q", "roeq_std_12q", "acc_profit_ttm"}
        if factor_id not in allowed:
            raise ValueError("Remaining financial policy does not cover this factor")
        raw, ttm = financials.raw, financials.cumulative_ttm
        if factor_id == "roe_ttm":
            value = safe_divide(ttm("net_profit"), raw("total_assets") - raw("total_liabilities"))
        elif factor_id == "npttm_chg":
            profit = ttm("net_profit")
            value = safe_divide(profit, financials.exact_quarter_lag(profit, 1)) - 1.0
        elif factor_id.startswith("rev_growth_com_"):
            years = 3 if factor_id.endswith("3y") else 2
            revenue = ttm("operating_revenue")
            ratio = safe_divide(revenue, financials.exact_quarter_lag(revenue, 4 * years))
            value = ratio.where(ratio.gt(0)).pow(1.0 / years) - 1.0
        elif factor_id == "sale_yoy":
            revenue = raw("operating_revenue")
            prior = financials.exact_quarter_lag(revenue, 4)
            value = safe_divide(revenue - prior, prior.abs())
        elif factor_id in {"tot_assets_cash_recovery", "factor_op_roa"}:
            value = safe_divide(ttm("operating_cash_flow" if factor_id == "tot_assets_cash_recovery" else "operating_profit"), raw("total_assets"))
        elif factor_id == "acc_profit_ttm":
            value = ttm("net_profit") - ttm("operating_cash_flow")
        elif factor_id.startswith("gross_margin_ttm"):
            revenue = ttm("operating_revenue")
            value = safe_divide(revenue - ttm("operating_cost"), revenue)
            if factor_id.endswith("chg"):
                value = (value - financials.exact_quarter_lag(value, 1)) * 100.0
        else:
            equity = raw("equity_parent")
            roe = safe_divide(2.0 * financials.single("net_profit_parent"), equity + financials.exact_quarter_lag(equity, 1))
            complete = (financials.ordinal - financials.ordinal.groupby(level="security_id").shift(11)).eq(11)
            # Recompute each 12-quarter window directly. Removing historical
            # extreme values from a rolling accumulator can lose precision.
            mean = financials.rolling(roe, 12, lambda values: float(np.mean(values)))
            deviation = financials.rolling(roe, 12, lambda values: float(np.std(values, ddof=1)))
            if factor_id in {"roeq_mean_3y", "roeq_mean_12q"}:
                value = mean * 100.0
            elif factor_id in {"roeq_std_3y", "roeq_std_12q"}:
                value = deviation * 100.0
            else:
                value = safe_divide(mean, deviation) if factor_id == "roeq_mean_div_std_3y" else mean - deviation
            value = value.where(complete)
        return _filter_universe(_latest_report_frame(value), context, date)
    if policy == "financial_core_same_report_v1":
        allowed = {"NET_PROFIT_INCL_MIN_INT_INC_TTM", "NET_PROFIT_EXCL_MIN_INT_INC_TTM", "NET_PROFIT_INCL_MIN_INT_INC_QUA", "NET_PROFIT_EXCL_MIN_INT_INC_QUA", "assets_turnover_ttm", "assets_turnoverTTM_chg", "profit_quality", "acc_profit_qua", "net_margin_ttm", "liab2assets", "liab_to_assets_chg", "goodwill2na", "goodwill2ta", "cfottm2curliab", "cfottm2curliab_chg", "cfo2tb_ttm", "gpmq_diff4", "gpmq_diff1", "npm_diff4", "npm_diff1", "cfo2rev_ttm_diff"}
        if factor_id not in allowed:
            raise ValueError("Financial core policy does not cover this factor")
        raw, ttm, single = financials.raw, financials.cumulative_ttm, financials.single
        difference = None
        if factor_id.startswith("NET_PROFIT_"):
            field = "net_profit_parent" if "EXCL" in factor_id else "net_profit"
            value = ttm(field) if factor_id.endswith("TTM") else single(field)
        elif factor_id.startswith("assets_turnover"):
            assets = raw("total_assets")
            value = safe_divide(ttm("operating_revenue"), (assets + financials.exact_quarter_lag(assets, 4)) / 2.0)
            difference = 1 if factor_id.endswith("chg") else None
        elif factor_id == "profit_quality":
            value = safe_divide(ttm("operating_cash_flow") - ttm("operating_profit"), raw("total_assets"))
        elif factor_id == "acc_profit_qua":
            value = single("net_profit") - single("operating_cash_flow")
        elif factor_id == "net_margin_ttm":
            value = safe_divide(ttm("net_profit_parent"), ttm("operating_revenue"))
        elif factor_id in {"liab2assets", "liab_to_assets_chg"}:
            value = safe_divide(raw("total_liabilities"), raw("total_assets"))
            difference = 1 if factor_id.endswith("chg") else None
        elif factor_id in {"goodwill2na", "goodwill2ta"}:
            value = safe_divide(raw("goodwill"), raw("equity_parent" if factor_id == "goodwill2na" else "total_assets"))
        elif factor_id in {"cfottm2curliab", "cfottm2curliab_chg", "cfo2tb_ttm"}:
            value = safe_divide(ttm("operating_cash_flow"), raw("total_liabilities" if factor_id == "cfo2tb_ttm" else "total_current_liabilities"))
            difference = 1 if factor_id.endswith("chg") else None
        elif factor_id.startswith(("gpmq_diff", "npm_diff")):
            revenue = single("operating_revenue")
            numerator = revenue - single("operating_cost") if factor_id.startswith("gpmq") else single("net_profit_parent")
            value = safe_divide(numerator, revenue)
            difference = 4 if factor_id.endswith("4") else 1
        else:
            value = safe_divide(ttm("operating_cash_flow"), ttm("operating_revenue"))
            difference = 1
        if difference is not None:
            value = value - financials.exact_quarter_lag(value, difference)
        return _filter_universe(_latest_report_frame(value), context, date)
    if policy == "native_ratios_report_clock_v1":
        if factor_id not in {"ROE", "quick_ratio"}:
            raise ValueError("Native ratio policy does not cover this factor")
        if "__report_clock" not in financials.frame:
            raise ValueError("Native ratios require a known financial report clock")
        # ROE is the explicitly named native percentage-point source field;
        # lowercase roe fractions or derived TTM returns are not substitutes.
        return _filter_universe(_latest_report_frame(financials.raw(factor_id).rename("native_ratio")), context, date)
    if policy == "financial_changes_exact_quarters_v1":
        if factor_id not in {"revq_yoy_diff4", "revq_yoy_diff1", "npq_yoy_diff4", "npq_yoy_diff1", "OPR_Skew", "cfo_cv_pit", "delta_roeq"}:
            raise ValueError("Financial change policy does not cover this factor")
        if factor_id.startswith(("revq_yoy_diff", "npq_yoy_diff")):
            primitive = "operating_revenue" if factor_id.startswith("revq") else "net_profit_parent"
            quarter = financials.single(primitive)
            prior = financials.exact_quarter_lag(quarter, 4)
            growth = safe_divide(quarter - prior, prior.abs())
            value = growth - financials.exact_quarter_lag(growth, 4 if factor_id.endswith("4") else 1)
        elif factor_id == "delta_roeq":
            equity = financials.raw("equity_parent")
            roe = safe_divide(2.0 * financials.single("net_profit_parent"), equity + financials.exact_quarter_lag(equity, 1))
            value = (roe - financials.exact_quarter_lag(roe, 4)) * 100.0
        else:
            complete = (financials.ordinal - financials.ordinal.groupby(level="security_id").shift(7)).eq(7)
            if factor_id == "cfo_cv_pit":
                quarter = financials.single("operating_cash_flow")
                value = safe_divide(financials.mean(quarter, 8), financials.std(quarter, 8)).where(complete)
            else:
                ratio = safe_divide(financials.cumulative_ttm("operating_profit"), financials.cumulative_ttm("operating_revenue"))
                value = financials.rolling(ratio, 8, _eight_quarter_skew).where(complete)
        return _filter_universe(_latest_report_frame(value), context, date)
    if policy in {"operating_leverage_current_assets_v2", "operating_leverage_expense_vintage_v3"}:
        if factor_id != "dol_ttm2":
            raise ValueError("Current-asset operating leverage policy only covers dol_ttm2")
        costs = _strict_sum([financials.cumulative_ttm(name) for name in
            ["operating_cost", "management_expense", "selling_expense", "rd_expense"]])
        if policy == "operating_leverage_expense_vintage_v3":
            costs = _expense_bundle_ttm(financials, ("operating_cost", "management_expense", "selling_expense"))
        value = safe_divide(costs, financials.raw("total_assets"))
        return _filter_universe(_latest_report_frame(value), context, date)
    if policy in {"operating_turnover_components_v1", "operating_margin_total_revenue_v2"}:
        if policy == "operating_margin_total_revenue_v2" and factor_id not in {"op2or_ttm_chg", "OPR_Skew"}:
            raise ValueError("Total-revenue margin policy only covers operating-margin change and skewness")
        if factor_id not in {"inventory_turnover_ttm", "receivable_turnover_ttm", "dol_ttm2", "op2or_ttm_chg"} and not (factor_id == "OPR_Skew" and policy == "operating_margin_total_revenue_v2"):
            raise ValueError("Operating turnover policy does not cover this factor")
        if factor_id in {"op2or_ttm_chg", "OPR_Skew"}:
            denominator = "total_revenue" if policy == "operating_margin_total_revenue_v2" else "operating_revenue"
            ratio = safe_divide(financials.cumulative_ttm("operating_profit"), financials.cumulative_ttm(denominator))
            if factor_id == "OPR_Skew":
                complete = (financials.ordinal - financials.ordinal.groupby(level="security_id").shift(7)).eq(7)
                value = financials.rolling(ratio, 8, _eight_quarter_skew).where(complete)
            else:
                value = (ratio - financials.exact_quarter_lag(ratio, 1)) * 100.0
            return _filter_universe(_latest_report_frame(value), context, date)
        if factor_id == "inventory_turnover_ttm":
            stock = financials.raw("inventory")
            numerator = financials.cumulative_ttm("operating_cost")
        elif factor_id == "receivable_turnover_ttm":
            stock = financials.raw("accounts_receivable_combined").combine_first(
                financials.raw("notes_receivable") + financials.raw("accounts_receivable"))
            numerator = financials.cumulative_ttm("operating_revenue")
        else:
            stock = financials.raw("total_assets")
            numerator = _strict_sum([financials.cumulative_ttm(name) for name in
                ["operating_cost", "management_expense", "selling_expense", "rd_expense"]])
        denominator = (stock + financials.exact_quarter_lag(stock, 4)) / 2.0
        return _filter_universe(_latest_report_frame(safe_divide(numerator, denominator)), context, date)
    if policy in {"operating_daa_disclosed_components_v1", "operating_daa_expense_vintage_v2"}:
        if policy == "operating_daa_expense_vintage_v2" and factor_id not in {"opex_ttm", "opex2ta_ttm"}:
            raise ValueError("Expense-vintage D&A policy only covers opex_ttm and opex2ta_ttm")
        if factor_id not in {"opex_ttm", "opex2ta_ttm", "da2rev_ttm", "da2gross_ttm", "capex_da_ratio_ttm"}:
            raise ValueError("Operating D&A policy does not cover this factor")
        component_da = _strict_sum([financials.cumulative_ttm(name) for name in
            ["depreciation_fixed_assets", "amortization_intangible_assets", "amortization_long_term_deferred_expenses", "depreciation_right_of_use_assets"]])
        da = financials.cumulative_ttm("depreciation_amortization").combine_first(component_da)
        revenue = financials.cumulative_ttm("operating_revenue")
        if factor_id in {"opex_ttm", "opex2ta_ttm"}:
            expenses = (_expense_bundle_ttm(financials, ("management_expense", "selling_expense"))
                if policy == "operating_daa_expense_vintage_v2" else
                _strict_sum([financials.cumulative_ttm(name) for name in ["management_expense", "selling_expense", "rd_expense"]]))
            value = da + expenses
            if factor_id == "opex2ta_ttm":
                value = safe_divide(value, financials.raw("total_assets"))
        elif factor_id == "da2rev_ttm":
            value = safe_divide(da, revenue)
        elif factor_id == "da2gross_ttm":
            value = safe_divide(da, revenue - financials.cumulative_ttm("operating_cost"))
        else:
            value = safe_divide(financials.cumulative_ttm("capital_expenditure_cash_paid"), da)
        return _filter_universe(_latest_report_frame(value), context, date)
    if policy == "authorized_four_field_expense_capital_v1":
        if factor_id != "pb_int":
            raise ValueError("Four-field expense capital policy is restricted to pb_int")
        capitals = {}
        fields = ["rd_expense", "selling_expense", "management_expense"]
        for code, reports in financials.frame.groupby(level="security_id", sort=False):
            annual = reports.droplevel("security_id").sort_index()
            annual = annual[annual.index.quarter == 4].reindex(columns=fields)
            complete = np.isfinite(annual).all(axis=1)
            capital = np.nan
            if complete.any():
                first = annual.index[complete][0]
                required = pd.DatetimeIndex([pd.Timestamp(y, 12, 31) for y in range(first.year, annual.index[-1].year + 1)])
                segment = annual.reindex(required)
                if np.isfinite(segment).all().all():
                    research = organization = 0.0
                    for row in segment.itertuples(index=False):
                        research = 0.85 * research + row.rd_expense
                        organization = 0.8 * organization + 0.30 * (row.selling_expense + row.management_expense)
                    capital = research + organization
            capitals[code] = capital
        cap = _latest_report_frame(financials.daily("total_market_cap").rename("market_cap"))
        value = safe_divide(pd.Series(capitals, dtype=float).rename_axis("security_id"), cap.where(cap.gt(0)))
        return _filter_universe(value, context, date)
    if policy == "cumulative_rd_annual_history_v1":
        if factor_id != "cum_rd_exp":
            raise ValueError("Cumulative R&D policy is restricted to cum_rd_exp")
        values = {}
        for code, reports in financials.frame.groupby(level="security_id", sort=False):
            reports = reports.droplevel("security_id").sort_index()
            rd = reports.get("rd_expense", pd.Series(np.nan, index=reports.index))
            known = rd.dropna()
            value = np.nan
            if not known.empty:
                latest = reports.index[-1]
                required = [pd.Timestamp(y, 12, 31) for y in range(known.index[0].year, latest.year)] + [latest]
                total = rd.reindex(required).sum(min_count=len(required))
                assets = reports.iloc[-1].get("total_assets", np.nan)
                if pd.notna(assets) and assets > 0:
                    value = total / assets
            values[code] = value
        return _filter_universe(pd.Series(values, dtype=float).rename_axis("security_id"), context, date)
    if policy == "capital_ratios_disclosed_components_v1":
        if factor_id not in {"roic_ttm", "ebitq2ev", "eyoev_ttm"}:
            raise ValueError("Capital ratio policy does not cover this factor")
        if factor_id == "roic_ttm":
            receivable = financials.raw("accounts_receivable_combined").combine_first(
                financials.raw("notes_receivable") + financials.raw("accounts_receivable"))
            invested = (financials.raw("inventory") + financials.raw("fixed_assets") + receivable
                        + financials.raw("other_receivables") - financials.raw("non_interest_current_liabilities"))
            value = safe_divide(financials.cumulative_ttm("ebit"), invested)
        else:
            cap = financials.daily("total_market_cap")
            denominator = cap.where(cap.gt(0)) + financials.raw("interest_bearing_debt")
            if factor_id == "eyoev_ttm":
                denominator = denominator + financials.raw("minority_interest") + financials.raw("other_equity_instruments") - financials.raw("cash")
            numerator = financials.single("ebit") if factor_id == "ebitq2ev" else financials.cumulative_ttm("ebit")
            value = safe_divide(numerator, denominator)
        return _filter_universe(_latest_report_frame(value), context, date)
    if policy == "deducted_growth_subtraction_clock_v1":
        if factor_id not in {"np_qua_chg_acc", "performances_trend", "factor_roe_deducted"}:
            raise ValueError("Deducted growth policy does not cover this factor")
        cumulative = financials.raw("net_profit_parent") - financials.raw("nonrecurring_profit_parent")
        quarter = cumulative.where(financials.quarter.eq(1), cumulative - financials.exact_quarter_lag(cumulative, 1))
        ttm = financials.cumulative_ttm("net_profit_parent") - financials.cumulative_ttm("nonrecurring_profit_parent")
        if factor_id == "factor_roe_deducted":
            value = safe_divide(ttm, financials.raw("equity_parent"))
        elif factor_id == "np_qua_chg_acc":
            complete = (financials.ordinal - financials.ordinal.groupby(level="security_id").shift(7)).eq(7)
            value = financials.rolling(quarter, 8, _quadratic_trend_coefficient).where(complete)
        else:
            # The workbook's TTM growth leg is generic net profit; only the
            # quarter acceleration leg uses deducted parent profit.
            generic_ttm = financials.cumulative_ttm("net_profit")
            prior = financials.exact_quarter_lag(generic_ttm, 1)
            growth = safe_divide(generic_ttm - prior, prior.abs())
            prior_quarter = financials.exact_quarter_lag(quarter, 4)
            yoy = safe_divide(quarter - prior_quarter, prior_quarter.abs())
            acceleration = yoy - financials.exact_quarter_lag(yoy, 1)
            pair = _latest_report_frame(pd.concat([growth.rename("growth"), acceleration.rename("acceleration")], axis=1))
            pair = pair.replace([np.inf, -np.inf], np.nan).dropna()
            pair = pair.loc[pair.index.intersection(_filter_cross_section_population(pair.growth, context, date).index)]
            return (_tercile(pair.growth) - 1.0) * 3.0 + _tercile(pair.acceleration)
        return _filter_universe(_latest_report_frame(value), context, date)
    if policy == "deducted_profit_subtraction_clock_v1":
        if factor_id not in {"sug", "sue", "NET_PROFIT_AFTER_DED_NR_LP_TTM", "NET_PROFIT_AFTER_DED_NR_LP_QUA"}:
            raise ValueError("Deducted profit policy does not cover this factor")
        cumulative = financials.raw("net_profit_parent") - financials.raw("nonrecurring_profit_parent")
        quarter = cumulative.where(financials.quarter.eq(1), cumulative - financials.exact_quarter_lag(cumulative, 1))
        if factor_id.endswith("_TTM"):
            value = financials.cumulative_ttm("net_profit_parent") - financials.cumulative_ttm("nonrecurring_profit_parent")
        elif factor_id.endswith("_QUA"):
            value = quarter
        else:
            increment = quarter - financials.exact_quarter_lag(quarter, 4)
            prior = financials.exact_quarter_lag(increment, 1)
            complete = (financials.ordinal - financials.ordinal.groupby(level="security_id").shift(7)).eq(7)
            deviation = financials.std(prior, 8)
            value = safe_divide(increment - financials.mean(prior, 8), deviation.where(deviation.gt(0))).where(complete)
        return _filter_universe(_latest_report_frame(value), context, date)
    if policy == "statement_earnings_history_v1":
        if factor_id not in {"ebit_ttm", "ebitda_ttm", "ebit2rev_ttm", "ebitda2rev_ttm", "ebit2interest_ttm"}:
            raise ValueError("Statement earnings policy does not cover this factor")
        numerator = financials.cumulative_ttm("ebitda" if factor_id.startswith("ebitda") else "ebit")
        if factor_id.endswith("2rev_ttm"):
            value = safe_divide(numerator, financials.cumulative_ttm("operating_revenue"))
        elif factor_id == "ebit2interest_ttm":
            if parameters.get("interest_source") != "income.fin_exp_int_exp" or parameters.get("interest_field") != "interest_expense":
                raise ValueError("Interest coverage requires the fixed native income gross-expense route")
            value = safe_divide(numerator, financials.cumulative_ttm("interest_expense"))
        else:
            value = numerator
        return _filter_universe(_latest_report_frame(value), context, date)
    if policy == "cashflow_history_reviewed_v1":
        if factor_id not in {"CFOG", "fcff_ttm", "fcffp_ttm", "fcff_yoy", "fcffG", "fcff_cv", "cfo_cv", "cfo2rev_ttm"}:
            raise ValueError("Cashflow history policy does not cover this factor")
        cash = financials.cumulative_ttm("operating_cash_flow")
        free_cash = cash - financials.cumulative_ttm("capital_expenditure_cash_paid")
        if factor_id in {"fcff_cv", "cfo_cv"}:
            single = financials.single("operating_cash_flow")
            if factor_id == "fcff_cv":
                single = single - financials.single("capital_expenditure_cash_paid")
            complete = (financials.ordinal - financials.ordinal.groupby(level="security_id").shift(7)).eq(7)
            deviation = financials.std(single, 8)
            value = safe_divide(financials.mean(single, 8), deviation.where(deviation.gt(0))).where(complete)
        elif factor_id in {"CFOG", "fcff_yoy", "fcffG"}:
            current = cash if factor_id == "CFOG" else free_cash
            prior = financials.exact_quarter_lag(current, 4)
            value = safe_divide(current - prior, prior.abs() if factor_id == "fcff_yoy" else prior)
        elif factor_id == "fcffp_ttm":
            cap = financials.daily("total_market_cap")
            value = safe_divide(free_cash, cap.where(cap.gt(0)))
        elif factor_id == "cfo2rev_ttm":
            value = safe_divide(cash, financials.cumulative_ttm("operating_revenue")) * 100.0
        else:
            value = free_cash
        return _filter_universe(_latest_report_frame(value), context, date)
    if policy == "per_share_own_report_clock_v1":
        if factor_id not in PER_SHARE_PRIMITIVE_FIELDS:
            raise ValueError("Per-share report policy does not cover this factor")
        primitive = PER_SHARE_PRIMITIVE_FIELDS[factor_id]
        numerator = financials.raw(primitive) if factor_id == "up_per_share" else financials.cumulative_ttm(primitive)
        shares = financials.raw("total_shares")
        value = safe_divide(numerator, shares.where(shares.gt(0)))
        return _filter_universe(_latest_report_frame(value), context, date)
    if policy == "cashflow_equity_history_v1":
        family, _, suffix = factor_id.partition("_")
        if family not in {"fcffoe", "cfooe", "cfoe"} or suffix not in {"ttm", "ttm_diff4", "ttm_diff1", "qua", "qua_delta"}:
            raise ValueError("Cashflow equity policy does not cover this factor")
        quarterly = suffix.startswith("qua")
        field_name = "cash_net_increase" if family == "cfoe" else "operating_cash_flow"
        get_value = financials.single if quarterly else financials.cumulative_ttm
        numerator = get_value(field_name)
        if family == "fcffoe":
            numerator = numerator - get_value("capital_expenditure_cash_paid")
        equity = financials.raw("equity_parent")
        denominator = equity + financials.exact_quarter_lag(equity, 1 if quarterly else 4)
        value = safe_divide(numerator, denominator)
        if quarterly:
            value = value.where(financials.quarter.ne(4))
        if suffix.endswith(("diff4", "delta")):
            value = value - financials.exact_quarter_lag(value, 4)
        elif suffix.endswith("diff1"):
            value = value - financials.exact_quarter_lag(value, 1)
        return _filter_universe(_latest_report_frame(value), context, date)
    if policy == "unexpected_own_report_clock_v1":
        if factor_id not in UNEXPECTED_PRIMITIVE_FIELDS:
            raise ValueError("Unexpected-value report policy does not cover this factor")
        raw = _standardized_unexpected(financials, financials.single(UNEXPECTED_PRIMITIVE_FIELDS[factor_id]))
        return _filter_universe(_latest_report_frame(raw), context, date)
    if policy == "strict_working_capital_v6":
        if factor_id not in {"net_wc", "nwc2rev_ttm"}:
            raise ValueError("Strict working capital policy does not cover this factor")
        value = _strict_net_working_capital(financials)
        if factor_id == "nwc2rev_ttm":
            value = safe_divide(value, financials.cumulative_ttm("operating_revenue"))
        return _filter_universe(_latest_report_frame(value), context, date)
    if policy == "profit_growth_history_abs_v1":
        if factor_id not in {"npyoy", "np_yoy", "npqyoy", "npttm_qoq", "salettm_yoy"}:
            raise ValueError("Profit growth history policy does not cover this factor")
        field_name = "operating_revenue" if factor_id == "salettm_yoy" else "net_profit_parent"
        if field_name not in financials.frame:
            raise ValueError(f"Profit growth history policy requires {field_name}")
        value = financials.raw(field_name)
        if factor_id == "npqyoy":
            value = value.where(financials.quarter.eq(1), value - financials.exact_quarter_lag(value, 1))
        elif factor_id in {"npttm_qoq", "salettm_yoy"}:
            value = financials.cumulative_ttm(field_name)
        prior = financials.exact_quarter_lag(value, 1 if factor_id == "npttm_qoq" else 4)
        raw = safe_divide(value - prior, prior.abs())
        return _filter_universe(_latest_report_frame(raw), context, date)
    if policy in {"cashflow_growth_report_clock_v3", "npttm_yoy_abs_history_v2"}:
        permitted = {"cfo_yoy", "cfo_ttm_yoy"} if policy == "cashflow_growth_report_clock_v3" else {"npttm_yoy"}
        if factor_id not in permitted:
            raise ValueError("Growth history policy does not cover this factor")
        field_name = "net_profit" if factor_id == "npttm_yoy" else "operating_cash_flow"
        required = {field_name}
        if not required.issubset(financials.frame.columns):
            raise ValueError(f"Growth history policy requires {sorted(required)}")
        get_value = financials.raw if factor_id == "cfo_yoy" else financials.cumulative_ttm
        value = get_value(field_name)
        prior = financials.exact_quarter_lag(value, 4)
        raw = safe_divide(value - prior, prior.abs())
        return _filter_universe(_latest_report_frame(raw), context, date)
    if factor_id in GROWTH_TREND_IDS:
        raw, report_flags = _growth_trend_values_and_flags(financials, factor_id)
        current = _latest_report_frame(raw)
        current_flags = _latest_report_frame(report_flags)
        allowed = _allowed_universe(context, date)
        if allowed is not None:
            current = current[current.index.isin(allowed)]
            current_flags = current_flags[current_flags.index.isin(allowed)]
        financials._cache[f"quality_flags:{factor_id}"] = current_flags
        return current.astype(float)

    if factor_id in PB_ROE_REGRESSION_IDS:
        return _pb_roe_regression_cross_section(
            financials, factor_id, context, parameters,
        )

    if factor_id in OCFA_RESIDUAL_IDS:
        raw, report_flags = _ocfa_residual_and_flags(financials)
        current = _latest_report_frame(raw)
        current_flags = _latest_report_frame(report_flags)
        allowed = _allowed_universe(context, date)
        if allowed is not None:
            current = current[current.index.isin(allowed)]
            current_flags = current_flags[current_flags.index.isin(allowed)]
        financials._cache[f"quality_flags:{factor_id}"] = current_flags
        return current.astype(float)

    if factor_id in IDIOSYNCRATIC_PROFIT_IDS:
        raw, report_flags = _idiosyncratic_profit_residual_and_flags(
            financials, factor_id, context, parameters,
        )
        current = _latest_report_frame(raw)
        current_flags = _latest_report_frame(report_flags)
        allowed = _all_a_share_universe(context, date)
        if allowed is not None:
            current = current[current.index.isin(allowed)]
            current_flags = current_flags[current_flags.index.isin(allowed)]
        financials._cache[f"quality_flags:{factor_id}"] = current_flags
        return current.astype(float)

    if factor_id in ABNORMAL_CFO_IDS:
        raw, report_flags = _abnormal_cfo_residual_and_flags(
            financials, factor_id, context, parameters,
        )
        current = _latest_report_frame(raw)
        current_flags = _latest_report_frame(report_flags)
        allowed = _all_a_share_universe(context, date)
        if allowed is not None:
            current = current[current.index.isin(allowed)]
            current_flags = current_flags[current_flags.index.isin(allowed)]
        financials._cache[f"quality_flags:{factor_id}"] = current_flags
        return current.astype(float)

    if factor_id in SHORT_LOAN_LONG_INVESTMENT_IDS:
        raw, report_flags = _sfli_values_and_flags(
            financials, factor_id, parameters,
        )
        current = _latest_report_frame(raw)
        current_flags = _latest_report_frame(report_flags)
        allowed = _all_a_share_universe(context, date)
        if allowed is not None:
            current = current[current.index.isin(allowed)]
            current_flags = current_flags[current_flags.index.isin(allowed)]
        financials._cache[f"quality_flags:{factor_id}"] = current_flags
        return current.astype(float)

    if factor_id in NET_OPERATING_LIABILITY_IDS:
        raw, report_flags = _net_operating_liability_values_and_flags(
            financials, factor_id, context, parameters,
        )
        current = _latest_report_frame(raw)
        current_flags = _latest_report_frame(report_flags)
        allowed = _all_a_share_universe(context, date)
        if allowed is not None:
            current = current[current.index.isin(allowed)]
            current_flags = current_flags[current_flags.index.isin(allowed)]
        financials._cache[f"quality_flags:{factor_id}"] = current_flags
        return current.astype(float)

    if factor_id == "performances_trend":
        profit_ttm = financials.ttm("net_profit_parent_excl_nr")
        ttm_growth = financials.signed_growth(profit_ttm, 1)
        quarter_growth = financials.signed_growth(financials.single("net_profit_parent_excl_nr"), 4)
        acceleration = financials.diff(quarter_growth, 1)
        pair = pd.concat([
            financials.latest(ttm_growth).rename("ttm"),
            financials.latest(acceleration).rename("acceleration"),
        ], axis=1).dropna()
        pair = pair.loc[pair.index.intersection(_filter_universe(pair["ttm"], context, date).index)]
        return (_tercile(pair["ttm"]) - 1.0) * 3.0 + _tercile(pair["acceleration"])

    if factor_id in {"pb_roe_quantile_diff", "pb_roe_rank_dfif"}:
        price_to_book = financials.latest(_price_to_book(financials))
        roe_ttm = financials.latest(_return_on_stock(
            financials,
            financials.ttm("net_profit_parent"),
            _equity(financials),
            4,
        ))
        pair = pd.concat([
            price_to_book.rename("pb"), roe_ttm.rename("roe"),
        ], axis=1).dropna()
        allowed = _allowed_universe(context, date)
        if allowed is not None:
            pair = pair[pair.index.isin(allowed)]
        percentile = factor_id == "pb_roe_quantile_diff"
        pb_rank = pair["pb"].rank(method="average", pct=percentile)
        roe_rank = pair["roe"].rank(method="average", pct=percentile)
        return pb_rank - roe_rank

    if factor_id == "value_bivariate_rg":
        ep = financials.latest(safe_divide(
            financials.ttm("net_profit_parent"), financials.daily("total_market_cap"),
        ))
        bp = financials.latest(safe_divide(_equity(financials), financials.daily("total_market_cap")))
        pair = pd.concat([ep.rename("ep"), bp.rename("bp")], axis=1).dropna()
        allowed = _allowed_universe(context, date)
        if allowed is not None:
            pair = pair[pair.index.isin(allowed)]
        raw = (pair["ep"].rank(method="average", pct=True) + pair["bp"].rank(method="average", pct=True)) / 2.0
        return _rank_cross_section(raw, date)

    raw = _raw_formula(financials, factor_id)
    if factor_id == "epttm_rg_zscore":
        # Output eligibility must not remove the stock's rolling inputs.
        return _latest_report_frame(raw)
    current = _filter_universe(financials.latest(raw), context, date)
    if factor_id == "Rel_POP_Skew":
        industry = _industry_series(financials).reindex(current.index)
        valid = pd.concat([current.rename("value"), industry.rename("industry")], axis=1).dropna()
        return valid["value"] - valid.groupby("industry")["value"].transform("median")
    if factor_id in RANK_GAUSS_IDS:
        return _rank_cross_section(current, date)
    return current


def _rank_cross_section(values: pd.Series, date: pd.Timestamp) -> pd.Series:
    index = pd.MultiIndex.from_arrays(
        [[date] * len(values), values.index.astype(str)], names=INDEX_NAMES,
    )
    ranked = cross_sectional_rank_gauss(pd.Series(values.to_numpy(dtype=float), index=index))
    return ranked.droplevel("date")


def _calculation_grid(context: FactorContext, factor_id: str) -> pd.DatetimeIndex:
    targets = pd.DatetimeIndex(context.calculation_dates).normalize().sort_values().unique()
    if factor_id not in TIME_TRANSFORM_IDS or context.daily.empty:
        return targets
    dates = context.daily.index.get_level_values("date").normalize().sort_values().unique()
    return dates[dates <= targets.max()]


def _rolling_last_rank(values: pd.Series, window: int) -> pd.Series:
    def rank_last(array: np.ndarray) -> float:
        finite = array[np.isfinite(array)]
        if len(finite) < window:
            return np.nan
        last = finite[-1]
        less = np.sum(finite < last)
        equal = np.sum(finite == last)
        return (less + (equal + 1.0) / 2.0) / len(finite)

    return group_transform(
        values,
        lambda group: group.rolling(window, min_periods=window).apply(rank_last, raw=True),
    )


def _trading_session_quantile(values: pd.Series, calendar: pd.DatetimeIndex, window: int = 720) -> pd.Series:
    """Average percentile rank over consecutive market sessions, preserving holes."""
    calendar = pd.DatetimeIndex(calendar).normalize().sort_values().unique()
    parts = []
    for security, group in values.groupby(level="security_id", sort=False):
        history = group.droplevel("security_id").reindex(calendar)
        ranked = history.rolling(window, min_periods=window).rank(method="average", pct=True)
        ranked.index = pd.MultiIndex.from_product([calendar, [str(security)]], names=INDEX_NAMES)
        parts.append(ranked)
    return pd.concat(parts).sort_index() if parts else values.iloc[:0]


def _post_time_transform(values: pd.Series, factor_id: str, context: FactorContext) -> pd.Series:
    if factor_id == "epttm_rg_zscore":
        calendar = _calculation_grid(context, factor_id)
        parts = []
        for security, group in values.groupby(level="security_id", sort=False):
            history = group.droplevel("security_id").reindex(calendar)
            rolling = history.rolling(720, min_periods=720)
            deviation = rolling.std(ddof=1)
            zscore = (history - rolling.mean()) / deviation.where(deviation.ne(0))
            zscore.index = pd.MultiIndex.from_product([calendar, [str(security)]], names=INDEX_NAMES)
            parts.append(zscore)
        transformed = pd.concat(parts).sort_index()
        ranked_parts = []
        for date, group in transformed.groupby(level="date", sort=False):
            current = _filter_cross_section_population(group.droplevel("date"), context, date)
            ranked = _filter_universe(_rank_cross_section(current, date), context, date)
            ranked.index = pd.MultiIndex.from_product([[date], ranked.index], names=INDEX_NAMES)
            ranked_parts.append(ranked)
        return pd.concat(ranked_parts).sort_index()
    if factor_id in {"EPTTM_3y_quantile", "CFPTTM_3y_quantile", "PB_3y_quantile"}:
        return _rolling_last_rank(values, 720)
    return values


@registry.register("financial_statement")
def compute_financial_statement(context: FactorContext, spec: FactorSpec) -> pd.Series:
    factor_id = str(spec.parameters.get("formula_id", spec.factor_id))
    if factor_id not in FORMULA_IDS:
        raise ValueError(f"{spec.factor_id}: unsupported financial formula_id {factor_id!r}")
    if spec.parameters.get("history_policy") == "native_dividend_rank_v1":
        if factor_id not in {"divp_ttm_rg", "pdiv_ttm_rg"}:
            raise ValueError("Native dividend rank policy does not cover this factor")
        column = "dividend_yield_ttm_fraction"
        if column not in context.daily:
            raise ValueError("Dividend ranks require native dividend_yield_ttm_fraction")
        output = []
        for day in pd.DatetimeIndex(context.calculation_dates):
            values = context.daily[column].xs(day, level="date").replace([np.inf, -np.inf], np.nan)
            values = _filter_cross_section_population(values, context, day)
            if factor_id == "pdiv_ttm_rg":
                values = 1.0 / values.where(values.ne(0))
            ranked = _rank_cross_section(values, day)
            ranked.index = pd.MultiIndex.from_arrays([[day] * len(ranked), ranked.index], names=INDEX_NAMES)
            output.append(ranked)
        result = pd.concat(output) if output else pd.Series(dtype=float, index=pd.MultiIndex.from_arrays([[], []], names=INDEX_NAMES))
        result.name = spec.factor_id
        return result
    if spec.parameters.get("history_policy") == "valuation_720_trading_sessions_v1" and factor_id == "PB_3y_quantile":
        column = next((name for name in DAILY_ALIASES["price_to_book"] if name in context.daily), None)
        if column is None:
            raise ValueError("PB quantile requires native daily PB")
        calendar = _calculation_grid(context, factor_id)
        values = context.daily[column].replace([np.inf, -np.inf], np.nan)
        result = _trading_session_quantile(values, calendar)
        result.name = spec.factor_id
        return result
    source_observations = context.financial_observations
    if spec.parameters.get("history_policy") in {"sfli_identity_v2", "sfli_identity_v3"}:
        # These inputs are monetary amounts derived from complete same-record
        # equity identities by the reviewed source adapter, never share counts.
        names = {"equity_financing": "equity_financing_cny", "long_debt": "long_debt_cny",
                 "assets": "total_assets", "capex": "capital_expenditure_cash_paid",
                 "cfo": "operating_cash_flow", "disposal": "cash_received_disposal_long_term_assets"}
        source_observations = source_observations.copy()
        if spec.parameters.get("history_policy") == "sfli_identity_v3":
            expected_scope = spec.parameters["monetary_capital_identity_policy"]["scope_version"]
            if ("sfli_identity_scope_version" not in source_observations or
                    not source_observations.sfli_identity_scope_version.eq(expected_scope).all()):
                raise ValueError("SFLI requires source renormalization under the current report scope")
        source_observations["field"] = source_observations.field.replace(names)
        source_observations = source_observations[source_observations.field.isin(names.values())]
        if "source_unit" not in source_observations or not source_observations.source_unit.eq("CNY").all():
            raise ValueError("SFLI identity inputs require explicit CNY source units")
    observations = _normalize_observations(source_observations)
    if spec.parameters.get("history_policy") == "ocfa_operating_cost_fixed_assets_clock_v1":
        if factor_id != "ocfa_res" or spec.parameters.get("source_fields") != ["operating_cost", "fixed_assets"]:
            raise ValueError("OCFA requires the fixed operating-cost/fixed-assets report clock")
        observations = observations[observations.field.isin(spec.parameters["source_fields"])]
    if spec.parameters.get("history_policy") == "report_price_operating_profit_skew_v1":
        observations = observations[observations.field.eq("operating_profit")]
    if spec.parameters.get("history_policy") == "native_ratios_report_clock_v1":
        observations = observations[observations.field.isin({"ROE", "quick_ratio", "__report_clock"})]
    if spec.parameters.get("history_policy") in {"operating_daa_expense_vintage_v2", "operating_leverage_expense_vintage_v3", "operating_leverage_current_assets_v2", "operating_daa_disclosed_components_v1", "operating_turnover_components_v1", "operating_margin_total_revenue_v2", "financial_changes_exact_quarters_v1", "financial_core_same_report_v1", "financial_remaining_disclosed_history_v1", "financial_extended_components_v1", "valuation_disclosed_rank_v1", "fcff_quarter_workbook_ev_v1", "pb_roe_current_report_ols_v2", "tax_payable_current_report_zscore_v1", "tax_payable_total_equity_zscore_v2", "fcff_ttm_strict_enterprise_v1", "capex_ten_complete_years_v2", "quarter_roe_parent_rank_v1", "valuation_720_trading_sessions_v1", "nol2_exact_quarter_model_v2", "roe_ttm_current_report_rank_v1", "nol_current_balance_v1"}:
        observations = observations[observations.field.isin(spec.parameters["source_fields"])]
    if spec.parameters.get("history_policy") == "authorized_four_field_expense_capital_v1":
        observations = observations[observations.field.isin({"rd_expense", "selling_expense", "management_expense"}) & observations.report_period.ge(pd.Timestamp("2007-01-01"))]
    if spec.parameters.get("history_policy") == "cumulative_rd_annual_history_v1":
        observations = observations[observations.field.isin({"rd_expense", "selling_expense", "management_expense", "total_assets"}) & observations.report_period.ge(pd.Timestamp("2007-01-01"))]
    if spec.parameters.get("history_policy") == "capital_ratios_disclosed_components_v1":
        observations = observations[observations.field.isin(spec.parameters["source_fields"])]
    if spec.parameters.get("history_policy") == "deducted_growth_subtraction_clock_v1":
        observations = observations[observations.field.isin({"net_profit_parent", "nonrecurring_profit_parent", "equity_parent"} | ({"net_profit"} if factor_id == "performances_trend" else set()))]
    if spec.parameters.get("history_policy") == "deducted_profit_subtraction_clock_v1":
        observations = observations[observations.field.isin({"net_profit_parent", "nonrecurring_profit_parent"})]
    if spec.parameters.get("history_policy") == "statement_earnings_history_v1":
        source_fields = set(STATEMENT_EARNINGS_SOURCE_FIELDS)
        if factor_id == "ebit2interest_ttm":
            if spec.parameters.get("interest_source") != "income.fin_exp_int_exp" or spec.parameters.get("interest_field") != "interest_expense":
                raise ValueError("Interest coverage requires the fixed native income gross-expense route")
            source_fields.remove("interst_income")
            source_fields.add("interest_expense")
        observations = observations[observations.field.isin(source_fields)]
    if spec.parameters.get("history_policy") == "cashflow_history_reviewed_v1":
        observations = observations[observations.field.isin(CASHFLOW_HISTORY_SOURCE_FIELDS)]
    if spec.parameters.get("history_policy") == "per_share_own_report_clock_v1":
        primitive = PER_SHARE_PRIMITIVE_FIELDS.get(factor_id)
        if primitive is None:
            raise ValueError("Per-share report policy does not cover this factor")
        observations = observations[observations.field.isin([primitive, "total_shares"])]
    if spec.parameters.get("history_policy") == "cashflow_equity_history_v1":
        observations = observations[observations.field.isin(CASHFLOW_EQUITY_SOURCE_FIELDS)]
    if spec.parameters.get("history_policy") == "unexpected_own_report_clock_v1":
        primitive = UNEXPECTED_PRIMITIVE_FIELDS.get(factor_id)
        if primitive is None:
            raise ValueError("Unexpected-value report policy does not cover this factor")
        observations = observations[observations.field.eq(primitive)]
    if spec.parameters.get("history_policy") == "strict_working_capital_v6":
        fields = spec.parameters.get("source_fields")
        if not isinstance(fields, list) or not fields:
            raise ValueError("Strict working capital policy requires explicit source_fields")
        observations = observations[observations.field.isin(fields)]
    if spec.parameters.get("history_policy") == "profit_growth_history_abs_v1":
        # The frozen source advances its report clock on these two income fields.
        observations = observations[observations.field.isin({"net_profit_parent", "operating_revenue"})]
    dates = _calculation_grid(context, factor_id)
    results: list[pd.Series] = []
    quality_results: list[pd.DataFrame] = []
    for calculation_date in dates:
        cutoff = (
            calculation_date + pd.to_timedelta(context.pit_calculation_time)
            if context.pit_allow_release_day else calculation_date
        )
        quarterly = _visible_quarterly_frame(observations, cutoff, expense_vintage=spec.parameters.get("history_policy") in {"operating_leverage_expense_vintage_v3", "operating_daa_expense_vintage_v2"})
        if quarterly.empty:
            continue
        financials = QuarterlyFinancials(quarterly, context, calculation_date)
        current = _current_cross_section(financials, factor_id, context, spec.parameters)
        if factor_id in QUALITY_FLAG_IDS:
            current_flags = financials._cache[f"quality_flags:{factor_id}"].copy()
            current_flags.insert(0, "factor_id", factor_id)
            current_flags.index = pd.MultiIndex.from_arrays(
                [
                    [calculation_date] * len(current_flags),
                    current_flags.index.astype(str),
                ],
                names=INDEX_NAMES,
            )
            quality_results.append(current_flags)
        if factor_id == "pb_roe_reg_rg":
            raw_current = financials._cache[f"raw_factor:{factor_id}"].copy()
            raw_current.index = pd.MultiIndex.from_arrays(
                [
                    [calculation_date] * len(raw_current),
                    raw_current.index.astype(str),
                ],
                names=INDEX_NAMES,
            )
            context.cache.setdefault(f"raw_factor_parts:{factor_id}", []).append(raw_current)
        current.index = pd.MultiIndex.from_arrays(
            [[calculation_date] * len(current), current.index.astype(str)], names=INDEX_NAMES,
        )
        results.append(current.astype(float))
    if factor_id in QUALITY_FLAG_IDS:
        quality_key = f"quality_flags:{factor_id}"
        if quality_results:
            context.cache[quality_key] = pd.concat(quality_results).sort_index()
        else:
            context.cache[quality_key] = pd.DataFrame(
                index=pd.MultiIndex.from_arrays([[], []], names=INDEX_NAMES),
            )
    if factor_id == "pb_roe_reg_rg":
        raw_parts = context.cache.pop(f"raw_factor_parts:{factor_id}", [])
        context.cache[f"raw_factor:{factor_id}"] = (
            pd.concat(raw_parts).sort_index()
            if raw_parts
            else pd.Series(
                dtype=float,
                index=pd.MultiIndex.from_arrays([[], []], names=INDEX_NAMES),
            )
        )
    if not results:
        index = pd.MultiIndex.from_arrays([[], []], names=INDEX_NAMES)
        return pd.Series(dtype=float, index=index, name=spec.factor_id)
    output = pd.concat(results).sort_index().replace([np.inf, -np.inf], np.nan)
    output = _trading_session_quantile(output, _calculation_grid(context, factor_id)) if spec.parameters.get("history_policy") == "valuation_720_trading_sessions_v1" else _post_time_transform(output, factor_id, context)
    output.name = spec.factor_id
    return output.sort_index()
