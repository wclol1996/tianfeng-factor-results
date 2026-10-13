from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
import re
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from factor_engine.catalog import FactorSpec
from factor_engine.config import PriceAdjustment
from factor_engine.data.base import DataProvider, FactorContext


APPROVED_DATA_ROOT = Path(
    "/Users/wclol/Documents/ChatGPT/Factor mining/outputs/"
    "gildata_tianfeng_2011_present"
).resolve()

HIGH_FREQUENCY_IMPLEMENTATIONS = frozenset({"intraday_microstructure", "apm"})
HIGH_FREQUENCY_FIELDS = frozenset({
    "1分钟复权价格",
    "1分钟成交量",
    "120分钟复权收盘价",
    "中证全指120分钟收盘价",
    "日内波动率",
})

# Unit conversions are part of the provider contract, not factor-specific
# post-processing.  Keep them explicit in lineage so downstream review never
# has to infer whether a Gildata percentage column is already a decimal ratio.
# The source units come from the exact vendor column labels; the canonical
# units come from the factor-planning data contract.
CANONICAL_UNIT_CONTRACTS: dict[str, dict[str, object]] = {
    "volume": {
        "source_dataset": "StockDailyQuote",
        "source_field": "成交量（万股）",
        "source_unit": "10,000 shares",
        "canonical_unit": "shares",
        "multiplier": 10_000.0,
    },
    "amount": {
        "source_dataset": "StockDailyQuote",
        "source_field": "成交额（万元）",
        "source_unit": "CNY 10,000",
        "canonical_unit": "CNY",
        "multiplier": 10_000.0,
    },
    "turnover_rate": {
        "source_dataset": "StockDailyQuote",
        "source_field": "换手率（%）",
        "source_unit": "percent points",
        "canonical_unit": "ratio",
        "multiplier": 0.01,
    },
    "total_market_cap": {
        "source_dataset": "StockValueAnalysis",
        "source_field": "总市值（亿元）",
        "source_unit": "CNY 100,000,000",
        "canonical_unit": "CNY",
        "multiplier": 100_000_000.0,
    },
    "float_market_cap": {
        "source_dataset": "StockValueAnalysis",
        "source_field": "流通市值（亿元）",
        "source_unit": "CNY 100,000,000",
        "canonical_unit": "CNY",
        "multiplier": 100_000_000.0,
    },
}

SUPPORTED_MAPPING_CATEGORIES = frozenset({
    "daily_quote",
    "valuation",
    "index_quote",
    "financial_statement",
    "financial_analysis",
    "financial_derived",
    "quarterly_financial",
    "ttm_derived",
    "pit_financial",
    "factor_derived",
    "consensus",
    "consensus_derived",
    "master",
    "status",
    "disclosure",
    "fund_flow",
})

# These fields have a file or apparent name match, but the downloaded package
# cannot establish the semantics needed by the factor engine.  They must never
# be silently substituted with a similarly named column.
MANUAL_REVIEW_FIELDS: dict[str, str] = {
    "自由流通市值": "缺少可确认的自由流通股本口径；不用流通市值代替",
    "自由流通股本": "Gildata 数据包未提供已确认的自由流通股本",
    "股票状态": "停牌状态可直读，但历史 ST 状态未提供",
    "涨跌停状态": "数据包没有供应商历史涨跌停标记；不从复权价格反推",
    "涨跌停价": "历史板块、ST 及新股无涨跌幅规则未形成经验证状态表",
    "行业分类": "现有行业表无生效起止日，不能当作 PIT 行业",
    "中信一级行业": "只有当前行业标签，无 PIT 生效日",
    "中信二级行业": "只有当前行业标签，无 PIT 生效日",
    "业绩预告日": "结构化业绩预告缺公告日，尚未与公告表人工核对匹配",
    "业绩预告公告日": "结构化业绩预告缺公告日，尚未与公告表人工核对匹配",
    "股东质押比例": "原表是单笔质押比例，因子所需的汇总/时点口径未确认",
    "一致预期每股现金流": "汇总与明细分区都未提供可核对的 CFPS",
    "Barra book_to_price": "聚源价值风格因子不保证等于指定 Barra 版本",
    "Barra size": "聚源市值风格因子不保证等于指定 Barra 版本",
    "size": "不读取未确认的 Gildata/Barra 同名风格因子；FF3 使用明确的市值构造",
    "一致预期记录-预测类型": "数据包只有明确预测年度，没有供应商预测类型原字段",
    "无涨跌幅限制状态": "历史新股/板块无涨跌幅限制状态未下载",
    "涨停状态": "未下载供应商历史涨停标记，不从价格反推",
    "跌停状态": "未下载供应商历史跌停标记，不从价格反推",
    "带息债务": "只能由已下载的短期借款、长期借款和应付债券构造，其他有息科目口径待核",
    "自由现金流": "企业自由现金流与股东自由现金流口径待确认",
    "净利润TTM（口径待确认）": "未确认应使用含或不含少数股东损益的净利润 TTM",
    "研发投入总额": "财务报表分区仅有研发费用，无可核对的研发投入总额",
    "费用化研究支出": "数据包无研发支出费用化/资本化拆分",
    "资本化研究支出": "数据包无研发支出费用化/资本化拆分",
    "管理费用中的研发费用": "数据包无该明细科目",
    "广告宣传推广费(销售费用)": "数据包无该销售费用明细",
    "工资薪酬(销售费用)": "数据包无该销售费用明细",
    "工资薪酬(管理费用)": "数据包无该管理费用明细",
}

EXPLICIT_FIELD_REVIEWS: dict[str, tuple[str, str, str, str]] = {
    "复权收盘价": (
        "CONFIRMED", "StockDailyQuote", "收盘价（元）",
        "PriceAdjustment 显式选择 front/back 分区",
    ),
    "PB": ("CONFIRMED", "valuation", "市净率PB(MRQ)", "显式对应数据库 PB(MRQ)"),
    "book_to_price": (
        "DERIVED_REVIEW",
        "derived/daily_factor_inputs",
        "book_to_price_归母权益口径",
        "使用披露日 PIT 归母权益/总市值，不冒充 Barra 商业版本",
    ),
    "公司成立日期": ("CONFIRMED", "CompanyBasicInfo", "成立日期", "公司主表静态字段"),
    "总股本": (
        "DERIVED_REVIEW", "StockValueAnalysis + StockDailyQuote",
        "总市值（亿元）/不复权收盘价（元）",
        "逐日构造总股本（股），避免用季度股本向前回填",
    ),
    "一致预期归母净利润": (
        "CONFIRMED", "ConsensusExpectation", "归属母公司净利润（万元）",
        "显式使用归属母公司口径，不用总净利润代替",
    ),
    "一致预期记录-证券代码": (
        "CONFIRMED", "ConsensusExpectation", "股票代码", "通过证券主表转为 security_id",
    ),
    "一致预期记录-预测年度": (
        "CONFIRMED", "ConsensusExpectation", "预测年度", "整数预测年度",
    ),
    "年报实际披露日": (
        "CONFIRMED", "FinanceReportDisclosure", "实际披露日",
        "仅选择 12-31 报告期；日期精度保守按当日收盘后可得",
    ),
    "归母净利润": (
        "CONFIRMED", "FinancialStatement", "NPParentCompanyOwners", "合并口径归母净利润",
    ),
    "横截面排名": (
        "DERIVED_REVIEW", "factor_engine", "cross-sectional rank", "因子公式内生计算，不是 Gildata 原始字段",
    ),
}

# Canonical names consumed by financial_statement.py.  Exact requested field
# aliases are added only for the current FactorSpec, which avoids multiplying
# every observation by all Chinese synonyms.
STATEMENT_CODE_CANONICAL: dict[str, str] = {
    "TotalAssets": "total_assets",
    "TotalLiability": "total_liabilities",
    "SEWithoutMI": "equity_parent",
    "Inventories": "inventory",
    "FixedAssets": "fixed_assets",
    "OtherReceivable": "other_receivables",
    "BillReceivable": "notes_receivable",
    "BillAccReceivable": "accounts_receivable_combined",
    "AccountReceivable": "accounts_receivable",
    "AdvancePayment": "prepayments",
    "OtherCurrentAssets": "other_current_assets",
    "NotesPayable": "notes_payable",
    "NotAccountsPayable": "accounts_payable_combined",
    "AccountsPayable": "accounts_payable",
    "SalariesPayable": "employee_compensation_payable",
    "TaxsPayable": "taxes_payable",
    "OtherPayable": "other_payables",
    "RetainedProfit": "undistributed_profit",
    "SurplusReserveFund": "surplus_reserve",
    "ContractLiability": "contract_liabilities",
    "AdvanceReceipts": "advance_receipts",
    "Cash": "cash",
    "MinorityInterests": "minority_interest",
    "OtherEquityinstruments": "other_equity_instruments",
    "PaidInCapital": "total_shares",
    "CapitalReserveFund": "capital_reserve",
    "LongtermLoan": "long_term_borrowing",
    "BondsPayable": "bonds_payable",
    "EndPeriodCashEquivalent": "cash_end",
    "OperatingRevenue": "operating_revenue",
    "SalesRevenue": "sales_revenue",
    "OperatingCost": "operating_cost",
    "OperatingProfit": "operating_profit",
    "NetProfit": "net_profit",
    "NPParentCompanyOwners": "net_profit_parent",
    "NetProfitCut": "net_profit_parent_excl_nr",
    "AdministrationExpense": "management_expense",
    "OperatingExpense": "selling_expense",
    "FinancialExpense": "financial_expense",
    "RAndD": "rd_expense",
    "NetOperateCashFlow": "operating_cash_flow",
    "OtherCashInRelatedOperate": "other_operating_cash_flow",
    "StaffBehalfPaid": "cash_paid_employees",
    "CashEquivalentIncrease": "cash_net_increase",
    "FixIntanOtherAssetAcquiCash": "capital_expenditure_cash_paid",
    "GoodsSaleServiceRenderCash": "cash_received_sales",
    "FixIntanOtherAssetDispoCash": "cash_received_disposal_long_term_assets",
}

DERIVED_COMPONENT_CODES: dict[str, tuple[str, ...]] = {
    "带息债务": ("ShortTermLoan", "LongtermLoan", "BondsPayable"),
    "无息流动负债(元)": ("TotalCurrentLiability", "ShortTermLoan"),
    "无息流动负债": ("TotalCurrentLiability", "ShortTermLoan"),
    "营运资本": ("TotalCurrentAssets", "TotalCurrentLiability"),
    "净营运资本": ("TotalCurrentAssets", "TotalCurrentLiability"),
    "折旧与摊销": (),
}

TTM_FIELD_COLUMNS: dict[str, str] = {
    "EBIT TTM": "EBIT_TTM",
    "EBITDA TTM": "EBITDA_TTM",
    "扣非净利润TTM": "NetProfitCut_TTM",
    "经营活动产生的现金流量净额TTM": "NetOperateCashFlow_TTM",
    "营业利润TTM": "OperatingProfit_TTM",
    "营业利润率TTM": "营业利润率TTM",
    "营业利润比营业总收入(TTM)": "营业利润比营业总收入TTM",
    "营业利润比营业总收入TTM": "营业利润比营业总收入TTM",
    "营业收入TTM": "OperatingRevenue_TTM",
    "归属母公司净利润TTM": "NPParentCompanyOwners_TTM",
    "归母净利润TTM": "NPParentCompanyOwners_TTM",
    "净利润TTM": "NetProfit_TTM",
    "财务费用(TTM)": "FinancialExpense_TTM",
    "财务费用TTM": "FinancialExpense_TTM",
    "销售毛利率TTM": "销售毛利率TTM",
}

ANALYSIS_FIELD_COLUMNS: dict[str, str] = {
    "速动比率": "QuickRatio",
}

CONSENSUS_FIELD_COLUMNS: dict[str, str] = {
    "operating_revenue": "营业收入（万元）",
    "operating_profit": "营业利润（万元）",
    "net_profit_parent": "归属母公司净利润（万元）",
    "eps": "每股收益EPS（元/股）",
    "roe": "净资产收益率ROE（%）",
    "pe": "市盈率PE",
    "pb": "市净率PB",
}


@dataclass(frozen=True)
class FieldReview:
    factor_id: str
    requested_field: str
    review_status: str
    source_api: str
    source_field: str
    mapping_status: str
    note: str


class ManualReviewRequiredError(ValueError):
    """Raised before I/O-heavy calculation when a mapping is not confirmed."""

    def __init__(self, factor_id: str, reviews: pd.DataFrame):
        self.factor_id = factor_id
        self.reviews = reviews.copy()
        fields = reviews.loc[
            reviews["review_status"].isin({"MANUAL_REVIEW", "NO_MATCH"}),
            "requested_field",
        ].astype(str).tolist()
        super().__init__(
            f"{factor_id}: manual Gildata field review required: {fields}"
        )


def _month_starts(start: date, end: date) -> Iterable[tuple[int, int]]:
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        yield year, month
        month += 1
        if month == 13:
            year, month = year + 1, 1


def _as_numeric(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")


class GildataParquetProvider(DataProvider):
    """Read-only adapter for the approved local Gildata Parquet package.

    The adapter intentionally has no API/network path and refuses any data root
    other than :data:`APPROVED_DATA_ROOT`.  Ambiguous source fields are surfaced
    through ``field_mapping_audit`` and, by default, stop the factor before its
    large input partitions are read.
    """

    def __init__(self, config: Mapping[str, object] | None = None) -> None:
        raw = dict(config or {})
        configured = Path(str(raw.get("data_root", APPROVED_DATA_ROOT))).expanduser().resolve()
        if configured != APPROVED_DATA_ROOT:
            raise ValueError(
                "GildataParquetProvider only permits the approved read-only data root: "
                f"{APPROVED_DATA_ROOT}"
            )
        if not configured.is_dir():
            raise FileNotFoundError(f"Gildata data root does not exist: {configured}")
        policy = str(raw.get("manual_review_policy", "raise")).strip().lower()
        if policy not in {"raise", "mark"}:
            raise ValueError("provider.manual_review_policy must be 'raise' or 'mark'")

        self.root = configured
        self.manual_review_policy = policy
        self._mapping = pd.read_csv(self.root / "field_mapping.csv", encoding="utf-8-sig")
        self._mapping["requested_field"] = self._mapping["requested_field"].astype(str).str.strip()
        self._master_cache: pd.DataFrame | None = None
        self._disclosure_cache: pd.DataFrame | None = None
        self._calendar_cache: pd.DatetimeIndex | None = None
        self._daily_cache: dict[tuple[date, date, str, bool, bool], pd.DataFrame] = {}
        self._status_cache: dict[tuple[date, date], pd.DataFrame] = {}
        self._financial_cache: dict[tuple[str, date, date, tuple[str, ...]], pd.DataFrame] = {}
        self._runtime_issues: list[dict[str, object]] = []
        self._code_to_id = self._load_code_map()
        self._requested_to_statement_codes = self._build_requested_statement_codes()

    def _record_issue(self, area: str, status: str, detail: str, **metadata: object) -> None:
        row = {"area": area, "status": status, "detail": detail, **metadata}
        if row not in self._runtime_issues:
            self._runtime_issues.append(row)

    def data_quality_issues(self) -> pd.DataFrame:
        fixed = [
            {
                "area": "security_status.is_st",
                "status": "MANUAL_REVIEW",
                "detail": "历史 ST 状态缺失；日行情中的股票名称是当前名称，不能回填历史",
            },
            {
                "area": "security_status.price_limit",
                "status": "MANUAL_REVIEW",
                "detail": "历史涨跌停与无涨跌幅限制状态未下载，保留为缺失",
            },
            {
                "area": "security_master.delist_date",
                "status": "MANUAL_REVIEW",
                "detail": "主表有终止上市标记但没有退市生效日，不用最后行情日猜测",
            },
            {
                "area": "financial_revision_history",
                "status": "MANUAL_REVIEW",
                "detail": "报表有实际披露日，但无供应商历史修订时间和版本链",
            },
            {
                "area": "industry_pit",
                "status": "MANUAL_REVIEW",
                "detail": "行业归属表无生效起止日，不注入因子的 PIT 行业字段",
            },
            {
                "area": "consensus_coverage",
                "status": "COVERAGE_LIMIT",
                "detail": "汇总从 2021-09 开始；明细从 2024-12 开始",
            },
        ]
        return pd.DataFrame([*fixed, *self._runtime_issues])

    @staticmethod
    def unit_conversion_contract() -> pd.DataFrame:
        """Return the source-to-canonical unit lineage used by this adapter."""

        rows = [
            {"canonical_field": field, **contract}
            for field, contract in CANONICAL_UNIT_CONTRACTS.items()
        ]
        return pd.DataFrame(rows, columns=[
            "canonical_field", "source_dataset", "source_field", "source_unit",
            "canonical_unit", "multiplier",
        ])

    def _load_code_map(self) -> dict[str, str]:
        master = pd.read_csv(
            self.root / "security_master_all.csv",
            encoding="utf-8-sig",
            dtype={"security_code": "string", "ref_code": "string"},
            usecols=["security_code", "ref_code"],
        )
        master["security_code"] = master["security_code"].str.slice(0, 6).str.zfill(6)
        if master["security_code"].duplicated().any():
            duplicates = master.loc[master["security_code"].duplicated(), "security_code"].unique()
            raise ValueError(f"security master has duplicate six-digit codes: {duplicates[:10].tolist()}")
        return master.set_index("security_code")["ref_code"].astype(str).to_dict()

    def _normalize_security_id(self, values: pd.Series, *, area: str) -> pd.Series:
        codes = values.astype("string").str.extract(r"(\d{6})", expand=False).str.zfill(6)
        mapped = codes.map(self._code_to_id).astype("string")
        missing = mapped.isna() & codes.notna()
        if missing.any():
            self._record_issue(
                area,
                "UNMAPPED_SECURITY",
                "rows were dropped because their six-digit code is absent from security_master_all.csv",
                rows=int(missing.sum()),
                examples="|".join(codes[missing].drop_duplicates().head(10).astype(str)),
            )
        return mapped

    def _build_requested_statement_codes(self) -> dict[str, tuple[str, ...]]:
        result: dict[str, set[str]] = {}
        for row in self._mapping.itertuples(index=False):
            api = str(row.source_api)
            source = str(row.source_field).strip()
            if "FinancialStatement" not in api:
                continue
            tokens = re.findall(r"[A-Z][A-Za-z0-9_]+", source)
            if not tokens:
                continue
            result.setdefault(str(row.requested_field).strip(), set()).update(tokens)
        for field, codes in DERIVED_COMPONENT_CODES.items():
            result.setdefault(field, set()).update(codes)
        return {key: tuple(sorted(value)) for key, value in result.items()}

    def field_mapping_audit(self, spec: FactorSpec) -> pd.DataFrame:
        reviews: list[FieldReview] = []
        if spec.implementation in HIGH_FREQUENCY_IMPLEMENTATIONS or spec.factor_id == "resvol60_ff3_iv":
            reviews.append(FieldReview(
                spec.factor_id,
                "__factor_implementation__",
                "SKIPPED_HF",
                "",
                "",
                "skipped",
                "因子依赖分钟/高频数据，按本轮要求不计算",
            ))

        for field in spec.required_fields:
            if field in HIGH_FREQUENCY_FIELDS:
                reviews.append(FieldReview(
                    spec.factor_id, field, "SKIPPED_HF", "", "", "skipped",
                    "该字段依赖分钟数据",
                ))
                continue
            if field in MANUAL_REVIEW_FIELDS:
                matched = self._mapping[self._mapping["requested_field"].eq(field)]
                source_api = "|".join(matched["source_api"].dropna().astype(str).unique())
                source_field = "|".join(matched["source_field"].dropna().astype(str).unique())
                mapping_status = "|".join(matched["status"].dropna().astype(str).unique()) or "unmapped"
                reviews.append(FieldReview(
                    spec.factor_id, field, "MANUAL_REVIEW", source_api, source_field,
                    mapping_status, MANUAL_REVIEW_FIELDS[field],
                ))
                continue
            if field in EXPLICIT_FIELD_REVIEWS:
                status, api, source, note = EXPLICIT_FIELD_REVIEWS[field]
                reviews.append(FieldReview(
                    spec.factor_id, field, status, api, source,
                    "direct" if status == "CONFIRMED" else "derived", note,
                ))
                continue

            matched = self._mapping[self._mapping["requested_field"].eq(field)]
            if matched.empty:
                reviews.append(FieldReview(
                    spec.factor_id, field, "NO_MATCH", "", "", "unmapped",
                    "因子字段名与 field_mapping.csv 无精确对应，禁止模糊猜测",
                ))
                continue

            statuses = set(matched["status"].dropna().astype(str).str.lower())
            categories = set(matched["category"].dropna().astype(str))
            source_api = "|".join(matched["source_api"].dropna().astype(str).unique())
            source_field = "|".join(matched["source_field"].dropna().astype(str).unique())
            if "skipped" in statuses:
                review_status = "SKIPPED_HF"
                note = "field_mapping.csv 明确标记本轮跳过"
            elif "partial" in statuses or not categories <= SUPPORTED_MAPPING_CATEGORIES:
                review_status = "MANUAL_REVIEW"
                note = "名称可对应，但原始字段/口径或适配器支持不完整"
            elif "derived" in statuses:
                review_status = "DERIVED_REVIEW"
                note = "使用 field_mapping.csv 中明示的派生公式"
            else:
                review_status = "CONFIRMED"
                note = "精确字段匹配"
            reviews.append(FieldReview(
                spec.factor_id,
                field,
                review_status,
                source_api,
                source_field,
                "|".join(sorted(statuses)),
                note,
            ))
        return pd.DataFrame([review.__dict__ for review in reviews], columns=[
            "factor_id", "requested_field", "review_status", "source_api",
            "source_field", "mapping_status", "note",
        ])

    def _enforce_mapping_policy(self, spec: FactorSpec, audit: pd.DataFrame) -> None:
        if not audit.empty and audit["review_status"].eq("SKIPPED_HF").any():
            fields = audit.loc[audit["review_status"].eq("SKIPPED_HF"), "requested_field"].tolist()
            raise NotImplementedError(
                f"{spec.factor_id}: high-frequency/minute inputs are intentionally disabled: {fields}"
            )
        blockers = audit["review_status"].isin({"MANUAL_REVIEW", "NO_MATCH"})
        if self.manual_review_policy == "raise" and blockers.any():
            raise ManualReviewRequiredError(spec.factor_id, audit)

    def _read_monthly(
        self,
        relative: str,
        start: date,
        end: date,
        columns: list[str],
        date_column: str,
    ) -> pd.DataFrame:
        pieces: list[pd.DataFrame] = []
        for year, month in _month_starts(start, end):
            path = self.root / relative / f"year={year}" / f"month={month:02d}" / "data.parquet"
            if path.is_file():
                pieces.append(pd.read_parquet(path, columns=columns))
        if not pieces:
            return pd.DataFrame(columns=columns)
        frame = pd.concat(pieces, ignore_index=True)
        frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce").dt.normalize()
        begin, finish = pd.Timestamp(start), pd.Timestamp(end)
        return frame[frame[date_column].between(begin, finish)].copy()

    def _read_daily_quote_source(self, start: date, end: date, restoration: str) -> pd.DataFrame:
        columns = [
            "股票名称", "股票代码", "交易日", "复权方式", "今开盘（元）",
            "最高价（元）", "最低价（元）", "收盘价（元）", "均价（元）",
            "成交量（万股）", "成交笔数（笔）", "成交额（万元）", "换手率（%）", "是否停牌",
        ]
        frame = self._read_monthly(
            f"partitions/daily_quote/restoration={restoration}",
            start,
            end,
            columns,
            "交易日",
        )
        if frame.empty:
            return frame
        expected = {"front": "前复权", "back": "后复权", "unadjusted": "不复权"}[restoration]
        unexpected = set(frame["复权方式"].dropna().astype(str)) - {expected}
        if unexpected:
            raise ValueError(
                f"daily quote restoration={restoration} contains unexpected labels: {sorted(unexpected)}"
            )
        frame["security_id"] = self._normalize_security_id(
            frame["股票代码"], area=f"daily_quote.{restoration}",
        )
        frame = frame.dropna(subset=["交易日", "security_id"])
        if frame.duplicated(["交易日", "security_id"]).any():
            raise ValueError(f"daily quote restoration={restoration} has duplicate date/security keys")
        return frame

    def _read_valuation(self, start: date, end: date) -> pd.DataFrame:
        columns = [
            "股票代码", "交易日", "总市值（亿元）", "流通市值（亿元）",
            "市盈率PE(TTM)", "市净率PB(MRQ)", "市净率PB(LF)", "市现率PCF(TTM)",
            "市销率PS(TTM)", "滚动股息率",
        ]
        frame = self._read_monthly(
            "partitions/valuation", start, end, columns, "交易日",
        )
        if frame.empty:
            return frame
        frame["security_id"] = self._normalize_security_id(frame["股票代码"], area="valuation")
        frame = frame.dropna(subset=["交易日", "security_id"])
        if frame.duplicated(["交易日", "security_id"]).any():
            raise ValueError("valuation data has duplicate date/security keys")
        return frame

    def _read_daily_factor_inputs(self, start: date, end: date) -> pd.DataFrame:
        pieces: list[pd.DataFrame] = []
        columns = ["股票代码", "交易日", "book_to_price_归母权益口径"]
        for year in range(start.year, end.year + 1):
            path = self.root / "derived" / "daily_factor_inputs" / f"year={year}" / "data.parquet"
            if path.is_file():
                pieces.append(pd.read_parquet(path, columns=columns))
        if not pieces:
            return pd.DataFrame(columns=["date", "security_id", "book_to_price"])
        frame = pd.concat(pieces, ignore_index=True)
        frame["date"] = pd.to_datetime(frame["交易日"], errors="coerce").dt.normalize()
        frame["security_id"] = self._normalize_security_id(frame["股票代码"], area="daily_factor_inputs")
        frame["book_to_price"] = pd.to_numeric(frame["book_to_price_归母权益口径"], errors="coerce")
        frame = frame[frame["date"].between(pd.Timestamp(start), pd.Timestamp(end))]
        return frame[["date", "security_id", "book_to_price"]].dropna(
            subset=["date", "security_id"]
        ).drop_duplicates(["date", "security_id"], keep="last")

    def _read_large_order_inflow_rate(self, start: date, end: date) -> pd.DataFrame:
        columns = ["股票代码", "交易日期", "大单净买入额(万元)", "总成交额(万元)"]
        frame = self._read_monthly(
            "partitions/fund_flow", start, end, columns, "交易日期",
        )
        if frame.empty:
            return pd.DataFrame(columns=["date", "security_id", "large_order_inflow_rate"])
        frame["security_id"] = self._normalize_security_id(frame["股票代码"], area="fund_flow")
        numerator = pd.to_numeric(frame["大单净买入额(万元)"], errors="coerce")
        denominator = pd.to_numeric(frame["总成交额(万元)"], errors="coerce")
        frame["large_order_inflow_rate"] = numerator.divide(denominator.where(denominator.ne(0)))
        return frame.rename(columns={"交易日期": "date"})[[
            "date", "security_id", "large_order_inflow_rate",
        ]].dropna(subset=["date", "security_id"]).drop_duplicates(
            ["date", "security_id"], keep="last",
        )

    def _load_daily(
        self,
        spec: FactorSpec,
        start: date,
        end: date,
        adjustment: PriceAdjustment,
    ) -> pd.DataFrame:
        need_book_to_price = spec.implementation in {"ff3_residual", "financial_statement"}
        need_fund_flow = spec.factor_id == "resvol60_ff3_lir"
        key = (start, end, adjustment.value, need_book_to_price, need_fund_flow)
        cached = self._daily_cache.get(key)
        if cached is not None:
            return cached.copy()

        restoration = {
            PriceAdjustment.BACKWARD: "back",
            PriceAdjustment.FORWARD: "front",
            PriceAdjustment.RAW: "unadjusted",
        }[adjustment]
        selected = self._read_daily_quote_source(start, end, restoration)
        if selected.empty:
            empty_index = pd.MultiIndex.from_arrays([[], []], names=["date", "security_id"])
            return pd.DataFrame(index=empty_index)
        raw = self._read_daily_quote_source(start, end, "unadjusted")

        daily = selected.rename(columns={
            "交易日": "date",
            "今开盘（元）": "open",
            "最高价（元）": "high",
            "最低价（元）": "low",
            "收盘价（元）": "close",
            "均价（元）": "vwap",
            "成交量（万股）": "volume",
            "成交笔数（笔）": "trade_count",
            "成交额（万元）": "amount",
            "换手率（%）": "turnover_rate",
        })[[
            "date", "security_id", "open", "high", "low", "close", "vwap",
            "volume", "trade_count", "amount", "turnover_rate",
        ]].copy()
        daily["_selected_is_suspended"] = self._yes_no(selected["是否停牌"]).array
        raw_prices = raw.rename(columns={
            "交易日": "date",
            "今开盘（元）": "raw_open",
            "最高价（元）": "raw_high",
            "最低价（元）": "raw_low",
            "收盘价（元）": "raw_close",
            "是否停牌": "_raw_is_suspended",
        })[["date", "security_id", "raw_open", "raw_high", "raw_low", "raw_close", "_raw_is_suspended"]]
        raw_prices["_raw_is_suspended"] = self._yes_no(raw_prices["_raw_is_suspended"]).array
        daily = daily.merge(raw_prices, on=["date", "security_id"], how="left", validate="1:1")
        _as_numeric(daily, [
            "open", "high", "low", "close", "vwap", "volume", "trade_count",
            "amount", "turnover_rate", "raw_open", "raw_high", "raw_low", "raw_close",
        ])
        suspended = daily.pop("_selected_is_suspended").fillna(False) | daily.pop(
            "_raw_is_suspended"
        ).fillna(False)
        if suspended.any():
            price_columns = [
                "open", "high", "low", "close", "vwap",
                "raw_open", "raw_high", "raw_low", "raw_close",
            ]
            daily.loc[suspended, price_columns] = np.nan
            self._record_issue(
                "daily_quote.suspended_prices", "NORMALIZED_TO_MISSING",
                "vendor suspension rows were retained but their zero/stale prices were set to missing",
                rows=int(suspended.sum()),
            )
        nonpositive = daily[["open", "high", "low", "close"]].le(0).any(axis=1)
        if nonpositive.any():
            self._record_issue(
                "daily_quote.nonpositive_price", "MANUAL_REVIEW",
                "non-suspension rows contain nonpositive adjusted prices and were not guessed or rewritten",
                rows=int(nonpositive.sum()),
            )
        # Gildata source units are 10,000 shares, CNY 10,000, and percent.
        # Canonical volume/amount use base shares/CNY and turnover uses a
        # decimal ratio, matching the engine's synthetic/test contract.
        daily["volume"] *= 10_000.0
        daily["amount"] *= 10_000.0
        daily["turnover_rate"] /= 100.0
        missing_raw = daily["raw_close"].isna()
        if missing_raw.any():
            self._record_issue(
                "daily_quote.raw_alignment", "MISSING_VALUE",
                "adjusted rows without a matching unadjusted close",
                rows=int(missing_raw.sum()),
            )

        valuation = self._read_valuation(start, end).rename(columns={"交易日": "date"})
        if not valuation.empty:
            valuation = valuation.rename(columns={
                "总市值（亿元）": "total_market_cap",
                "流通市值（亿元）": "float_market_cap",
                "市盈率PE(TTM)": "pe_ttm",
                "市净率PB(MRQ)": "pb",
                "市净率PB(LF)": "pb_lf",
                "市现率PCF(TTM)": "pcf_ttm",
                "市销率PS(TTM)": "ps_ttm",
                "滚动股息率": "dividend_yield_ttm",
            })
            valuation_columns = [
                "date", "security_id", "total_market_cap", "float_market_cap",
                "pe_ttm", "pb", "pb_lf", "pcf_ttm", "ps_ttm", "dividend_yield_ttm",
            ]
            valuation = valuation[valuation_columns]
            _as_numeric(valuation, valuation_columns[2:])
            valuation["total_market_cap"] *= 100_000_000.0
            valuation["float_market_cap"] *= 100_000_000.0
            daily = daily.merge(valuation, on=["date", "security_id"], how="left", validate="1:1")
            missing_valuation = daily["total_market_cap"].isna()
            if missing_valuation.any():
                self._record_issue(
                    "valuation.alignment", "MISSING_VALUE",
                    "daily quote rows without total market capitalization",
                    rows=int(missing_valuation.sum()),
                )
        else:
            self._record_issue(
                "valuation.coverage", "MISSING_PARTITION",
                "no valuation rows were found for the requested range",
                requested_start=start.isoformat(), requested_end=end.isoformat(),
            )

        if "total_market_cap" in daily:
            daily["total_shares"] = daily["total_market_cap"].divide(
                daily["raw_close"].where(daily["raw_close"].gt(0))
            )
        if "float_market_cap" in daily:
            daily["float_shares"] = daily["float_market_cap"].divide(
                daily["raw_close"].where(daily["raw_close"].gt(0))
            )
        if need_book_to_price:
            book = self._read_daily_factor_inputs(start, end)
            daily = daily.merge(book, on=["date", "security_id"], how="left", validate="1:1")
            missing_book = daily["book_to_price"].isna()
            if missing_book.any():
                self._record_issue(
                    "book_to_price.coverage", "MISSING_VALUE",
                    "daily rows without PIT parent-equity book_to_price",
                    rows=int(missing_book.sum()),
                )
        if need_fund_flow:
            flow = self._read_large_order_inflow_rate(start, end)
            daily = daily.merge(flow, on=["date", "security_id"], how="left", validate="1:1")
            if start < date(2025, 6, 1):
                self._record_issue(
                    "fund_flow.coverage", "COVERAGE_LIMIT",
                    "fund-flow partitions are empty before 2025-06",
                    requested_start=start.isoformat(),
                )

        daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
        output = daily.set_index(["date", "security_id"]).sort_index()
        if len(self._daily_cache) >= 3:
            self._daily_cache.pop(next(iter(self._daily_cache)))
        self._daily_cache[key] = output.copy()
        return output

    def trading_calendar(self, start: date, end: date) -> pd.DatetimeIndex:
        if self._calendar_cache is None:
            index = pd.read_parquet(
                self.root / "index_000985_daily_2011_present.parquet",
                columns=["交易日"],
            )
            dates = pd.to_datetime(index["交易日"], errors="coerce").dropna().dt.normalize()
            self._calendar_cache = pd.DatetimeIndex(dates).sort_values().unique()
        begin, finish = pd.Timestamp(start), pd.Timestamp(end)
        calendar = self._calendar_cache[
            (self._calendar_cache >= begin) & (self._calendar_cache <= finish)
        ]
        # The index download can lag the stock quote package by one session.
        # Supplement only from exact stock quote dates; never invent weekdays.
        quote = self._read_monthly(
            "partitions/daily_quote/restoration=unadjusted",
            start,
            end,
            ["交易日"],
            "交易日",
        )
        if not quote.empty:
            calendar = calendar.union(pd.DatetimeIndex(quote["交易日"].dropna().unique()))
        return calendar.sort_values().unique()

    def security_master(self, start: date, end: date) -> pd.DataFrame:
        del start
        if self._master_cache is None:
            source = pd.read_csv(
                self.root / "security_master_all.csv",
                encoding="utf-8-sig",
                dtype={"security_code": "string", "ref_code": "string"},
            )
            master = source.rename(columns={
                "ref_code": "security_id",
                "listing_date": "list_date",
                "security_name": "security_name",
                "listing_status": "source_listing_status",
                "sw1_industry": "source_sw1_industry_current",
            }).copy()
            master["security_id"] = master["security_id"].astype(str)
            master["list_date"] = pd.to_datetime(master["list_date"], errors="coerce").dt.normalize()
            master["delist_date"] = pd.NaT
            company = pd.read_parquet(
                self.root / "company_basic_all.parquet",
                columns=["股票代码", "成立日期"],
            )
            company["security_id"] = self._normalize_security_id(company["股票代码"], area="company_basic")
            company["establishment_date"] = pd.to_datetime(company["成立日期"], errors="coerce").dt.normalize()
            company = company.dropna(subset=["security_id"]).drop_duplicates("security_id", keep="last")
            master = master.merge(
                company[["security_id", "establishment_date"]],
                on="security_id",
                how="left",
                validate="1:1",
            )
            self._master_cache = master
        return self._master_cache[self._master_cache["list_date"] <= pd.Timestamp(end)].copy()

    @staticmethod
    def _yes_no(values: pd.Series) -> pd.Series:
        normalized = values.astype("string").str.strip()
        return normalized.map({"是": True, "否": False, "1": True, "0": False}).astype("boolean")

    def security_status(self, start: date, end: date) -> pd.DataFrame:
        key = (start, end)
        cached = self._status_cache.get(key)
        if cached is not None:
            return cached.copy()
        source = self._read_daily_quote_source(start, end, "unadjusted")
        if source.empty:
            return pd.DataFrame(columns=["date", "security_id", "is_st", "is_suspended", "is_trading"])
        suspended = self._yes_no(source["是否停牌"])
        unknown = suspended.isna()
        if unknown.any():
            self._record_issue(
                "security_status.is_suspended", "UNKNOWN_VALUE",
                "unrecognized suspension labels remain missing", rows=int(unknown.sum()),
            )
        frame = pd.DataFrame({
            "date": pd.to_datetime(source["交易日"]).dt.normalize(),
            "security_id": source["security_id"].astype(str),
            "is_st": pd.array([pd.NA] * len(source), dtype="boolean"),
            "is_suspended": suspended.array,
            "is_trading": (~suspended).array,
            "is_limit_up": pd.array([pd.NA] * len(source), dtype="boolean"),
            "is_limit_down": pd.array([pd.NA] * len(source), dtype="boolean"),
            "is_no_price_limit": pd.array([pd.NA] * len(source), dtype="boolean"),
            "has_price_limit": pd.array([pd.NA] * len(source), dtype="boolean"),
            "limit_up_price": np.nan,
            "limit_down_price": np.nan,
        }).sort_values(["date", "security_id"]).reset_index(drop=True)
        self._status_cache[key] = frame.copy()
        return frame

    def index_constituents(self, index_id: str, start: date, end: date) -> pd.DataFrame:
        del index_id, start, end
        self._record_issue(
            "index_constituents", "UNAVAILABLE",
            "the approved package contains index prices but no PIT constituent intervals",
        )
        return pd.DataFrame(columns=["index_id", "security_id", "effective_from", "effective_to"])

    def _index_daily(self, start: date, end: date, benchmark_id: str) -> pd.DataFrame:
        if benchmark_id not in {"000985", "000985.CSI"}:
            raise ValueError(
                f"approved Gildata package only contains 000985.CSI; requested {benchmark_id!r}"
            )
        source = pd.read_parquet(
            self.root / "index_000985_daily_2011_present.parquet",
            columns=["交易日", "收盘价(点)", "今开盘(点)", "最高价(点)", "最低价(点)"],
        ).rename(columns={
            "交易日": "date",
            "收盘价(点)": "close",
            "今开盘(点)": "open",
            "最高价(点)": "high",
            "最低价(点)": "low",
        })
        source["date"] = pd.to_datetime(source["date"], errors="coerce").dt.normalize()
        source = source[source["date"].between(pd.Timestamp(start), pd.Timestamp(end))].copy()
        _as_numeric(source, ["open", "high", "low", "close"])
        source["index_id"] = "000985.CSI"
        if source.duplicated(["date", "index_id"]).any():
            raise ValueError("000985 index data has duplicate dates")
        return source.sort_values(["date", "index_id"]).reset_index(drop=True)

    def _disclosures(self) -> pd.DataFrame:
        if self._disclosure_cache is None:
            source = pd.read_parquet(
                self.root / "finance_disclosure_2011_present.parquet",
                columns=["股票代码", "报告期", "实际披露日"],
            )
            source["security_id"] = self._normalize_security_id(source["股票代码"], area="finance_disclosure")
            source["report_period"] = pd.to_datetime(source["报告期"], errors="coerce").dt.normalize()
            disclosure_date = pd.to_datetime(source["实际披露日"], errors="coerce").dt.normalize()
            # Only a date is supplied.  Treat it as available after that day's
            # close, so a 15:00 calculation cannot see a potentially post-close
            # release.  This conservative rule avoids look-ahead.
            source["available_at"] = disclosure_date + pd.Timedelta(hours=23, minutes=59, seconds=59)
            self._disclosure_cache = source[[
                "security_id", "report_period", "available_at",
            ]].dropna().sort_values("available_at").drop_duplicates(
                ["security_id", "report_period"], keep="last",
            ).reset_index(drop=True)
        return self._disclosure_cache

    def _required_statement_codes(self, spec: FactorSpec) -> set[str]:
        codes: set[str] = set()
        for field in spec.required_fields:
            codes.update(self._requested_to_statement_codes.get(field, ()))
            codes.update(DERIVED_COMPONENT_CODES.get(field, ()))
        if spec.implementation == "delta_roe_rank_gauss":
            codes.update({"TotalAssets", "TotalLiability", "NPParentCompanyOwners"})
        if spec.implementation == "consensus_expectation":
            codes.update({"NPParentCompanyOwners", "OperatingRevenue"})
        if spec.implementation == "intangible_value":
            codes.update({"RAndD", "GoodWill"})
        return {code for code in codes if code}

    def _statement_aliases_for_spec(self, spec: FactorSpec) -> dict[str, set[str]]:
        aliases = {code: {canonical} for code, canonical in STATEMENT_CODE_CANONICAL.items()}
        for requested in spec.required_fields:
            if spec.implementation == "fundamental_formula" and (
                "TTM" in requested or requested == "净利润TTM（口径待确认）"
            ):
                continue
            for code in self._requested_to_statement_codes.get(requested, ()):
                aliases.setdefault(code, set()).add(requested)
        if spec.implementation == "delta_roe_rank_gauss":
            aliases.setdefault("TotalAssets", set()).add("资产总计")
            aliases.setdefault("TotalLiability", set()).add("负债合计")
            aliases.setdefault("NPParentCompanyOwners", set()).add("净利润(不含少数股东损益)")
        return aliases

    def _read_statement_rows(
        self,
        spec: FactorSpec,
        start: date,
        end: date,
        codes: set[str],
    ) -> pd.DataFrame:
        if not codes:
            return pd.DataFrame()
        # The engine already expands start for the factor lookback.  One extra
        # year is retained so the latest report before load_start is visible.
        report_start = pd.Timestamp(start) - pd.Timedelta(days=550)
        pieces: list[pd.DataFrame] = []
        columns = ["股票代码", "时间", "核算方式", "财务科目代码", "财务科目数额"]
        for year in range(report_start.year, end.year + 1):
            directory = self.root / "partitions" / "financial_statement_A" / f"year={year}"
            for path in sorted(directory.glob("part_*.parquet")):
                piece = pd.read_parquet(
                    path,
                    columns=columns,
                    filters=[("财务科目代码", "in", sorted(codes))],
                )
                if not piece.empty:
                    pieces.append(piece)
        if not pieces:
            return pd.DataFrame()
        source = pd.concat(pieces, ignore_index=True)
        source["security_id"] = self._normalize_security_id(source["股票代码"], area="financial_statement")
        source["report_period"] = pd.to_datetime(source["时间"], errors="coerce").dt.normalize()
        source = source[source["report_period"].between(report_start, pd.Timestamp(end))]
        source["value"] = pd.to_numeric(source["财务科目数额"], errors="coerce")
        source = source.dropna(subset=["security_id", "report_period"])
        disclosures = self._disclosures()
        source = source.merge(
            disclosures,
            on=["security_id", "report_period"],
            how="left",
            validate="m:1",
        )
        missing_disclosure = source["available_at"].isna()
        if missing_disclosure.any():
            self._record_issue(
                "financial_statement.available_at", "MISSING_DISCLOSURE",
                "financial rows without an actual disclosure date were excluded",
                rows=int(missing_disclosure.sum()),
            )
            source = source[~missing_disclosure]
        source = source[source["available_at"] <= pd.Timestamp(end) + pd.Timedelta(days=1)]
        return source

    @staticmethod
    def _empty_observations() -> pd.DataFrame:
        return pd.DataFrame(columns=[
            "security_id", "report_period", "available_at", "revision_at",
            "field", "value", "accounting_method", "source_field",
        ])

    def _statement_observations(self, spec: FactorSpec, start: date, end: date) -> pd.DataFrame:
        codes = self._required_statement_codes(spec)
        cache_key = (spec.factor_id, start, end, tuple(sorted(codes)))
        cached = self._financial_cache.get(cache_key)
        if cached is not None:
            return cached.copy()
        source = self._read_statement_rows(spec, start, end, codes)
        if source.empty:
            return self._empty_observations()

        aliases = self._statement_aliases_for_spec(spec)
        mapping_rows: list[dict[str, str]] = []
        for code in codes:
            fields = aliases.get(code, {STATEMENT_CODE_CANONICAL.get(code, code)})
            for field in fields:
                mapping_rows.append({"source_field": code, "field": field})
        alias_frame = pd.DataFrame(mapping_rows)
        observed = source.rename(columns={"财务科目代码": "source_field"}).merge(
            alias_frame, on="source_field", how="inner", validate="m:m",
        )
        observed["revision_at"] = pd.NaT
        observed["accounting_method"] = "A"
        output = observed[[
            "security_id", "report_period", "available_at", "revision_at",
            "field", "value", "accounting_method", "source_field",
        ]].drop_duplicates(
            ["security_id", "report_period", "available_at", "field"], keep="last",
        )

        # Deterministic derived balance-sheet fields are emitted even when the
        # semantic review is configured to mark rather than raise.
        requested_derived = set(spec.required_fields) & set(DERIVED_COMPONENT_CODES)
        if requested_derived:
            base = source.pivot_table(
                index=["security_id", "report_period", "available_at"],
                columns="财务科目代码",
                values="value",
                aggfunc="last",
            )
            derived: list[pd.DataFrame] = []
            for field in sorted(requested_derived):
                if field == "带息债务":
                    parts = [base.get(name, pd.Series(np.nan, index=base.index)) for name in DERIVED_COMPONENT_CODES[field]]
                    values = pd.concat(parts, axis=1).sum(axis=1, min_count=1)
                elif field in {"无息流动负债(元)", "无息流动负债"}:
                    values = base.get("TotalCurrentLiability", pd.Series(np.nan, index=base.index)) - base.get(
                        "ShortTermLoan", pd.Series(0.0, index=base.index)
                    ).fillna(0.0)
                elif field in {"营运资本", "净营运资本"}:
                    values = base.get("TotalCurrentAssets", pd.Series(np.nan, index=base.index)) - base.get(
                        "TotalCurrentLiability", pd.Series(np.nan, index=base.index)
                    )
                else:
                    continue
                frame = values.rename("value").reset_index()
                frame["field"] = field
                frame["revision_at"] = pd.NaT
                frame["accounting_method"] = "A"
                frame["source_field"] = "+".join(DERIVED_COMPONENT_CODES[field])
                derived.append(frame)
            if derived:
                output = pd.concat([output, *derived], ignore_index=True, sort=False)

        # EBITDA minus EBIT is an explicit data-package derivation.  It is
        # loaded only for factors that request depreciation/amortization.
        if "折旧与摊销" in spec.required_fields:
            analysis = self._analysis_derived_observation(start, end, "折旧与摊销")
            output = pd.concat([output, analysis], ignore_index=True, sort=False)
        if len(self._financial_cache) >= 3:
            self._financial_cache.pop(next(iter(self._financial_cache)))
        self._financial_cache[cache_key] = output.copy()
        return output

    def _analysis_derived_observation(self, start: date, end: date, field: str) -> pd.DataFrame:
        path = self.root / "derived" / "financial_analysis_wide.parquet"
        if field == "折旧与摊销":
            columns = ["股票代码", "时间", "核算方式", "实际披露日", "EBIT", "EBITDA"]
            source = pd.read_parquet(path, columns=columns)
            source = source[source["核算方式"].eq("累计值")]
            source["value"] = pd.to_numeric(source["EBITDA"], errors="coerce") - pd.to_numeric(
                source["EBIT"], errors="coerce"
            )
            source_field = "EBITDA-EBIT"
        else:
            column = ANALYSIS_FIELD_COLUMNS[field]
            columns = ["股票代码", "时间", "核算方式", "实际披露日", column]
            source = pd.read_parquet(path, columns=columns)
            source = source[source["核算方式"].eq("累计值")]
            source["value"] = pd.to_numeric(source[column], errors="coerce")
            source_field = column
        source["security_id"] = self._normalize_security_id(source["股票代码"], area="financial_analysis")
        source["report_period"] = pd.to_datetime(source["时间"], errors="coerce").dt.normalize()
        available = pd.to_datetime(source["实际披露日"], errors="coerce").dt.normalize()
        source["available_at"] = available + pd.Timedelta(hours=23, minutes=59, seconds=59)
        source = source[
            source["report_period"].between(pd.Timestamp(start) - pd.Timedelta(days=550), pd.Timestamp(end))
            & source["available_at"].le(pd.Timestamp(end) + pd.Timedelta(days=1))
        ]
        source["revision_at"] = pd.NaT
        source["field"] = field
        source["accounting_method"] = "A"
        source["source_field"] = source_field
        return source[[
            "security_id", "report_period", "available_at", "revision_at",
            "field", "value", "accounting_method", "source_field",
        ]].dropna(subset=["security_id", "report_period", "available_at"])

    def _ttm_observations(self, spec: FactorSpec, start: date, end: date) -> pd.DataFrame:
        requested = [field for field in spec.required_fields if field in TTM_FIELD_COLUMNS]
        if not requested:
            return self._empty_observations()
        columns = ["股票代码", "时间", "实际披露日", *sorted({TTM_FIELD_COLUMNS[x] for x in requested})]
        source = pd.read_parquet(self.root / "derived" / "financial_ttm_derived.parquet", columns=columns)
        source["security_id"] = self._normalize_security_id(source["股票代码"], area="financial_ttm")
        source["report_period"] = pd.to_datetime(source["时间"], errors="coerce").dt.normalize()
        available = pd.to_datetime(source["实际披露日"], errors="coerce").dt.normalize()
        source["available_at"] = available + pd.Timedelta(hours=23, minutes=59, seconds=59)
        source = source[
            source["report_period"].between(pd.Timestamp(start) - pd.Timedelta(days=550), pd.Timestamp(end))
            & source["available_at"].le(pd.Timestamp(end) + pd.Timedelta(days=1))
        ]
        pieces: list[pd.DataFrame] = []
        for field in requested:
            piece = source[["security_id", "report_period", "available_at", TTM_FIELD_COLUMNS[field]]].rename(
                columns={TTM_FIELD_COLUMNS[field]: "value"}
            )
            piece["field"] = field
            piece["revision_at"] = pd.NaT
            piece["accounting_method"] = "A"
            piece["source_field"] = TTM_FIELD_COLUMNS[field]
            pieces.append(piece)
        return pd.concat(pieces, ignore_index=True)[[
            "security_id", "report_period", "available_at", "revision_at",
            "field", "value", "accounting_method", "source_field",
        ]].dropna(subset=["security_id", "report_period", "available_at"])

    def _analysis_observations(self, spec: FactorSpec, start: date, end: date) -> pd.DataFrame:
        requested = [field for field in spec.required_fields if field in ANALYSIS_FIELD_COLUMNS]
        if not requested:
            return self._empty_observations()
        return pd.concat(
            [self._analysis_derived_observation(start, end, field) for field in requested],
            ignore_index=True,
        )

    def _financial_observations(self, spec: FactorSpec, start: date, end: date) -> pd.DataFrame:
        frames = [self._statement_observations(spec, start, end)]
        if spec.implementation == "fundamental_formula":
            frames.extend([
                self._ttm_observations(spec, start, end),
                self._analysis_observations(spec, start, end),
            ])
        nonempty = [frame for frame in frames if not frame.empty]
        if not nonempty:
            return self._empty_observations()
        return pd.concat(nonempty, ignore_index=True, sort=False).drop_duplicates(
            ["security_id", "report_period", "available_at", "field"], keep="last",
        ).sort_values(["available_at", "security_id", "report_period", "field"]).reset_index(drop=True)

    def _consensus_observations(self, start: date, end: date) -> pd.DataFrame:
        lookback = start - timedelta(days=370)
        columns = ["股票代码", "截止日期", "预测年度", "data_source", *CONSENSUS_FIELD_COLUMNS.values()]
        source = self._read_monthly(
            "derived/consensus_aggregate_best_available_180d",
            lookback,
            end,
            columns,
            "截止日期",
        )
        if source.empty:
            return pd.DataFrame(columns=[
                "security_id", "available_at", "revision_at", "forecast_year",
                "field", "value", "data_source",
            ])
        source["security_id"] = self._normalize_security_id(source["股票代码"], area="consensus")
        source["available_at"] = pd.to_datetime(source["截止日期"]).dt.normalize() + pd.Timedelta(
            hours=23, minutes=59, seconds=59
        )
        source["forecast_year"] = pd.to_numeric(source["预测年度"], errors="coerce").astype("Int64")
        pieces: list[pd.DataFrame] = []
        for field, column in CONSENSUS_FIELD_COLUMNS.items():
            piece = source[["security_id", "available_at", "forecast_year", "data_source", column]].rename(
                columns={column: "value"}
            )
            piece["field"] = field
            piece["revision_at"] = pd.NaT
            pieces.append(piece)
        result = pd.concat(pieces, ignore_index=True)
        result["value"] = pd.to_numeric(result["value"], errors="coerce")
        return result.dropna(subset=["security_id", "available_at", "forecast_year"])[[
            "security_id", "available_at", "revision_at", "forecast_year",
            "field", "value", "data_source",
        ]].sort_values(["available_at", "security_id", "forecast_year", "field"]).reset_index(drop=True)

    def _annual_report_disclosures(self, start: date, end: date) -> pd.DataFrame:
        disclosure = self._disclosures().copy()
        disclosure = disclosure[disclosure["report_period"].dt.strftime("%m-%d").eq("12-31")]
        disclosure = disclosure[
            disclosure["available_at"].between(
                pd.Timestamp(start) - pd.Timedelta(days=3 * 366),
                pd.Timestamp(end) + pd.Timedelta(days=1),
            )
        ]
        disclosure["report_year"] = disclosure["report_period"].dt.year
        return disclosure[["security_id", "report_year", "available_at"]].reset_index(drop=True)

    def _financial_report_events(self, start: date, end: date) -> pd.DataFrame:
        disclosure = self._disclosures().copy()
        disclosure = disclosure[
            disclosure["available_at"].between(
                pd.Timestamp(start) - pd.Timedelta(days=3 * 366),
                pd.Timestamp(end) + pd.Timedelta(days=1),
            )
        ]
        return disclosure.rename(columns={"available_at": "announcement_at"})[[
            "security_id", "report_period", "announcement_at",
        ]].reset_index(drop=True)

    def load_factor_context(
        self,
        spec: FactorSpec,
        start: date,
        end: date,
        calculation_dates: pd.DatetimeIndex,
        price_adjustment: PriceAdjustment,
        benchmark_id: str,
    ) -> FactorContext:
        audit = self.field_mapping_audit(spec)
        self._enforce_mapping_policy(spec, audit)
        daily = self._load_daily(spec, start, end, price_adjustment)
        index_daily = self._index_daily(start, end, benchmark_id)
        if not daily.empty:
            stock_dates = pd.DatetimeIndex(daily.index.get_level_values("date").unique())
            index_dates = pd.DatetimeIndex(index_daily["date"].unique())
            missing_index_dates = stock_dates.difference(index_dates)
            if len(missing_index_dates):
                self._record_issue(
                    "index_daily.coverage", "MISSING_TRADING_DATES",
                    "stock trading dates without 000985.CSI daily prices",
                    dates="|".join(missing_index_dates.strftime("%Y-%m-%d")[:20]),
                    count=len(missing_index_dates),
                )
        financial = (
            self._financial_observations(spec, start, end)
            if spec.implementation in {
                "financial_statement", "fundamental_formula", "delta_roe_rank_gauss",
                "consensus_expectation", "intangible_value",
            }
            else self._empty_observations()
        )
        consensus = (
            self._consensus_observations(start, end)
            if spec.implementation == "consensus_expectation"
            else pd.DataFrame()
        )
        events: dict[str, pd.DataFrame] = {}
        if spec.implementation == "event_price":
            events["financial_report_disclosures"] = self._financial_report_events(start, end)

        context = FactorContext(
            calculation_dates=pd.DatetimeIndex(calculation_dates).normalize(),
            daily=daily,
            index_daily=index_daily,
            financial_observations=financial,
            consensus_observations=consensus,
            events=events,
            extras={
                "field_mapping_audit": audit.copy(),
                "unit_conversion_contract": self.unit_conversion_contract(),
                "manual_review": audit[
                    audit["review_status"].isin({"MANUAL_REVIEW", "NO_MATCH"})
                ].copy(),
                "data_quality_issues": self.data_quality_issues(),
                "annual_report_disclosures": self._annual_report_disclosures(start, end),
            },
        )
        context.validate()
        return context
