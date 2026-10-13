from __future__ import annotations

import argparse
from datetime import date
import importlib
import json

from factor_engine.catalog import FactorCatalog
from factor_engine.config import EngineConfig
from factor_engine.engine import FactorEngine


def _load_provider(class_path: str, provider_config: dict):
    if ":" not in class_path:
        raise ValueError("provider.class_path must use 'package.module:ClassName'")
    module_name, class_name = class_path.split(":", 1)
    provider_class = getattr(importlib.import_module(module_name), class_name)
    return provider_class(provider_config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="factor-engine")
    subparsers = parser.add_subparsers(dest="command", required=True)
    summary = subparsers.add_parser("catalog-summary", help="show factor catalog status counts")
    summary.add_argument("--config", required=True)

    run = subparsers.add_parser("run", help="incrementally calculate factors")
    run.add_argument("--config", required=True)
    run.add_argument("--start", required=True, type=date.fromisoformat)
    run.add_argument("--end", required=True, type=date.fromisoformat)
    run.add_argument("--factor-id", action="append", dest="factor_ids")
    run.add_argument("--strict-pending", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = EngineConfig.from_toml(args.config)
    if args.command == "catalog-summary":
        print(json.dumps(FactorCatalog.from_csv(config.catalog_path).summary(), ensure_ascii=False, indent=2))
        return
    provider = _load_provider(config.provider_class_path, config.raw.get("provider", {}))
    results = FactorEngine(config, provider).run(
        args.start,
        args.end,
        factor_ids=args.factor_ids,
        strict_pending=args.strict_pending,
    )
    print(json.dumps([result.__dict__ for result in results], ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
