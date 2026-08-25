"""Dump AWS Price List API entries for Bedrock service codes to JSON.

Examples:
    uv run scripts/dump_pricing.py --all-services
    uv run scripts/dump_pricing.py AmazonBedrock --output-dir data
    uv run scripts/dump_pricing.py AmazonBedrockService -r us-west-2 -o out.json
    uv run scripts/dump_pricing.py AmazonBedrock  # prints to stdout
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import boto3

BEDROCK_SERVICE_CODES = (
    "AmazonBedrock",
    "AmazonBedrockFoundationModels",
    "AmazonBedrockService",
)


def fetch_all_products(
    service_code: str,
    region_filter: str | None = None,
    *,
    pricing_client: Any | None = None,
) -> list[dict]:
    """Fetch and deterministically sort all products for a service code.

    PriceList entries are parsed from JSON strings into nested dictionaries.
    A client can be supplied by callers that need custom boto3 configuration.
    """
    client = pricing_client or boto3.client("pricing", region_name="us-east-1")
    paginator = client.get_paginator("get_products")

    kwargs: dict = {"ServiceCode": service_code}
    if region_filter:
        kwargs["Filters"] = [
            {"Type": "TERM_MATCH", "Field": "regionCode", "Value": region_filter}
        ]

    products: list[dict] = []
    for page in paginator.paginate(**kwargs):
        for item in page.get("PriceList", []):
            products.append(json.loads(item) if isinstance(item, str) else item)

    return sorted(products, key=_product_sort_key)


def _product_sort_key(product: dict) -> tuple[str, str, str, str]:
    """Return stable source identifiers for deterministic local diffs."""
    product_data = product.get("product", {})
    attributes = product_data.get("attributes", {})
    return (
        product.get("serviceCode") or attributes.get("servicecode", ""),
        attributes.get("regionCode", ""),
        product_data.get("sku", ""),
        product.get("version", ""),
    )


def _dated_output_path(output_dir: Path, service_code: str, date_tag: str) -> Path:
    """Build the standard ignored-data filename for one service dump."""
    return output_dir / f"prices.{service_code}-{date_tag}.json"


def _write_products(path: Path, products: list[dict]) -> None:
    """Write one dump atomically with deterministic JSON formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(products, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _validate_date_tag(parser: argparse.ArgumentParser, date_tag: str) -> None:
    """Require the YYYYMMDD date convention used by local pricing history."""
    if not re.fullmatch(r"\d{8}", date_tag):
        parser.error("--date must use YYYYMMDD format")
    try:
        datetime.strptime(date_tag, "%Y%m%d")
    except ValueError:
        parser.error("--date must be a valid calendar date")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dump AWS Pricing API entries for Bedrock service code(s)."
    )
    parser.add_argument(
        "service_code",
        nargs="?",
        help="One service code to fetch (for example, AmazonBedrock).",
    )
    parser.add_argument(
        "--all-services",
        action="store_true",
        help=(
            "Fetch AmazonBedrock, AmazonBedrockFoundationModels, and "
            "AmazonBedrockService service data into dated files."
        ),
    )
    parser.add_argument(
        "--region",
        "-r",
        default=None,
        help="Filter by regionCode (e.g. us-east-1). Omit for all regions.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output file for one service. Defaults to stdout.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Write dated service-named files to this directory. "
            "Defaults to data/ with --all-services."
        ),
    )
    parser.add_argument(
        "--date",
        default=datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        help="Date tag for --output-dir filenames in YYYYMMDD format.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing output file for the same service and date.",
    )
    args = parser.parse_args()

    if args.all_services and args.service_code:
        parser.error("service_code and --all-services are mutually exclusive")
    if not args.all_services and not args.service_code:
        parser.error("provide a service_code to fetch, or --all-services")
    if args.all_services and args.output:
        parser.error("--output cannot be used with --all-services; use --output-dir")
    if args.output and args.output_dir:
        parser.error("--output and --output-dir are mutually exclusive")
    _validate_date_tag(parser, args.date)

    service_codes = BEDROCK_SERVICE_CODES if args.all_services else (args.service_code,)
    output_dir = args.output_dir or (Path("data") if args.all_services else None)

    output_paths: dict[str, Path] = {}
    if output_dir is not None:
        output_paths = {
            service_code: _dated_output_path(output_dir, service_code, args.date)
            for service_code in service_codes
        }
    elif args.output is not None:
        output_paths[args.service_code] = args.output

    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not args.force:
        rendered = ", ".join(str(path) for path in existing)
        parser.error(f"output already exists: {rendered}; use --force to replace it")

    products_by_service = {
        service_code: fetch_all_products(service_code, args.region)
        for service_code in service_codes
    }

    # Complete every network fetch before publishing any file. This prevents a
    # later service failure from leaving a partial same-date capture that blocks
    # an ordinary retry.
    for service_code, products in products_by_service.items():
        output_path = output_paths.get(service_code)
        if output_path is None:
            print(json.dumps(products, indent=2))
            continue
        _write_products(output_path, products)
        print(f"Wrote {len(products)} entries to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
