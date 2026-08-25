"""Pricing API interaction and pagination.

This module provides two fetcher classes:
- UsagetypeFetcher: for AmazonBedrock and AmazonBedrockService (usagetype-based)
- MarketplaceFetcher: for AmazonBedrockFoundationModels (servicename-based)
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import boto3

from .marketplace_mapper import normalize_servicename
from .types import MarketplaceEntry, UsagetypeEntry

if TYPE_CHECKING:
    pass


# Service codes queried for Bedrock pricing data.
SERVICE_CODE_BEDROCK = "AmazonBedrock"
SERVICE_CODE_BEDROCK_SERVICE = "AmazonBedrockService"
SERVICE_CODE_FOUNDATION_MODELS = "AmazonBedrockFoundationModels"


# ---------------------------------------------------------------------------
# Shared pagination helper
# ---------------------------------------------------------------------------


def _paginate_products(
    session: boto3.Session,
    service_code: str,
    region: str,
) -> list[dict]:
    """Paginate through all products for a service code and region.

    Returns parsed JSON dicts (not raw strings). Handles the
    JSON-string-within-JSON format the Pricing API returns.
    """
    client = session.client("pricing", region_name="us-east-1")
    paginator = client.get_paginator("get_products")

    pages = paginator.paginate(
        ServiceCode=service_code,
        Filters=[
            {
                "Type": "TERM_MATCH",
                "Field": "regionCode",
                "Value": region,
            },
        ],
    )

    products: list[dict] = []
    for page in pages:
        for item in page.get("PriceList", []):
            if isinstance(item, str):
                products.append(json.loads(item))
            else:
                products.append(item)

    return products


# ---------------------------------------------------------------------------
# UsagetypeFetcher: AmazonBedrock + AmazonBedrockService
# ---------------------------------------------------------------------------


class UsagetypeFetcher:
    """Fetches pricing from usagetype-based service codes.

    Handles AmazonBedrock and AmazonBedrockService, which both use the same
    product structure (usagetype field for model identification, standard
    OnDemand terms with unit/pricePerUnit).
    """

    def __init__(self, session: boto3.Session | None = None):
        self._session = session or boto3.Session()

    def fetch(self, region: str, service_code: str) -> list[UsagetypeEntry]:
        """Fetch all pricing entries for a usagetype-based service code."""
        products = _paginate_products(self._session, service_code, region)

        entries: list[UsagetypeEntry] = []
        for product in products:
            parsed = self._parse_product(product)
            if parsed is not None:
                entries.extend(parsed)

        return entries

    def fetch_bedrock(self, region: str) -> list[UsagetypeEntry]:
        """Fetch from AmazonBedrock service code."""
        return self.fetch(region, SERVICE_CODE_BEDROCK)

    def fetch_bedrock_service(self, region: str) -> list[UsagetypeEntry]:
        """Fetch from AmazonBedrockService service code."""
        return self.fetch(region, SERVICE_CODE_BEDROCK_SERVICE)

    @staticmethod
    def _parse_product(product: dict) -> list[UsagetypeEntry] | None:
        """Parse a single product entry into UsagetypeEntry list.

        Returns None if the entry has no usagetype or no OnDemand terms.
        """
        attributes = product.get("product", {}).get("attributes", {})
        usagetype = attributes.get("usagetype")
        if usagetype is None:
            return None

        inference_type = attributes.get("inferenceType") or None
        provider = attributes.get("provider") or None
        model = attributes.get("model") or None
        service_tier = attributes.get("service_tier") or None

        on_demand = product.get("terms", {}).get("OnDemand", {})
        if not on_demand:
            return None

        entries: list[UsagetypeEntry] = []

        for offer_term in on_demand.values():
            offer_term_code = offer_term.get("offerTermCode", "")
            price_dimensions = offer_term.get("priceDimensions", {})

            for dimension in price_dimensions.values():
                unit = dimension.get("unit", "")
                rate_code = dimension.get("rateCode", "")

                # Validate price — defer error rather than silently defaulting.
                price_str = dimension.get("pricePerUnit", {}).get("USD")
                parse_errors: list[str] = []
                if price_str is None:
                    parse_errors.append(
                        f"Missing pricePerUnit.USD in dimension "
                        f"(rateCode={rate_code}, usagetype={usagetype})"
                    )
                    price = 0.0
                else:
                    try:
                        price = float(price_str)
                    except (ValueError, TypeError):
                        parse_errors.append(
                            f"Invalid pricePerUnit.USD={price_str!r} in dimension "
                            f"(rateCode={rate_code}, usagetype={usagetype})"
                        )
                        price = 0.0

                entries.append(
                    UsagetypeEntry(
                        usagetype=usagetype,
                        inference_type=inference_type,
                        provider=provider,
                        model=model,
                        service_tier=service_tier,
                        unit=unit,
                        price_per_unit=price,
                        offer_term_code=offer_term_code,
                        rate_code=rate_code,
                        parse_errors=tuple(parse_errors),
                    )
                )

        return entries if entries else None


# ---------------------------------------------------------------------------
# MarketplaceFetcher: AmazonBedrockFoundationModels
# ---------------------------------------------------------------------------


class MarketplaceFetcher:
    """Fetches pricing from AmazonBedrockFoundationModels.

    This service code uses a different format: model identification is via
    the `servicename` attribute, and prices are already per-million tokens.
    Returns a dict mapping normalized servicenames to MarketplaceEntry lists.
    """

    def __init__(self, session: boto3.Session | None = None):
        self._session = session or boto3.Session()

    def fetch(self, region: str) -> dict[str, list[MarketplaceEntry]]:
        """Fetch all foundation model pricing for a region.

        Returns entries keyed by normalized servicename. The mapping from
        model IDs to servicenames is handled by MarketplaceMapper at
        resolve time.
        """
        products = _paginate_products(
            self._session, SERVICE_CODE_FOUNDATION_MODELS, region
        )

        results: dict[str, list[MarketplaceEntry]] = {}
        for product in products:
            records = self._parse_product(product)
            if records:
                for norm_sn, record in records:
                    results.setdefault(norm_sn, []).append(record)

        return results

    @staticmethod
    def _parse_product(
        product: dict,
    ) -> list[tuple[str, MarketplaceEntry]] | None:
        """Parse a foundation model product entry.

        Returns list of (normalized_servicename, MarketplaceEntry) tuples,
        or None if the entry has no servicename or no OnDemand terms.
        """

        attributes = product.get("product", {}).get("attributes", {})
        servicename = attributes.get("servicename", "")
        if not servicename:
            return None

        norm_sn = normalize_servicename(servicename)

        on_demand = product.get("terms", {}).get("OnDemand", {})
        if not on_demand:
            return None

        results: list[tuple[str, MarketplaceEntry]] = []

        for offer_term in on_demand.values():
            offer_term_code = offer_term.get("offerTermCode", "")
            for dimension in offer_term.get("priceDimensions", {}).values():
                desc = dimension.get("description", "")
                rate_code = dimension.get("rateCode", "")
                raw_unit = dimension.get("unit", "")

                # Validate price — defer error rather than silently defaulting.
                price_str = dimension.get("pricePerUnit", {}).get("USD")
                parse_errors: list[str] = []
                if price_str is None:
                    parse_errors.append(
                        f"Missing pricePerUnit.USD in dimension "
                        f"(rateCode={rate_code}, servicename={servicename})"
                    )
                    price = 0.0
                else:
                    try:
                        price = float(price_str)
                    except (ValueError, TypeError):
                        parse_errors.append(
                            f"Invalid pricePerUnit.USD={price_str!r} in dimension "
                            f"(rateCode={rate_code}, servicename={servicename})"
                        )
                        price = 0.0

                # Extract the dimension portion of the description for the
                # classifier (strip the "AWS Marketplace software usage|region|"
                # prefix if present).
                desc_parts = desc.split("|")
                dimension_desc = desc_parts[-1].strip() if desc_parts else desc

                # The API's unit field is often a useless "Units" placeholder.
                # When that's the case, pass through the description's dimension
                # portion as the unit — the classifier will parse it.
                if raw_unit == "Units":
                    effective_unit = dimension_desc
                else:
                    effective_unit = raw_unit

                results.append(
                    (
                        norm_sn,
                        MarketplaceEntry(
                            servicename=norm_sn,
                            description=dimension_desc,
                            unit=effective_unit,
                            price_per_unit=price,
                            offer_term_code=offer_term_code,
                            rate_code=rate_code,
                            parse_errors=tuple(parse_errors),
                        ),
                    )
                )

        return results if results else None
