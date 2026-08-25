"""Tests for the pricing MarketplaceFetcher."""

import json
from unittest.mock import MagicMock

from model_info_amazon_bedrock.pricing.fetcher import MarketplaceFetcher
from model_info_amazon_bedrock.pricing.types import MarketplaceEntry

_SAMPLE_DESC = "AWS Marketplace software usage|us-east-1|Million Input Tokens Regional"


def _make_marketplace_product(
    servicename="Claude Sonnet 4 (Amazon Bedrock Edition)",
    description=_SAMPLE_DESC,
    unit="Units",
    price_usd="3.0000000000",
    rate_code="ABC123.JRTCKXETXF.6YS6EN2CT7",
    offer_term_code="JRTCKXETXF",
    sku="ABC123",
):
    """Helper to build a marketplace product dict matching the Pricing API format."""
    return {
        "product": {
            "productFamily": "Machine Learning",
            "attributes": {
                "servicename": servicename,
                "regionCode": "us-east-1",
                "servicecode": "AmazonBedrockFoundationModels",
            },
            "sku": sku,
        },
        "serviceCode": "AmazonBedrockFoundationModels",
        "terms": {
            "OnDemand": {
                f"{sku}.{offer_term_code}": {
                    "priceDimensions": {
                        rate_code: {
                            "unit": unit,
                            "endRange": "Inf",
                            "description": description,
                            "appliesTo": [],
                            "rateCode": rate_code,
                            "beginRange": "0",
                            "pricePerUnit": {"USD": price_usd},
                        }
                    },
                    "sku": sku,
                    "effectiveDate": "2026-03-01T00:00:00Z",
                    "offerTermCode": offer_term_code,
                    "termAttributes": {},
                }
            }
        },
    }


def _mock_paginator(products):
    """Create a mock session with paginator returning products as JSON strings."""
    mock_session = MagicMock()
    mock_client = MagicMock()
    mock_paginator = MagicMock()

    mock_session.client.return_value = mock_client
    mock_client.get_paginator.return_value = mock_paginator
    mock_paginator.paginate.return_value = [
        {"PriceList": [json.dumps(p) for p in products]}
    ]

    return mock_session


class TestMarketplaceFetcherBasic:
    """Tests for basic MarketplaceFetcher behavior."""

    def test_returns_entries_keyed_by_normalized_servicename(self):
        product = _make_marketplace_product()
        mock_session = _mock_paginator([product])
        fetcher = MarketplaceFetcher(session=mock_session)

        result = fetcher.fetch("us-east-1")

        assert "claude-sonnet-4" in result
        assert len(result["claude-sonnet-4"]) == 1

    def test_entry_has_correct_fields(self):
        product = _make_marketplace_product(
            description=_SAMPLE_DESC,
            price_usd="3.0000000000",
        )
        mock_session = _mock_paginator([product])
        fetcher = MarketplaceFetcher(session=mock_session)

        result = fetcher.fetch("us-east-1")
        entry = result["claude-sonnet-4"][0]

        assert isinstance(entry, MarketplaceEntry)
        assert entry.servicename == "claude-sonnet-4"
        assert entry.description == "Million Input Tokens Regional"
        assert entry.price_per_unit == 3.0
        assert entry.offer_term_code == "JRTCKXETXF"
        assert entry.rate_code == "ABC123.JRTCKXETXF.6YS6EN2CT7"
        assert entry.parse_errors == ()

    def test_unit_substituted_when_units_placeholder(self):
        """When API unit is 'Units', the description dimension is used instead."""
        desc = "AWS Marketplace software usage|us-east-1|Million Response Tokens Global"
        product = _make_marketplace_product(unit="Units", description=desc)
        mock_session = _mock_paginator([product])
        fetcher = MarketplaceFetcher(session=mock_session)

        result = fetcher.fetch("us-east-1")
        entry = result["claude-sonnet-4"][0]

        assert entry.unit == "Million Response Tokens Global"

    def test_unit_preserved_when_not_units_placeholder(self):
        """When API unit is something specific, it's preserved as-is."""
        product = _make_marketplace_product(unit="1K tokens")
        mock_session = _mock_paginator([product])
        fetcher = MarketplaceFetcher(session=mock_session)

        result = fetcher.fetch("us-east-1")
        entry = result["claude-sonnet-4"][0]

        assert entry.unit == "1K tokens"

    def test_skips_entries_without_servicename(self):
        product = _make_marketplace_product()
        product["product"]["attributes"]["servicename"] = ""

        mock_session = _mock_paginator([product])
        fetcher = MarketplaceFetcher(session=mock_session)

        result = fetcher.fetch("us-east-1")
        assert result == {}

    def test_skips_entries_without_on_demand_terms(self):
        product = _make_marketplace_product()
        product["terms"]["OnDemand"] = {}

        mock_session = _mock_paginator([product])
        fetcher = MarketplaceFetcher(session=mock_session)

        result = fetcher.fetch("us-east-1")
        assert result == {}

    def test_multiple_dimensions_produce_multiple_entries(self):
        product = _make_marketplace_product()
        # Add a second price dimension
        on_demand_key = list(product["terms"]["OnDemand"].keys())[0]
        product["terms"]["OnDemand"][on_demand_key]["priceDimensions"][
            "SECOND.RATE"
        ] = {
            "unit": "Units",
            "description": (
                "AWS Marketplace software usage|us-east-1"
                "|Million Response Tokens Regional"
            ),
            "rateCode": "SECOND.RATE",
            "pricePerUnit": {"USD": "15.0"},
        }

        mock_session = _mock_paginator([product])
        fetcher = MarketplaceFetcher(session=mock_session)

        result = fetcher.fetch("us-east-1")
        assert len(result["claude-sonnet-4"]) == 2

    def test_r_plus_normalized_in_servicename(self):
        """R+ in servicename is normalized to r-plus."""
        product = _make_marketplace_product(
            servicename="Cohere Command R+ (Amazon Bedrock Edition)"
        )
        mock_session = _mock_paginator([product])
        fetcher = MarketplaceFetcher(session=mock_session)

        result = fetcher.fetch("us-east-1")
        assert "cohere-command-r-plus" in result


class TestMarketplaceFetcherParseErrors:
    """Tests for deferred error handling in MarketplaceFetcher."""

    def test_missing_usd_price_produces_parse_error(self):
        product = _make_marketplace_product()
        # Remove the USD price
        on_demand_key = list(product["terms"]["OnDemand"].keys())[0]
        dim_key = list(
            product["terms"]["OnDemand"][on_demand_key]["priceDimensions"].keys()
        )[0]
        product["terms"]["OnDemand"][on_demand_key]["priceDimensions"][dim_key][
            "pricePerUnit"
        ] = {}

        mock_session = _mock_paginator([product])
        fetcher = MarketplaceFetcher(session=mock_session)

        result = fetcher.fetch("us-east-1")
        entry = result["claude-sonnet-4"][0]

        assert len(entry.parse_errors) == 1
        assert "Missing pricePerUnit.USD" in entry.parse_errors[0]
        assert entry.price_per_unit == 0.0

    def test_invalid_usd_price_produces_parse_error(self):
        product = _make_marketplace_product(price_usd="not-a-number")
        mock_session = _mock_paginator([product])
        fetcher = MarketplaceFetcher(session=mock_session)

        result = fetcher.fetch("us-east-1")
        entry = result["claude-sonnet-4"][0]

        assert len(entry.parse_errors) == 1
        assert "Invalid pricePerUnit.USD" in entry.parse_errors[0]
        assert entry.price_per_unit == 0.0

    def test_description_extracted_as_dimension_portion(self):
        """The description field stores only the dimension portion (after last |)."""
        desc = (
            "AWS Marketplace software usage|us-east-1"
            "|Million Cache Read Input Tokens Regional CRIS"
        )
        product = _make_marketplace_product(description=desc)
        mock_session = _mock_paginator([product])
        fetcher = MarketplaceFetcher(session=mock_session)

        result = fetcher.fetch("us-east-1")
        entry = result["claude-sonnet-4"][0]

        assert entry.description == "Million Cache Read Input Tokens Regional CRIS"
