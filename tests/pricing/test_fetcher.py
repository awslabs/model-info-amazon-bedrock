"""Tests for the pricing UsagetypeFetcher."""

import json
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import (
    ClientError,
    EndpointConnectionError,
    NoCredentialsError,
)
from hypothesis import given, settings
from hypothesis import strategies as st

from model_info_amazon_bedrock.pricing.fetcher import UsagetypeFetcher
from model_info_amazon_bedrock.pricing.types import UsagetypeEntry


def _make_product(
    usagetype="USE1-anthropic.claude-sonnet-4-20250514-v1-input-tokens",
    inference_type="Input tokens",
    provider="Anthropic",
    model="Claude Sonnet 4",
    service_tier=None,
    unit="1K tokens",
    price_usd="0.0030000000",
    rate_code="ABC123.JRTCKXETXF.6YS6EN2CT7",
    offer_term_code="JRTCKXETXF",
    sku="ABC123",
):
    """Helper to build a product dict matching the Pricing API format."""
    attributes = {
        "usagetype": usagetype,
        "regionCode": "us-east-1",
        "servicecode": "AmazonBedrock",
    }
    if inference_type is not None:
        attributes["inferenceType"] = inference_type
    if provider is not None:
        attributes["provider"] = provider
    if model is not None:
        attributes["model"] = model
    if service_tier is not None:
        attributes["service_tier"] = service_tier

    return {
        "product": {
            "productFamily": "Amazon Bedrock",
            "attributes": attributes,
            "sku": sku,
        },
        "serviceCode": "AmazonBedrock",
        "terms": {
            "OnDemand": {
                f"{sku}.{offer_term_code}": {
                    "priceDimensions": {
                        rate_code: {
                            "unit": unit,
                            "endRange": "Inf",
                            "description": "test",
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
    """Create a mock paginator that returns products as JSON strings."""
    mock_session = MagicMock()
    mock_client = MagicMock()
    mock_paginator = MagicMock()

    mock_session.client.return_value = mock_client
    mock_client.get_paginator.return_value = mock_paginator
    mock_paginator.paginate.return_value = [
        {"PriceList": [json.dumps(p) for p in products]}
    ]

    return mock_session


class TestUsagetypeFetcherInit:
    """Tests for UsagetypeFetcher initialization."""

    def test_uses_provided_session(self):
        mock_session = MagicMock()
        fetcher = UsagetypeFetcher(session=mock_session)
        assert fetcher._session is mock_session

    @patch("model_info_amazon_bedrock.pricing.fetcher.boto3.Session")
    def test_creates_default_session_when_none(self, mock_session_cls):
        fetcher = UsagetypeFetcher()
        mock_session_cls.assert_called_once()
        assert fetcher._session is mock_session_cls.return_value


class TestFetchBedrockPricing:
    """Tests for fetch_bedrock method."""

    def test_creates_client_in_us_east_1(self):
        mock_session = _mock_paginator([])
        fetcher = UsagetypeFetcher(session=mock_session)
        fetcher.fetch_bedrock("us-west-2")

        mock_session.client.assert_called_once_with("pricing", region_name="us-east-1")

    def test_uses_correct_service_code_and_filter(self):
        mock_session = _mock_paginator([])
        fetcher = UsagetypeFetcher(session=mock_session)
        fetcher.fetch_bedrock("us-west-2")

        mock_client = mock_session.client.return_value
        mock_paginator = mock_client.get_paginator.return_value
        mock_paginator.paginate.assert_called_once_with(
            ServiceCode="AmazonBedrock",
            Filters=[
                {
                    "Type": "TERM_MATCH",
                    "Field": "regionCode",
                    "Value": "us-west-2",
                },
            ],
        )

    def test_returns_raw_pricing_entries(self):
        product = _make_product()
        mock_session = _mock_paginator([product])
        fetcher = UsagetypeFetcher(session=mock_session)

        result = fetcher.fetch_bedrock("us-east-1")

        assert len(result) == 1
        entry = result[0]
        assert isinstance(entry, UsagetypeEntry)
        expected_ut = "USE1-anthropic.claude-sonnet-4-20250514-v1-input-tokens"
        assert entry.usagetype == expected_ut
        assert entry.inference_type == "Input tokens"
        assert entry.provider == "Anthropic"
        assert entry.model == "Claude Sonnet 4"
        assert entry.service_tier is None
        assert entry.unit == "1K tokens"
        assert entry.price_per_unit == 0.003
        assert entry.offer_term_code == "JRTCKXETXF"
        assert entry.rate_code == "ABC123.JRTCKXETXF.6YS6EN2CT7"

    def test_handles_missing_optional_attributes(self):
        product = _make_product(
            inference_type=None,
            provider=None,
            model=None,
            service_tier=None,
        )
        mock_session = _mock_paginator([product])
        fetcher = UsagetypeFetcher(session=mock_session)

        result = fetcher.fetch_bedrock("us-east-1")

        assert len(result) == 1
        entry = result[0]
        assert entry.inference_type is None
        assert entry.provider is None
        assert entry.model is None
        assert entry.service_tier is None

    def test_handles_service_tier_present(self):
        product = _make_product(service_tier="flex")
        mock_session = _mock_paginator([product])
        fetcher = UsagetypeFetcher(session=mock_session)

        result = fetcher.fetch_bedrock("us-east-1")

        assert result[0].service_tier == "flex"

    def test_paginates_multiple_pages(self):
        product1 = _make_product(usagetype="USE1-model-a-input-tokens", sku="SKU1")
        product2 = _make_product(usagetype="USE1-model-b-output-tokens", sku="SKU2")

        mock_session = MagicMock()
        mock_client = MagicMock()
        mock_paginator = MagicMock()

        mock_session.client.return_value = mock_client
        mock_client.get_paginator.return_value = mock_paginator
        # Simulate two pages
        mock_paginator.paginate.return_value = [
            {"PriceList": [json.dumps(product1)]},
            {"PriceList": [json.dumps(product2)]},
        ]

        fetcher = UsagetypeFetcher(session=mock_session)
        result = fetcher.fetch_bedrock("us-east-1")

        assert len(result) == 2
        assert result[0].usagetype == "USE1-model-a-input-tokens"
        assert result[1].usagetype == "USE1-model-b-output-tokens"

    def test_skips_entries_without_usagetype(self):
        product = _make_product()
        del product["product"]["attributes"]["usagetype"]

        mock_session = _mock_paginator([product])
        fetcher = UsagetypeFetcher(session=mock_session)

        result = fetcher.fetch_bedrock("us-east-1")
        assert result == []

    def test_skips_entries_without_on_demand_terms(self):
        product = _make_product()
        product["terms"]["OnDemand"] = {}

        mock_session = _mock_paginator([product])
        fetcher = UsagetypeFetcher(session=mock_session)

        result = fetcher.fetch_bedrock("us-east-1")
        assert result == []

    def test_handles_product_as_dict_not_json_string(self):
        """PriceList items can be dicts (not JSON strings) in some cases."""
        product = _make_product()

        mock_session = MagicMock()
        mock_client = MagicMock()
        mock_paginator_obj = MagicMock()

        mock_session.client.return_value = mock_client
        mock_client.get_paginator.return_value = mock_paginator_obj
        # Return product as a dict (not JSON string)
        mock_paginator_obj.paginate.return_value = [{"PriceList": [product]}]

        fetcher = UsagetypeFetcher(session=mock_session)
        result = fetcher.fetch_bedrock("us-east-1")

        assert len(result) == 1

    def test_price_per_unit_converted_to_float(self):
        product = _make_product(price_usd="0.0011000000")
        mock_session = _mock_paginator([product])
        fetcher = UsagetypeFetcher(session=mock_session)

        result = fetcher.fetch_bedrock("us-east-1")

        assert result[0].price_per_unit == 0.0011

    def test_empty_optional_string_treated_as_none(self):
        """Empty string values for optional fields should become None."""
        product = _make_product()
        product["product"]["attributes"]["inferenceType"] = ""
        product["product"]["attributes"]["provider"] = ""
        product["product"]["attributes"]["model"] = ""

        mock_session = _mock_paginator([product])
        fetcher = UsagetypeFetcher(session=mock_session)

        result = fetcher.fetch_bedrock("us-east-1")

        assert result[0].inference_type is None
        assert result[0].provider is None
        assert result[0].model is None


class TestParseProduct:
    """Tests for the _parse_product static method."""

    def test_returns_none_for_missing_usagetype(self):
        product = {"product": {"attributes": {}}, "terms": {"OnDemand": {}}}
        assert UsagetypeFetcher._parse_product(product) is None

    def test_returns_none_for_empty_on_demand(self):
        product = {
            "product": {"attributes": {"usagetype": "USE1-test-input-tokens"}},
            "terms": {"OnDemand": {}},
        }
        assert UsagetypeFetcher._parse_product(product) is None

    def test_returns_none_for_missing_terms(self):
        product = {
            "product": {"attributes": {"usagetype": "USE1-test-input-tokens"}},
            "terms": {},
        }
        assert UsagetypeFetcher._parse_product(product) is None

    def test_multiple_price_dimensions(self):
        """A product with multiple price dimensions returns multiple entries."""
        product = _make_product()
        # Add a second price dimension
        on_demand_key = list(product["terms"]["OnDemand"].keys())[0]
        product["terms"]["OnDemand"][on_demand_key]["priceDimensions"][
            "SECOND.RATE.CODE"
        ] = {
            "unit": "image",
            "endRange": "Inf",
            "description": "test2",
            "appliesTo": [],
            "rateCode": "SECOND.RATE.CODE",
            "beginRange": "0",
            "pricePerUnit": {"USD": "0.05"},
        }

        result = UsagetypeFetcher._parse_product(product)
        assert result is not None
        assert len(result) == 2


class TestBoto3ExceptionPropagation:
    """Tests that boto3/botocore exceptions propagate without being caught."""

    def test_client_error_propagates(self):
        """ClientError from paginator should propagate to caller."""
        mock_session = MagicMock()
        mock_client = MagicMock()
        mock_paginator = MagicMock()

        mock_session.client.return_value = mock_client
        mock_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value.__iter__ = MagicMock(
            side_effect=ClientError(
                {"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}},
                "GetProducts",
            )
        )

        fetcher = UsagetypeFetcher(session=mock_session)

        with pytest.raises(ClientError) as exc_info:
            fetcher.fetch_bedrock("us-east-1")

        assert "ThrottlingException" in str(exc_info.value)

    def test_endpoint_connection_error_propagates(self):
        """EndpointConnectionError from client creation should propagate."""
        mock_session = MagicMock()
        mock_client = MagicMock()
        mock_paginator = MagicMock()

        mock_session.client.return_value = mock_client
        mock_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value.__iter__ = MagicMock(
            side_effect=EndpointConnectionError(
                endpoint_url="https://pricing.us-east-1.amazonaws.com"
            )
        )

        fetcher = UsagetypeFetcher(session=mock_session)

        with pytest.raises(EndpointConnectionError):
            fetcher.fetch_bedrock("us-east-1")

    def test_no_credentials_error_propagates(self):
        """NoCredentialsError should propagate to caller."""
        mock_session = MagicMock()
        mock_session.client.side_effect = NoCredentialsError()

        fetcher = UsagetypeFetcher(session=mock_session)

        with pytest.raises(NoCredentialsError):
            fetcher.fetch_bedrock("us-east-1")


# Feature: Model Info for Amazon Bedrock, Property 1:
# Field extraction preserves all data from valid entries


def _api_response_strategy():
    """Strategy to generate valid Pricing API response dicts.

    Generates random subsets of optional fields.
    """
    # Required fields
    usagetype_st = st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz0123456789.-",
        min_size=3,
        max_size=50,
    ).map(lambda s: f"USE1-{s}-input-tokens")

    unit_st = st.sampled_from(["1K tokens", "image", "requests", "hours"])
    price_usd_st = st.floats(
        min_value=0.0,
        max_value=100.0,
        allow_nan=False,
        allow_infinity=False,
    ).map(lambda f: f"{f:.10f}")
    rate_code_st = st.text(
        alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.",
        min_size=5,
        max_size=30,
    )
    offer_term_code_st = st.text(
        alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        min_size=5,
        max_size=20,
    )

    # Optional fields - each is either present (a non-empty string) or absent
    optional_str_st = st.text(
        alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -_",
        min_size=1,
        max_size=30,
    )

    return st.fixed_dictionaries(
        {
            "usagetype": usagetype_st,
            "unit": unit_st,
            "price_usd": price_usd_st,
            "rate_code": rate_code_st,
            "offer_term_code": offer_term_code_st,
            "inference_type": st.one_of(st.none(), optional_str_st),
            "provider": st.one_of(st.none(), optional_str_st),
            "model": st.one_of(st.none(), optional_str_st),
            "service_tier": st.one_of(st.none(), optional_str_st),
        }
    )


def _build_api_product(data: dict) -> dict:
    """Build a Pricing API product dict from strategy-generated data."""
    attributes = {"usagetype": data["usagetype"]}
    if data["inference_type"] is not None:
        attributes["inferenceType"] = data["inference_type"]
    if data["provider"] is not None:
        attributes["provider"] = data["provider"]
    if data["model"] is not None:
        attributes["model"] = data["model"]
    if data["service_tier"] is not None:
        attributes["service_tier"] = data["service_tier"]

    offer_key = f"SKU.{data['offer_term_code']}"
    dim_key = data["rate_code"]

    return {
        "product": {
            "attributes": attributes,
        },
        "terms": {
            "OnDemand": {
                offer_key: {
                    "offerTermCode": data["offer_term_code"],
                    "priceDimensions": {
                        dim_key: {
                            "unit": data["unit"],
                            "pricePerUnit": {"USD": data["price_usd"]},
                            "rateCode": data["rate_code"],
                        }
                    },
                }
            }
        },
    }


class TestUsagetypeFetcherParseErrors:
    """Tests for deferred error handling in UsagetypeFetcher."""

    def test_missing_usd_price_produces_parse_error(self):
        product = _make_product()
        # Remove the USD price
        on_demand_key = list(product["terms"]["OnDemand"].keys())[0]
        dim_key = list(
            product["terms"]["OnDemand"][on_demand_key]["priceDimensions"].keys()
        )[0]
        product["terms"]["OnDemand"][on_demand_key]["priceDimensions"][dim_key][
            "pricePerUnit"
        ] = {}

        mock_session = _mock_paginator([product])
        fetcher = UsagetypeFetcher(session=mock_session)

        result = fetcher.fetch_bedrock("us-east-1")
        assert len(result) == 1
        entry = result[0]
        assert len(entry.parse_errors) == 1
        assert "Missing pricePerUnit.USD" in entry.parse_errors[0]
        assert entry.price_per_unit == 0.0

    def test_invalid_usd_price_produces_parse_error(self):
        product = _make_product(price_usd="not-a-number")
        mock_session = _mock_paginator([product])
        fetcher = UsagetypeFetcher(session=mock_session)

        result = fetcher.fetch_bedrock("us-east-1")
        assert len(result) == 1
        entry = result[0]
        assert len(entry.parse_errors) == 1
        assert "Invalid pricePerUnit.USD" in entry.parse_errors[0]
        assert entry.price_per_unit == 0.0

    def test_valid_price_has_no_parse_errors(self):
        product = _make_product(price_usd="0.003")
        mock_session = _mock_paginator([product])
        fetcher = UsagetypeFetcher(session=mock_session)

        result = fetcher.fetch_bedrock("us-east-1")
        assert result[0].parse_errors == ()


class TestFieldExtractionProperty:
    """Property 1: Field extraction preserves all data from valid entries."""

    @given(data=_api_response_strategy())
    @settings(max_examples=200)
    def test_field_extraction_preserves_all_data(self, data):
        """
        For any valid Pricing API response entry containing usagetype, unit, and
        pricePerUnit.USD fields (with optional inferenceType, provider, model,
        service_tier), parsing it into a UsagetypeEntry preserves all present
        field values exactly, and sets any missing optional fields to None.

        **Validates: Requirements 1.3, 1.4**
        """
        product = _build_api_product(data)
        result = UsagetypeFetcher._parse_product(product)

        # Should always produce a result since we have usagetype and OnDemand terms
        assert result is not None
        assert len(result) == 1

        entry = result[0]

        # Required fields are always preserved
        assert entry.usagetype == data["usagetype"]
        assert entry.unit == data["unit"]
        assert entry.price_per_unit == float(data["price_usd"])
        assert entry.offer_term_code == data["offer_term_code"]
        assert entry.rate_code == data["rate_code"]

        # Optional fields: present values are preserved exactly, missing are None
        if data["inference_type"] is not None:
            assert entry.inference_type == data["inference_type"]
        else:
            assert entry.inference_type is None

        if data["provider"] is not None:
            assert entry.provider == data["provider"]
        else:
            assert entry.provider is None

        if data["model"] is not None:
            assert entry.model == data["model"]
        else:
            assert entry.model is None

        if data["service_tier"] is not None:
            assert entry.service_tier == data["service_tier"]
        else:
            assert entry.service_tier is None
