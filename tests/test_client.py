"""Tests for the BedrockModelInfoClient."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import (
    ClientError,
    EndpointConnectionError,
    NoCredentialsError,
    NoRegionError,
)
from hypothesis import given, settings
from hypothesis import strategies as st

from model_info_amazon_bedrock import BedrockModelInfoClient, PricingNotFoundError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
from model_info_amazon_bedrock.pricing.types import UsagetypeEntry
from model_info_amazon_bedrock.types import (
    Direction,
    InferenceScope,
    ModelPricing,
    PriceDimension,
    PricingUnit,
    ServiceTier,
)

MODEL_ID = "anthropic.claude-sonnet-4-20250514-v1:0"
# The usagetype model segment must match the *normalized* model ID.
MODEL_USAGETYPE = "USE1-anthropic.claude-sonnet-4-20250514-input-tokens"


@pytest.fixture(autouse=True)
def _stable_default_aws_region(monkeypatch: pytest.MonkeyPatch):
    """Keep unit tests independent of the developer's boto3 configuration."""
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")


def _make_entry(
    usagetype: str = MODEL_USAGETYPE,
    unit: str = "1K tokens",
    price: float = 0.003,
    inference_type: str | None = "Input tokens",
    service_tier: str | None = None,
    rate_code: str = "SKU.OFFER.RATE",
) -> UsagetypeEntry:
    return UsagetypeEntry(
        usagetype=usagetype,
        inference_type=inference_type,
        provider="Anthropic",
        model="Claude Sonnet 4",
        service_tier=service_tier,
        unit=unit,
        price_per_unit=price,
        offer_term_code="OFFER",
        rate_code=rate_code,
    )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestGetModelPricingInputValidation:
    """Validates: Requirements 8.3"""

    def test_none_raises_value_error(self):
        client = BedrockModelInfoClient()
        with pytest.raises(ValueError, match="model_id must be a non-empty string"):
            client.get_model_pricing(None)  # type: ignore[arg-type]

    def test_empty_string_raises_value_error(self):
        client = BedrockModelInfoClient()
        with pytest.raises(ValueError, match="model_id must be a non-empty string"):
            client.get_model_pricing("")

    def test_whitespace_only_raises_value_error(self):
        client = BedrockModelInfoClient()
        with pytest.raises(ValueError, match="model_id must be a non-empty string"):
            client.get_model_pricing("   ")

    def test_non_string_raises_value_error(self):
        client = BedrockModelInfoClient()
        with pytest.raises(ValueError, match="model_id must be a non-empty string"):
            client.get_model_pricing(123)  # type: ignore[arg-type]

    @patch("model_info_amazon_bedrock.client.UsagetypeFetcher")
    def test_validation_before_api_call(self, mock_usagetype_fetcher_class: MagicMock):
        """Validation must happen before any API interaction."""
        mock_fetch_bedrock = mock_usagetype_fetcher_class.return_value.fetch_bedrock
        client = BedrockModelInfoClient()
        with pytest.raises(ValueError):
            client.get_model_pricing("")
        mock_fetch_bedrock.assert_not_called()


# ---------------------------------------------------------------------------
# ModelPricing output structure
# ---------------------------------------------------------------------------


class TestGetModelPricingResult:
    """Test that get_model_pricing returns a ModelPricing object."""

    @patch("model_info_amazon_bedrock.client.UsagetypeFetcher")
    def test_returns_model_pricing(self, mock_usagetype_fetcher_class: MagicMock):
        mock_fetch_bedrock = mock_usagetype_fetcher_class.return_value.fetch_bedrock
        entry = _make_entry(unit="1K tokens", price=0.003)
        mock_fetch_bedrock.return_value = [entry]

        client = BedrockModelInfoClient()
        result = client.get_model_pricing(MODEL_ID)

        assert isinstance(result, ModelPricing)
        assert result.model_id == MODEL_ID
        assert result.region == "us-east-1"

    @patch("model_info_amazon_bedrock.client.UsagetypeFetcher")
    def test_dimensions_are_price_dimensions(
        self, mock_usagetype_fetcher_class: MagicMock
    ):
        mock_fetch_bedrock = mock_usagetype_fetcher_class.return_value.fetch_bedrock
        entry = _make_entry(unit="1K tokens", price=0.003)
        mock_fetch_bedrock.return_value = [entry]

        client = BedrockModelInfoClient()
        result = client.get_model_pricing(MODEL_ID)

        assert len(result.dimensions) >= 1
        dim = result.dimensions[0]
        assert isinstance(dim, PriceDimension)
        assert dim.direction == Direction.INPUT
        assert dim.tier == ServiceTier.STANDARD
        assert dim.scope == InferenceScope.REGIONAL
        assert dim.unit == PricingUnit.MILLION_TOKENS
        assert dim.price == 3.0  # 0.003 * 1000

    @patch("model_info_amazon_bedrock.client.UsagetypeFetcher")
    def test_input_tokens_convenience(self, mock_usagetype_fetcher_class: MagicMock):
        """Convenience accessor for standard input token price."""
        mock_fetch_bedrock = mock_usagetype_fetcher_class.return_value.fetch_bedrock
        entries = [
            _make_entry(
                usagetype="USE1-anthropic.claude-sonnet-4-20250514-input-tokens",
                unit="1K tokens",
                price=0.003,
            ),
            _make_entry(
                usagetype="USE1-anthropic.claude-sonnet-4-20250514-output-tokens",
                unit="1K tokens",
                price=0.015,
            ),
        ]
        mock_fetch_bedrock.return_value = entries

        client = BedrockModelInfoClient()
        result = client.get_model_pricing(MODEL_ID)

        assert result.input_tokens == 3.0
        assert result.output_tokens == 15.0

    @patch("model_info_amazon_bedrock.client.UsagetypeFetcher")
    def test_image_pricing(self, mock_usagetype_fetcher_class: MagicMock):
        mock_fetch_bedrock = mock_usagetype_fetcher_class.return_value.fetch_bedrock
        entry = _make_entry(
            usagetype="USE1-stability.stable-image-ultra-input-image-count",
            unit="image",
            price=0.08,
        )
        mock_fetch_bedrock.return_value = [entry]

        client = BedrockModelInfoClient()
        result = client.get_model_pricing("stability.stable-image-ultra-v1:0")

        assert result.input_images == 0.08
        assert result.input_tokens is None


# ---------------------------------------------------------------------------
# Cache behavior
# ---------------------------------------------------------------------------


class TestGetModelPricingCache:
    """Validates: Requirements 5.1, 5.2, 5.3, 5.4"""

    @patch("model_info_amazon_bedrock.client.UsagetypeFetcher")
    def test_cache_prevents_redundant_fetches(
        self, mock_usagetype_fetcher_class: MagicMock
    ):
        mock_fetch_bedrock = mock_usagetype_fetcher_class.return_value.fetch_bedrock
        entry = _make_entry()
        mock_fetch_bedrock.return_value = [entry]

        client = BedrockModelInfoClient()
        client.get_model_pricing(MODEL_ID)
        client.get_model_pricing(MODEL_ID)

        mock_fetch_bedrock.assert_called_once()

    @patch("model_info_amazon_bedrock.client.UsagetypeFetcher")
    def test_refresh_bypasses_cache(self, mock_usagetype_fetcher_class: MagicMock):
        mock_fetch_bedrock = mock_usagetype_fetcher_class.return_value.fetch_bedrock
        entry = _make_entry()
        mock_fetch_bedrock.return_value = [entry]

        client = BedrockModelInfoClient()
        client.get_model_pricing(MODEL_ID)
        client.get_model_pricing(MODEL_ID, refresh=True)

        assert mock_fetch_bedrock.call_count == 2

    @patch("model_info_amazon_bedrock.client.UsagetypeFetcher")
    def test_invalidate_pricing_cache_forces_refetch(
        self, mock_usagetype_fetcher_class: MagicMock
    ):
        mock_fetch_bedrock = mock_usagetype_fetcher_class.return_value.fetch_bedrock
        entry = _make_entry()
        mock_fetch_bedrock.return_value = [entry]

        client = BedrockModelInfoClient()
        client.get_model_pricing(MODEL_ID)
        client.invalidate_pricing_cache("us-east-1")
        client.get_model_pricing(MODEL_ID)

        assert mock_fetch_bedrock.call_count == 2

    @patch("model_info_amazon_bedrock.client.UsagetypeFetcher")
    def test_failed_refresh_preserves_cache(
        self, mock_usagetype_fetcher_class: MagicMock
    ):
        mock_fetch_bedrock = mock_usagetype_fetcher_class.return_value.fetch_bedrock
        entry = _make_entry()
        mock_fetch_bedrock.return_value = [entry]

        client = BedrockModelInfoClient()
        # First call populates cache.
        client.get_model_pricing(MODEL_ID)

        # Second call with refresh fails.
        mock_fetch_bedrock.side_effect = RuntimeError("API error")
        with pytest.raises(RuntimeError, match="API error"):
            client.get_model_pricing(MODEL_ID, refresh=True)

        # Cache should still be intact for non-refresh calls.
        mock_fetch_bedrock.side_effect = None
        mock_fetch_bedrock.return_value = [entry]
        # Reset call count to verify no new fetch happens.
        mock_fetch_bedrock.reset_mock()
        result = client.get_model_pricing(MODEL_ID)
        mock_fetch_bedrock.assert_not_called()
        assert len(result.dimensions) >= 1

    @patch("model_info_amazon_bedrock.client.UsagetypeFetcher")
    def test_different_regions_cached_independently(
        self, mock_usagetype_fetcher_class: MagicMock
    ):
        mock_fetch_bedrock = mock_usagetype_fetcher_class.return_value.fetch_bedrock
        entry = _make_entry()
        mock_fetch_bedrock.return_value = [entry]

        client = BedrockModelInfoClient()
        client.get_model_pricing(MODEL_ID, region="us-east-1")
        client.get_model_pricing(MODEL_ID, region="us-west-2")

        assert mock_fetch_bedrock.call_count == 2


# ---------------------------------------------------------------------------
# PricingNotFoundError
# ---------------------------------------------------------------------------


class TestGetModelPricingNotFound:
    """Validates: Requirements 3.6, 8.5"""

    @patch("model_info_amazon_bedrock.client.MarketplaceFetcher")
    @patch("model_info_amazon_bedrock.client.UsagetypeFetcher")
    def test_raises_when_no_entries_match(
        self,
        mock_usagetype_fetcher_class: MagicMock,
        mock_marketplace_fetcher_class: MagicMock,
    ):
        mock_fetch_bedrock = mock_usagetype_fetcher_class.return_value.fetch_bedrock
        mock_fetch_bedrock_service = (
            mock_usagetype_fetcher_class.return_value.fetch_bedrock_service
        )
        mock_marketplace_fetch = mock_marketplace_fetcher_class.return_value.fetch
        # Return entries that don't match the requested model.
        entry = _make_entry(usagetype="USE1-some.other-model-input-tokens")
        mock_fetch_bedrock.return_value = [entry]
        mock_fetch_bedrock_service.return_value = []
        mock_marketplace_fetch.return_value = {}

        session = MagicMock()
        session.region_name = "eu-west-1"
        client = BedrockModelInfoClient(session=session)
        with pytest.raises(PricingNotFoundError) as exc_info:
            client.get_model_pricing(MODEL_ID)

        assert exc_info.value.model_id == MODEL_ID
        assert exc_info.value.region == "eu-west-1"


# ---------------------------------------------------------------------------
# Default region
# ---------------------------------------------------------------------------


class TestGetModelPricingDefaultRegion:
    """Validates: Requirements 4.2

    target Region resolution from explicit and session settings.
    """

    @patch("model_info_amazon_bedrock.client.UsagetypeFetcher")
    def test_uses_session_default_region(self, mock_usagetype_fetcher_class: MagicMock):
        mock_fetch_bedrock = mock_usagetype_fetcher_class.return_value.fetch_bedrock
        mock_fetch_bedrock.return_value = [_make_entry()]
        session = MagicMock()
        session.region_name = "eu-west-1"

        client = BedrockModelInfoClient(session=session)
        result = client.get_model_pricing(MODEL_ID)

        mock_fetch_bedrock.assert_called_once_with("eu-west-1")
        assert result.region == "eu-west-1"

    @patch("model_info_amazon_bedrock.client.UsagetypeFetcher")
    def test_explicit_region_overrides_session_default(
        self, mock_usagetype_fetcher_class: MagicMock
    ):
        mock_fetch_bedrock = mock_usagetype_fetcher_class.return_value.fetch_bedrock
        mock_fetch_bedrock.return_value = [_make_entry()]
        session = MagicMock()
        session.region_name = "eu-west-1"

        client = BedrockModelInfoClient(session=session)
        result = client.get_model_pricing(MODEL_ID, region="ap-southeast-2")

        mock_fetch_bedrock.assert_called_once_with("ap-southeast-2")
        assert result.region == "ap-southeast-2"

    @patch("model_info_amazon_bedrock.client.UsagetypeFetcher")
    def test_raises_when_no_target_region_is_available(
        self, mock_usagetype_fetcher_class: MagicMock
    ):
        mock_fetch_bedrock = mock_usagetype_fetcher_class.return_value.fetch_bedrock

        for session_region in (None, ""):
            session = MagicMock()
            session.region_name = session_region
            client = BedrockModelInfoClient(session=session)

            with pytest.raises(NoRegionError):
                client.get_model_pricing(MODEL_ID)

        mock_fetch_bedrock.assert_not_called()


# ---------------------------------------------------------------------------
# Boto3 exception propagation
# ---------------------------------------------------------------------------


class TestGetModelPricingAwsErrors:
    """Validates: Requirements 8.1, 8.2, 8.4"""

    @patch("model_info_amazon_bedrock.client.UsagetypeFetcher")
    def test_client_error_propagates(self, mock_usagetype_fetcher_class: MagicMock):
        """boto3 ClientError propagates without suppression."""
        mock_fetch_bedrock = mock_usagetype_fetcher_class.return_value.fetch_bedrock
        mock_fetch_bedrock.side_effect = ClientError(
            {"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}},
            "GetProducts",
        )

        client = BedrockModelInfoClient()
        with pytest.raises(ClientError, match="Rate exceeded"):
            client.get_model_pricing(MODEL_ID)

    @patch("model_info_amazon_bedrock.client.UsagetypeFetcher")
    def test_no_credentials_error_propagates(
        self, mock_usagetype_fetcher_class: MagicMock
    ):
        """boto3 NoCredentialsError propagates without suppression."""
        mock_fetch_bedrock = mock_usagetype_fetcher_class.return_value.fetch_bedrock
        mock_fetch_bedrock.side_effect = NoCredentialsError()

        client = BedrockModelInfoClient()
        with pytest.raises(NoCredentialsError):
            client.get_model_pricing(MODEL_ID)

    @patch("model_info_amazon_bedrock.client.UsagetypeFetcher")
    def test_endpoint_connection_error_propagates(
        self, mock_usagetype_fetcher_class: MagicMock
    ):
        """boto3 EndpointConnectionError propagates without suppression."""
        mock_fetch_bedrock = mock_usagetype_fetcher_class.return_value.fetch_bedrock
        mock_fetch_bedrock.side_effect = EndpointConnectionError(
            endpoint_url="https://pricing.us-east-1.amazonaws.com"
        )

        client = BedrockModelInfoClient()
        with pytest.raises(EndpointConnectionError):
            client.get_model_pricing(MODEL_ID)


# ---------------------------------------------------------------------------
# Property-based tests for input validation and caching
# ---------------------------------------------------------------------------


# Feature: Model Info for Amazon Bedrock, Property 8: Invalid model ID rejection
class TestGetModelPricingInvalidModelIdProperty:
    """Property 8: Invalid model ID rejection.

    For any value that is not a non-empty string (including None, empty string,
    whitespace-only strings, and non-string types), calling get_model_pricing SHALL
    raise a ValueError without making any API calls.

    **Validates: Requirements 8.3**
    """

    @settings(max_examples=200)
    @given(
        ws=st.text(
            alphabet=st.sampled_from([" ", "\t", "\n", "\r", "\x0b", "\x0c"]),
            min_size=1,
            max_size=20,
        )
    )
    @patch("model_info_amazon_bedrock.client.UsagetypeFetcher")
    def test_whitespace_only_strings_raise_value_error(
        self, mock_usagetype_fetcher_class: MagicMock, ws: str
    ):
        """Whitespace-only strings of any length raise ValueError."""
        mock_fetch_bedrock = mock_usagetype_fetcher_class.return_value.fetch_bedrock
        client = BedrockModelInfoClient()
        with pytest.raises(ValueError, match="model_id must be a non-empty string"):
            client.get_model_pricing(ws)
        mock_fetch_bedrock.assert_not_called()

    @settings(max_examples=200)
    @given(
        val=st.one_of(
            st.integers(),
            st.floats(allow_nan=False),
            st.lists(st.integers(), max_size=3),
            st.dictionaries(st.text(max_size=3), st.integers(), max_size=3),
            st.booleans(),
        )
    )
    @patch("model_info_amazon_bedrock.client.UsagetypeFetcher")
    def test_non_string_types_raise_value_error(
        self, mock_usagetype_fetcher_class: MagicMock, val: object
    ):
        """Non-string types (ints, floats, lists, dicts, booleans) raise ValueError."""
        mock_fetch_bedrock = mock_usagetype_fetcher_class.return_value.fetch_bedrock
        client = BedrockModelInfoClient()
        with pytest.raises(ValueError, match="model_id must be a non-empty string"):
            client.get_model_pricing(val)  # type: ignore[arg-type]
        mock_fetch_bedrock.assert_not_called()


# Feature: Model Info for Amazon Bedrock, Property 9: Cache prevents redundant fetches
class TestGetModelPricingCacheReuseProperty:
    """Property 9: Cache prevents redundant fetches.

    For any region and sequence of get_model_pricing calls for that region (with
    refresh=False), the Pricing API SHALL be called at most once (on the first
    lookup), and all subsequent lookups for the same region SHALL return results
    from cache without additional API calls.

    **Validates: Requirements 5.1**
    """

    @settings(max_examples=200)
    @given(
        region=st.text(
            alphabet=st.characters(
                whitelist_categories=("Ll", "Nd"),
                whitelist_characters="-",
            ),
            min_size=3,
            max_size=15,
        ),
        num_calls=st.integers(min_value=2, max_value=10),
    )
    @patch("model_info_amazon_bedrock.client.UsagetypeFetcher")
    def test_fetcher_called_at_most_once_per_region(
        self,
        mock_usagetype_fetcher_class: MagicMock,
        region: str,
        num_calls: int,
    ):
        """Fetcher called at most once for N calls with refresh=False."""
        mock_fetch_bedrock = mock_usagetype_fetcher_class.return_value.fetch_bedrock
        # Set up mock to return entries matching our known model ID
        entry = _make_entry()
        mock_fetch_bedrock.return_value = [entry]

        client = BedrockModelInfoClient()
        for _ in range(num_calls):
            client.get_model_pricing(MODEL_ID, region=region)

        mock_fetch_bedrock.assert_called_once_with(region)
