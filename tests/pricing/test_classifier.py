"""Tests for the price record classification layer."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from model_info_amazon_bedrock.pricing.classifier import (
    classify_marketplace_entry,
    classify_usagetype_entry,
)
from model_info_amazon_bedrock.pricing.types import (
    MarketplaceEntry,
    UsagetypeEntry,
)
from model_info_amazon_bedrock.types import (
    CacheOperation,
    ContextLength,
    Direction,
    InferenceScope,
    Modality,
    PricingUnit,
    ServiceTier,
)

from ..conftest import INFERENCE_SUFFIXES


def _make_entry(
    usagetype: str = "USE1-test.model-input-tokens",
    unit: str = "1K tokens",
    price: float = 0.003,
    rate_code: str = "SKU.OFFER.RATE",
) -> UsagetypeEntry:
    return UsagetypeEntry(
        usagetype=usagetype,
        inference_type=None,
        provider="Test",
        model="Test Model",
        service_tier=None,
        unit=unit,
        price_per_unit=price,
        offer_term_code="OFFER",
        rate_code=rate_code,
    )


class TestClassifyDirection:
    """Test direction parsing from inference suffixes."""

    @pytest.mark.parametrize(
        "suffix,expected",
        [
            ("input-tokens", Direction.INPUT),
            ("input-tokens-standard", Direction.INPUT),
            ("input-tokens-batch", Direction.INPUT),
            ("input-tokens-cross-region-global", Direction.INPUT),
            ("output-tokens", Direction.OUTPUT),
            ("output-tokens-standard", Direction.OUTPUT),
            ("output-tokens-batch", Direction.OUTPUT),
            ("output-tokens-cross-region-global", Direction.OUTPUT),
            ("input-image-count", Direction.INPUT),
            ("output-image-count", Direction.OUTPUT),
            ("input-image-token-count", Direction.INPUT),
            ("output-image-token-count", Direction.OUTPUT),
        ],
    )
    def test_direction(self, suffix, expected):
        entry = _make_entry()
        result = classify_usagetype_entry(entry, suffix)
        assert result is not None
        assert result.direction == expected


class TestClassifyModality:
    """Test modality parsing from inference suffixes."""

    @pytest.mark.parametrize(
        "suffix,expected",
        [
            ("input-tokens", Modality.TOKENS),
            ("output-tokens-batch", Modality.TOKENS),
            ("cache-read-input-token-count", Modality.TOKENS),
            ("input-image-count", Modality.IMAGE),
            ("output-image-count", Modality.IMAGE),
            ("input-image-token-count", Modality.IMAGE),
        ],
    )
    def test_modality(self, suffix, expected):
        entry = _make_entry()
        result = classify_usagetype_entry(entry, suffix)
        assert result is not None
        assert result.modality == expected


class TestClassifyCache:
    """Test cache operation parsing from inference suffixes."""

    @pytest.mark.parametrize(
        "suffix,expected",
        [
            ("input-tokens", CacheOperation.NONE),
            ("output-tokens", CacheOperation.NONE),
            ("cache-read-input-token-count", CacheOperation.READ),
            ("cache-read-input-token-count-cross-region-global", CacheOperation.READ),
            ("cache-write-input-token-count", CacheOperation.WRITE),
            ("cache-write-input-token-count-cross-region-geo", CacheOperation.WRITE),
        ],
    )
    def test_cache(self, suffix, expected):
        entry = _make_entry()
        result = classify_usagetype_entry(entry, suffix)
        assert result is not None
        assert result.cache == expected


class TestClassifyTier:
    """Test tier parsing from inference suffixes."""

    @pytest.mark.parametrize(
        "suffix,expected",
        [
            ("input-tokens", ServiceTier.STANDARD),
            ("input-tokens-standard", ServiceTier.STANDARD),
            ("input-tokens-batch", ServiceTier.BATCH),
            ("output-tokens-batch", ServiceTier.BATCH),
            ("input-tokens-flex", ServiceTier.FLEX),
            ("output-tokens-flex", ServiceTier.FLEX),
            ("input-tokens-priority", ServiceTier.PRIORITY),
            ("output-tokens-priority", ServiceTier.PRIORITY),
            ("input-tokens-cross-region-global-batch", ServiceTier.BATCH),
        ],
    )
    def test_tier(self, suffix, expected):
        entry = _make_entry()
        result = classify_usagetype_entry(entry, suffix)
        assert result is not None
        assert result.tier == expected


class TestClassifyScope:
    """Test scope parsing from inference suffixes."""

    @pytest.mark.parametrize(
        "suffix,expected",
        [
            ("input-tokens", InferenceScope.REGIONAL),
            ("input-tokens-standard", InferenceScope.REGIONAL),
            ("input-tokens-cross-region-global", InferenceScope.CROSS_REGION_GLOBAL),
            ("output-tokens-cross-region-global", InferenceScope.CROSS_REGION_GLOBAL),
            ("input-tokens-cross-region-geo", InferenceScope.CROSS_REGION_GEO),
            (
                "cache-read-input-token-count-cross-region-global",
                InferenceScope.CROSS_REGION_GLOBAL,
            ),  # noqa: E501
        ],
    )
    def test_scope(self, suffix, expected):
        entry = _make_entry()
        result = classify_usagetype_entry(entry, suffix)
        assert result is not None
        assert result.scope == expected


class TestClassifyContext:
    """Test context length parsing from inference suffixes."""

    @pytest.mark.parametrize(
        "suffix,expected",
        [
            ("input-tokens", ContextLength.STANDARD),
            ("input-tokens-cross-region-global", ContextLength.STANDARD),
            (
                "input-tokens-long-context-cross-region-global",
                ContextLength.LONG,
            ),
            (
                "output-tokens-long-context-cross-region-global",
                ContextLength.LONG,
            ),
            (
                "cache-read-input-token-count-long-context-cross-region-global",
                ContextLength.LONG,
            ),
        ],
    )
    def test_context(self, suffix, expected):
        entry = _make_entry()
        result = classify_usagetype_entry(entry, suffix)
        assert result is not None
        assert result.context == expected


class TestClassifyReserved:
    """Reserved suffixes should return None."""

    @pytest.mark.parametrize(
        "suffix",
        [
            "reserved-1-month-input-tokens-per-minute-cross-region-global",
            "reserved-3-month-output-tokens-per-minute-cross-region-geo",
        ],
    )
    def test_reserved_returns_none(self, suffix):
        entry = _make_entry()
        result = classify_usagetype_entry(entry, suffix)
        assert result is None


class TestClassifyPriceNormalization:
    """Test price normalization in classification."""

    def test_1k_tokens_normalized_to_per_million(self):
        entry = _make_entry(unit="1K tokens", price=0.003)
        result = classify_usagetype_entry(entry, "input-tokens")
        assert result is not None
        assert result.unit == PricingUnit.MILLION_TOKENS
        assert result.price == 3.0

    def test_image_passthrough(self):
        entry = _make_entry(unit="image", price=0.08)
        result = classify_usagetype_entry(entry, "input-image-count")
        assert result is not None
        assert result.unit == PricingUnit.IMAGE
        assert result.price == 0.08

    def test_unknown_unit(self):
        entry = _make_entry(unit="requests", price=0.01)
        result = classify_usagetype_entry(entry, "input-tokens")
        assert result is not None
        assert result.unit == PricingUnit.UNKNOWN
        assert result.price == 0.01


class TestClassifyProperty:
    """Property: classification always produces valid enum combinations."""

    @given(
        suffix=st.sampled_from(
            [s for s in INFERENCE_SUFFIXES if not s.startswith("reserved")]
        ),
        price=st.floats(min_value=0, max_value=100, allow_nan=False),
        unit=st.sampled_from(["1K tokens", "image", "requests"]),
    )
    @settings(max_examples=200)
    def test_produces_valid_enums(self, suffix, price, unit):
        """Every non-reserved suffix produces a valid PriceDimension."""
        entry = _make_entry(unit=unit, price=price)
        result = classify_usagetype_entry(entry, suffix)
        assert result is not None
        assert isinstance(result.direction, Direction)
        assert isinstance(result.modality, Modality)
        assert isinstance(result.cache, CacheOperation)
        assert isinstance(result.tier, ServiceTier)
        assert isinstance(result.scope, InferenceScope)
        assert isinstance(result.context, ContextLength)
        assert isinstance(result.unit, PricingUnit)


class TestClassifyMarketplace:
    """Test marketplace record classification."""

    def test_input_tokens(self):
        record = MarketplaceEntry(
            servicename="test",
            description="Million Input Tokens Regional",
            unit="Million Input Tokens Regional",
            price_per_unit=3.0,
            offer_term_code="OFFER",
            rate_code="R1",
        )
        result = classify_marketplace_entry(record)
        assert result is not None
        assert result.direction == Direction.INPUT
        assert result.modality == Modality.TOKENS
        assert result.cache == CacheOperation.NONE
        assert result.tier == ServiceTier.STANDARD
        assert result.unit == PricingUnit.MILLION_TOKENS
        assert result.price == 3.0

    def test_output_tokens(self):
        record = MarketplaceEntry(
            servicename="test",
            description="Million Response Tokens Regional",
            unit="Million Response Tokens Regional",
            price_per_unit=15.0,
            offer_term_code="OFFER",
            rate_code="R1b",
        )
        result = classify_marketplace_entry(record)
        assert result is not None
        assert result.direction == Direction.OUTPUT
        assert result.price == 15.0

    def test_cache_read(self):
        record = MarketplaceEntry(
            servicename="test",
            description="Million Cache Read Input Tokens Regional",
            unit="Million Cache Read Input Tokens Regional",
            price_per_unit=0.3,
            offer_term_code="OFFER",
            rate_code="R2",
        )
        result = classify_marketplace_entry(record)
        assert result is not None
        assert result.direction == Direction.INPUT
        assert result.cache == CacheOperation.READ

    def test_cache_write_1h(self):
        record = MarketplaceEntry(
            servicename="test",
            description="Million 1 hour Cache Write Input Tokens Global",
            unit="Million 1 hour Cache Write Input Tokens Global",
            price_per_unit=5.0,
            offer_term_code="OFFER",
            rate_code="R3",
        )
        result = classify_marketplace_entry(record)
        assert result is not None
        assert result.cache == CacheOperation.WRITE_1H

    def test_reserved_returns_none(self):
        record = MarketplaceEntry(
            servicename="test",
            description="Per Hour per 1K Input TPM Reserved 1 Month Regional",
            unit="Per Hour per 1K Input TPM Reserved 1 Month Regional",
            price_per_unit=0.2,
            offer_term_code="OFFER",
            rate_code="R4",
        )
        result = classify_marketplace_entry(record)
        assert result is None

    def test_unrecognized_description_returns_none(self):
        record = MarketplaceEntry(
            servicename="test",
            description="Something Completely Unknown",
            unit="Something Completely Unknown",
            price_per_unit=0.0,
            offer_term_code="OFFER",
            rate_code="R5",
        )
        result = classify_marketplace_entry(record)
        assert result is None

    def test_parse_error_raises(self):
        """Entries with parse_errors raise ValueError at classification time."""
        record = MarketplaceEntry(
            servicename="test",
            description="Million Input Tokens Regional",
            unit="Million Input Tokens Regional",
            price_per_unit=0.0,
            offer_term_code="OFFER",
            rate_code="R6",
            parse_errors=("Missing pricePerUnit.USD",),
        )
        with pytest.raises(ValueError, match="Missing pricePerUnit.USD"):
            classify_marketplace_entry(record)

    def test_unrecognized_unit_produces_unknown(self):
        """Entries with unrecognized units produce PricingUnit.UNKNOWN."""
        record = MarketplaceEntry(
            servicename="test",
            description="Million Input Tokens Regional",
            unit="widgets",
            price_per_unit=1.0,
            offer_term_code="OFFER",
            rate_code="R7",
        )
        result = classify_marketplace_entry(record)
        assert result is not None
        assert result.unit == PricingUnit.UNKNOWN
        assert result.price == 1.0
