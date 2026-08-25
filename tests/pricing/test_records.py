"""Tests for internal pricing data models (UsagetypeEntry, ParsedUsagetype)."""

import pytest

from model_info_amazon_bedrock.pricing.types import (
    ParsedUsagetype,
    UsagetypeEntry,
)


class TestUsagetypeEntry:
    """Tests for UsagetypeEntry dataclass."""

    def test_creation_with_all_fields(self):
        entry = UsagetypeEntry(
            usagetype="USE1-anthropic.claude-sonnet-4-20250514-v1-input-tokens-standard",
            inference_type="Input tokens",
            provider="Anthropic",
            model="Claude Sonnet 4",
            service_tier="standard",
            unit="1K tokens",
            price_per_unit=0.003,
            offer_term_code="23AVXUXVTBTRAACY.JRTCKXETXF",
            rate_code="23AVXUXVTBTRAACY.JRTCKXETXF.6YS6EN2CT7",
        )
        expected = "USE1-anthropic.claude-sonnet-4-20250514-v1-input-tokens-standard"
        assert entry.usagetype == expected
        assert entry.inference_type == "Input tokens"
        assert entry.provider == "Anthropic"
        assert entry.model == "Claude Sonnet 4"
        assert entry.service_tier == "standard"
        assert entry.unit == "1K tokens"
        assert entry.price_per_unit == 0.003
        assert entry.offer_term_code == "23AVXUXVTBTRAACY.JRTCKXETXF"
        assert entry.rate_code == "23AVXUXVTBTRAACY.JRTCKXETXF.6YS6EN2CT7"

    def test_optional_fields_none(self):
        entry = UsagetypeEntry(
            usagetype="USE1-some-model-input-tokens",
            inference_type=None,
            provider=None,
            model=None,
            service_tier=None,
            unit="1K tokens",
            price_per_unit=0.001,
            offer_term_code="ABC.DEF",
            rate_code="ABC.DEF.GHI",
        )
        assert entry.inference_type is None
        assert entry.provider is None
        assert entry.model is None
        assert entry.service_tier is None

    def test_frozen(self):
        entry = UsagetypeEntry(
            usagetype="USE1-model-input-tokens",
            inference_type=None,
            provider=None,
            model=None,
            service_tier=None,
            unit="1K tokens",
            price_per_unit=0.001,
            offer_term_code="ABC.DEF",
            rate_code="ABC.DEF.GHI",
        )
        with pytest.raises(AttributeError):
            entry.price_per_unit = 0.999  # type: ignore[misc]


class TestParsedUsagetype:
    """Tests for ParsedUsagetype dataclass."""

    def test_creation(self):
        parsed = ParsedUsagetype(
            region_prefix="USE1",
            model_segment="anthropic.claude-sonnet-4-20250514-v1",
            has_mantle=False,
            inference_suffix="input-tokens-standard",
        )
        assert parsed.region_prefix == "USE1"
        assert parsed.model_segment == "anthropic.claude-sonnet-4-20250514-v1"
        assert parsed.has_mantle is False
        assert parsed.inference_suffix == "input-tokens-standard"

    def test_with_mantle(self):
        parsed = ParsedUsagetype(
            region_prefix="USW2",
            model_segment="zai.glm-4.7",
            has_mantle=True,
            inference_suffix="output-tokens-batch",
        )
        assert parsed.has_mantle is True

    def test_frozen(self):
        parsed = ParsedUsagetype(
            region_prefix="USE1",
            model_segment="model",
            has_mantle=False,
            inference_suffix="input-tokens",
        )
        with pytest.raises(AttributeError):
            parsed.region_prefix = "USW2"  # type: ignore[misc]
