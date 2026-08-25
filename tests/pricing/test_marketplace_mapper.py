"""Tests for the MarketplaceMapper and model ID component parsing."""

import pytest

from model_info_amazon_bedrock.pricing.marketplace_mapper import (
    MarketplaceMapper,
    _ModelIdComponents,
    normalize_servicename,
)
from model_info_amazon_bedrock.pricing.types import MarketplaceEntry

# ---------------------------------------------------------------------------
# _ModelIdComponents.parse
# ---------------------------------------------------------------------------


class TestModelIdComponentsParse:
    """Test parsing model IDs into structured components."""

    def test_simple_model_id(self):
        p = _ModelIdComponents.parse("cohere.command-r")
        assert p.provider == "cohere"
        assert p.model_name == "command-r"
        assert p.geo == ""
        assert p.date == ""
        assert p.version == ""
        assert p.context == ""

    def test_model_with_version_suffix(self):
        p = _ModelIdComponents.parse("anthropic.claude-sonnet-4-20250514-v1:0")
        assert p.provider == "anthropic"
        assert p.model_name == "claude-sonnet-4"
        assert p.date == "20250514"
        assert p.version == "1"
        assert p.context == ""

    def test_model_with_bare_version(self):
        """Bare :0 suffix (no -v prefix) is stripped."""
        p = _ModelIdComponents.parse("cohere.rerank-v3-5:0")
        assert p.provider == "cohere"
        assert p.model_name == "rerank-v3-5"
        assert p.version == ""
        assert p.context == ""

    def test_model_with_context_length(self):
        p = _ModelIdComponents.parse("amazon.nova-pro-v1:0:24k")
        assert p.provider == "amazon"
        assert p.model_name == "nova-pro"
        assert p.version == "1"
        assert p.context == "24k"

    def test_model_with_large_context(self):
        p = _ModelIdComponents.parse("amazon.nova-pro-v1:0:300k")
        assert p.provider == "amazon"
        assert p.model_name == "nova-pro"
        assert p.version == "1"
        assert p.context == "300k"

    def test_geo_prefix(self):
        p = _ModelIdComponents.parse("global.anthropic.claude-opus-4-5-20251101-v1:0")
        assert p.geo == "global"
        assert p.provider == "anthropic"
        assert p.model_name == "claude-opus-4-5"
        assert p.date == "20251101"
        assert p.version == "1"

    def test_us_geo_prefix(self):
        p = _ModelIdComponents.parse("us.anthropic.claude-opus-4-5-20251101-v1:0")
        assert p.geo == "us"
        assert p.provider == "anthropic"
        assert p.model_name == "claude-opus-4-5"
        assert p.date == "20251101"

    def test_model_no_provider(self):
        """Single-segment model ID (no dots)."""
        p = _ModelIdComponents.parse("ray2")
        assert p.provider == ""
        assert p.model_name == "ray2"

    def test_model_with_dots_in_name(self):
        """Provider.model where model has no special suffixes."""
        p = _ModelIdComponents.parse("luma.ray2")
        assert p.provider == "luma"
        assert p.model_name == "ray2"

    def test_date_stripped_from_middle(self):
        """Date in the middle of the model name is stripped cleanly."""
        p = _ModelIdComponents.parse("anthropic.claude-3-5-haiku-20241022")
        assert p.provider == "anthropic"
        assert p.model_name == "claude-3-5-haiku"
        assert p.date == "20241022"

    def test_model_with_plus_in_name(self):
        """R+ becomes r- after symbol normalization."""
        p = _ModelIdComponents.parse("cohere.command-r-plus")
        assert p.provider == "cohere"
        assert p.model_name == "command-r-plus"

    def test_embed_model(self):
        p = _ModelIdComponents.parse("cohere.embed-english-v3")
        assert p.provider == "cohere"
        assert p.model_name == "embed-english"
        assert p.version == "3"

    def test_embed_model_v4(self):
        p = _ModelIdComponents.parse("cohere.embed-v4")
        assert p.provider == "cohere"
        assert p.model_name == "embed"
        assert p.version == "4"

    def test_jamba_model(self):
        p = _ModelIdComponents.parse("ai21.jamba-1-5-large")
        assert p.provider == "ai21"
        assert p.model_name == "jamba-1-5-large"

    def test_twelvelabs_model(self):
        p = _ModelIdComponents.parse("twelvelabs.marengo-embed-2-7")
        assert p.provider == "twelvelabs"
        assert p.model_name == "marengo-embed-2-7"

    def test_writer_model(self):
        p = _ModelIdComponents.parse("writer.palmyra-x4")
        assert p.provider == "writer"
        assert p.model_name == "palmyra-x4"


# ---------------------------------------------------------------------------
# normalize_servicename
# ---------------------------------------------------------------------------


class TestNormalizeServicename:
    """Test servicename normalization."""

    def test_strips_bedrock_edition_suffix(self):
        result = normalize_servicename("Claude Opus 4 (Amazon Bedrock Edition)")
        assert result == "claude-opus-4"

    def test_handles_r_plus(self):
        result = normalize_servicename("Cohere Command R+ (Amazon Bedrock Edition)")
        assert result == "cohere-command-r-plus"

    def test_normalizes_dots_to_hyphens(self):
        result = normalize_servicename("Claude Opus 4.5 (Amazon Bedrock Edition)")
        assert result == "claude-opus-4-5"

    def test_normalizes_spaces_to_hyphens(self):
        result = normalize_servicename("Jamba 1.5 Large (Amazon Bedrock Edition)")
        assert result == "jamba-1-5-large"

    def test_collapses_multiple_hyphens(self):
        result = normalize_servicename(
            "Cohere Embed 3 Model - English (Amazon Bedrock Edition)"
        )
        assert result == "cohere-embed-3-model-english"

    def test_no_suffix(self):
        """Works even without the Bedrock Edition suffix."""
        result = normalize_servicename("Claude Opus 4")
        assert result == "claude-opus-4"


# ---------------------------------------------------------------------------
# MarketplaceMapper.match
# ---------------------------------------------------------------------------


def _make_entry(
    description: str = "Million Input Tokens Regional",
    price: float = 3.0,
) -> MarketplaceEntry:
    return MarketplaceEntry(
        servicename="test",
        description=description,
        unit="Million Input Tokens Regional",
        price_per_unit=price,
        offer_term_code="OFFER",
        rate_code="R1",
    )


class TestMarketplaceMapperMatch:
    """Test the full matching pipeline."""

    @pytest.fixture
    def sample_data(self) -> dict[str, list[MarketplaceEntry]]:
        """Simulated marketplace data keyed by normalized servicename."""
        return {
            "claude-opus-4": [_make_entry(price=15.0)],
            "claude-opus-4-1": [_make_entry(price=15.0)],
            "claude-opus-4-5": [_make_entry(price=15.0)],
            "claude-sonnet-4": [_make_entry(price=3.0)],
            "claude-sonnet-4-5": [_make_entry(price=3.0)],
            "claude-3-5-haiku": [_make_entry(price=0.8)],
            "cohere-command-r": [_make_entry(price=0.5)],
            "cohere-command-r-plus": [_make_entry(price=3.0)],
            "cohere-embed-3-model-english": [_make_entry(price=0.1)],
            "cohere-embed-model-3-multilingual": [_make_entry(price=0.1)],
            "cohere-embed-4-model": [_make_entry(price=0.1)],
            "cohere-rerank-v3-5": [_make_entry(price=0.002)],
            "jamba-1-5-large": [_make_entry(price=2.0)],
            "palmyra-x4": [_make_entry(price=5.0)],
            "twelvelabs-marengo-embed-2-7": [_make_entry(price=0.0)],
            "luma-ray2": [_make_entry(price=0.0)],
        }

    def test_exact_algorithmic_match(self, sample_data):
        result = MarketplaceMapper.match(
            "anthropic.claude-opus-4-20250514",
            "anthropic.claude-opus-4-20250514",
            sample_data,
        )
        assert result is not None
        assert result[0].price_per_unit == 15.0

    def test_versioned_model_id(self, sample_data):
        result = MarketplaceMapper.match(
            "anthropic.claude-sonnet-4-20250514-v1:0",
            "anthropic.claude-sonnet-4-20250514",
            sample_data,
        )
        assert result is not None
        assert result[0].price_per_unit == 3.0

    def test_disambiguates_opus_4_vs_4_5(self, sample_data):
        result = MarketplaceMapper.match(
            "anthropic.claude-opus-4-5-20251101",
            "anthropic.claude-opus-4-5-20251101",
            sample_data,
        )
        assert result is not None
        assert result[0].price_per_unit == 15.0

    def test_cohere_command_r_plus(self, sample_data):
        result = MarketplaceMapper.match(
            "cohere.command-r-plus",
            "cohere.command-r-plus",
            sample_data,
        )
        assert result is not None
        assert result[0].price_per_unit == 3.0

    def test_cohere_command_r(self, sample_data):
        result = MarketplaceMapper.match(
            "cohere.command-r",
            "cohere.command-r",
            sample_data,
        )
        assert result is not None
        assert result[0].price_per_unit == 0.5

    def test_cohere_embed_uses_exception_table(self, sample_data):
        """Cohere Embed models use the exceptions table."""
        result = MarketplaceMapper.match(
            "cohere.embed-english-v3",
            "cohere.embed-english-v3",
            sample_data,
        )
        assert result is not None
        assert result[0].price_per_unit == 0.1

    def test_cohere_embed_v4_uses_exception_table(self, sample_data):
        result = MarketplaceMapper.match(
            "cohere.embed-v4",
            "cohere.embed-v4",
            sample_data,
        )
        assert result is not None

    def test_jamba(self, sample_data):
        result = MarketplaceMapper.match(
            "ai21.jamba-1-5-large",
            "ai21.jamba-1-5-large",
            sample_data,
        )
        assert result is not None
        assert result[0].price_per_unit == 2.0

    def test_luma(self, sample_data):
        result = MarketplaceMapper.match(
            "luma.ray2",
            "luma.ray2",
            sample_data,
        )
        assert result is not None

    def test_no_match_returns_none(self, sample_data):
        result = MarketplaceMapper.match(
            "nonexistent.model-xyz",
            "nonexistent.model-xyz",
            sample_data,
        )
        assert result is None

    def test_empty_data_returns_none(self):
        result = MarketplaceMapper.match(
            "anthropic.claude-opus-4-20250514",
            "anthropic.claude-opus-4-20250514",
            {},
        )
        assert result is None

    def test_geo_prefix_stripped(self, sample_data):
        """Geo-prefixed model IDs still match."""
        result = MarketplaceMapper.match(
            "global.anthropic.claude-opus-4-5-20251101-v1:0",
            "anthropic.claude-opus-4-5-20251101",
            sample_data,
        )
        assert result is not None

    def test_rerank_with_version_suffix(self, sample_data):
        """cohere.rerank-v3-5:0 matches cohere-rerank-v3-5."""
        result = MarketplaceMapper.match(
            "cohere.rerank-v3-5:0",
            "cohere.rerank-v3-5",
            sample_data,
        )
        assert result is not None
        assert result[0].price_per_unit == 0.002
