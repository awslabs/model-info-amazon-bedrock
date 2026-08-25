"""Internal source and intermediate records for pricing resolution."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UsagetypeEntry:
    """Raw pricing entry from a usagetype-based service code.

    Produced by UsagetypeFetcher from AmazonBedrock and AmazonBedrockService.
    Model identification comes from parsing the `usagetype` field; prices are
    raw per-unit values (e.g. per 1K tokens) that need normalization.

    If critical fields were missing or malformed during parsing, `parse_errors`
    is populated with descriptions. The entry is still stored (so valid entries from
    the same API page aren't lost), but will raise at classification time if
    it matches a user's query.
    """

    usagetype: str
    inference_type: str | None
    provider: str | None
    model: str | None
    service_tier: str | None
    unit: str
    price_per_unit: float
    offer_term_code: str
    rate_code: str
    parse_errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class ParsedUsagetype:
    """Result of parsing a UsagetypeEntry's usagetype string into components."""

    region_prefix: str
    model_segment: str
    has_mantle: bool
    inference_suffix: str


@dataclass(frozen=True)
class MarketplaceEntry:
    """Raw pricing entry from AmazonBedrockFoundationModels service code.

    Produced by MarketplaceFetcher. Entries are keyed by normalized
    servicename (e.g. "claude-opus-4") and the MarketplaceMapper handles
    matching model IDs to servicenames at resolve time.

    The `description` field carries the raw dimension description from the
    API (e.g. "Million Input Tokens Regional"). The classifier parses this
    to determine inference type, service tier, and unit.

    If critical fields were missing or malformed during parsing, `parse_errors`
    is populated with descriptions. The entry is still stored (so valid entries from
    the same API page aren't lost), but will raise at classification time if
    it matches a user's query.
    """

    servicename: str  # Normalized servicename (e.g. "claude-opus-4")
    description: str  # Raw dimension description from the API
    unit: str  # Raw unit from API, or description portion if API gives "Units"
    price_per_unit: float  # Raw price per unit from API
    offer_term_code: str
    rate_code: str
    parse_errors: tuple[str, ...] = ()
