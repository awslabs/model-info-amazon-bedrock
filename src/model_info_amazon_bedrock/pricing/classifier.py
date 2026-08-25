"""Classification layer: maps raw pricing entries to structured PriceDimensions.

For usagetype-based entries (AmazonBedrock, AmazonBedrockService), parses the
inference suffix into orthogonal enum components (direction, modality, cache,
tier, scope, context).

For marketplace entries (AmazonBedrockFoundationModels), parses the dimension
description string to determine the same set of enum axes.

Both paths share the same price normalization logic and produce the same
PriceDimension output type.
"""

from __future__ import annotations

from ..types import (
    CacheOperation,
    ContextLength,
    Direction,
    InferenceScope,
    Modality,
    PriceDimension,
    PricingUnit,
    ServiceTier,
)
from .types import MarketplaceEntry, UsagetypeEntry


def classify_usagetype_entry(
    entry: UsagetypeEntry,
    inference_suffix: str,
    source_service: str = "",
) -> PriceDimension | None:
    """Classify a UsagetypeEntry into a structured PriceDimension.

    Parses the inference suffix into its component axes. Returns None
    for reserved pricing entries or unrecognized suffixes.

    Raises:
        ValueError: If the entry has deferred parse errors (e.g. missing
            or malformed price data from the API).
    """
    if entry.parse_errors:
        raise ValueError("; ".join(entry.parse_errors))

    # Skip reserved pricing entries entirely.
    if inference_suffix.startswith("reserved-"):
        return None

    direction = _parse_direction(inference_suffix)
    if direction is None:
        return None

    modality = _parse_modality(inference_suffix)
    cache = _parse_cache(inference_suffix)
    tier = _parse_tier(inference_suffix)
    scope = _parse_scope(inference_suffix)
    context = _parse_context(inference_suffix)
    unit, price = _normalize_price(entry.unit, entry.price_per_unit, modality)

    return PriceDimension(
        direction=direction,
        modality=modality,
        cache=cache,
        tier=tier,
        scope=scope,
        context=context,
        unit=unit,
        price=price,
        rate_code=entry.rate_code,
        source_service=source_service,
    )


def classify_marketplace_entry(
    record: MarketplaceEntry,
    source_service: str = "AmazonBedrockFoundationModels",
) -> PriceDimension | None:
    """Classify a MarketplaceEntry into a structured PriceDimension.

    Parses the entry's description field to determine all dimension axes
    (direction, modality, cache, tier, scope) in a single pass.
    Returns None for entries that should be skipped (reserved, provisioned, etc.).

    Raises:
        ValueError: If the entry has deferred parse errors (e.g. missing
            or malformed price data from the API).
    """
    if record.parse_errors:
        raise ValueError("; ".join(record.parse_errors))

    result = _classify_marketplace_description(record.description)
    if result is None:
        return None

    direction, modality, cache, tier, scope = result
    unit, price = _normalize_price(record.unit, record.price_per_unit, modality)

    return PriceDimension(
        direction=direction,
        modality=modality,
        cache=cache,
        tier=tier,
        scope=scope,
        context=ContextLength.STANDARD,
        unit=unit,
        price=price,
        rate_code=record.rate_code,
        source_service=source_service,
    )


# ---------------------------------------------------------------------------
# Suffix component parsing (no lookup tables needed)
# ---------------------------------------------------------------------------


def _parse_direction(suffix: str) -> Direction | None:
    """Extract direction from suffix.

    Cache dimensions are input-token operations even when their suffix omits
    the literal ``input`` token.
    """
    if suffix.startswith("cache-") or "input" in suffix:
        return Direction.INPUT
    if "output" in suffix:
        return Direction.OUTPUT
    return None


def _parse_modality(suffix: str) -> Modality:
    """Extract modality from suffix."""
    if "image" in suffix:
        return Modality.IMAGE
    return Modality.TOKENS


def _parse_cache(suffix: str) -> CacheOperation:
    """Extract cache operation from suffix."""
    if suffix.startswith("cache-read"):
        return CacheOperation.READ
    if suffix.startswith("cache-write"):
        # Both the established cache-write form and Mantle's explicit 30-minute
        # form map to the standard cache-write operation. The public model only
        # distinguishes the separate one-hour marketplace variant.
        return CacheOperation.WRITE
    return CacheOperation.NONE


def _parse_tier(suffix: str) -> ServiceTier:
    """Extract service tier from suffix."""
    if "-batch" in suffix:
        return ServiceTier.BATCH
    if "-flex" in suffix:
        return ServiceTier.FLEX
    if "-priority" in suffix:
        return ServiceTier.PRIORITY
    return ServiceTier.STANDARD


def _parse_scope(suffix: str) -> InferenceScope:
    """Extract inference scope from suffix."""
    if "cross-region-global" in suffix:
        return InferenceScope.CROSS_REGION_GLOBAL
    if "cross-region-geo" in suffix:
        return InferenceScope.CROSS_REGION_GEO
    return InferenceScope.REGIONAL


def _parse_context(suffix: str) -> ContextLength:
    """Extract context length from suffix."""
    if "long-context" in suffix or "long-ctx" in suffix:
        return ContextLength.LONG
    return ContextLength.STANDARD


# ---------------------------------------------------------------------------
# Price normalization
# ---------------------------------------------------------------------------


def _normalize_price(
    raw_unit: str, raw_price: float, modality: Modality
) -> tuple[PricingUnit, float]:
    """Normalize a raw price to the standard unit.

    Recognized units:
    - "1K tokens": multiply by 1000 → per-million tokens
    - "1M tokens": per-million passthrough
    - Description strings containing "million" and "token": per-million passthrough
    - "image": per-image passthrough
    - Anything else: UNKNOWN with raw price preserved
    """
    if raw_unit == "1K tokens":
        return PricingUnit.MILLION_TOKENS, raw_price * 1000
    if raw_unit == "1M tokens":
        return PricingUnit.MILLION_TOKENS, raw_price
    if raw_unit == "image":
        return PricingUnit.IMAGE, raw_price

    # Handle description-derived unit strings from marketplace entries
    # (e.g. "Million Input Tokens Global", "Million Response Tokens Regional")
    unit_lower = raw_unit.lower()
    if "million" in unit_lower and "token" in unit_lower:
        return PricingUnit.MILLION_TOKENS, raw_price

    return PricingUnit.UNKNOWN, raw_price


# ---------------------------------------------------------------------------
# Marketplace description classification
# ---------------------------------------------------------------------------

_MARKETPLACE_SKIP_KEYWORDS = ("reserved", "provisioned", "customiz", "storage")


def _classify_marketplace_description(
    desc: str,
) -> tuple[Direction, Modality, CacheOperation, ServiceTier, InferenceScope] | None:
    """Parse a marketplace description directly into enum values.

    The description is the dimension portion of the API's description field
    (e.g. "Million Input Tokens Regional", "Million Response Tokens Batch").

    Returns a tuple of (direction, modality, cache, tier, scope) or None for
    entries that should be skipped (reserved, provisioned, storage, etc.) or
    that can't be classified.
    """
    d = desc.lower()

    # Skip entries we don't handle
    if any(kw in d for kw in _MARKETPLACE_SKIP_KEYWORDS):
        return None

    # --- Direction + Cache + Modality (determined together from content type) ---
    if "cache read" in d and "token" in d:
        direction = Direction.INPUT
        cache = CacheOperation.READ
        modality = Modality.TOKENS
    elif "cache write" in d and "token" in d:
        direction = Direction.INPUT
        cache = (
            CacheOperation.WRITE_1H
            if ("1h" in d or "1 hour" in d)
            else CacheOperation.WRITE
        )
        modality = Modality.TOKENS
    elif "input token" in d or "input tokens" in d:
        direction = Direction.INPUT
        cache = CacheOperation.NONE
        modality = Modality.TOKENS
    elif "response token" in d or "output token" in d or "output tokens" in d:
        direction = Direction.OUTPUT
        cache = CacheOperation.NONE
        modality = Modality.TOKENS
    elif "image" in d:
        direction = Direction.OUTPUT if "generation" in d else Direction.INPUT
        cache = CacheOperation.NONE
        modality = Modality.IMAGE
    elif "video" in d:
        direction = Direction.OUTPUT
        cache = CacheOperation.NONE
        modality = Modality.IMAGE
    elif "search unit" in d:
        direction = Direction.INPUT
        cache = CacheOperation.NONE
        modality = Modality.TOKENS
    elif "audio" in d:
        direction = Direction.INPUT
        cache = CacheOperation.NONE
        modality = Modality.TOKENS
    else:
        return None

    # --- Service tier ---
    if "batch" in d:
        tier = ServiceTier.BATCH
    elif "latency optimized" in d:
        tier = ServiceTier.LATENCY_OPTIMIZED
    else:
        tier = ServiceTier.STANDARD

    # --- Inference scope ---
    if "global" in d:
        scope = InferenceScope.CROSS_REGION_GLOBAL
    elif "regional cris" in d:
        scope = InferenceScope.CROSS_REGION_GEO
    else:
        # The default treatment also covers descriptions containing "regional".
        scope = InferenceScope.REGIONAL

    return direction, modality, cache, tier, scope
