"""Pricing resolution chain.

Each PricingSource attempts to resolve pricing for a model ID from a
specific AWS Pricing API service code. The client iterates through sources
in priority order until one succeeds.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..types import PriceDimension
from .classifier import classify_marketplace_entry, classify_usagetype_entry
from .mapper import UsagetypeMapper
from .marketplace_mapper import MarketplaceMapper
from .types import MarketplaceEntry, UsagetypeEntry


class PricingSource(ABC):
    """Protocol for a pricing data source."""

    @abstractmethod
    def resolve(
        self,
        model_id: str,
        normalized_id: str,
        cache: RegionCache,
        region: str,
        *,
        refresh: bool = False,
    ) -> list[PriceDimension] | None:
        """Attempt to resolve pricing for a model.

        Returns list of PriceDimension if this source can resolve the model,
        or None to indicate the next source should be tried.
        """


class BedrockSource(PricingSource):
    """Resolves pricing from AmazonBedrock entries."""

    SOURCE_SERVICE = "AmazonBedrock"

    def resolve(
        self,
        model_id: str,
        normalized_id: str,
        cache: RegionCache,
        region: str,
        *,
        refresh: bool = False,
    ) -> list[PriceDimension] | None:
        entries = cache.get_bedrock(region, refresh=refresh)
        matched = UsagetypeMapper.match(model_id, entries)
        if not matched:
            return None
        return _classify_entries(matched, self.SOURCE_SERVICE)


class BedrockServiceSource(PricingSource):
    """Resolves pricing from AmazonBedrockService entries."""

    SOURCE_SERVICE = "AmazonBedrockService"

    def resolve(
        self,
        model_id: str,
        normalized_id: str,
        cache: RegionCache,
        region: str,
        *,
        refresh: bool = False,
    ) -> list[PriceDimension] | None:
        entries = cache.get_service(region, refresh=refresh)
        matched = UsagetypeMapper.match(model_id, entries)
        if not matched:
            return None
        return _classify_entries(matched, self.SOURCE_SERVICE)


class MarketplaceSource(PricingSource):
    """Resolves pricing from AmazonBedrockFoundationModels."""

    SOURCE_SERVICE = "AmazonBedrockFoundationModels"

    def resolve(
        self,
        model_id: str,
        normalized_id: str,
        cache: RegionCache,
        region: str,
        *,
        refresh: bool = False,
    ) -> list[PriceDimension] | None:
        fm_data = cache.get_foundation(region, refresh=refresh)

        entries = MarketplaceMapper.match(model_id, normalized_id, fm_data)
        if not entries:
            return None

        dimensions = []
        for entry in entries:
            dim = classify_marketplace_entry(entry, self.SOURCE_SERVICE)
            if dim is not None:
                dimensions.append(dim)
        return dimensions if dimensions else None


# ---------------------------------------------------------------------------
# Pricing API Data Cache
# ---------------------------------------------------------------------------


class RegionCache:
    """Unified cache for all pricing sources, keyed by region.

    Each source is cached independently. Supports refresh with
    preserve-on-failure semantics.
    """

    def __init__(
        self,
        fetch_bedrock,
        fetch_bedrock_service,
        fetch_foundation_models,
    ):
        self._fetch_bedrock = fetch_bedrock
        self._fetch_bedrock_service = fetch_bedrock_service
        self._fetch_foundation_models = fetch_foundation_models
        self._bedrock: dict[str, list[UsagetypeEntry]] = {}
        self._service: dict[str, list[UsagetypeEntry]] = {}
        self._foundation: dict[str, dict[str, list[MarketplaceEntry]]] = {}

    def get_bedrock(
        self, region: str, *, refresh: bool = False
    ) -> list[UsagetypeEntry]:
        """Get AmazonBedrock entries for a region."""
        return self._fetch_cached(
            region, self._bedrock, self._fetch_bedrock, refresh=refresh
        )

    def get_service(
        self, region: str, *, refresh: bool = False
    ) -> list[UsagetypeEntry]:
        """Get AmazonBedrockService entries for a region."""
        return self._fetch_cached(
            region, self._service, self._fetch_bedrock_service, refresh=refresh
        )

    def get_foundation(
        self, region: str, *, refresh: bool = False
    ) -> dict[str, list[MarketplaceEntry]]:
        """Get AmazonBedrockFoundationModels data for a region."""
        return self._fetch_cached(
            region,
            self._foundation,
            self._fetch_foundation_models,
            refresh=refresh,
        )

    def invalidate(self, region: str) -> None:
        """Remove cached data for a region."""
        self._bedrock.pop(region, None)
        self._service.pop(region, None)
        self._foundation.pop(region, None)

    @staticmethod
    def _fetch_cached(region, cache, fetcher, *, refresh):
        """Generic fetch-with-cache logic."""
        if not refresh and region in cache:
            return cache[region]
        try:
            data = fetcher(region)
        except Exception:
            if refresh and region in cache:
                raise
            raise
        cache[region] = data
        return data


# ---------------------------------------------------------------------------
# Default source chain
# ---------------------------------------------------------------------------

DEFAULT_SOURCES: list[PricingSource] = [
    BedrockSource(),
    BedrockServiceSource(),
    MarketplaceSource(),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _classify_entries(
    entries: list[UsagetypeEntry], source_service: str
) -> list[PriceDimension] | None:
    """Classify a list of matched UsagetypeEntry into PriceDimensions.

    Raises ValueError if any matched entry has a deferred parse error or
    an unrecognized unit — these indicate data problems that should surface
    to the caller rather than being silently swallowed.
    """
    dimensions = []
    for entry in entries:
        parsed = UsagetypeMapper.parse_usagetype(entry.usagetype)
        if parsed is None:
            continue
        dim = classify_usagetype_entry(entry, parsed.inference_suffix, source_service)
        if dim is not None:
            dimensions.append(dim)
    return dimensions if dimensions else None
