"""Main client entry point"""

from __future__ import annotations

import boto3
from botocore.exceptions import NoRegionError

from .exceptions import PricingNotFoundError
from .pricing.fetcher import MarketplaceFetcher, UsagetypeFetcher
from .pricing.mapper import UsagetypeMapper
from .pricing.resolver import (
    DEFAULT_SOURCES,
    RegionCache,
)
from .types import ModelPricing


class BedrockModelInfoClient:
    """Main client for looking up information about models on Amazon Bedrock"""

    def __init__(self, session: boto3.Session | None = None):
        """Initialize with an optional boto3 session.

        The session supplies credentials and the default target Region used by
        pricing lookups that omit ``region``.
        """
        self._session = session if session is not None else boto3.Session()
        self._pricing_usagetype_fetcher = UsagetypeFetcher(session=self._session)
        self._pricing_marketplace_fetcher = MarketplaceFetcher(session=self._session)
        self._pricing_cache = RegionCache(
            fetch_bedrock=(lambda r: self._pricing_usagetype_fetcher.fetch_bedrock(r)),
            fetch_bedrock_service=(
                lambda r: self._pricing_usagetype_fetcher.fetch_bedrock_service(r)
            ),
            fetch_foundation_models=lambda r: self._pricing_marketplace_fetcher.fetch(
                r
            ),
        )
        self._pricing_sources = DEFAULT_SOURCES

    def get_model_pricing(
        self,
        model_id: str,
        region: str | None = None,
        *,
        refresh: bool = False,
    ) -> ModelPricing:
        """Look up pricing for a model ID in a region.

        The target Region is passed to the AWS Price List API as ``regionCode``;
        the Pricing API client endpoint remains in ``us-east-1``.

        Args:
            model_id: Customer-facing Bedrock model ID.
            region: Target AWS Region code. If ``None``, use the client's boto3
                session default.
            refresh: If True, bypass the cache and fetch fresh data.

        Returns:
            ModelPricing object with structured pricing dimensions and
            convenience accessors for standard on-demand pricing.

        Raises:
            ValueError: If model_id is empty, None, or non-string.
            NoRegionError: If neither region nor a session default is available.
            PricingNotFoundError: If no pricing found for the model.
            botocore exceptions: If API call fails.
        """
        # Input validation — before any API calls.
        if not isinstance(model_id, str):
            raise ValueError(
                "model_id must be a non-empty string, "
                f"got {type(model_id).__name__}: {model_id!r}"
            )
        if not model_id.strip():
            raise ValueError(
                "model_id must be a non-empty string, "
                f"got empty/whitespace: {model_id!r}"
            )

        if not region:
            region = self._session.region_name
        if not region:
            # Need to explicitly raise an error when region not specified, because
            # price list API queries will target us-east-1 anyway so would appear to
            # work but then not return any data matching the region we want:
            raise NoRegionError()

        # Normalize the model ID for resolution.
        normalized = UsagetypeMapper.normalize_model_id(model_id).lower()

        # Try each source in priority order.
        for source in self._pricing_sources:
            dimensions = source.resolve(
                model_id,
                normalized,
                self._pricing_cache,
                region,
                refresh=refresh,
            )
            if dimensions is not None:
                return ModelPricing(
                    model_id=model_id,
                    region=region,
                    dimensions=tuple(dimensions),
                )

        raise PricingNotFoundError(model_id, region)

    def invalidate_pricing_cache(self, region: str) -> None:
        """Remove cached pricing data for a region."""
        self._pricing_cache.invalidate(region)
