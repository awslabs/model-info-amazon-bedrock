"""Public types for the Model Info for Amazon Bedrock library.

External output types use strict enum values rather than open-ended strings
from the source APIs. Each pricing dimension is decomposed into orthogonal
axes: direction, modality, cache operation, tier, scope, and context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ._inference_profile_ids import (
    KNOWN_GEO_INFERENCE_PROFILE_PREFIXES,
    KnownInferenceProfilePrefix,
    split_known_inference_profile_id,
)


class Direction(Enum):
    """Whether tokens/images are going in or coming out."""

    INPUT = "input"
    OUTPUT = "output"


class Modality(Enum):
    """What unit of content is being priced."""

    TOKENS = "tokens"
    IMAGE = "image"


class CacheOperation(Enum):
    """Whether this dimension involves prompt caching."""

    NONE = "none"
    READ = "read"
    WRITE = "write"
    WRITE_1H = "write_1h"


class ServiceTier(Enum):
    """The service tier or processing mode for inference."""

    STANDARD = "standard"
    BATCH = "batch"
    FLEX = "flex"
    PRIORITY = "priority"
    LATENCY_OPTIMIZED = "latency_optimized"


class InferenceScope(Enum):
    """The geographic scope of the inference endpoint."""

    REGIONAL = "regional"
    CROSS_REGION_GLOBAL = "cross_region_global"
    CROSS_REGION_GEO = "cross_region_geo"


class ContextLength(Enum):
    """Whether this is standard or long-context pricing."""

    STANDARD = "standard"
    LONG = "long_context"


class PricingUnit(Enum):
    """The unit in which a price is expressed."""

    MILLION_TOKENS = "million_tokens"
    IMAGE = "image"
    SECOND = "second"
    SEARCH_UNIT = "search_unit"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class PriceDimension:
    """A single pricing dimension with fully decomposed metadata.

    Each field represents an independent axis of the pricing structure.
    For token-based pricing, `price` is always per-million tokens.
    For images, it's per-image.
    """

    direction: Direction
    modality: Modality
    cache: CacheOperation
    tier: ServiceTier
    scope: InferenceScope
    context: ContextLength
    unit: PricingUnit
    price: float
    rate_code: str
    source_service: str = ""  # Which AWS service code this came from


@dataclass(frozen=True)
class ModelPricing:
    """Aggregate pricing for a single model.

    Provides convenience accessors for the most common use case (standard
    on-demand token pricing) while exposing the full set of pricing
    dimensions for power users.
    """

    model_id: str
    region: str
    dimensions: tuple[PriceDimension, ...] = field(default_factory=tuple)

    @property
    def input_tokens(self) -> float | None:
        """Standard on-demand price per million input tokens, or None."""
        return self._standard_token_price(Direction.INPUT)

    @property
    def output_tokens(self) -> float | None:
        """Standard on-demand price per million output tokens, or None."""
        return self._standard_token_price(Direction.OUTPUT)

    @property
    def input_images(self) -> float | None:
        """Standard on-demand price per input image, or None."""
        return self.get_price(
            direction=Direction.INPUT,
            modality=Modality.IMAGE,
            cache=CacheOperation.NONE,
            tier=ServiceTier.STANDARD,
        )

    @property
    def output_images(self) -> float | None:
        """Standard on-demand price per output image, or None."""
        return self.get_price(
            direction=Direction.OUTPUT,
            modality=Modality.IMAGE,
            cache=CacheOperation.NONE,
            tier=ServiceTier.STANDARD,
        )

    def get_price(
        self,
        direction: Direction,
        *,
        modality: Modality = Modality.TOKENS,
        cache: CacheOperation = CacheOperation.NONE,
        tier: ServiceTier = ServiceTier.STANDARD,
        scope: InferenceScope | None = None,
        context: ContextLength = ContextLength.STANDARD,
    ) -> float | None:
        """Look up a specific price by its dimension axes.

        If scope is None, prefer the scope encoded by an inference profile ID:
        global profiles prefer CROSS_REGION_GLOBAL, geographic profiles prefer
        CROSS_REGION_GEO, and ordinary model IDs prefer REGIONAL. Other scopes
        remain fallbacks for compatibility with incomplete source records.
        """
        if scope is not None:
            return self._find_price(
                direction=direction,
                modality=modality,
                cache=cache,
                tier=tier,
                scope=scope,
                context=context,
            )

        profile_prefix, _ = split_known_inference_profile_id(self.model_id)
        if profile_prefix == KnownInferenceProfilePrefix.GLOBAL:
            scope_preference = (
                InferenceScope.CROSS_REGION_GLOBAL,
                InferenceScope.REGIONAL,
                InferenceScope.CROSS_REGION_GEO,
            )
        elif profile_prefix in KNOWN_GEO_INFERENCE_PROFILE_PREFIXES:
            scope_preference = (
                InferenceScope.CROSS_REGION_GEO,
                InferenceScope.REGIONAL,
                InferenceScope.CROSS_REGION_GLOBAL,
            )
        else:
            scope_preference = (
                InferenceScope.REGIONAL,
                InferenceScope.CROSS_REGION_GLOBAL,
                InferenceScope.CROSS_REGION_GEO,
            )

        for inferred_scope in scope_preference:
            price = self._find_price(
                direction=direction,
                modality=modality,
                cache=cache,
                tier=tier,
                scope=inferred_scope,
                context=context,
            )
            if price is not None:
                return price
        return None

    def filter_dimensions(
        self,
        *,
        direction: Direction | None = None,
        modality: Modality | None = None,
        cache: CacheOperation | None = None,
        tier: ServiceTier | None = None,
        scope: InferenceScope | None = None,
        context: ContextLength | None = None,
    ) -> list[PriceDimension]:
        """Return all dimensions matching the given filters."""
        results = []
        for dim in self.dimensions:
            if direction is not None and dim.direction != direction:
                continue
            if modality is not None and dim.modality != modality:
                continue
            if cache is not None and dim.cache != cache:
                continue
            if tier is not None and dim.tier != tier:
                continue
            if scope is not None and dim.scope != scope:
                continue
            if context is not None and dim.context != context:
                continue
            results.append(dim)
        return results

    def validate(self) -> list[str]:
        """Check for asymmetric or unexpected pricing coverage.

        Returns a list of warning messages. Empty list means no issues.
        """
        warnings: list[str] = []

        # Check that non-standard tiers have both input and output
        tier_scope_directions: dict[
            tuple[ServiceTier, InferenceScope], set[Direction]
        ] = {}
        for dim in self.dimensions:
            if dim.cache != CacheOperation.NONE:
                continue  # Cache dimensions don't need I/O symmetry
            if dim.modality != Modality.TOKENS:
                continue  # Only check token pricing symmetry
            key = (dim.tier, dim.scope)
            tier_scope_directions.setdefault(key, set()).add(dim.direction)

        # Check for asymmetry in non-standard tiers
        for (tier, scope), directions in tier_scope_directions.items():
            if tier == ServiceTier.STANDARD:
                continue  # Standard tier may legitimately have only input
            has_input = Direction.INPUT in directions
            has_output = Direction.OUTPUT in directions
            if has_input and not has_output:
                warnings.append(
                    f"{tier.value}/{scope.value}: has input pricing "
                    f"but no output pricing"
                )
            elif has_output and not has_input:
                warnings.append(
                    f"{tier.value}/{scope.value}: has output pricing "
                    f"but no input pricing"
                )

        return warnings

    def _standard_token_price(self, direction: Direction) -> float | None:
        """Find the standard on-demand token price for a direction."""
        return self.get_price(
            direction,
            modality=Modality.TOKENS,
            cache=CacheOperation.NONE,
            tier=ServiceTier.STANDARD,
        )

    def _find_price(
        self,
        direction: Direction,
        modality: Modality = Modality.TOKENS,
        cache: CacheOperation = CacheOperation.NONE,
        tier: ServiceTier = ServiceTier.STANDARD,
        scope: InferenceScope | None = None,
        context: ContextLength = ContextLength.STANDARD,
    ) -> float | None:
        """Find a single price matching the given criteria."""
        for dim in self.dimensions:
            if dim.direction != direction:
                continue
            if dim.modality != modality:
                continue
            if dim.cache != cache:
                continue
            if dim.tier != tier:
                continue
            if scope is not None and dim.scope != scope:
                continue
            if dim.context != context:
                continue
            return dim.price
        return None
