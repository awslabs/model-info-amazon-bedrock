"""Tests for the structured output types."""

import pytest

from model_info_amazon_bedrock.types import (
    CacheOperation,
    ContextLength,
    Direction,
    InferenceScope,
    Modality,
    ModelPricing,
    PriceDimension,
    PricingUnit,
    ServiceTier,
)


def _dim(
    direction=Direction.INPUT,
    modality=Modality.TOKENS,
    cache=CacheOperation.NONE,
    tier=ServiceTier.STANDARD,
    scope=InferenceScope.REGIONAL,
    context=ContextLength.STANDARD,
    unit=PricingUnit.MILLION_TOKENS,
    price=3.0,
    rate_code="R1",
) -> PriceDimension:
    """Helper to create a PriceDimension with sensible defaults."""
    return PriceDimension(
        direction=direction,
        modality=modality,
        cache=cache,
        tier=tier,
        scope=scope,
        context=context,
        unit=unit,
        price=price,
        rate_code=rate_code,
    )


class TestPriceDimension:
    """Tests for PriceDimension dataclass."""

    def test_frozen(self):
        dim = _dim()
        with pytest.raises(AttributeError):
            dim.price = 5.0  # type: ignore[misc]

    def test_all_fields_accessible(self):
        dim = _dim(
            direction=Direction.OUTPUT,
            modality=Modality.IMAGE,
            cache=CacheOperation.READ,
            tier=ServiceTier.BATCH,
            scope=InferenceScope.CROSS_REGION_GLOBAL,
            context=ContextLength.LONG,
            price=7.5,
            rate_code="ABC",
        )
        assert dim.direction == Direction.OUTPUT
        assert dim.modality == Modality.IMAGE
        assert dim.cache == CacheOperation.READ
        assert dim.tier == ServiceTier.BATCH
        assert dim.scope == InferenceScope.CROSS_REGION_GLOBAL
        assert dim.context == ContextLength.LONG
        assert dim.price == 7.5


class TestModelPricingConvenience:
    """Tests for ModelPricing convenience accessors."""

    @pytest.fixture
    def standard_token_model(self) -> ModelPricing:
        return ModelPricing(
            model_id="test.model",
            region="us-east-1",
            dimensions=(
                _dim(direction=Direction.INPUT, price=3.0),
                _dim(direction=Direction.OUTPUT, price=15.0),
                _dim(direction=Direction.INPUT, tier=ServiceTier.BATCH, price=1.5),
                _dim(direction=Direction.OUTPUT, tier=ServiceTier.BATCH, price=7.5),
            ),
        )

    @pytest.fixture
    def cross_region_only_model(self) -> ModelPricing:
        return ModelPricing(
            model_id="test.model",
            region="us-east-1",
            dimensions=(
                _dim(
                    direction=Direction.INPUT,
                    scope=InferenceScope.CROSS_REGION_GLOBAL,
                    price=5.0,
                ),
                _dim(
                    direction=Direction.OUTPUT,
                    scope=InferenceScope.CROSS_REGION_GLOBAL,
                    price=25.0,
                ),
            ),
        )

    @pytest.fixture
    def image_model(self) -> ModelPricing:
        return ModelPricing(
            model_id="test.model",
            region="us-east-1",
            dimensions=(
                _dim(
                    direction=Direction.OUTPUT,
                    modality=Modality.IMAGE,
                    unit=PricingUnit.IMAGE,
                    price=0.08,
                ),
            ),
        )

    def test_input_tokens(self, standard_token_model):
        assert standard_token_model.input_tokens == 3.0

    def test_output_tokens(self, standard_token_model):
        assert standard_token_model.output_tokens == 15.0

    def test_cross_region_fallback(self, cross_region_only_model):
        assert cross_region_only_model.input_tokens == 5.0
        assert cross_region_only_model.output_tokens == 25.0

    def test_image_model_no_tokens(self, image_model):
        assert image_model.input_tokens is None
        assert image_model.output_images == 0.08

    def test_empty_model(self):
        model = ModelPricing(model_id="empty", region="us-east-1")
        assert model.input_tokens is None
        assert model.output_tokens is None
        assert model.input_images is None
        assert model.output_images is None


class TestModelPricingGetPrice:
    """Tests for ModelPricing.get_price() explicit lookups."""

    @pytest.fixture
    def model(self) -> ModelPricing:
        return ModelPricing(
            model_id="test",
            region="us-east-1",
            dimensions=(
                _dim(direction=Direction.INPUT, tier=ServiceTier.STANDARD, price=3.0),
                _dim(direction=Direction.INPUT, tier=ServiceTier.BATCH, price=1.5),
                _dim(direction=Direction.INPUT, tier=ServiceTier.FLEX, price=2.0),
                _dim(
                    direction=Direction.INPUT,
                    scope=InferenceScope.CROSS_REGION_GLOBAL,
                    price=3.3,
                ),
                _dim(
                    direction=Direction.INPUT,
                    cache=CacheOperation.READ,
                    price=0.3,
                ),
            ),
        )

    def test_standard_regional(self, model):
        assert model.get_price(Direction.INPUT, tier=ServiceTier.STANDARD) == 3.0

    def test_batch(self, model):
        assert model.get_price(Direction.INPUT, tier=ServiceTier.BATCH) == 1.5

    def test_cache_read(self, model):
        assert model.get_price(Direction.INPUT, cache=CacheOperation.READ) == 0.3

    def test_nonexistent(self, model):
        assert model.get_price(Direction.OUTPUT) is None


class TestModelPricingFilter:
    """Tests for ModelPricing.filter_dimensions()."""

    @pytest.fixture
    def model(self) -> ModelPricing:
        return ModelPricing(
            model_id="test",
            region="us-east-1",
            dimensions=(
                _dim(direction=Direction.INPUT, price=3.0),
                _dim(direction=Direction.OUTPUT, price=15.0),
                _dim(direction=Direction.INPUT, tier=ServiceTier.BATCH, price=1.5),
                _dim(direction=Direction.INPUT, cache=CacheOperation.READ, price=0.3),
            ),
        )

    def test_filter_by_direction(self, model):
        results = model.filter_dimensions(direction=Direction.INPUT)
        assert len(results) == 3

    def test_filter_by_cache(self, model):
        results = model.filter_dimensions(cache=CacheOperation.READ)
        assert len(results) == 1
        assert results[0].price == 0.3

    def test_filter_no_match(self, model):
        results = model.filter_dimensions(tier=ServiceTier.FLEX)
        assert results == []

    def test_filter_no_args_returns_all(self, model):
        assert len(model.filter_dimensions()) == 4


class TestModelPricingValidation:
    """Tests for ModelPricing.validate() asymmetry detection."""

    def test_symmetric_no_warnings(self):
        model = ModelPricing(
            model_id="test",
            region="us-east-1",
            dimensions=(
                _dim(direction=Direction.INPUT, tier=ServiceTier.BATCH, price=1.5),
                _dim(direction=Direction.OUTPUT, tier=ServiceTier.BATCH, price=7.5),
            ),
        )
        assert model.validate() == []

    def test_asymmetric_batch_warns(self):
        model = ModelPricing(
            model_id="test",
            region="us-east-1",
            dimensions=(
                _dim(direction=Direction.INPUT, tier=ServiceTier.BATCH, price=1.5),
            ),
        )
        warnings = model.validate()
        assert len(warnings) == 1
        assert "batch" in warnings[0]
        assert "no output" in warnings[0]

    def test_standard_not_checked(self):
        model = ModelPricing(
            model_id="test",
            region="us-east-1",
            dimensions=(_dim(direction=Direction.INPUT, price=0.1),),
        )
        assert model.validate() == []

    def test_cache_dimensions_not_checked(self):
        """Cache dimensions don't need I/O symmetry."""
        model = ModelPricing(
            model_id="test",
            region="us-east-1",
            dimensions=(
                _dim(
                    direction=Direction.INPUT,
                    cache=CacheOperation.READ,
                    tier=ServiceTier.BATCH,
                    price=0.1,
                ),
            ),
        )
        assert model.validate() == []


class TestEnumValues:
    """Verify enum values are stable strings."""

    def test_direction(self):
        assert Direction.INPUT.value == "input"
        assert Direction.OUTPUT.value == "output"

    def test_modality(self):
        assert Modality.TOKENS.value == "tokens"
        assert Modality.IMAGE.value == "image"

    def test_cache_operation(self):
        assert CacheOperation.NONE.value == "none"
        assert CacheOperation.READ.value == "read"
        assert CacheOperation.WRITE.value == "write"
        assert CacheOperation.WRITE_1H.value == "write_1h"

    def test_service_tier(self):
        assert ServiceTier.STANDARD.value == "standard"
        assert ServiceTier.BATCH.value == "batch"
        assert ServiceTier.FLEX.value == "flex"
        assert ServiceTier.PRIORITY.value == "priority"

    def test_context_length(self):
        assert ContextLength.STANDARD.value == "standard"
        assert ContextLength.LONG.value == "long_context"
