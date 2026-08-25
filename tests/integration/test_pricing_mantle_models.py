"""Verify pricing coverage against the live Mantle API model list per --target-region"""

import os
import warnings

import pytest
from aws_bedrock_token_generator import provide_token
from openai import OpenAI

from model_info_amazon_bedrock import BedrockModelInfoClient, PricingNotFoundError


@pytest.mark.integration
def test_pricing_coverage_for_mantle_models(
    target_region: str,
    boto_session,
):
    """Fetch model IDs from Mantle API and attempt pricing lookup for each.

    Models with no pricing are recorded as warnings, not failures. Missing
    credentials, token generation, connectivity, or live model data fail the
    selected test.
    """
    base_url = os.environ.get(
        "OPENAI_API_BASE_URL",
        f"https://bedrock-mantle.{target_region}.api.aws/v1",
    )
    api_key = os.environ.get("OPENAI_API_KEY") or provide_token(region=target_region)
    assert base_url, "Mantle API base URL is empty"
    assert api_key, "Mantle API credentials are empty"

    client = OpenAI(base_url=base_url, api_key=api_key)
    models_response = client.models.list()

    model_ids = [model.id for model in models_response]
    assert model_ids, "No models returned from Mantle API"

    pricing_client = BedrockModelInfoClient(session=boto_session)
    models_without_pricing: list[str] = []

    for model_id in model_ids:
        try:
            result = pricing_client.get_model_pricing(
                model_id,
                region=target_region,
            )
            assert result.region == target_region
            assert len(result.dimensions) > 0, f"Empty pricing for {model_id}"
        except PricingNotFoundError:
            models_without_pricing.append(model_id)

    for model_id in models_without_pricing:
        warnings.warn(
            f"No pricing found for model: {model_id}",
            stacklevel=1,
        )
