"""Integration tests: verify pricing coverage against live Bedrock model list."""

import warnings

import boto3
import pytest

from model_info_amazon_bedrock import BedrockModelInfoClient, PricingNotFoundError


@pytest.mark.integration
def test_pricing_coverage_for_bedrock_models():
    """Fetch model IDs from list_foundation_models and attempt pricing lookup for each.

    Models with no pricing are recorded as warnings, not failures. Missing AWS
    credentials, service access, or live model data fail the selected test.
    """
    region = "us-east-1"

    session = boto3.Session(region_name=region)
    bedrock = session.client("bedrock")
    response = bedrock.list_foundation_models()

    model_summaries = response.get("modelSummaries", [])
    assert model_summaries, "No foundation models returned from Bedrock API"

    model_ids = [m["modelId"] for m in model_summaries if "modelId" in m]
    assert model_ids, "No model IDs found in Bedrock response"

    pricing_client = BedrockModelInfoClient(session=session)
    models_without_pricing: list[str] = []

    for model_id in model_ids:
        try:
            result = pricing_client.get_model_pricing(model_id)
            assert result.region == region
            assert len(result.dimensions) > 0, f"Empty pricing list for {model_id}"
        except PricingNotFoundError:
            models_without_pricing.append(model_id)

    for model_id in models_without_pricing:
        warnings.warn(
            f"No pricing found for model: {model_id}",
            stacklevel=1,
        )


@pytest.mark.integration
def test_pricing_coverage_for_inference_profiles():
    """Fetch cross-region inference profile IDs and attempt pricing lookup.

    Uses list_inference_profiles with typeEquals=SYSTEM_DEFINED to get all
    system-defined cross-region inference profiles (CRIS profiles).
    Models with no pricing are recorded as warnings, not failures. Missing AWS
    credentials, service access, or live profile data fail the selected test.
    """
    region = "us-east-1"

    bedrock = boto3.client("bedrock", region_name=region)
    response = bedrock.list_inference_profiles(typeEquals="SYSTEM_DEFINED")

    profiles = response.get("inferenceProfileSummaries", [])
    assert profiles, "No inference profiles returned from Bedrock API"

    model_ids = list(
        {p["inferenceProfileId"] for p in profiles if "inferenceProfileId" in p}
    )
    assert model_ids, "No inference profile IDs found"

    pricing_client = BedrockModelInfoClient()
    models_without_pricing: list[str] = []

    for model_id in model_ids:
        try:
            result = pricing_client.get_model_pricing(model_id, region=region)
            assert len(result.dimensions) > 0, f"Empty pricing list for {model_id}"
        except PricingNotFoundError:
            models_without_pricing.append(model_id)

    for model_id in models_without_pricing:
        warnings.warn(
            f"No pricing found for inference profile: {model_id}",
            stacklevel=1,
        )
