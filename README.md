# Model Info for Amazon Bedrock

[![Latest Version](https://img.shields.io/pypi/v/model-info-amazon-bedrock.svg)](https://pypi.python.org/pypi/model-info-amazon-bedrock)
[![Supported Python Versions](https://img.shields.io/badge/dynamic/json?query=info.requires_python&label=python&url=https%3A%2F%2Fpypi.org%2Fpypi%2Fllmeter%2Fjson)](https://pypi.python.org/pypi/llmeter)
[![Code Style: Ruff](https://img.shields.io/badge/code_style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

Model Info for Amazon Bedrock is a lightweight Python helper to simplify retrieving metadata about AI models on Amazon Bedrock.

This early release is focussed on mapping pricing information in particular, although we hope to expand to other metadata in future and contributions are welcome!

> ⚠️ **WARNING:** This helper library is provided "as is", as described in the [LICENSE](LICENSE):
>
> 1. The price data mapping it performs is non-trivial and (despite our best efforts) may contain bugs.
> 2. No warranty is provided or liability accepted for the correctness of summarized pricing data it provides.
>
> For truly **authoritative** pricing information, refer to the underlying [AWS Price List API](https://docs.aws.amazon.com/awsaccountbilling/latest/aboutv2/price-changes.html) and the [Amazon Bedrock Pricing page](https://aws.amazon.com/bedrock/pricing/).

While the AWS Price List API provides authoritative pricing information, its listings for Amazon Bedrock are distributed across multiple service codes, and their attributes do not generally correspond exactly to unique model or inference profile IDs used by Amazon Bedrock inference APIs. This library implements matching logic to provide a simpler way to look up current pricing records by model ID. Callers should validate results before using them for contractual, billing-critical, or other high-impact decisions.

## Installation

Because this library uses the AWS Price List API, the environment where it runs needs [AWS credentials configured](https://docs.aws.amazon.com/boto3/latest/guide/credentials.html#guide-credentials) with appropriate [IAM permissions](https://docs.aws.amazon.com/service-authorization/latest/reference/list_awspricelist.html) to list services and products.

Install on **Python 3.11+** with your preferred package manager. For example, with [uv](https://docs.astral.sh/uv/):

```sh
uv add model-info-amazon-bedrock
```

Or with [pip](https://pypi.org/project/pip/):

```sh
pip install model-info-amazon-bedrock
```

## Usage

Many factors can affect the final price of a foundation model on Amazon Bedrock, including for example:

- Which [service tier](https://aws.amazon.com/bedrock/service-tiers/) you use for to balance between cost, speed, and predictability.
- Which Region or inference scope you target, including [global cross-region inference](https://docs.aws.amazon.com/bedrock/latest/userguide/global-cross-region-inference.html) which can be cheaper.
- Whether you use a longer-context variant of a model or the standard context length.
- Private pricing agreements or any other account-specific arrangements that are not represented by public list prices.

It's important to consider these complexities for detailed assessments and high-impact decisions.

However, this library mainly helps retrieve **indicative list pricing**, including standard on-demand pricing, by foundation model or cross-Region inference profile ID:

```python
from model_info_amazon_bedrock import BedrockModelInfoClient

client = BedrockModelInfoClient()

pricing = client.get_model_pricing(
    "anthropic.claude-sonnet-4-20250514-v1:0",
    region="us-east-1",
)

# Standard on-demand token pricing, when available.
print(f"Input:  ${pricing.input_tokens:.2f} per million tokens")
print(f"Output: ${pricing.output_tokens:.2f} per million tokens")
```

Example output:

```text
Input:  $3.00 per million tokens
Output: $15.00 per million tokens
```

You can inspect all pricing dimensions matched and classified for the model:

```python
for dim in pricing.dimensions:
    print(
        f"{dim.direction.value} ({dim.tier.value}, {dim.scope.value}): "
        f"${dim.price:.4f} per {dim.unit.value}"
    )
```

### Price Records

`get_model_pricing` returns a `ModelPricing` object. Each `PriceDimension` in `ModelPricing.dimensions` contains:

| Field | Description |
|---|---|
| `direction` | `Direction.INPUT` or `Direction.OUTPUT` |
| `modality` | `Modality.TOKENS` or `Modality.IMAGE` |
| `cache` | `CacheOperation.NONE`, `.READ`, `.WRITE`, or `.WRITE_1H` |
| `tier` | `ServiceTier.STANDARD`, `.BATCH`, `.FLEX`, `.PRIORITY`, or `.LATENCY_OPTIMIZED` |
| `scope` | `InferenceScope.REGIONAL`, `.CROSS_REGION_GLOBAL`, or `.CROSS_REGION_GEO` |
| `context` | `ContextLength.STANDARD` or `.LONG` |
| `unit` | `PricingUnit.MILLION_TOKENS`, `.IMAGE`, `.SECOND`, `.SEARCH_UNIT`, or `.UNKNOWN` |
| `price` | Normalized price for the classified unit, or the raw value for an unknown unit |
| `rate_code` | AWS Price List API rate code for traceability |
| `source_service` | AWS service code from which the dimension was resolved |

The convenience properties `input_tokens`, `output_tokens`, `input_images`, and `output_images` return the standard pricing value selected by the library, or `None` when that value is unavailable. Use `get_price` or `filter_dimensions` when you need a specific tier, scope, cache operation, modality, or context length.

Matching and classification are heuristic. The returned dimensions reflect records the current resolver, mapper, and classifier recognize; they are not a guarantee that every AWS pricing dimension for every model is represented.

### Caching

Pricing data is fetched lazily by source and cached per Region on each client instance:

```python
client = BedrockModelInfoClient()

# First lookup fetches the required AWS pricing source data.
pricing = client.get_model_pricing("anthropic.claude-sonnet-4-20250514-v1:0")

# Later lookups in the same Region reuse cached source data when possible.
pricing = client.get_model_pricing("meta.llama3-70b-instruct-v1:0")

# Force fresh source data for this lookup.
pricing = client.get_model_pricing(
    "anthropic.claude-sonnet-4-20250514-v1:0",
    refresh=True,
)

# Clear cached pricing data for a Region.
client.invalidate_pricing_cache("us-east-1")
```

A failed refresh raises the underlying error and preserves existing cached data.

### Custom boto3 Session

```python
import boto3

from model_info_amazon_bedrock import BedrockModelInfoClient

session = boto3.Session(
    profile_name="my-profile",
    region_name="us-west-2",
)
client = BedrockModelInfoClient(session=session)
```

### Error Handling

```python
from model_info_amazon_bedrock import (
    BedrockModelInfoClient,
    PricingNotFoundError,
)

client = BedrockModelInfoClient()

try:
    pricing = client.get_model_pricing("nonexistent.model-id")
except PricingNotFoundError as exc:
    print(f"No pricing for {exc.model_id} in {exc.region}")
except ValueError as exc:
    print(f"Invalid input: {exc}")
```

AWS credential, service, throttling, and network exceptions from boto3/botocore propagate to the caller.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidance including how best to engage with our community, setting up your local development environment for the project, tips on testing & debugging, and architectural guidance.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.
