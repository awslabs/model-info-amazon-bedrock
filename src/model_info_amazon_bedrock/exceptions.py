"""Custom exceptions for Model Info for Amazon Bedrock library"""


class PricingNotFoundError(Exception):
    """Raised when no pricing data could be found matching a given Bedrock model ID."""

    def __init__(self, model_id: str, region: str):
        self.model_id = model_id
        self.region = region
        super().__init__(
            f"No pricing found for model '{model_id}' in region '{region}'"
        )
