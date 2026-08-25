"""Static lookup tables for exception model IDs that fail algorithmic matching.

The AWS Pricing API uses inconsistent naming conventions across service codes.
This module provides static mappings for cases where algorithmic matching fails:
Mostly in cases where the word ordering is materially different between the
pricing API and model IDs.

To update: run the integration tests and check which models produce warnings.
Cross-reference the failing model IDs with the unmatched pricing segments
(visible via UsagetypeMapper.parse_usagetype on raw entries) or servicenames
(visible in the fetched marketplace data keys).
"""

# ---------------------------------------------------------------------------
# Usagetype aliases: AmazonBedrock + AmazonBedrockService
#
# Some models use marketing-style names in their usagetype field (e.g.
# "Claude4Sonnet") that bear no algorithmic relationship to the customer-facing
# model ID. The UsagetypeMapper consults this table when both exact matching
# and fallback heuristics fail.
#
# Key: normalized model ID (lowercase, no version suffix, no geo prefix)
# Value: usagetype segment(s) used in the Pricing API
# ---------------------------------------------------------------------------

MODEL_ID_TO_SEGMENTS: dict[str, str | list[str]] = {
    # --- Anthropic Claude (AmazonBedrockService: cross-region inference) ---
    "anthropic.claude-sonnet-4-20250514": "Claude4Sonnet",
    "anthropic.claude-sonnet-4-5-20250929": "Claude4.5Sonnet",
    "anthropic.claude-sonnet-4-6": "Claude4Sonnet",
    # --- Amazon Nova ---
    "amazon.nova-sonic": ["NovaSonic-text", "NovaSonic-speech"],
    "amazon.nova-2-lite": "Nova2.0Lite",
    "amazon.nova-2-sonic": ["NovaSonic2.0-text", "NovaSonic2.0-speech"],
    # --- Amazon Titan Embeddings ---
    "amazon.titan-embed-text-v2": "TitanEmbeddingV2-Text",
    "amazon.titan-embed-image-v1": "TitanEmbeddingsG1-Image",
    "amazon.titan-embed-image": "TitanEmbeddingsG1-Image",
    "amazon.titan-embed-text-v1": "TitanEmbeddingsG1-Text",
    "amazon.titan-embed-text": "TitanEmbeddingsG1-Text",
    "amazon.titan-embed-g1-text-02": "TitanEmbeddingsG1-Text",
    # --- Amazon Nova Multimodal Embeddings ---
    "amazon.nova-2-multimodal-embeddings": "NovaMultiModalEmbeddings",
}


# ---------------------------------------------------------------------------
# Marketplace aliases: AmazonBedrockFoundationModels
#
# Some models have servicenames with material word-ordering differences from
# their model IDs, making algorithmic matching impossible. The
# MarketplaceMapper consults this table before attempting algorithmic matching.
#
# Key: normalized model ID (lowercase, no version suffix, no geo prefix)
# Value: normalized servicename (as produced by normalize_servicename)
# ---------------------------------------------------------------------------

MODEL_ID_TO_SERVICENAME: dict[str, str] = {
    # Cohere Embed: servicenames use "Model" and reorder version/language
    "cohere.embed-english-v3": "cohere-embed-3-model-english",
    "cohere.embed-english": "cohere-embed-3-model-english",
    "cohere.embed-multilingual-v3": "cohere-embed-model-3-multilingual",
    "cohere.embed-multilingual": "cohere-embed-model-3-multilingual",
    "cohere.embed-v4": "cohere-embed-4-model",
    "cohere.embed": "cohere-embed-4-model",
}
