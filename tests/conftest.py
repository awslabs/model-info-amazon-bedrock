"""Shared test fixtures for Model Info for Amazon Bedrock."""

# Enumerated list of known inference suffixes, used as a Hypothesis strategy
# source for property-based tests. The actual parsing uses a structural regex
# in mapper.py — this list provides concrete examples for test generation.
INFERENCE_SUFFIXES = [
    # Cross-region long-context variants:
    "cache-read-input-token-count-long-context-cross-region-global",
    "cache-write-input-token-count-long-context-cross-region-global",
    "input-tokens-long-context-cross-region-global",
    "output-tokens-long-context-cross-region-global",
    # Cross-region batch variants:
    "input-tokens-cross-region-global-batch",
    "output-tokens-cross-region-global-batch",
    # Cross-region standard variants:
    "cache-read-input-token-count-cross-region-global",
    "cache-write-input-token-count-cross-region-global",
    "input-tokens-cross-region-global",
    "output-tokens-cross-region-global",
    "cache-read-input-token-count-cross-region-geo",
    "cache-write-input-token-count-cross-region-geo",
    "input-tokens-cross-region-geo",
    "output-tokens-cross-region-geo",
    # Reserved pricing suffixes (cross-region):
    "reserved-1-month-input-tokens-per-minute-cross-region-global",
    "reserved-1-month-output-tokens-per-minute-cross-region-global",
    "reserved-3-month-input-tokens-per-minute-cross-region-global",
    "reserved-3-month-output-tokens-per-minute-cross-region-global",
    "reserved-1-month-input-tokens-per-minute-cross-region-geo",
    "reserved-1-month-output-tokens-per-minute-cross-region-geo",
    "reserved-3-month-input-tokens-per-minute-cross-region-geo",
    "reserved-3-month-output-tokens-per-minute-cross-region-geo",
    # Standard suffixes:
    "cache-read-input-token-count",
    "cache-write-input-token-count",
    "input-image-token-count",
    "output-image-token-count",
    "input-tokens-standard",
    "output-tokens-standard",
    "input-tokens-priority",
    "output-tokens-priority",
    "output-tokens-batch",
    "input-tokens-batch",
    "output-tokens-flex",
    "input-tokens-flex",
    "input-image-count",
    "output-image-count",
    "output-tokens",
    "input-tokens",
]
