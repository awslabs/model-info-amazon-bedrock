"""Known Amazon Bedrock inference profile ID prefixes.

Since some model ID name components (like 'kimi-k2.5') contain periods, we're not
able to reliably differentiate CRIS profile IDs (like 'global.moonshotai.kimi-k2.5')
from model IDs (like 'moonshotai.kimi-k2.5') using dot-separator counts alone.
Although it's not ideal, and should not be relied upon except where really necessary,
this leads us to maintain a list of known cross-Region inference profile prefixes in
this module.
"""

from __future__ import annotations

from enum import StrEnum


class KnownInferenceProfilePrefix(StrEnum):
    """Known ID prefix for a cross-Region inference profile"""

    APAC = "apac"  # Asia-Pacific
    AU = "au"  # Australia
    EU = "eu"  # Europe
    GLOBAL = "global"  # Global/commercial
    JP = "jp"  # Japan
    US = "us"  # United States


KNOWN_GEO_INFERENCE_PROFILE_PREFIXES = frozenset(
    {
        KnownInferenceProfilePrefix.APAC,
        KnownInferenceProfilePrefix.AU,
        KnownInferenceProfilePrefix.EU,
        KnownInferenceProfilePrefix.JP,
        KnownInferenceProfilePrefix.US,
    }
)


def split_known_inference_profile_id(
    model_id: str,
) -> tuple[KnownInferenceProfilePrefix | None, str]:
    """Split a known inference profile prefix from a model ID, if present

    Matching is case-insensitive, while the returned remainder preserves its
    original casing. Unknown prefixes are not stripped because provider and
    model names can themselves contain dots.
    """
    prefix_text, separator, remainder = model_id.partition(".")
    if not separator:
        return None, model_id

    try:
        prefix = KnownInferenceProfilePrefix(prefix_text.lower())
    except ValueError:
        return None, model_id

    return prefix, remainder
