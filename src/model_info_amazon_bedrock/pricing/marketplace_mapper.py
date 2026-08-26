"""Marketplace servicename matching logic.

Given a model ID, finds the matching servicename in the
AmazonBedrockFoundationModels pricing data. Uses algorithmic matching
(parse model ID into components, find servicename via containment +
disambiguation) with a small exceptions table for cases where the
naming conventions are too different for algorithmic matching.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .._inference_profile_ids import (
    KnownInferenceProfilePrefix,
    split_known_inference_profile_id,
)
from .aliases import MODEL_ID_TO_SERVICENAME
from .types import MarketplaceEntry

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

_SYMBOLS_RE = re.compile(r"[^-\w]")
_MULTI_DASH_RE = re.compile(r"-+")


def normalize_servicename(s: str) -> str:
    """Normalize a servicename for matching.

    Strips the '(Amazon Bedrock Edition)' suffix, handles special cases
    like 'R+', and normalizes all symbols to hyphens.
    """
    # Strip common suffix
    s = s.removesuffix(" (Amazon Bedrock Edition)")
    # Handle R+ before symbol normalization
    s = s.replace("R+", "r-plus")
    # Normalize symbols to hyphens, collapse runs
    s = _SYMBOLS_RE.sub("-", s).lower()
    return _MULTI_DASH_RE.sub("-", s).strip("-")


# ---------------------------------------------------------------------------
# Model ID parsing
# ---------------------------------------------------------------------------

_VERSION_CONTEXT_RE = re.compile(
    r"(?:-v(\d+)(?::(\d+)(?::(\d+.*))?)?|:(\d+)(?::(\d+.*))?)$"
)
_DATE_RE = re.compile(r"-(\d{8})")


@dataclass(frozen=True)
class _ModelIdComponents:
    """Parsed components of a model ID relevant for servicename matching."""

    provider: str
    model_name: str  # Normalized: hyphens, no date, no version suffix
    inference_profile_prefix: KnownInferenceProfilePrefix | None
    date: str  # 8-digit date suffix (e.g. "20250514") or ""
    version: str  # Version suffix (e.g. "1") or ""
    context: str  # Context length suffix (e.g. "24k") or ""

    @classmethod
    def parse(cls, model_id: str) -> _ModelIdComponents:
        """Parse a model ID into its components for matching."""
        profile_prefix, model_id_without_prefix = split_known_inference_profile_id(
            model_id
        )
        s = model_id_without_prefix.lower()

        # A model ID starts with its provider; the remaining model name can
        # itself contain dots, so split only once.
        provider, separator, model_name = s.partition(".")
        if not separator:
            model_name = provider
            provider = ""

        # Strip version/context suffix: -v1:0:24k or bare :0:24k
        version = ""
        context = ""
        m = _VERSION_CONTEXT_RE.search(model_name)
        if m:
            model_name = model_name[: m.start()]
            # Extract version and context from whichever branch matched
            if m.group(1) is not None:
                # Matched -v<major>:<minor>:<context>
                version = m.group(1)
                context = m.group(3) or ""
            else:
                # Matched bare :<minor>:<context>
                version = ""
                context = m.group(5) or ""

        # Strip date suffix: -20250514
        date = ""
        dm = _DATE_RE.search(model_name)
        if dm:
            date = dm.group(1)
            model_name = model_name[: dm.start()] + model_name[dm.end() :]

        # Normalize symbols
        model_name = _SYMBOLS_RE.sub("-", model_name)
        model_name = _MULTI_DASH_RE.sub("-", model_name).strip("-")

        return cls(
            provider=provider,
            model_name=model_name,
            inference_profile_prefix=profile_prefix,
            date=date,
            version=version,
            context=context,
        )


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


class MarketplaceMapper:
    """Matches model IDs to marketplace servicenames."""

    @staticmethod
    def match(
        model_id: str,
        normalized_id: str,
        entries_by_servicename: dict[str, list[MarketplaceEntry]],
    ) -> list[MarketplaceEntry] | None:
        """Find marketplace entries matching a model ID.

        Args:
            model_id: Original model ID from the user.
            normalized_id: Model ID after UsagetypeMapper normalization
                (lowercase, no version suffix, no geo prefix).
            entries_by_servicename: Marketplace data keyed by normalized
                servicename.

        Returns:
            List of matching MarketplaceEntry, or None if no match found.
        """

        # Step 1: Check exceptions table.
        exception_sn = MODEL_ID_TO_SERVICENAME.get(normalized_id)
        if exception_sn is not None:
            entries = entries_by_servicename.get(exception_sn)
            if entries:
                return entries

        # Step 2: Algorithmic matching.
        servicenames = list(entries_by_servicename.keys())
        if not servicenames:
            return None

        parsed = _ModelIdComponents.parse(model_id)
        matched_sn = _find_servicename(parsed, servicenames)
        if matched_sn is None:
            return None

        entries = entries_by_servicename.get(matched_sn)
        return entries if entries else None


def _find_servicename(
    parsed: _ModelIdComponents, servicenames: list[str]
) -> str | None:
    """Find the best matching servicename for a parsed model ID.

    Uses containment matching with progressive disambiguation:
    1. model_name contained in servicename
    2. Disambiguate by provider
    3. Disambiguate by ends-with (separates e.g. 'model-3' from 'model-3-5')

    Returns the matched servicename, or None if no unambiguous match found.
    """
    model_name = parsed.model_name
    if not model_name:
        return None

    # Find all servicenames containing the model name.
    candidates = [sn for sn in servicenames if model_name in sn]

    if not candidates:
        return None
    if len(candidates) == 1:
        return _validate_match(parsed, candidates[0])

    # Disambiguate by provider.
    if parsed.provider:
        provider_matches = [sn for sn in candidates if parsed.provider in sn]
        if len(provider_matches) == 1:
            return provider_matches[0]
        if provider_matches:
            candidates = provider_matches

    # Disambiguate by ends-with (handles 'opus-4' vs 'opus-4-1').
    endswith_matches = [sn for sn in candidates if sn.endswith(model_name)]
    if len(endswith_matches) == 1:
        return endswith_matches[0]
    if endswith_matches:
        candidates = endswith_matches

    # Still ambiguous — no match.
    return None


def _validate_match(parsed: _ModelIdComponents, servicename: str) -> str | None:
    """Validate a single candidate match.

    For short model names (<=5 chars), require the provider to also appear
    in the servicename to avoid false positives.
    """
    if len(parsed.model_name) > 5:
        return servicename
    if parsed.provider and parsed.provider in servicename:
        return servicename
    # Too short and no provider confirmation — reject.
    return None
