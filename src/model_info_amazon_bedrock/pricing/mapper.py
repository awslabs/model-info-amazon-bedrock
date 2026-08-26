"""Usagetype parsing and model ID matching logic."""

from __future__ import annotations

import re

from .._inference_profile_ids import split_known_inference_profile_id
from .aliases import MODEL_ID_TO_SEGMENTS
from .types import ParsedUsagetype, UsagetypeEntry

_VERSION_SUFFIX_RE = re.compile(r"(-v\d+)?:\d+$")
_CONTEXT_SUFFIX_RE = re.compile(r":\d+[kKmM]?$|:mm$")

# Structural regex that matches any valid inference suffix at the end of a
# usagetype remainder (after the region prefix has been stripped).
# This replaces the manually-maintained INFERENCE_SUFFIXES list with a pattern
# that automatically handles new tier/scope combinations.
_INFERENCE_SUFFIX_RE = re.compile(
    r"-("
    # Reserved pricing variants
    r"reserved-\d+-month-(?:input|output)-tokens-per-minute"
    r"-cross-region-(?:global|geo)"
    r"|"
    # Cache variants used by established Bedrock pricing records.
    r"cache-(?:read|write)-input-token-count"
    r"(?:-long-context)?(?:-cross-region-(?:global|geo))?"
    r"|"
    # Mantle cache variants, including the explicit 30-minute write TTL.
    r"cache-(?:read-tokens|write-tokens-30m)"
    r"(?:-long-ctx)?-(?:standard|flex|priority)"
    r"|"
    # Image variants
    r"(?:input|output)-image-(?:token-count|count)"
    r"|"
    # Mantle long-context token variants.
    r"(?:input|output)-tokens-long-ctx-(?:standard|flex|priority)"
    r"|"
    # Token variants (most common)
    r"(?:input|output)-tokens"
    r"(?:-(?:standard|batch|flex|priority))?"
    r"(?:-(?:long-context-)?cross-region-(?:global|geo))?"
    r"(?:-batch)?"
    r")$"
)


class UsagetypeMapper:
    """Parses usagetype strings and matches them to model IDs."""

    @staticmethod
    def normalize_model_id(model_id: str) -> str:
        """Strip version suffix, context-window suffix, and geographic prefix.

        Handles:
        - Context-window suffixes: for IDs with multiple colons, strips the
          trailing context portion (e.g. ':24k', ':200k', ':512', ':mm')
        - Version suffixes: removes trailing (-v\\d+)?:\\d+ (e.g. ':0', '-v1:0')
        - Geographic prefixes: removes leading 'us.', 'eu.', 'global.' etc.
          (cross-region inference profile IDs)
        """
        # Strip a recognized inference profile prefix from the model ID.
        _, normalized = split_known_inference_profile_id(model_id)
        # Strip context-window suffix. Only applies when there are 2+ colons
        # (version:context pattern like "v1:0:24k" or "v3:0:512").
        if normalized.count(":") >= 2:
            normalized = _CONTEXT_SUFFIX_RE.sub("", normalized)
        # Strip version suffix.
        return _VERSION_SUFFIX_RE.sub("", normalized)

    @staticmethod
    def parse_usagetype(usagetype: str) -> ParsedUsagetype | None:
        """Parse a usagetype string into its components.

        Returns None if the usagetype doesn't match the expected model pattern
        (e.g. Guardrail usagetypes).
        """
        # Step 1: Split on the first '-' to extract region prefix.
        dash_idx = usagetype.find("-")
        if dash_idx == -1:
            return None

        region_prefix = usagetype[:dash_idx]
        remainder = usagetype[dash_idx + 1 :]

        # Step 2: Find the inference suffix using structural regex.
        m = _INFERENCE_SUFFIX_RE.search(remainder)
        if m is None:
            return None

        matched_suffix = m.group(1)  # Group 1 is the suffix without leading '-'

        # Step 3: Extract the model segment (between region prefix and suffix).
        suffix_start = m.start()  # Position of the '-' before the suffix
        model_segment = remainder[:suffix_start] if suffix_start > 0 else ""

        if not model_segment:
            return None

        # Step 4: Detect and strip '-mantle' marker at the end of model segment.
        # In the raw usagetype the marker appears as '-mantle-' between model
        # and suffix; after stripping the separator dash it ends with '-mantle'.
        has_mantle = False
        if model_segment.endswith("-mantle"):
            model_segment = model_segment[: -len("-mantle")]
            has_mantle = True

        return ParsedUsagetype(
            region_prefix=region_prefix,
            model_segment=model_segment,
            has_mantle=has_mantle,
            inference_suffix=matched_suffix,
        )

    @staticmethod
    def match(model_id: str, entries: list[UsagetypeEntry]) -> list[UsagetypeEntry]:
        """Return all entries whose usagetype matches the given model ID.

        Matching strategy:
        1. Exact match (case-insensitive) of normalized model ID against the
           usagetype model segment (after stripping mantle marker).
        2. Static alias lookup for known model ID → segment mappings.
        3. Fallback heuristic matching by stripping provider prefix and common
           suffixes, then checking containment.
        4. Fallback only succeeds if all matched segments are equivalent
           (fail-safe: ambiguous matches return empty list).
        """
        normalized = UsagetypeMapper.normalize_model_id(model_id).lower()

        # Build parsed index for all entries.
        parsed_entries: list[tuple[ParsedUsagetype, UsagetypeEntry]] = []
        for entry in entries:
            parsed = UsagetypeMapper.parse_usagetype(entry.usagetype)
            if parsed is not None:
                parsed_entries.append((parsed, entry))

        # Step 1: Exact match.
        results: list[UsagetypeEntry] = []
        for parsed, entry in parsed_entries:
            if parsed.model_segment.lower() == normalized:
                results.append(entry)

        if results:
            return results

        # Step 2: Static alias lookup.
        results = UsagetypeMapper._alias_match(normalized, parsed_entries)
        if results:
            return results

        # Step 3: Fallback heuristic matching for inconsistent API naming.
        return UsagetypeMapper._fallback_match(normalized, parsed_entries)

    @staticmethod
    def _alias_match(
        normalized_id: str,
        parsed_entries: list[tuple[ParsedUsagetype, UsagetypeEntry]],
    ) -> list[UsagetypeEntry]:
        """Match using the static alias lookup table."""

        segments_value = MODEL_ID_TO_SEGMENTS.get(normalized_id)
        if segments_value is None:
            return []

        # Normalize to a list of target segments (case-insensitive).
        if isinstance(segments_value, str):
            target_segments = {segments_value.lower()}
        else:
            target_segments = {s.lower() for s in segments_value}

        return [
            entry
            for parsed, entry in parsed_entries
            if parsed.model_segment.lower() in target_segments
        ]

    @staticmethod
    def _fallback_match(
        normalized_id: str,
        parsed_entries: list[tuple[ParsedUsagetype, UsagetypeEntry]],
    ) -> list[UsagetypeEntry]:
        """Fallback matching for inconsistent pricing API naming conventions.

        Strips provider prefix and common suffixes from the model ID, then
        performs a normalized containment check against pricing segments.

        Only returns results if all matched segments normalize to the same key
        (ambiguous matches across truly different models are treated as no match
        for safety).
        """
        # Strip provider prefix (e.g. "meta." from "meta.llama4-scout-17b")
        if "." in normalized_id:
            without_provider = normalized_id.split(".", 1)[1]
        else:
            without_provider = normalized_id

        # Normalize the model name: strip common suffixes and separators
        model_key = _normalize_for_fallback(without_provider)

        if not model_key:
            return []

        # Find all distinct segments that match via containment
        matching_segments: dict[str, str] = {}  # segment_lower -> its normalized key
        for parsed, _entry in parsed_entries:
            seg_lower = parsed.model_segment.lower()
            if seg_lower in matching_segments:
                continue
            segment_key = _normalize_for_fallback(seg_lower)
            if not segment_key:
                continue
            # Check containment in both directions
            if segment_key in model_key or model_key in segment_key:
                matching_segments[seg_lower] = segment_key

        if not matching_segments:
            return []

        # Fail-safe: all matched segments must be equivalent after accounting
        # for provider prefixes. Two segments are considered equivalent if one
        # normalized key is a suffix of the other (provider prefixes always
        # appear at the start). This allows the same model appearing under
        # different naming conventions (e.g. "Kimi-K2-Thinking" and
        # "moonshotai.kimi-k2-thinking") while rejecting truly ambiguous
        # matches (e.g. matching both "llama3-8b" and "llama3-80b").
        keys = list(matching_segments.values())
        if not _all_keys_equivalent(keys):
            return []

        # Return all entries with any of the matched segments
        matched_segment_set = set(matching_segments.keys())
        return [
            entry
            for parsed, entry in parsed_entries
            if parsed.model_segment.lower() in matched_segment_set
        ]


# Suffixes commonly present in model IDs but absent from pricing segments.
_STRIP_SUFFIXES = ["-instruct", "-chat", "-it"]


def _normalize_for_fallback(name: str) -> str:
    """Normalize a model name for fallback comparison.

    Strips common suffixes (-instruct, -chat, -it) and removes all
    non-alphanumeric characters to allow flexible matching between
    different naming conventions.
    """
    for suffix in _STRIP_SUFFIXES:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    # Remove all non-alphanumeric characters (hyphens, dots, underscores)
    return re.sub(r"[^a-z0-9]", "", name)


def _all_keys_equivalent(keys: list[str]) -> bool:
    """Check if all normalized keys refer to the same model.

    Two keys are considered equivalent if one is a suffix of the other.
    This handles the case where the same model appears with and without
    a provider prefix (e.g. 'moonshotaikimik2thinking' vs 'kimik2thinking').

    Returns False if the shortest key is fewer than 4 characters, since
    very short keys produce unreliable suffix matches.
    """
    if len(keys) <= 1:
        return True

    # Find the shortest key — all others must end with it
    shortest = min(keys, key=len)
    if len(shortest) < 4:
        return False
    return all(k.endswith(shortest) or shortest.endswith(k) for k in keys)
