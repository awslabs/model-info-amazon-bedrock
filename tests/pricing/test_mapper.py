"""Tests for the pricing UsagetypeMapper."""

import re

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from model_info_amazon_bedrock.pricing.mapper import UsagetypeMapper
from model_info_amazon_bedrock.pricing.types import UsagetypeEntry

from ..conftest import INFERENCE_SUFFIXES


class TestNormalizeModelId:
    """Tests for UsagetypeMapper.normalize_model_id()."""

    @pytest.mark.parametrize(
        "model_id,expected",
        [
            # Strips -v1:0 suffix
            (
                "anthropic.claude-sonnet-4-20250514-v1:0",
                "anthropic.claude-sonnet-4-20250514",
            ),
            ("amazon.titan-text-express-v1:0", "amazon.titan-text-express"),
            ("meta.llama3-70b-instruct-v1:0", "meta.llama3-70b-instruct"),
            ("stability.stable-diffusion-xl-v1:0", "stability.stable-diffusion-xl"),
            # Strips -v2:0 suffix
            ("cohere.command-r-plus-v2:0", "cohere.command-r-plus"),
            # Strips :0 suffix without -vN
            (
                "anthropic.claude-3-haiku-20240307:0",
                "anthropic.claude-3-haiku-20240307",
            ),
            # No version suffix — unchanged
            ("nvidia.nemotron-nano-12b-v2", "nvidia.nemotron-nano-12b-v2"),
            (
                "anthropic.claude-sonnet-4-20250514",
                "anthropic.claude-sonnet-4-20250514",
            ),
            # Multi-digit version numbers
            ("some.model-v12:99", "some.model"),
        ],
    )
    def test_normalize_model_id(self, model_id: str, expected: str) -> None:
        assert UsagetypeMapper.normalize_model_id(model_id) == expected


# --- Strategies for Property 2 ---

# Region prefixes: uppercase alphanumeric, 3-4 characters (e.g. "USE1", "USW2", "EUW1")
_region_prefix_st = st.from_regex(r"[A-Z][A-Z0-9]{2,3}", fullmatch=True)

# Characters typical of model IDs: lowercase letters, digits, dots, hyphens
_model_id_chars = st.sampled_from(list("abcdefghijklmnopqrstuvwxyz0123456789.-"))


@st.composite
def _model_segments(draw: st.DrawFn) -> str:
    """Generate valid model segments that won't confuse the parser.

    Constraints:
    - Must not be empty
    - Must not end with '-mantle' (would confuse mantle detection)
    - Must not end with any INFERENCE_SUFFIX (would confuse suffix detection)
    """
    # Generate a base segment of reasonable length
    length = draw(st.integers(min_value=3, max_value=40))
    chars = draw(st.lists(_model_id_chars, min_size=length, max_size=length))
    segment = "".join(chars)

    # Ensure it doesn't end with '-mantle'
    if segment.endswith("-mantle"):
        segment = segment + "x"

    # Ensure it doesn't end with any inference suffix
    for suffix in INFERENCE_SUFFIXES:
        if segment.endswith(suffix):
            segment = segment + "x"

    # Ensure it's not empty
    if not segment:
        segment = "model.id"

    # Geographic inference profile prefixes are stripped from query model IDs,
    # so they are not valid literal pricing model segments for this property.
    assume(not re.match(r"^(us|eu|ap|global)\.", segment))

    return segment


# Feature: Model Info for Amazon Bedrock, Property 2: Usagetype parsing round-trip
class TestUsagetypeParsingRoundTrip:
    """Property 2: Usagetype parsing round-trip.

    For any valid usagetype string composed of a region prefix, a model-id-like
    segment, an optional '-mantle-' marker, and a known inference-type suffix,
    parsing the usagetype into a ParsedUsagetype and reconstructing the original
    string from its components SHALL produce the original usagetype.

    Validates: Requirements 2.2
    """

    @given(
        region_prefix=_region_prefix_st,
        model_segment=_model_segments(),
        has_mantle=st.booleans(),
        inference_suffix=st.sampled_from(INFERENCE_SUFFIXES),
    )
    @settings(max_examples=200)
    def test_usagetype_parsing_round_trip(
        self,
        region_prefix: str,
        model_segment: str,
        has_mantle: bool,
        inference_suffix: str,
    ) -> None:
        """**Validates: Requirements 2.2**"""
        # Construct the usagetype string from components
        mantle_part = "-mantle" if has_mantle else ""
        usagetype = f"{region_prefix}-{model_segment}{mantle_part}-{inference_suffix}"

        # Parse it
        parsed = UsagetypeMapper.parse_usagetype(usagetype)

        # Verify the parsed result is not None
        assert parsed is not None, f"Failed to parse usagetype: {usagetype}"

        # Verify individual components
        assert parsed.region_prefix == region_prefix
        assert parsed.model_segment == model_segment
        assert parsed.has_mantle == has_mantle
        assert parsed.inference_suffix == inference_suffix

        # Verify reconstruction matches original
        reconstructed = (
            f"{parsed.region_prefix}-{parsed.model_segment}"
            f"{'-mantle' if parsed.has_mantle else ''}-{parsed.inference_suffix}"
        )
        assert reconstructed == usagetype


# Feature: Model Info for Amazon Bedrock, Property 3: Model ID normalization round-trip


def _base_model_id_strategy():
    """Generate base model IDs that do NOT contain a version suffix pattern.

    Model IDs look like: provider.model-name-variant
    Characters: lowercase letters, digits, dots, hyphens.
    Must not already end with (or contain) the pattern (-v\\d+)?:\\d+.
    """
    # Generate a provider segment (e.g. "anthropic", "meta", "amazon")
    provider = st.from_regex(r"[a-z][a-z0-9]{1,10}", fullmatch=True)
    # Generate a model name segment (e.g. "claude-sonnet-4-20250514", "llama3-70b")
    model_name = st.from_regex(r"[a-z][a-z0-9\-]{2,30}[a-z0-9]", fullmatch=True)

    @st.composite
    def strategy(draw):
        p = draw(provider)
        m = draw(model_name)
        base_id = f"{p}.{m}"
        # Ensure the base ID does not contain a version suffix pattern
        assume(not re.search(r"(-v\d+)?:\d+", base_id))
        # Also ensure it doesn't end with -v followed by digits (which would
        # be partially stripped by a suffix like :0)
        assume(not re.search(r"-v\d+$", base_id))
        # Ensure it doesn't start with a geographic prefix that would be
        # stripped by normalize_model_id (us., eu., ap., global.)
        assume(not re.match(r"^(us|eu|ap|global)\.", base_id))
        return base_id

    return strategy()


def _version_suffix_strategy():
    """Generate valid version suffixes matching (-v\\d+)?:\\d+.

    Either just `:N` or `-vN:M` where N and M are non-negative integers.
    """
    # Just `:digits`
    colon_only = st.integers(min_value=0, max_value=99).map(lambda n: f":{n}")
    # `-vN:M`
    with_v = st.tuples(
        st.integers(min_value=1, max_value=99),
        st.integers(min_value=0, max_value=99),
    ).map(lambda t: f"-v{t[0]}:{t[1]}")

    return st.one_of(colon_only, with_v)


class TestNormalizeModelIdProperty:
    """Property-based tests for model ID normalization."""

    @given(base_id=_base_model_id_strategy(), suffix=_version_suffix_strategy())
    @settings(max_examples=200)
    def test_normalization_strips_version_suffix(
        self, base_id: str, suffix: str
    ) -> None:
        """Appending a version suffix and normalizing returns the base ID.

        **Validates: Requirements 2.6**
        """
        combined = base_id + suffix
        result = UsagetypeMapper.normalize_model_id(combined)
        assert result == base_id, (
            f"normalize_model_id({combined!r}) == {result!r}, expected {base_id!r}"
        )

    @given(base_id=_base_model_id_strategy())
    @settings(max_examples=200)
    def test_normalization_identity_for_base_ids(self, base_id: str) -> None:
        """A base model ID without any version suffix normalizes to itself.

        **Validates: Requirements 2.6**
        """
        result = UsagetypeMapper.normalize_model_id(base_id)
        assert result == base_id, (
            f"normalize_model_id({base_id!r}) == {result!r}, expected itself"
        )


# Feature: Model Info for Amazon Bedrock, Property 4: Mapper matching correctness


def _make_entry(usagetype: str) -> UsagetypeEntry:
    """Create a UsagetypeEntry with the given usagetype and dummy values."""
    return UsagetypeEntry(
        usagetype=usagetype,
        inference_type="Input tokens",
        provider="test-provider",
        model="test-model",
        service_tier="standard",
        unit="1K tokens",
        price_per_unit=0.001,
        offer_term_code="OFFER123",
        rate_code="RATE123",
    )


class TestMapperMatchingCorrectness:
    """Property 4: Mapper matching correctness.

    For any model ID and set of pricing entries, the mapper SHALL return
    exactly those entries whose usagetype model segment matches the
    normalized model ID under case-insensitive comparison after stripping
    the '-mantle-' marker.

    Validates: Requirements 2.1, 2.3, 2.4
    """

    @given(
        region_prefix=_region_prefix_st,
        model_segment=_model_segments(),
        inference_suffix=st.sampled_from(INFERENCE_SUFFIXES),
    )
    @settings(max_examples=200)
    def test_positive_matching(
        self,
        region_prefix: str,
        model_segment: str,
        inference_suffix: str,
    ) -> None:
        """A usagetype constructed from a model ID should match that ID.

        **Validates: Requirements 2.1, 2.3, 2.4**
        """
        usagetype = f"{region_prefix}-{model_segment}-{inference_suffix}"
        entry = _make_entry(usagetype)
        results = UsagetypeMapper.match(model_segment, [entry])
        assert entry in results, (
            f"Expected match for model_segment={model_segment!r} "
            f"in usagetype={usagetype!r}, got {results}"
        )

    @given(
        region_prefix=_region_prefix_st,
        model_segment=_model_segments(),
        inference_suffix=st.sampled_from(INFERENCE_SUFFIXES),
        data=st.data(),
    )
    @settings(max_examples=200)
    def test_case_insensitive_matching(
        self,
        region_prefix: str,
        model_segment: str,
        inference_suffix: str,
        data: st.DataObject,
    ) -> None:
        """Matching should be case-insensitive on the model segment.

        **Validates: Requirements 2.1, 2.3, 2.4**
        """
        # Randomly change case of each character in the model segment
        randomized_segment = "".join(
            data.draw(
                st.sampled_from([c.upper(), c.lower()]) if c.isalpha() else st.just(c)
            )
            for c in model_segment
        )
        usagetype = f"{region_prefix}-{randomized_segment}-{inference_suffix}"
        entry = _make_entry(usagetype)
        results = UsagetypeMapper.match(model_segment, [entry])
        assert entry in results, (
            f"Expected case-insensitive match for "
            f"model_segment={model_segment!r} "
            f"with randomized={randomized_segment!r} "
            f"in usagetype={usagetype!r}"
        )

    @given(
        region_prefix=_region_prefix_st,
        model_segment_a=_model_segments(),
        model_segment_b=_model_segments(),
        inference_suffix=st.sampled_from(INFERENCE_SUFFIXES),
    )
    @settings(max_examples=200)
    def test_negative_matching(
        self,
        region_prefix: str,
        model_segment_a: str,
        model_segment_b: str,
        inference_suffix: str,
    ) -> None:
        """A usagetype for one model should not match a different model ID.

        **Validates: Requirements 2.1, 2.3, 2.4**
        """
        # Ensure the two model segments are actually different
        assume(model_segment_a.lower() != model_segment_b.lower())
        # Ensure fallback matching won't trigger (neither normalized form
        # is a substring of the other, even after provider prefix stripping)
        key_a = re.sub(r"[^a-z0-9]", "", model_segment_a.lower())
        key_b = re.sub(r"[^a-z0-9]", "", model_segment_b.lower())
        # Also account for provider prefix stripping in fallback
        if "." in model_segment_b.lower():
            key_b_no_provider = re.sub(
                r"[^a-z0-9]", "", model_segment_b.lower().split(".", 1)[1]
            )
        else:
            key_b_no_provider = key_b
        assume(key_a not in key_b and key_b not in key_a)
        assume(key_a not in key_b_no_provider and key_b_no_provider not in key_a)

        usagetype = f"{region_prefix}-{model_segment_a}-{inference_suffix}"
        entry = _make_entry(usagetype)
        results = UsagetypeMapper.match(model_segment_b, [entry])
        assert entry not in results, (
            f"Expected no match for model_segment_b={model_segment_b!r} "
            f"against usagetype from model_segment_a={model_segment_a!r}"
        )

    @given(
        region_prefix=_region_prefix_st,
        model_segment=_model_segments(),
        inference_suffix=st.sampled_from(INFERENCE_SUFFIXES),
    )
    @settings(max_examples=200)
    def test_mantle_stripping(
        self,
        region_prefix: str,
        model_segment: str,
        inference_suffix: str,
    ) -> None:
        """A usagetype with '-mantle-' should still match the model ID.

        **Validates: Requirements 2.1, 2.3, 2.4**
        """
        usagetype = f"{region_prefix}-{model_segment}-mantle-{inference_suffix}"
        entry = _make_entry(usagetype)
        results = UsagetypeMapper.match(model_segment, [entry])
        assert entry in results, (
            f"Expected match with mantle for model_segment={model_segment!r} "
            f"in usagetype={usagetype!r}, got {results}"
        )


# --- Unit tests for mapper edge cases (Task 3.7) ---


class TestMapperEdgeCases:
    """Unit tests for mapper edge cases.

    Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5
    """

    # --- Guardrail usagetypes return None from parse ---

    def test_guardrail_text_unit_returns_none(self) -> None:
        """Guardrail usagetypes don't match the model pattern."""
        result = UsagetypeMapper.parse_usagetype(
            "USE1-Guardrail-ContentPolicyUnitsConsumed"
        )
        assert result is None

    def test_guardrail_automated_reasoning_returns_none(self) -> None:
        """Guardrail automated reasoning usagetype returns None."""
        result = UsagetypeMapper.parse_usagetype(
            "USE1-Guardrail-AutomatedReasoningPolicyUnitsConsumed"
        )
        assert result is None

    def test_guardrail_sensitive_info_returns_none(self) -> None:
        """Guardrail sensitive information usagetype returns None."""
        result = UsagetypeMapper.parse_usagetype(
            "USE1-Guardrail-SensitiveInformationPolicyPaidUnitsConsumed"
        )
        assert result is None

    def test_data_automation_returns_none(self) -> None:
        """DataAutomation usagetypes don't match the model pattern."""
        result = UsagetypeMapper.parse_usagetype(
            "USE1-DataAutomation-Standard-ImagesProcessed"
        )
        assert result is None

    # --- Case-insensitive matching ---

    def test_match_case_insensitive_uppercase_model_id(self) -> None:
        """match() finds entries regardless of case in model_id."""
        entries = [_make_entry("USE1-anthropic.claude-3-haiku-input-tokens")]
        # Query with uppercase characters in model_id
        results = UsagetypeMapper.match("Anthropic.Claude-3-Haiku", entries)
        assert len(results) == 1
        assert results[0].usagetype == "USE1-anthropic.claude-3-haiku-input-tokens"

    def test_match_case_insensitive_mixed_case_usagetype(self) -> None:
        """match() finds entries with mixed-case model segments."""
        entries = [_make_entry("USE1-ZAI.GLM-4.7-output-tokens-batch")]
        results = UsagetypeMapper.match("zai.glm-4.7", entries)
        assert len(results) == 1

    def test_match_case_insensitive_both_sides(self) -> None:
        """Case-insensitive comparison works with mixed case on both."""
        entries = [_make_entry("USE1-Meta.Llama3-70B-output-tokens")]
        results = UsagetypeMapper.match("META.LLAMA3-70B", entries)
        assert len(results) == 1

    # --- Mantle marker stripping with real examples ---

    def test_mantle_stripping_nvidia_nemotron(self) -> None:
        """Real mantle example: nvidia.nemotron-super-3-120b."""
        entries = [
            _make_entry("USE1-nvidia.nemotron-super-3-120b-mantle-output-tokens-flex")
        ]
        results = UsagetypeMapper.match("nvidia.nemotron-super-3-120b", entries)
        assert len(results) == 1

    def test_mantle_stripping_mistral_magistral(self) -> None:
        """Real mantle example: mistral.magistral-small-2509."""
        entries = [
            _make_entry("USE1-mistral.magistral-small-2509-mantle-input-tokens-flex")
        ]
        results = UsagetypeMapper.match("mistral.magistral-small-2509", entries)
        assert len(results) == 1

    def test_mantle_stripping_moonshotai_kimi(self) -> None:
        """Real mantle example: moonshotai.kimi-k2-thinking."""
        entries = [
            _make_entry("USE1-moonshotai.kimi-k2-thinking-mantle-output-tokens-flex")
        ]
        results = UsagetypeMapper.match("moonshotai.kimi-k2-thinking", entries)
        assert len(results) == 1

    def test_mantle_stripping_minimax(self) -> None:
        """Real mantle example: minimax.minimax-m2.1."""
        entries = [
            _make_entry("USE1-minimax.minimax-m2.1-mantle-output-tokens-priority")
        ]
        results = UsagetypeMapper.match("minimax.minimax-m2.1", entries)
        assert len(results) == 1

    # --- Empty entries list returns empty match list ---

    def test_match_empty_entries_returns_empty(self) -> None:
        """match() with empty entries list returns empty list."""
        results = UsagetypeMapper.match("any-model", [])
        assert results == []

    # --- No matches found returns empty list ---

    def test_match_nonexistent_model_returns_empty(self) -> None:
        """match() returns empty list when no entries contain the model."""
        entries = [
            _make_entry("USE1-anthropic.claude-3-haiku-input-tokens"),
            _make_entry("USE1-zai.glm-4.7-output-tokens-batch"),
            _make_entry("USE1-nvidia.nemotron-super-3-120b-mantle-output-tokens-flex"),
        ]
        results = UsagetypeMapper.match("nonexistent.model-xyz", entries)
        assert results == []
