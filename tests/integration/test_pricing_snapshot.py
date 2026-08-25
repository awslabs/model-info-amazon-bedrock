"""Snapshot tests: record and detect changes in basic model pricing output.

These tests capture the standard on-demand input/output token prices for each
model into a local snapshot file. On subsequent runs, they compare against the
stored snapshot and fail if the output has changed — helping detect unintended
logic changes that alter pricing results.

Snapshot files and their local history are gitignored because they contain
derived pricing data that should not be checked in.

Usage:
    # First run, or add/remove models while retaining the previous local baseline.
    pytest tests/integration/test_pricing_snapshot.py -m integration --update-snapshots

    # Permit reviewed changes to prices for IDs already in the baseline.
    pytest tests/integration/test_pricing_snapshot.py -m integration \
        --update-snapshots --allow-snapshot-price-changes

    # Subsequent runs: detect changes.
    pytest tests/integration/test_pricing_snapshot.py -m integration
"""

from __future__ import annotations

import csv
import io
import warnings
from datetime import UTC, datetime
from pathlib import Path

import pytest

from model_info_amazon_bedrock import BedrockModelInfoClient, PricingNotFoundError

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"
PRICE_FIELDS = ("input_tokens", "output_tokens", "input_images", "output_images")


def _format_price(price: float | None) -> str:
    """Format a price for snapshot output, or '-' if None."""
    if price is None:
        return "-"
    return f"{price:.6f}"


def _build_pricing_table(
    model_ids: list[str],
    pricing_client: BedrockModelInfoClient,
    region: str,
) -> str:
    """Build a stable CSV table of basic prices for a list of model IDs.

    Columns: model_id, input_tokens, output_tokens, input_images, output_images
    Sorted by model_id for stable output.
    Models that raise PricingNotFoundError get a row with all '-' prices.
    """
    rows: list[dict[str, str]] = []

    for model_id in sorted(model_ids):
        try:
            result = pricing_client.get_model_pricing(model_id, region=region)
            rows.append(
                {
                    "model_id": model_id,
                    "input_tokens": _format_price(result.input_tokens),
                    "output_tokens": _format_price(result.output_tokens),
                    "input_images": _format_price(result.input_images),
                    "output_images": _format_price(result.output_images),
                }
            )
        except PricingNotFoundError:
            rows.append(
                {
                    "model_id": model_id,
                    "input_tokens": "-",
                    "output_tokens": "-",
                    "input_images": "-",
                    "output_images": "-",
                }
            )

    # Write as CSV for easy reading and diffing.
    # Use lineterminator="\n" to avoid platform-dependent \r\n issues.
    output = io.StringIO(newline="")
    fieldnames = ["model_id", *PRICE_FIELDS]
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _snapshot_rows(content: str) -> dict[str, dict[str, str]]:
    """Index snapshot CSV rows by model ID for semantic comparison."""
    return {row["model_id"]: row for row in csv.DictReader(io.StringIO(content))}


def _snapshot_changes(
    previous_content: str,
    current_content: str,
) -> tuple[list[str], list[str], list[tuple[str, str, str, str]]]:
    """Return added IDs, removed IDs, and changed existing price fields."""
    previous = _snapshot_rows(previous_content)
    current = _snapshot_rows(current_content)
    added = sorted(current.keys() - previous.keys())
    removed = sorted(previous.keys() - current.keys())
    price_changes = [
        (model_id, field, previous[model_id][field], current[model_id][field])
        for model_id in sorted(previous.keys() & current.keys())
        for field in PRICE_FIELDS
        if previous[model_id][field] != current[model_id][field]
    ]
    return added, removed, price_changes


def _record_snapshot_capture(snapshot_path: Path, content: str) -> Path:
    """Save the accepted current capture under its generation timestamp."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    history_dir = snapshot_path.parent / "history" / snapshot_path.stem
    history_dir.mkdir(parents=True, exist_ok=True)
    history_path = history_dir / f"{timestamp}.csv"
    history_path.write_text(content, encoding="utf-8")
    return history_path


def _format_price_changes(
    price_changes: list[tuple[str, str, str, str]],
) -> str:
    """Render changed existing-model prices for an actionable failure."""
    lines = [
        f"  {model_id} {field}: {old_value} -> {new_value}"
        for model_id, field, old_value, new_value in price_changes[:20]
    ]
    if len(price_changes) > 20:
        lines.append(f"  ... and {len(price_changes) - 20} more price changes")
    return "\n".join(lines)


def _compare_or_write_snapshot(
    snapshot_path: Path,
    current_content: str,
    update: bool,
    allow_price_changes: bool = False,
) -> None:
    """Compare or safely update a snapshot with local capture history."""
    if update:
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        previous_content = (
            snapshot_path.read_text(encoding="utf-8")
            if snapshot_path.exists()
            else None
        )

        if previous_content is None:
            added = sorted(_snapshot_rows(current_content))
            removed: list[str] = []
        else:
            added, removed, price_changes = _snapshot_changes(
                previous_content, current_content
            )
            if price_changes and not allow_price_changes:
                pytest.fail(
                    "Snapshot update would change prices for existing model IDs.\n"
                    "Review these changes, then rerun with "
                    "--allow-snapshot-price-changes if they are intentional:\n"
                    f"{_format_price_changes(price_changes)}"
                )

        # The history timestamp describes the captured contents themselves,
        # while the canonical path remains an identical latest-state baseline.
        history_path = _record_snapshot_capture(snapshot_path, current_content)
        temporary_path = snapshot_path.with_suffix(f"{snapshot_path.suffix}.tmp")
        temporary_path.write_text(current_content, encoding="utf-8")
        temporary_path.replace(snapshot_path)

        warnings.warn(
            f"Updated {snapshot_path.name}; recorded the accepted capture at "
            f"{history_path}. Added {len(added)} model(s), removed "
            f"{len(removed)} model(s).",
            stacklevel=2,
        )
        return

    if not snapshot_path.exists():
        pytest.fail(
            f"Snapshot file not found: {snapshot_path}\n"
            f"Run with --update-snapshots to generate it."
        )

    stored = snapshot_path.read_text(encoding="utf-8")
    if stored != current_content:
        # Build a helpful diff message.
        stored_lines = stored.splitlines()
        current_lines = current_content.splitlines()
        changes: list[str] = []
        max_lines = max(len(stored_lines), len(current_lines))
        for index in range(max_lines):
            old = stored_lines[index] if index < len(stored_lines) else "<missing>"
            new = current_lines[index] if index < len(current_lines) else "<missing>"
            if old != new:
                changes.append(f"  line {index + 1}:\n    - {old}\n    + {new}")

        diff_summary = "\n".join(changes[:20])
        if len(changes) > 20:
            diff_summary += f"\n  ... and {len(changes) - 20} more changes"

        pytest.fail(
            f"Pricing snapshot has changed ({len(changes)} lines differ).\n"
            f"Snapshot: {snapshot_path}\n\n"
            f"Changes:\n{diff_summary}\n\n"
            f"If this is expected, run with --update-snapshots to update."
        )


@pytest.mark.integration
def test_bedrock_foundation_model_pricing_snapshot(
    update_snapshots: bool,
    allow_snapshot_price_changes: bool,
):
    """Snapshot basic pricing for all Bedrock foundation models."""
    import boto3  # noqa: PLC0415

    region = "us-east-1"
    bedrock = boto3.client("bedrock", region_name=region)
    response = bedrock.list_foundation_models()

    model_summaries = response.get("modelSummaries", [])
    assert model_summaries, "No foundation models returned from Bedrock API"

    model_ids = [m["modelId"] for m in model_summaries if "modelId" in m]
    assert model_ids, "No model IDs found in Bedrock response"

    pricing_client = BedrockModelInfoClient()
    current = _build_pricing_table(model_ids, pricing_client, region)

    snapshot_path = SNAPSHOT_DIR / "bedrock_foundation_models.csv"
    _compare_or_write_snapshot(
        snapshot_path,
        current,
        update_snapshots,
        allow_snapshot_price_changes,
    )


@pytest.mark.integration
def test_inference_profile_pricing_snapshot(
    update_snapshots: bool,
    allow_snapshot_price_changes: bool,
):
    """Snapshot basic pricing for all system-defined inference profiles."""
    import boto3  # noqa: PLC0415

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
    current = _build_pricing_table(model_ids, pricing_client, region)

    snapshot_path = SNAPSHOT_DIR / "inference_profiles.csv"
    _compare_or_write_snapshot(
        snapshot_path,
        current,
        update_snapshots,
        allow_snapshot_price_changes,
    )
