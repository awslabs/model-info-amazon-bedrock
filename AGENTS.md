# Project guidance for coding agents

## Scope and source of truth

This repository implements a Python 3.11+ library to simplify fetching information about models on Amazon Bedrock, with a primary initial focus on pricing (which is distributed across multiple service codes in the AWS Price List API with no directly corresponding unique identifiers).

- This file is the canonical guidance/steering across different coding assistant tools.
- CONTRIBUTING.md contains our human developer-facing information, including some notes on the architecture and debugging.
- Use the `pricing-snapshot-refresh` skill when updating AWS Price List API dumps, refreshing pricing snapshots, or diagnosing unexpected discrepancies in reported model pricing.
- We aim to pull all model information (including pricing) from authoritative APIs at run-time. Caching at run-time for performance is okay, but this library avoids distributing internal copies of data in our package unless absolutely necessary and carefully reviewed.

## Commands quick reference

We recommend using uv for env management, so you'll usually need to prefix commands with `uv run` or ensure your venv is already activated.

- Install, with all developer and optional dependencies: `uv sync --all-extras --all-groups`
- Lint: `uv run ruff check`
- Format: `uv run ruff format`
- Run unit tests: `uv run pytest`
- Run integration tests: `uv run pytest -m integration`
- Fetch/dump raw pricing data for debugging: `uv run scripts/dump_pricing.py --all-services`
- Explicitly update snapshot tests (after you verified the changes are expected): `uv run pytest -m integration --update-snapshots`

## Local-only artifacts

The following artifacts are deliberately git ignored and should not be checked in:

- `data/` is used for storing raw price list API data for debugging and development
- `tests/integration/snapshots/` stores local test snapshots to protect against unexpected price resolution regressions introduced during development.

## Pricing invariants

- The AWS Price List API is the source of truth for pricing data (since available offers may differ between AWS Accounts).
- The Pricing API client endpoint is always `us-east-1`, but `regionCode` must equal the caller's requested target Region. Never substitute or infer a price from a different Region or partition.
- Resolve pricing sources in fixed order. Do not merge lower-priority dimensions after a source succeeds.
- Ground mapper and classifier changes in observed source records. Prefer exact structural parsing, reviewed aliases, and ambiguity rejection over guesses.
- Preserve raw numeric values for unknown units. Normalize only units whose semantics are explicit in the source data.

## Change discipline

- Make the smallest source-grounded change, retain fail-safe behavior for absent or ambiguous prices, and run relevant unit tests plus Ruff.
- Do not claim a model is priced merely because its ID appears in a Bedrock model list; a matching pricing record must exist for the exact requested Region.
- In general we don't line-wrap plain text in .md files (only the embedded code samples in them): rely on users' editors to handle plain text wrapping by default, to avoid unnecessary diffs for re-wrapping edited text.

Unless explicitly instructed by the user, avoid directly staging or committing changes in git by default.
