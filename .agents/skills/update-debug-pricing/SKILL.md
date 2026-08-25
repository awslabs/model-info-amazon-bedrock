---
name: update-debug-pricing
description: Debug discrepancies in resolved model pricing data; Refresh local integration test snapshots or price API data dumps; audit model pricing additions/removals and existing-price changes; diagnose unresolved model IDs; implement source-grounded price mapper or classifier fixes.
compatibility: Requires Python 3.11+, uv, AWS credentials with Price List and Bedrock read access, and standard shell/CSV diff tools.
metadata:
  project: model-info-amazon-bedrock
  workflow: pricing-maintenance
---

# Update and Debug Model Pricing

Use this skill when a model reports unresolved '-' prices in testing, a snapshot test suggests a pricing change, or otherwise a discrepancy emerges to debug between expected output price for a model and the results resolved by our library.

## Safety rules

1. Treat AWS Price List API records as authoritative inputs and this library's mapping as heuristic.
2. Match the requested `regionCode` exactly. Never copy, infer, or reuse a price from another Region or AWS partition.
3. A model-list entry does not prove that a pricing record exists. If no source publishes the model in the target Region, preserve `PricingNotFoundError`.
4. Keep `data/` and `tests/integration/snapshots/` local and uncommitted.
5. Do not add an alias until exact structural matching has been ruled out and the alias is grounded in an observed source naming difference.

## 1. Understand the ask or discrepancy

Are you just performing a routine refresh, or responding to a particular reported issue?

You MUST establish the exact target Region or selector set first because bare integration commands target only the default `us-east-1`. For any other or multi-Region investigation, you MUST pass the same `--target-regions` value throughout diagnosis, validation, and snapshot updates. The option accepts comma-separated, case-sensitive `fnmatchcase` patterns resolved against available Bedrock Regions and raises a usage error when nothing matches.

For a routine refresh, confirm AWS credentials are read-only or least-privileged, then run the integration and snapshot tests:

```bash
# Default Region only (us-east-1)
uv run pytest -m integration

# Explicit Regions and/or selectors (quote shell wildcards e.g. 'us-west-2,eu-*')
uv run pytest -m integration --target-regions <REGIONS>
```

The output and Region-scoped reports in `tests/integration/snapshots` should indicate which models, if any, are missing or unexpectedly changed in pricing since the last snapshot for each selected Region. At the time of writing there are several models for which we don't yet have pricing lookup working, so confirm with the user which one(s) you want to tackle.

If responding to a particular reported issue, you might already have enough context to understand the ask and the expected pricing figures.

## 2. Capture source pricing data

(Unless the `data/` folder already contains up-to-date pricing API dumps) Confirm AWS credentials are read-only or least-privileged, then run:

```bash
uv run scripts/dump_pricing.py --all-services
```

This captures these AWS Price List API records for the following service codes into deterministic, date-tagged files under `data/`:

- `AmazonBedrock`
- `AmazonBedrockFoundationModels`
- `AmazonBedrockService`

Use `--force` only when deliberately replacing a same-day local capture. For a
new exploratory service, use:

```bash
uv run scripts/dump_pricing.py SomeServiceCode --output-dir data
```

The source data dump is diagnostic. You can explore it as needed, to understand specific discrepancies between observed and expected prices resolved by this library for particular model IDs. Run-time lookups continue to fetch live, region-filtered data through `pricing/fetcher.py`.

## 3. Triage unresolved or inaccurately-resolved models

(Models for which pricing is missing will appear with `-` entries in pricing snapshot CSVs, or raise `PricingNotFoundError`)

For each model ID with issues:

1. Search all three dated pricing API data/ dumps for potential matches to the model ID or recognizable variants to its name, to identify relevant pricing records.
    - Note that observed variations can sometimes be quite complex, but
    - Distinct models can sometimes also carry very similar names
    - Feel free to clarify via the web (if you have access) and/or with the user, if uncertain whether candidates are matches or separate.
2. If you have access to the web, consider also fetching the official public pricing page at https://aws.amazon.com/bedrock/pricing/ as a secondary source to check against.
3. Check `product.attributes.regionCode`. A record in another Region is not a fix for your target Region.
3. Identify the pricing records' schema:
   - `usagetype` records flow through `UsagetypeMapper` and `classify_usagetype_entry`.
   - `servicename` marketplace records flow through `MarketplaceMapper` and `classify_marketplace_entry`.
4. Consider the following general guides for common failure modes:
   - Absent from every source in the target Region and no relevant records found: no code change; document the source gap and retain unresolved behavior.
   - `parse_usagetype` returns `None`: extend the structural suffix grammar in `pricing/mapper.py` using the exact observed form.
   - Parsing succeeds but matching fails: compare normalized model segments; add a reviewed alias only for a demonstrated source-name mismatch.
   - Matching succeeds but no dimension is emitted: update classification of direction, modality, cache, tier, scope, context, or unit based on source semantics.
   - Dimensions exist but convenience prices are `None`: inspect all axes and confirm whether standard tier, uncached, standard-context pricing really exists.

## 4. Implement narrowly

When implementing fixes to matching logic:

- Preserve established suffixes while adding only observed new forms.
- Normalize exact `1K tokens` by multiplying by 1,000 and exact `1M tokens` by preserving the numeric value; both become `PricingUnit.MILLION_TOKENS`.
- Preserve rate codes and source service for traceability.
- Do not broaden fallback matching in a way that can combine distinct models.
- Keep source priority and lazy per-Region caching unchanged unless the source data demonstrates a separate defect.

## 5. Validate

Run the focused unit suites for each changed pricing layer, followed by:

```bash
# Full unit test suite
uv run pytest

# Integration & snapshot tests for the default us-east-1
uv run pytest -m integration

# Explicit Regions and/or selectors (quote shell wildcards e.g. 'us-west-2,eu-*')
uv run pytest -m integration --target-regions <REGIONS>
```

You MUST reuse the exact Region selection from diagnosis because omitting `--target-regions` validates only `us-east-1`. Never validate one Region set and accept snapshots for another.

Confirm that **only** the intended model behaviours are changed, before moving on to...

## 6. Where necessary, update snapshots safely

Always confirm with the user before updating test snapshots.

To replace the local snapshots with your updated results after validating that all changes are expected, run the applicable command with the same `--target-regions` selection used during diagnosis and validation:

```bash
# If you *only* need to add/remove models:
uv run pytest -m integration --update-snapshots

# If you *also* need to enable updating existing model prices:
uv run pytest -m integration --update-snapshots --allow-snapshot-price-changes
```

Always confirm that additions/removals are expected and that every existing-ID price change is supported by current source data **before** running `--update-snapshots`.

Every accepted snapshot update writes the canonical Region-specific file and saves the same contents under its own capture timestamp:

```text
tests/integration/snapshots/pricing.models.{region}.csv
tests/integration/snapshots/pricing.profiles.{region}.csv
tests/integration/snapshots/history/pricing.models.{region}/{UTC timestamp}.csv
tests/integration/snapshots/history/pricing.profiles.{region}/{UTC timestamp}.csv
```

You can compare previous snapshots with `diff -u`, if available, to understand what changed. Repeated accepted runs are retained even when unchanged, so the timestamps record when captures actually occurred.


## 7. Final packaging

Remember to run the linter/formatter after making changes:

```bash
uv run ruff check
uv run ruff format --check
```

Report separately:

- models fixed in the exact target Region;
- models still unresolved because source data is absent or cross-Region only;
- added/removed snapshot IDs;
- any explicitly approved changes to existing prices.
