# Contributing Guidelines

Thank you for your interest in contributing to our project. Whether it's a bug report, new feature, correction, or additional documentation, we greatly value feedback and contributions from our community.

Please read through this document before submitting any issues or pull requests to ensure we have all the necessary information to effectively respond to your bug report or contribution.

## Table of Contents

- [Contributing Guidelines](#contributing-guidelines)
  - [Table of Contents](#table-of-contents)
  - [Reporting Bugs/Feature Requests](#reporting-bugsfeature-requests)
  - [Contributing via Pull Requests](#contributing-via-pull-requests)
    - [Best Practices](#best-practices)
    - [Getting Started](#getting-started)
    - [AI Coding Assistants](#ai-coding-assistants)
    - [Linting/Formatting](#lintingformatting)
    - [Testing](#testing)
    - [Updating and Debugging Price Resolutions](#updating-and-debugging-price-resolutions)
    - [Architecture](#architecture)
  - [Finding Contributions to Work On](#finding-contributions-to-work-on)
  - [Code of Conduct](#code-of-conduct)
  - [Security Issue Notifications](#security-issue-notifications)
  - [Licensing](#licensing)

## Reporting Bugs/Feature Requests

We welcome you to use the GitHub issue tracker to report bugs or suggest features.

When filing an issue, please check existing open, or recently closed, issues to make sure somebody else hasn't already reported the issue. Please try to include as much information as you can. Details like these are incredibly useful:

- A reproducible test case or series of steps
- The version of our code being used
- Any modifications you've made relevant to the bug
- Anything unusual about your environment or deployment

## Contributing via Pull Requests

Contributions via pull requests are much appreciated. Before sending us a pull request, please ensure that:

1. You are working against the latest source on the `main` branch.
2. You check existing open, and recently merged, pull requests to make sure someone else hasn't addressed the problem already.
3. You open an issue to discuss any significant work - we would hate for your time to be wasted.

### Best Practices

1. Fork the repository.
2. Commit to your fork using clear commit messages that follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification.
3. Ensure that linting, formatting and tests are are passing *prior* to raising the pull request.
4. If you are introducing new functionality, please commit the appropriate unit tests.
5. Answer any default questions in the pull request interface.
6. Pay attention to any automated CI failures reported in the pull request, and stay involved in the conversation.
7. Update `CHANGELOG.md` with any notable changes you make. Be sure to add these changes under `Unreleased`.

### Getting Started

We recommend installing the package locally in editable mode for ease of development.

First, ensure you have [uv](https://docs.astral.sh/uv/) installed. Then, to install the package in editable mode along with the development dependencies and all optional dependencies, run:

```bash
uv sync --all-extras --all-groups
```

...Or for a more minimal install, you could also take just:

```bash
uv sync --group dev
```

Consider running the linter and tests (see below) *before* first starting development, to check for any pre-existing issues and save your first snapshot data.

### AI Coding Assistants

`AGENTS.md` is our canonical, harness-agnostic project guidance and portable skills live in `.agents/skills/`.

Codex and OpenCode should discover those locations directly; while Claude Code and Kiro use repository symlinks: `CLAUDE.md` and `.kiro/steering/project.md` resolve to `AGENTS.md`, and `.claude/skills` and `.kiro/skills` to `.agents/skills`.

**On Windows**, enable Developer Mode and Git symlink support before cloning (for example, `git config --global core.symlinks true`). If symlinks are unavailable, replace each adapter with a local copy of its canonical target and resync it after changes; `.agents/skills/` and `AGENTS.md` remain the sources of truth.

### Linting/Formatting

The project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting the codebase. To check for linting and formatting issues, you can run the following (or drop the `uv run` prefixes if not using uv):

```bash
uv run ruff check && uv run ruff format
```

### Testing

The project uses [pytest](https://docs.pytest.org/en/) for testing.

```bash
# Run unit tests (⚠️ excludes integration tests by default, per pyproject.toml)
uv run pytest

# Run integration & snapshot tests (requires configured AWS IAM credentials)
uv run pytest -m integration
```

The integration tests check current pricing coverage against live model lists (across base Bedrock models, system-defined/cross-Region inference profiles, and OpenAI-compatible model APIs).

Models without matched pricing are reported as **warnings** rather than failures at this time, as some are still known to be missing. These tests help identify coverage gaps; they do not establish that every matched price is correct.

#### Snapshot Tests

The **snapshot** tests maintain a developer-local (git ignored) state of previous prices resolved for each model ID, which helps guard against any unintentional changes to price resolution/matching behaviour across the catalog. 

To deliberately update the saved snapshot, instead of comparing against it and failing on differences, run:

```bash
# To allow *only* model additions or removals (changed prices still error):
uv run pytest -m integration --update-snapshots

# To *also* allow changes to the resolved prices for models:
uv run pytest -m integration --update-snapshots --allow-snapshot-price-changes
```

Every accepted update records the *new* snapshot under `tests/integration/snapshots/history/<snapshot-name>/<UTC timestamp>.csv`, then overwrites the same contents to the canonical CSV in `tests/integration/snapshots`.

> ⚠️ This means historical record-keeping is at generation time, not overwrite time: If you clear your `history/` folder and run an update, there'll be no separate record kept of what the previous version of the snapshot was.

You can compare historical versions using `diff -u` or an equivalent CSV-aware diff tool. Do not commit snapshots to the repository. Outputs are sorted by stable source identifiers so local diffs are not affected by paginator order.

### Updating and Debugging Price Resolutions

We provide an agent skill (see [.agents/skills/pricing-snapshot-refresh/SKILL.md](.agents/skills/pricing-snapshot-refresh/SKILL.md)) to help accelerate refreshing pricing data and debugging matching errors, but if working manually you can also refer to that or the shorter guidance here:

The git ignored `data/` directory can be populated with local diagnostic copies of AWS Price List API responses, via our `dump_pricing` helper script:

```bash
# Fetch raw pricing data for all three service codes used by the resolver:
uv run scripts/dump_pricing.py --all-services

# Or get guidance on the other options of the script:
uv run scripts/dump_pricing.py --help
```

> Note: The library does not load or generate these files at runtime: production lookups fetch live data and filter it to the requested `regionCode`. They're just for you to help investigate and debug price record matching behaviour.

This writes date-tagged files using the service-code names:

```text
data/prices.AmazonBedrock-YYYYMMDD.json
data/prices.AmazonBedrockFoundationModels-YYYYMMDD.json
data/prices.AmazonBedrockService-YYYYMMDD.json
```

You can also explore a new service code by fetching it independently:

```bash
# Save some new service code's pricing to the folder:
uv run scripts/dump_pricing.py SomeServiceCode --output-dir data

# ...Or with a custom filename:
uv run scripts/dump_pricing.py SomeServiceCode --region us-east-1 -o data/exploration.json
```

These files contain each contain a JSON array of the pricing records returned for the given service code - which you can search and review to diagnose unexpected discrepancies between what prices the library is returning for a particular model/region/parameter; what you expect to see; and what's coming from the source API.

### Architecture

The public client, exceptions, and output types in the top-level package are scoped to support more general model information lookup capabilities in future - though at the moment, our main implemented functionality is just simplified pricing lookup which is mostly encapsulated in the `pricing/` module with no eager re-exports.

Matching the correct pricing records for a model ID is complex, and we tackle it with a chain of multiple candidate resolvers and a layered component architecture:

```text
BedrockModelInfoClient (public API)
    └── get_model_pricing
        ├── RegionCache (per-Region caching with lazy per-source fetching)
        ├── Resolver chain
        │   ├── BedrockSource → UsagetypeMapper → Classifier
        │   ├── BedrockServiceSource → UsagetypeMapper → Classifier
        │   └── MarketplaceSource → MarketplaceMapper → Classifier
        ├── Fetchers (raw AWS Price List API interaction)
        │   ├── UsagetypeFetcher (AmazonBedrock + AmazonBedrockService)
        │   └── MarketplaceFetcher (AmazonBedrockFoundationModels)
        └── Classifier (raw entries → structured PriceDimension objects)
```

- **Fetchers** extract raw fields from the AWS Price List API with minimal transformation. They validate critical fields and defer errors for later surfacing.
- **Mappers** heuristically match a model ID to relevant raw entries. `UsagetypeMapper` parses usagetype strings, while `MarketplaceMapper` matches model IDs to service names algorithmically.
- **Classifier** interprets raw entries as structured `PriceDimension` objects with orthogonal enum axes and normalizes recognized units.
- **Resolver chain** tries each pricing source in priority order until one produces dimensions.

## Finding Contributions to Work On

Looking at the existing issues is a great way to find something to contribute on. As our projects, by default, use the default GitHub issue labels (enhancement/bug/duplicate/help wanted/invalid/question/wontfix), looking at any 'help wanted' issues is a great place to start.

## Code of Conduct

This project has adopted the [Amazon Open Source Code of Conduct](https://aws.github.io/code-of-conduct). For more information see the [Code of Conduct FAQ](https://aws.github.io/code-of-conduct-faq) or contact [opensource-codeofconduct@amazon.com](opensource-codeofconduct@amazon.com) with any additional questions or comments.

## Security Issue Notifications

If you discover a potential security issue in this project we ask that you notify AWS/Amazon Security via our [vulnerability reporting page](http://aws.amazon.com/security/vulnerability-reporting/). Please do **not** create a public github issue.

## Licensing

See the [LICENSE](LICENSE) file for our project's licensing. We will ask you to confirm the licensing of your contribution.
