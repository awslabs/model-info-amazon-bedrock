"""Project-wide pytest configuration."""

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add snapshot options before pytest parses CLI arguments."""
    parser.addoption(
        "--update-snapshots",
        action="store_true",
        default=False,
        help="Update pricing snapshot files instead of comparing against current.",
    )
    parser.addoption(
        "--allow-snapshot-price-changes",
        action="store_true",
        default=False,
        help=(
            "Allow --update-snapshots to change prices for existing model IDs. "
            "Without this flag, only model additions and removals are accepted."
        ),
    )
