"""Shared fixtures for integration tests."""

import pytest


@pytest.fixture
def update_snapshots(request: pytest.FixtureRequest) -> bool:
    """Whether to update snapshot files (True) or test against current (False)."""
    return request.config.getoption("--update-snapshots")


@pytest.fixture
def allow_snapshot_price_changes(request: pytest.FixtureRequest) -> bool:
    """Whether intentional changes to existing model prices may be recorded."""
    return request.config.getoption("--allow-snapshot-price-changes")
