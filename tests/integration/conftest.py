"""Shared fixtures and parametrization for integration tests."""

from fnmatch import fnmatchcase

import boto3
import pytest

_BOTO_SESSION_KEY = pytest.StashKey[boto3.Session]()
_TARGET_REGIONS_KEY = pytest.StashKey[list[str]]()


def _get_boto_session(config: pytest.Config) -> boto3.Session:
    """Return the current boto session shared by integration tests."""
    if _BOTO_SESSION_KEY not in config.stash:
        config.stash[_BOTO_SESSION_KEY] = boto3.Session()
    return config.stash[_BOTO_SESSION_KEY]


def _get_target_regions(config: pytest.Config) -> list[str]:
    """Resolve configured Region selectors to concrete Bedrock Regions."""
    if _TARGET_REGIONS_KEY in config.stash:
        return config.stash[_TARGET_REGIONS_KEY]

    configured = config.getoption("--target-regions")
    selectors = (
        ["us-east-1"]
        if configured is None
        else [
            selector.strip() for selector in configured.split(",") if selector.strip()
        ]
    )
    available_regions = _get_boto_session(config).get_available_regions("bedrock")
    target_regions = sorted(
        {
            region
            for region in available_regions
            if any(fnmatchcase(region, selector) for selector in selectors)
        }
    )
    if not target_regions:
        requested = ", ".join(repr(selector) for selector in selectors) or "<none>"
        raise pytest.UsageError(
            "--target-regions matched no available Bedrock Regions for selectors: "
            f"{requested}"
        )

    config.stash[_TARGET_REGIONS_KEY] = target_regions
    return target_regions


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Automatically parametrize integration tests against all target_regions

    This is the dynamic equivalent of putting a decorator like this on every test:
        @pytest.mark.parametrize(
            "target_region",
            REGION_LIST,
            indirect=True,
        )
    """
    if "target_region" in metafunc.fixturenames:
        target_regions = _get_target_regions(metafunc.config)
        metafunc.parametrize(
            "target_region",
            target_regions,
            ids=target_regions,
            indirect=True,
        )


@pytest.fixture(scope="session")
def boto_session(pytestconfig: pytest.Config) -> boto3.Session:
    """Return the current boto session used to discover and access Regions."""
    return _get_boto_session(pytestconfig)


@pytest.fixture(scope="session")
def target_regions(pytestconfig: pytest.Config) -> list[str]:
    """Return concrete Regions matched by --target-regions."""
    return _get_target_regions(pytestconfig)


@pytest.fixture
def target_region(request: pytest.FixtureRequest) -> str:
    """Return one concrete target Region for a parameterized integration test."""
    return request.param


@pytest.fixture
def update_snapshots(request: pytest.FixtureRequest) -> bool:
    """Whether to update snapshot files (True) or test against current (False)."""
    return request.config.getoption("--update-snapshots")


@pytest.fixture
def allow_snapshot_price_changes(request: pytest.FixtureRequest) -> bool:
    """Whether intentional changes to existing model prices may be recorded."""
    return request.config.getoption("--allow-snapshot-price-changes")
