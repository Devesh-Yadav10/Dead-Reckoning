"""
test_osm_road_cache.py — Unit tests for src/navigate/osm_road_cache.py.

Offline unit tests verifying in-memory spatial and temporal caching around
OSMRoadProvider, ensuring no unnecessary network calls during navigation.

Coverage:
  1. First request calls provider and populates cache.
  2. Second request inside valid cache radius does not call provider again.
  3. Request outside cache radius triggers a provider refresh.
  4. Expired cache (> max_age_s) triggers a provider refresh.
  5. Fresh cache (< max_age_s) does not refresh.
  6. Successful refresh updates cached roads, coordinates, and timestamp.
  7. Provider failure with usable stale cache returns stale roads without raising.
  8. Provider failure with no existing cache returns [] cleanly without raising.
  9. Cache age evaluation logic.
 10. Spatial distance evaluation logic.
 11. Custom constructor parameters (radius, max_age, time_fn) are respected.
 12. Empty provider results (0 roads found) are cached and handled properly.
 13. High-frequency repeated queries inside coverage do not call provider.
 14. Force refresh bypasses fresh cache.
 15. Invalid coordinates and constructor parameters raise ValueError.
 16. Cache clear() resets state.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Make src importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from navigate.dead_reckoning import haversine_distance_m
from navigate.osm_road_cache import (
    CacheEntry,
    CacheStatus,
    DEFAULT_CACHE_RADIUS_M,
    DEFAULT_MAX_AGE_S,
    DEFAULT_QUERY_RADIUS_M,
    OSMRoadCache,
)
from navigate.osm_road_provider import (
    OSMNetworkError,
    OSMProviderError,
    OSMRoadProvider,
    OSMRoadWay,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def sample_ways() -> list[OSMRoadWay]:
    """Sample list of OSMRoadWay instances."""
    return [
        OSMRoadWay(
            way_id=101,
            highway_type="primary",
            coordinates=[(48.8566, 2.3522), (48.8576, 2.3532)],
            name="Boulevard Saint-Germain",
        ),
        OSMRoadWay(
            way_id=102,
            highway_type="residential",
            coordinates=[(48.8566, 2.3522), (48.8556, 2.3512)],
            name="Rue de Seine",
        ),
    ]


@pytest.fixture
def mock_provider(sample_ways: list[OSMRoadWay]) -> MagicMock:
    """Mock OSMRoadProvider returning sample ways."""
    provider = MagicMock(spec=OSMRoadProvider)
    provider.get_nearby_roads.return_value = sample_ways
    return provider


# ===========================================================================
# Test Cases
# ===========================================================================

def test_first_request_calls_provider(mock_provider: MagicMock, sample_ways: list[OSMRoadWay]) -> None:
    """Test that the initial request queries the provider and caches the result."""
    cache = OSMRoadCache(provider=mock_provider, cache_radius_m=400.0, query_radius_m=500.0)
    roads = cache.get_roads(48.8566, 2.3522)

    assert roads == sample_ways
    assert mock_provider.get_nearby_roads.call_count == 1
    mock_provider.get_nearby_roads.assert_called_once_with(
        lat=48.8566,
        lon=2.3522,
        radius_m=500.0,
        raise_on_error=True,
    )
    assert cache.last_status == CacheStatus.FETCHED_NEW
    assert cache.entry is not None
    assert cache.entry.roads == sample_ways


def test_second_request_inside_cache_radius_hits_cache(
    mock_provider: MagicMock,
    sample_ways: list[OSMRoadWay],
) -> None:
    """Test that subsequent requests within cache_radius_m do not query the provider."""
    cache = OSMRoadCache(provider=mock_provider, cache_radius_m=400.0)

    # Initial query at (48.8566, 2.3522)
    roads1 = cache.get_roads(48.8566, 2.3522)
    assert mock_provider.get_nearby_roads.call_count == 1

    # Small displacement ~100m away (lat + 0.0009 deg is ~100m North)
    dist = haversine_distance_m(48.8566, 2.3522, 48.8575, 2.3522)
    assert dist < 400.0

    roads2 = cache.get_roads(48.8575, 2.3522)
    assert roads2 == sample_ways
    assert mock_provider.get_nearby_roads.call_count == 1  # No additional provider call
    assert cache.last_status == CacheStatus.HIT_FRESH


def test_request_outside_cache_radius_refreshes(
    mock_provider: MagicMock,
    sample_ways: list[OSMRoadWay],
) -> None:
    """Test that a request beyond cache_radius_m triggers a new provider query."""
    cache = OSMRoadCache(provider=mock_provider, cache_radius_m=400.0)

    cache.get_roads(48.8566, 2.3522)
    assert mock_provider.get_nearby_roads.call_count == 1

    # Large displacement ~1100m away (lat + 0.01 deg is ~1.1km North)
    dist = haversine_distance_m(48.8566, 2.3522, 48.8666, 2.3522)
    assert dist > 400.0

    cache.get_roads(48.8666, 2.3522)
    assert mock_provider.get_nearby_roads.call_count == 2
    assert cache.last_status == CacheStatus.FETCHED_NEW
    assert cache.entry is not None
    assert cache.entry.center_lat == 48.8666


def test_expired_cache_causes_refresh(sample_ways: list[OSMRoadWay]) -> None:
    """Test that an expired cache entry (> max_age_s) causes a refresh."""
    current_time = 1000.0

    def mock_time() -> float:
        return current_time

    provider = MagicMock(spec=OSMRoadProvider)
    provider.get_nearby_roads.return_value = sample_ways

    cache = OSMRoadCache(
        provider=provider,
        cache_radius_m=400.0,
        max_age_s=60.0,
        time_fn=mock_time,
    )

    # Initial query at t=1000.0
    cache.get_roads(48.8566, 2.3522)
    assert provider.get_nearby_roads.call_count == 1

    # Same location at t=1030.0 (age=30s <= 60s) -> HIT_FRESH
    current_time = 1030.0
    cache.get_roads(48.8566, 2.3522)
    assert provider.get_nearby_roads.call_count == 1
    assert cache.last_status == CacheStatus.HIT_FRESH

    # Same location at t=1065.0 (age=65s > 60s) -> Expired, triggers refresh
    current_time = 1065.0
    cache.get_roads(48.8566, 2.3522)
    assert provider.get_nearby_roads.call_count == 2
    assert cache.last_status == CacheStatus.FETCHED_NEW


def test_successful_refresh_replaces_cache(sample_ways: list[OSMRoadWay]) -> None:
    """Test that a successful refresh replaces existing cached ways and timestamp."""
    current_time = 1000.0
    provider = MagicMock(spec=OSMRoadProvider)

    initial_ways = [sample_ways[0]]
    refreshed_ways = [sample_ways[1]]
    provider.get_nearby_roads.side_effect = [initial_ways, refreshed_ways]

    cache = OSMRoadCache(provider=provider, max_age_s=50.0, time_fn=lambda: current_time)

    res1 = cache.get_roads(48.8566, 2.3522)
    assert res1 == initial_ways
    assert cache.entry.timestamp == 1000.0

    # Advance time past expiration
    current_time = 1100.0
    res2 = cache.get_roads(48.8566, 2.3522)
    assert res2 == refreshed_ways
    assert cache.entry.timestamp == 1100.0
    assert cache.entry.roads == refreshed_ways


def test_provider_failure_with_stale_cache_returns_stale_data(
    sample_ways: list[OSMRoadWay],
) -> None:
    """Test fallback to stale cache when provider fails during refresh."""
    current_time = 1000.0
    provider = MagicMock(spec=OSMRoadProvider)
    provider.get_nearby_roads.side_effect = [
        sample_ways,                             # 1st call: success
        OSMNetworkError("HTTP 504 Gateway Timeout"), # 2nd call: failure
    ]

    cache = OSMRoadCache(
        provider=provider,
        cache_radius_m=400.0,
        max_age_s=60.0,
        time_fn=lambda: current_time,
    )

    # Initial query succeeds
    roads1 = cache.get_roads(48.8566, 2.3522)
    assert roads1 == sample_ways
    assert cache.last_status == CacheStatus.FETCHED_NEW

    # Expire cache
    current_time = 1100.0
    # Provider fails on refresh, but returns stale cache safely
    roads2 = cache.get_roads(48.8566, 2.3522)
    assert roads2 == sample_ways
    assert cache.last_status == CacheStatus.FETCH_FAILED_STALE


def test_provider_failure_with_no_cache_returns_empty_list() -> None:
    """Test that provider failure on initial query returns [] without raising exceptions."""
    provider = MagicMock(spec=OSMRoadProvider)
    provider.get_nearby_roads.side_effect = OSMNetworkError("Connection refused")

    cache = OSMRoadCache(provider=provider)
    roads = cache.get_roads(48.8566, 2.3522)

    assert roads == []
    assert cache.last_status == CacheStatus.FETCH_FAILED_EMPTY
    assert cache.entry is None


def test_empty_provider_result_handling() -> None:
    """Test that a provider returning 0 roads is handled cleanly as EMPTY_RESULT."""
    provider = MagicMock(spec=OSMRoadProvider)
    provider.get_nearby_roads.return_value = []

    cache = OSMRoadCache(provider=provider)
    roads = cache.get_roads(48.8566, 2.3522)

    assert roads == []
    assert cache.last_status == CacheStatus.EMPTY_RESULT
    assert cache.entry is not None
    assert cache.entry.roads == []


def test_high_frequency_queries_do_not_generate_extra_requests(
    mock_provider: MagicMock,
    sample_ways: list[OSMRoadWay],
) -> None:
    """Test simulating 100 10Hz navigation steps within a small ~50m area."""
    cache = OSMRoadCache(provider=mock_provider, cache_radius_m=400.0)

    base_lat, base_lon = 48.8566, 2.3522
    for i in range(100):
        # Simulate moving slightly (0.5m per step)
        lat = base_lat + (i * 0.000005)
        lon = base_lon + (i * 0.000005)
        roads = cache.get_roads(lat, lon)
        assert len(roads) == len(sample_ways)

    # All 100 queries inside radius must have triggered only 1 provider call
    assert mock_provider.get_nearby_roads.call_count == 1


def test_force_refresh_bypasses_valid_cache(
    mock_provider: MagicMock,
    sample_ways: list[OSMRoadWay],
) -> None:
    """Test that force_refresh=True forces a provider query even if cache is fresh."""
    cache = OSMRoadCache(provider=mock_provider, cache_radius_m=400.0)

    cache.get_roads(48.8566, 2.3522)
    assert mock_provider.get_nearby_roads.call_count == 1

    cache.get_roads(48.8566, 2.3522, force_refresh=True)
    assert mock_provider.get_nearby_roads.call_count == 2
    assert cache.last_status == CacheStatus.FETCHED_NEW


def test_is_cached_helper(sample_ways: list[OSMRoadWay]) -> None:
    """Test is_cached() method accurately reflects spatial and temporal coverage."""
    current_time = 500.0
    provider = MagicMock(spec=OSMRoadProvider)
    provider.get_nearby_roads.return_value = sample_ways

    cache = OSMRoadCache(
        provider=provider,
        cache_radius_m=400.0,
        max_age_s=100.0,
        time_fn=lambda: current_time,
    )

    assert cache.is_cached(48.8566, 2.3522) is False

    cache.get_roads(48.8566, 2.3522)
    assert cache.is_cached(48.8566, 2.3522) is True
    assert cache.is_cached(48.8575, 2.3522) is True   # ~100m away
    assert cache.is_cached(48.8666, 2.3522) is False  # ~1.1km away

    # Advance time past expiration
    current_time = 650.0
    assert cache.is_cached(48.8566, 2.3522) is False


def test_clear_cache(mock_provider: MagicMock) -> None:
    """Test clear() resets cache state."""
    cache = OSMRoadCache(provider=mock_provider)
    cache.get_roads(48.8566, 2.3522)
    assert cache.entry is not None

    cache.clear()
    assert cache.entry is None
    assert cache.last_status is None


def test_invalid_constructor_parameters() -> None:
    """Test that non-positive constructor parameters raise ValueError."""
    with pytest.raises(ValueError, match="cache_radius_m must be positive"):
        OSMRoadCache(cache_radius_m=0.0)

    with pytest.raises(ValueError, match="query_radius_m must be positive"):
        OSMRoadCache(query_radius_m=-10.0)

    with pytest.raises(ValueError, match="max_age_s must be positive"):
        OSMRoadCache(max_age_s=0.0)


def test_invalid_coordinates() -> None:
    """Test that invalid coordinates raise ValueError."""
    cache = OSMRoadCache()

    with pytest.raises(ValueError, match="Coordinates must be finite"):
        cache.get_roads(float("nan"), 2.3522)

    with pytest.raises(ValueError, match="Coordinates must be finite"):
        cache.get_roads(48.8566, float("inf"))

    with pytest.raises(ValueError, match="Coordinates out of bounds"):
        cache.get_roads(95.0, 2.3522)

    with pytest.raises(ValueError, match="Coordinates out of bounds"):
        cache.get_roads(48.8566, 200.0)

