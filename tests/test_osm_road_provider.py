"""
test_osm_road_provider.py — Unit tests for OpenStreetMap / Overpass road provider.

Tests cover:
1. Correct construction of the Overpass QL query.
2. Correct filtering of drivable vs excluded highway types.
3. Correct parsing of representative Overpass JSON response.
4. Correct extraction of way_id.
5. Correct extraction of highway_type.
6. Correct extraction of ordered geometry waypoints.
7. Safe handling of missing / incomplete geometry.
8. Safe handling of empty Overpass response.
9. Safe handling of HTTP, network, timeout, and malformed JSON errors.
10. Offline testing with mocked network responses (no live internet dependency).
"""

import io
import json
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from navigate.osm_road_provider import (
    DEFAULT_DRIVABLE_HIGHWAY_TYPES,
    DEFAULT_OVERPASS_ENDPOINT,
    EXCLUDED_HIGHWAY_TYPES,
    OSMNetworkError,
    OSMProviderError,
    OSMResponseError,
    OSMRoadProvider,
    OSMRoadWay,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def provider() -> OSMRoadProvider:
    """Default OSMRoadProvider instance."""
    return OSMRoadProvider(
        endpoint="https://overpass-api.de/api/interpreter",
        default_radius_m=500.0,
        timeout_s=15.0,
    )


@pytest.fixture
def sample_overpass_json() -> dict:
    """Representative Overpass JSON response with multiple ways and geometry."""
    return {
        "version": 0.6,
        "generator": "Overpass API 0.7.62.1",
        "elements": [
            {
                "type": "way",
                "id": 1001,
                "nodes": [10, 11, 12],
                "geometry": [
                    {"lat": 51.5074, "lon": -0.1278},
                    {"lat": 51.5076, "lon": -0.1280},
                    {"lat": 51.5078, "lon": -0.1282},
                ],
                "tags": {
                    "highway": "primary",
                    "name": "Trafalgar Road",
                    "oneway": "yes",
                    "maxspeed": "30 mph",
                },
            },
            {
                "type": "way",
                "id": 1002,
                "nodes": [20, 21],
                "geometry": [
                    {"lat": 51.5080, "lon": -0.1290},
                    {"lat": 51.5085, "lon": -0.1295},
                ],
                "tags": {
                    "highway": "residential",
                    "name": "St Martin's Lane",
                },
            },
            {
                # Excluded highway type: footway
                "type": "way",
                "id": 1003,
                "geometry": [
                    {"lat": 51.5090, "lon": -0.1300},
                    {"lat": 51.5092, "lon": -0.1302},
                ],
                "tags": {
                    "highway": "footway",
                    "name": "Pedestrian Path",
                },
            },
            {
                # Non-way element: node
                "type": "node",
                "id": 9999,
                "lat": 51.5074,
                "lon": -0.1278,
            },
        ],
    }


# ===========================================================================
# 1. Query Construction Tests
# ===========================================================================

def test_build_overpass_query_default_radius(provider: OSMRoadProvider):
    """Verifies default radius and coordinates in query."""
    query = provider.build_overpass_query(51.5074, -0.1278)
    assert "[out:json]" in query
    assert "[timeout:15]" in query
    assert "way(around:500.0,51.5074000,-0.1278000)" in query
    assert "out geom" in query
    for hw in DEFAULT_DRIVABLE_HIGHWAY_TYPES:
        assert hw in query


def test_build_overpass_query_custom_radius(provider: OSMRoadProvider):
    """Verifies custom radius in query."""
    query = provider.build_overpass_query(51.5074, -0.1278, radius_m=250.0)
    assert "way(around:250.0,51.5074000,-0.1278000)" in query


def test_build_overpass_query_invalid_radius(provider: OSMRoadProvider):
    """Invalid negative or zero radius should raise ValueError."""
    with pytest.raises(ValueError, match="radius"):
        provider.build_overpass_query(51.5074, -0.1278, radius_m=-10.0)
    with pytest.raises(ValueError, match="radius"):
        provider.build_overpass_query(51.5074, -0.1278, radius_m=0.0)


# ===========================================================================
# 2. Highway Type Filtering Tests
# ===========================================================================

def test_highway_types_filtering_exclusions():
    """Explicitly excluded types must be stripped from highway_types."""
    custom_types = ["primary", "footway", "residential", "cycleway", "steps"]
    p = OSMRoadProvider(highway_types=custom_types)
    assert "primary" in p.highway_types
    assert "residential" in p.highway_types
    assert "footway" not in p.highway_types
    assert "cycleway" not in p.highway_types
    assert "steps" not in p.highway_types


def test_query_includes_drivable_types_only(provider: OSMRoadProvider):
    """Query regex must only include configured drivable types."""
    query = provider.build_overpass_query(51.0, 0.0)
    for excluded in EXCLUDED_HIGHWAY_TYPES:
        assert f"|{excluded}|" not in query
        assert f"({excluded}|" not in query
        assert f"|{excluded})" not in query


# ===========================================================================
# 3. Response Parsing Tests (way_id, highway_type, geometry, tags)
# ===========================================================================

def test_parse_overpass_response(provider: OSMRoadProvider, sample_overpass_json: dict):
    """Verifies parsing of standard elements into OSMRoadWay objects."""
    roads = provider.parse_overpass_response(sample_overpass_json)
    # 2 drivable ways (primary, residential); footway and node must be ignored
    assert len(roads) == 2

    # Check Way 1001
    r1 = roads[0]
    assert r1.way_id == 1001
    assert r1.highway_type == "primary"
    assert r1.name == "Trafalgar Road"
    assert r1.oneway == "yes"
    assert r1.num_points == 3
    assert r1.coordinates == [
        (51.5074, -0.1278),
        (51.5076, -0.1280),
        (51.5078, -0.1282),
    ]
    assert r1.tags.get("maxspeed") == "30 mph"

    # Check Way 1002
    r2 = roads[1]
    assert r2.way_id == 1002
    assert r2.highway_type == "residential"
    assert r2.name == "St Martin's Lane"
    assert r2.oneway is None
    assert r2.num_points == 2
    assert r2.coordinates == [
        (51.5080, -0.1290),
        (51.5085, -0.1295),
    ]


# ===========================================================================
# 4. Incomplete / Malformed Geometry Handling
# ===========================================================================

def test_missing_or_empty_geometry(provider: OSMRoadProvider):
    """Ways with missing or single-point geometry should be safely skipped."""
    json_data = {
        "elements": [
            {
                "type": "way",
                "id": 2001,
                "tags": {"highway": "primary", "name": "No Geom Road"},
                # Missing 'geometry' key
            },
            {
                "type": "way",
                "id": 2002,
                "geometry": [{"lat": 51.5, "lon": -0.1}],  # Only 1 point
                "tags": {"highway": "primary", "name": "Single Point Road"},
            },
            {
                "type": "way",
                "id": 2003,
                "geometry": [
                    {"lat": "invalid", "lon": -0.1},
                    {"lat": 51.5, "lon": -0.1},
                ],
                "tags": {"highway": "primary", "name": "Corrupted Point Road"},
            },
            {
                "type": "way",
                "id": 2004,
                "geometry": [
                    {"lat": 51.5000, "lon": -0.1000},
                    {"lat": 51.5010, "lon": -0.1010},
                ],
                "tags": {"highway": "primary", "name": "Valid Road"},
            },
        ]
    }
    roads = provider.parse_overpass_response(json_data)
    assert len(roads) == 1
    assert roads[0].way_id == 2004
    assert roads[0].name == "Valid Road"


def test_empty_elements_response(provider: OSMRoadProvider):
    """Empty elements list should return an empty list without error."""
    json_data = {"version": 0.6, "elements": []}
    roads = provider.parse_overpass_response(json_data)
    assert roads == []


def test_invalid_json_structure(provider: OSMRoadProvider):
    """Non-dict JSON or missing elements field should raise OSMResponseError."""
    with pytest.raises(OSMResponseError):
        provider.parse_overpass_response(["not", "a", "dict"])

    with pytest.raises(OSMResponseError):
        provider.parse_overpass_response({"elements": "not a list"})


# ===========================================================================
# 5. Network & Mocked HTTP Tests (No Live Internet Dependency)
# ===========================================================================

def test_get_nearby_roads_mocked_success(provider: OSMRoadProvider, sample_overpass_json: dict):
    """Successful mocked HTTP request returns parsed road ways."""
    mock_response = MagicMock()
    mock_response.getcode.return_value = 200
    mock_response.read.return_value = json.dumps(sample_overpass_json).encode("utf-8")
    mock_response.__enter__.return_value = mock_response

    with patch("urllib.request.urlopen", return_value=mock_response):
        roads = provider.get_nearby_roads(51.5074, -0.1278, radius_m=300.0)

    assert len(roads) == 2
    assert roads[0].way_id == 1001
    assert roads[1].way_id == 1002


def test_get_nearby_roads_http_error_graceful(provider: OSMRoadProvider):
    """HTTP errors should log and return empty list by default."""
    http_err = urllib.error.HTTPError(
        url="https://overpass-api.de/api/interpreter",
        code=429,
        msg="Too Many Requests",
        hdrs=None,
        fp=io.BytesIO(b"Rate limit exceeded"),
    )

    with patch("urllib.request.urlopen", side_effect=http_err):
        roads = provider.get_nearby_roads(51.5074, -0.1278, raise_on_error=False)
    assert roads == []


def test_get_nearby_roads_http_error_raise(provider: OSMRoadProvider):
    """HTTP errors should raise OSMNetworkError when raise_on_error=True."""
    http_err = urllib.error.HTTPError(
        url="https://overpass-api.de/api/interpreter",
        code=504,
        msg="Gateway Timeout",
        hdrs=None,
        fp=io.BytesIO(b"Gateway Timeout"),
    )

    with patch("urllib.request.urlopen", side_effect=http_err):
        with pytest.raises(OSMNetworkError, match="504"):
            provider.get_nearby_roads(51.5074, -0.1278, raise_on_error=True)


def test_get_nearby_roads_timeout_error(provider: OSMRoadProvider):
    """TimeoutError should return empty list or raise OSMNetworkError."""
    with patch("urllib.request.urlopen", side_effect=TimeoutError("Connection timed out")):
        roads = provider.get_nearby_roads(51.5074, -0.1278, raise_on_error=False)
        assert roads == []

    with patch("urllib.request.urlopen", side_effect=TimeoutError("Connection timed out")):
        with pytest.raises(OSMNetworkError, match="timed out"):
            provider.get_nearby_roads(51.5074, -0.1278, raise_on_error=True)


def test_get_nearby_roads_malformed_json_response(provider: OSMRoadProvider):
    """Malformed non-JSON body should return empty list or raise OSMResponseError."""
    mock_response = MagicMock()
    mock_response.getcode.return_value = 200
    mock_response.read.return_value = b"<html><body>Internal Server Error</body></html>"
    mock_response.__enter__.return_value = mock_response

    with patch("urllib.request.urlopen", return_value=mock_response):
        roads = provider.get_nearby_roads(51.5074, -0.1278, raise_on_error=False)
        assert roads == []

    with patch("urllib.request.urlopen", return_value=mock_response):
        with pytest.raises(OSMResponseError):
            provider.get_nearby_roads(51.5074, -0.1278, raise_on_error=True)
