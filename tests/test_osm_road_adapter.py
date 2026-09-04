"""
test_osm_road_adapter.py — Unit tests for src/navigate/osm_road_adapter.py.

Offline unit tests verifying conversion of OSMRoadWay instances (WGS84 lat/lon)
into RoadPolyline instances (local ENU [East, North] in metres).

Test coverage:
  1. Basic conversion of an OSMRoadWay into RoadPolyline.
  2. Output shape is (N, 2) and dtype is float64.
  3. Reference coordinate (ref_lat, ref_lon) maps exactly to (0.0, 0.0) in local ENU.
  4. East/North displacements match expected values using lat_lon_to_enu_m().
  5. Original waypoint geometry ordering is strictly preserved.
  6. Road name is preserved when present.
  7. Deterministic fallback name ("osm_way_<way_id>") when name is missing/empty.
  8. Malformed and non-finite coordinates are filtered out safely.
  9. Ways with fewer than 2 valid points return None (rejected cleanly).
 10. osm_ways_to_road_polylines converts multiple ways independently and drops invalid ones.
 11. Invalid reference coordinates raise ValueError.
 12. None/empty way inputs return None or empty list.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Make src importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from navigate.ai_iekf_pipeline import lat_lon_to_enu_m
from navigate.map_matching import RoadPolyline
from navigate.osm_road_adapter import (
    _is_valid_coord,
    osm_way_to_road_polyline,
    osm_ways_to_road_polylines,
)
from navigate.osm_road_provider import OSMRoadWay


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def ref_origin() -> tuple[float, float]:
    """Arbitrary reference origin (Paris: lat 48.8566, lon 2.3522)."""
    return 48.8566, 2.3522


@pytest.fixture
def sample_way(ref_origin: tuple[float, float]) -> OSMRoadWay:
    """A 3-point road way starting at the reference origin."""
    ref_lat, ref_lon = ref_origin
    return OSMRoadWay(
        way_id=123456,
        highway_type="primary",
        coordinates=[
            (ref_lat, ref_lon),
            (ref_lat + 0.001, ref_lon),
            (ref_lat + 0.001, ref_lon + 0.002),
        ],
        name="Rue de Rivoli",
        oneway=None,
        tags={"highway": "primary", "name": "Rue de Rivoli"},
    )


# ===========================================================================
# Test Cases
# ===========================================================================

def test_basic_conversion_and_type(sample_way: OSMRoadWay, ref_origin: tuple[float, float]) -> None:
    """Test converting a valid OSMRoadWay produces a RoadPolyline instance."""
    ref_lat, ref_lon = ref_origin
    poly = osm_way_to_road_polyline(sample_way, ref_lat, ref_lon)

    assert poly is not None
    assert isinstance(poly, RoadPolyline)
    assert isinstance(poly.vertices, np.ndarray)


def test_output_shape_and_dtype(sample_way: OSMRoadWay, ref_origin: tuple[float, float]) -> None:
    """Test that vertices have shape (N, 2) and float64 dtype."""
    ref_lat, ref_lon = ref_origin
    poly = osm_way_to_road_polyline(sample_way, ref_lat, ref_lon)

    assert poly is not None
    assert poly.vertices.shape == (3, 2)
    assert poly.vertices.dtype == np.float64


def test_reference_coordinate_mapping(sample_way: OSMRoadWay, ref_origin: tuple[float, float]) -> None:
    """Test that a coordinate matching (ref_lat, ref_lon) maps to (0.0, 0.0) ENU."""
    ref_lat, ref_lon = ref_origin
    poly = osm_way_to_road_polyline(sample_way, ref_lat, ref_lon)

    assert poly is not None
    np.testing.assert_allclose(poly.vertices[0], [0.0, 0.0], atol=1e-9)


def test_east_north_displacement_values(ref_origin: tuple[float, float]) -> None:
    """
    Test that displacement matches the project's lat_lon_to_enu_m conversion exactly.
    """
    ref_lat, ref_lon = ref_origin
    d_lat = 0.001
    d_lon = 0.002

    way = OSMRoadWay(
        way_id=999,
        highway_type="secondary",
        coordinates=[
            (ref_lat, ref_lon),
            (ref_lat + d_lat, ref_lon),
            (ref_lat, ref_lon + d_lon),
        ],
        name="Test Way",
    )

    poly = osm_way_to_road_polyline(way, ref_lat, ref_lon)
    assert poly is not None

    # Expected displacements directly from lat_lon_to_enu_m
    expected_p0 = lat_lon_to_enu_m(ref_lat, ref_lon, ref_lat, ref_lon)
    expected_p1 = lat_lon_to_enu_m(ref_lat + d_lat, ref_lon, ref_lat, ref_lon)
    expected_p2 = lat_lon_to_enu_m(ref_lat, ref_lon + d_lon, ref_lat, ref_lon)

    np.testing.assert_allclose(poly.vertices[0], expected_p0, atol=1e-9)
    np.testing.assert_allclose(poly.vertices[1], expected_p1, atol=1e-9)
    np.testing.assert_allclose(poly.vertices[2], expected_p2, atol=1e-9)

    # Verify physical expectations:
    # p1 is purely North: East ~ 0, North > 0
    assert abs(poly.vertices[1, 0]) < 1e-9
    assert poly.vertices[1, 1] > 0.0
    # p2 is purely East: East > 0, North ~ 0
    assert poly.vertices[2, 0] > 0.0
    assert abs(poly.vertices[2, 1]) < 1e-9


def test_geometry_ordering_preservation(ref_origin: tuple[float, float]) -> None:
    """Test that waypoint order is strictly preserved without sorting or reversing."""
    ref_lat, ref_lon = ref_origin
    coords = [
        (ref_lat, ref_lon),
        (ref_lat + 0.001, ref_lon + 0.001),
        (ref_lat - 0.002, ref_lon + 0.003),
        (ref_lat + 0.004, ref_lon - 0.001),
    ]
    way = OSMRoadWay(way_id=777, highway_type="tertiary", coordinates=coords)

    poly = osm_way_to_road_polyline(way, ref_lat, ref_lon)
    assert poly is not None
    assert len(poly.vertices) == 4

    for i, (lat, lon) in enumerate(coords):
        expected_e, expected_n = lat_lon_to_enu_m(lat, lon, ref_lat, ref_lon)
        assert pytest.approx(poly.vertices[i, 0], rel=1e-6) == expected_e
        assert pytest.approx(poly.vertices[i, 1], rel=1e-6) == expected_n


def test_road_name_preserved_when_present(sample_way: OSMRoadWay, ref_origin: tuple[float, float]) -> None:
    """Test that existing OSM road name is used as the RoadPolyline name."""
    ref_lat, ref_lon = ref_origin
    poly = osm_way_to_road_polyline(sample_way, ref_lat, ref_lon)
    assert poly is not None
    assert poly.name == "Rue de Rivoli"


def test_deterministic_fallback_name_when_missing_or_blank(ref_origin: tuple[float, float]) -> None:
    """Test that missing/empty road names fallback to 'osm_way_<way_id>'."""
    ref_lat, ref_lon = ref_origin

    # Case 1: name is None
    way1 = OSMRoadWay(
        way_id=8849102,
        highway_type="residential",
        coordinates=[(ref_lat, ref_lon), (ref_lat + 0.001, ref_lon)],
        name=None,
    )
    poly1 = osm_way_to_road_polyline(way1, ref_lat, ref_lon)
    assert poly1 is not None
    assert poly1.name == "osm_way_8849102"

    # Case 2: name is empty / whitespace
    way2 = OSMRoadWay(
        way_id=456,
        highway_type="residential",
        coordinates=[(ref_lat, ref_lon), (ref_lat + 0.001, ref_lon)],
        name="   ",
    )
    poly2 = osm_way_to_road_polyline(way2, ref_lat, ref_lon)
    assert poly2 is not None
    assert poly2.name == "osm_way_456"


def test_malformed_and_non_finite_coordinates_handling(ref_origin: tuple[float, float]) -> None:
    """Test that non-finite, out-of-range, and malformed coordinates are skipped safely."""
    ref_lat, ref_lon = ref_origin
    raw_coords = [
        (ref_lat, ref_lon),                          # valid
        (float("nan"), ref_lon + 0.001),              # invalid NaN
        (ref_lat + 0.001, float("inf")),              # invalid Inf
        (120.0, ref_lon),                             # invalid lat > 90
        (ref_lat, -200.0),                            # invalid lon < -180
        ("invalid", "coords"),                        # invalid types
        (ref_lat + 0.002, ref_lon + 0.002, 100.0),    # invalid len != 2
        (ref_lat + 0.003, ref_lon + 0.003),           # valid
    ]

    way = OSMRoadWay(way_id=101, highway_type="trunk", coordinates=raw_coords)
    poly = osm_way_to_road_polyline(way, ref_lat, ref_lon)

    assert poly is not None
    assert poly.vertices.shape == (2, 2)
    # The 2 valid vertices should be the first and last
    np.testing.assert_allclose(poly.vertices[0], lat_lon_to_enu_m(ref_lat, ref_lon, ref_lat, ref_lon), atol=1e-9)
    np.testing.assert_allclose(
        poly.vertices[1],
        lat_lon_to_enu_m(ref_lat + 0.003, ref_lon + 0.003, ref_lat, ref_lon),
        atol=1e-9,
    )


def test_fewer_than_two_valid_points_rejected(ref_origin: tuple[float, float]) -> None:
    """Test that ways with < 2 valid points return None."""
    ref_lat, ref_lon = ref_origin

    # Empty coordinates
    way_empty = OSMRoadWay(way_id=1, highway_type="primary", coordinates=[])
    assert osm_way_to_road_polyline(way_empty, ref_lat, ref_lon) is None

    # Single coordinate point
    way_single = OSMRoadWay(way_id=2, highway_type="primary", coordinates=[(ref_lat, ref_lon)])
    assert osm_way_to_road_polyline(way_single, ref_lat, ref_lon) is None

    # Two coordinates but one is NaN
    way_one_valid = OSMRoadWay(
        way_id=3,
        highway_type="primary",
        coordinates=[(ref_lat, ref_lon), (float("nan"), ref_lon + 0.001)],
    )
    assert osm_way_to_road_polyline(way_one_valid, ref_lat, ref_lon) is None


def test_none_or_invalid_way_input(ref_origin: tuple[float, float]) -> None:
    """Test handling of None or objects without coordinates."""
    ref_lat, ref_lon = ref_origin
    assert osm_way_to_road_polyline(None, ref_lat, ref_lon) is None  # type: ignore[arg-type]


def test_invalid_reference_coordinates() -> None:
    """Test that non-finite or out-of-bounds reference coordinates raise ValueError."""
    way = OSMRoadWay(way_id=1, highway_type="primary", coordinates=[(0.0, 0.0), (0.001, 0.001)])

    with pytest.raises(ValueError, match="Reference coordinates must be finite"):
        osm_way_to_road_polyline(way, float("nan"), 0.0)

    with pytest.raises(ValueError, match="Reference coordinates must be finite"):
        osm_way_to_road_polyline(way, 0.0, float("inf"))

    with pytest.raises(ValueError, match="Reference coordinates out of bounds"):
        osm_way_to_road_polyline(way, 95.0, 0.0)

    with pytest.raises(ValueError, match="Reference coordinates out of bounds"):
        osm_way_to_road_polyline(way, 0.0, 190.0)


def test_osm_ways_to_road_polylines_batch(ref_origin: tuple[float, float]) -> None:
    """Test batch conversion of a sequence of OSMRoadWay instances."""
    ref_lat, ref_lon = ref_origin

    way_valid_1 = OSMRoadWay(
        way_id=10,
        highway_type="primary",
        coordinates=[(ref_lat, ref_lon), (ref_lat + 0.001, ref_lon)],
        name="Road 1",
    )
    way_invalid = OSMRoadWay(
        way_id=20,
        highway_type="secondary",
        coordinates=[(ref_lat, ref_lon)],  # only 1 point
        name="Road 2",
    )
    way_valid_2 = OSMRoadWay(
        way_id=30,
        highway_type="residential",
        coordinates=[(ref_lat, ref_lon), (ref_lat, ref_lon + 0.001)],
        name=None,
    )

    polylines = osm_ways_to_road_polylines(
        [way_valid_1, way_invalid, way_valid_2],
        ref_lat,
        ref_lon,
    )

    assert len(polylines) == 2
    assert polylines[0].name == "Road 1"
    assert polylines[0].vertices.shape == (2, 2)
    assert polylines[1].name == "osm_way_30"
    assert polylines[1].vertices.shape == (2, 2)


def test_osm_ways_to_road_polylines_empty_or_none(ref_origin: tuple[float, float]) -> None:
    """Test batch conversion with empty list or None."""
    ref_lat, ref_lon = ref_origin
    assert osm_ways_to_road_polylines([], ref_lat, ref_lon) == []
    assert osm_ways_to_road_polylines(None, ref_lat, ref_lon) == []  # type: ignore[arg-type]


def test_is_valid_coord_helper() -> None:
    """Unit tests for coordinate validation helper."""
    assert _is_valid_coord((48.8566, 2.3522)) is True
    assert _is_valid_coord([-90.0, 180.0]) is True
    assert _is_valid_coord((90.0, -180.0)) is True

    # Invalid
    assert _is_valid_coord(None) is False
    assert _is_valid_coord((48.8566,)) is False
    assert _is_valid_coord((48.8566, 2.3522, 100.0)) is False
    assert _is_valid_coord((float("nan"), 2.3522)) is False
    assert _is_valid_coord((48.8566, float("inf"))) is False
    assert _is_valid_coord((91.0, 0.0)) is False
    assert _is_valid_coord((0.0, -181.0)) is False
    assert _is_valid_coord("not_a_coord") is False

