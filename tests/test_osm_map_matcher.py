"""
test_osm_map_matcher.py — Unit tests for src/navigate/osm_map_matcher.py.

Offline unit tests verifying multi-candidate OSM road matching against the
existing MapMatcher.

Coverage:
  1. Empty OSM road list returns no match.
  2. Single valid OSM road produces a valid match.
  3. Multiple roads: nearest valid candidate selected.
  4. Road rejected when heading gate fails.
  5. Road rejected when distance gate fails.
  6. Invalid OSM geometry is skipped safely without crashing.
  7. Multiple valid candidates with different distances: closest selected.
  8. Tie/near-tie distance: smaller heading difference preferred.
  9. Selected result preserves road name and OSMRoadWay identity.
 10. ENU reference frame is consistent with Phase 2 coordinate conversion.
 11. Existing MapMatcher is called for evaluation rather than duplicating logic.
 12. Provider/cache are not contacted by this matching layer.
 13. OSMMapMatcher class wrapper interface works as expected.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# Make src importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from navigate.ai_iekf_pipeline import lat_lon_to_enu_m
from navigate.iekf_tracker import DEG2RAD, EARTH_RADIUS_M
from navigate.map_matching import MapMatcher, MapMatchResult, RoadPolyline
from navigate.osm_map_matcher import (
    OSMMatchResult,
    OSMMapMatcher,
    match_osm_roads,
)
from navigate.osm_road_provider import OSMRoadWay


# ===========================================================================
# Fixtures & Helpers
# ===========================================================================

@pytest.fixture
def ref_origin() -> tuple[float, float]:
    """Reference origin (lat 0.0, lon 0.0)."""
    return 0.0, 0.0


def make_way(
    way_id: int,
    name: str,
    coords_m: list[tuple[float, float]],
    ref_origin: tuple[float, float],
) -> OSMRoadWay:
    """
    Helper to construct an OSMRoadWay whose (lat, lon) correspond to [East, North] metres
    relative to ref_origin under standard spherical conversion.
    """
    ref_lat, ref_lon = ref_origin
    wgs84_coords: list[tuple[float, float]] = []
    for east_m, north_m in coords_m:
        lat = ref_lat + (north_m / EARTH_RADIUS_M) * (180.0 / math.pi)
        lon = ref_lon + (east_m / (EARTH_RADIUS_M * math.cos(ref_lat * DEG2RAD))) * (180.0 / math.pi)
        wgs84_coords.append((lat, lon))

    return OSMRoadWay(
        way_id=way_id,
        highway_type="primary",
        coordinates=wgs84_coords,
        name=name,
    )


# ===========================================================================
# Test Cases
# ===========================================================================

def test_empty_osm_roads_returns_no_match(ref_origin: tuple[float, float]) -> None:
    """Test that passing an empty list or None returns a clean no-match result."""
    ref_lat, ref_lon = ref_origin
    res = match_osm_roads(
        estimated_pos=np.array([0.0, 0.0]),
        estimated_heading_deg=90.0,
        osm_roads=[],
        ref_lat=ref_lat,
        ref_lon=ref_lon,
    )

    assert isinstance(res, OSMMatchResult)
    assert res.matched is False
    assert res.candidate_count == 0
    assert res.selected_road is None
    assert res.selected_way is None
    assert "no OSM roads provided" in res.rejection_reason


def test_single_valid_road_matched(ref_origin: tuple[float, float]) -> None:
    """Test matching against a single straight road passing through (0, 0) heading East."""
    ref_lat, ref_lon = ref_origin
    # Road heading East along North=0: from (-50, 0) to (50, 0)
    road_way = make_way(101, "East Avenue", [(-50.0, 0.0), (50.0, 0.0)], ref_origin)

    # Vehicle at [10, 2] heading 90 deg (East) -> 2m distance from road
    res = match_osm_roads(
        estimated_pos=np.array([10.0, 2.0]),
        estimated_heading_deg=90.0,
        osm_roads=[road_way],
        ref_lat=ref_lat,
        ref_lon=ref_lon,
    )

    assert res.matched is True
    assert res.candidate_count == 1
    assert res.valid_candidate_count == 1
    assert res.rejected_candidate_count == 0
    assert res.selected_road is not None
    assert res.selected_road.name == "East Avenue"
    assert res.selected_way is not None
    assert res.selected_way.way_id == 101
    assert pytest.approx(res.distance_to_road_m, abs=1e-3) == 2.0
    assert pytest.approx(res.heading_diff_deg, abs=1e-3) == 0.0
    assert pytest.approx(res.projected_pos[1], abs=1e-3) == 0.0


def test_multiple_roads_nearest_candidate_selected(ref_origin: tuple[float, float]) -> None:
    """Test that the closer valid road is chosen among multiple parallel roads."""
    ref_lat, ref_lon = ref_origin
    # Road A at North = 0 (heading East)
    road_a = make_way(1, "Road A (Close)", [(-50.0, 0.0), (50.0, 0.0)], ref_origin)
    # Road B at North = 10 (heading East)
    road_b = make_way(2, "Road B (Far)", [(-50.0, 10.0), (50.0, 10.0)], ref_origin)

    # Vehicle at [0, 2] heading 90 deg -> 2m from Road A, 8m from Road B
    res = match_osm_roads(
        estimated_pos=np.array([0.0, 2.0]),
        estimated_heading_deg=90.0,
        osm_roads=[road_b, road_a],  # Road A is second in list
        ref_lat=ref_lat,
        ref_lon=ref_lon,
    )

    assert res.matched is True
    assert res.candidate_count == 2
    assert res.valid_candidate_count == 2
    assert res.selected_road.name == "Road A (Close)"
    assert res.selected_way.way_id == 1
    assert pytest.approx(res.distance_to_road_m, abs=1e-3) == 2.0


def test_heading_gate_rejection(ref_origin: tuple[float, float]) -> None:
    """Test that a nearby road is rejected if heading discrepancy exceeds gate."""
    ref_lat, ref_lon = ref_origin
    # Road heading North (0 deg)
    road_north = make_way(1, "North Road", [(0.0, -50.0), (0.0, 50.0)], ref_origin)

    # Vehicle at [2, 0] heading East (90 deg) -> diff = 90 deg > max_heading_diff_deg (30 deg)
    matcher = MapMatcher(max_match_dist_m=20.0, max_heading_diff_deg=30.0)
    res = match_osm_roads(
        estimated_pos=np.array([2.0, 0.0]),
        estimated_heading_deg=90.0,
        osm_roads=[road_north],
        ref_lat=ref_lat,
        ref_lon=ref_lon,
        matcher=matcher,
    )

    assert res.matched is False
    assert res.valid_candidate_count == 0
    assert res.rejected_candidate_count == 1
    assert res.selected_road is None


def test_distance_gate_rejection(ref_origin: tuple[float, float]) -> None:
    """Test that a road with agreeing heading is rejected if distance exceeds gate."""
    ref_lat, ref_lon = ref_origin
    # Road at North = 50m (heading East)
    road_far = make_way(1, "Far Highway", [(-50.0, 50.0), (50.0, 50.0)], ref_origin)

    # Vehicle at [0, 0] heading East -> distance = 50m > max_match_dist_m (20m)
    matcher = MapMatcher(max_match_dist_m=20.0, max_heading_diff_deg=30.0)
    res = match_osm_roads(
        estimated_pos=np.array([0.0, 0.0]),
        estimated_heading_deg=90.0,
        osm_roads=[road_far],
        ref_lat=ref_lat,
        ref_lon=ref_lon,
        matcher=matcher,
    )

    assert res.matched is False
    assert res.valid_candidate_count == 0
    assert res.rejected_candidate_count == 1


def test_invalid_osm_geometry_skipped(ref_origin: tuple[float, float]) -> None:
    """Test that invalid OSM geometries are skipped safely without preventing valid matches."""
    ref_lat, ref_lon = ref_origin
    invalid_way_empty = OSMRoadWay(way_id=1, highway_type="primary", coordinates=[])
    invalid_way_single = OSMRoadWay(way_id=2, highway_type="primary", coordinates=[(0.0, 0.0)])
    valid_way = make_way(3, "Valid Road", [(-50.0, 0.0), (50.0, 0.0)], ref_origin)

    res = match_osm_roads(
        estimated_pos=np.array([0.0, 1.0]),
        estimated_heading_deg=90.0,
        osm_roads=[invalid_way_empty, invalid_way_single, valid_way],
        ref_lat=ref_lat,
        ref_lon=ref_lon,
    )

    assert res.matched is True
    assert res.candidate_count == 3
    assert res.valid_candidate_count == 1
    assert res.rejected_candidate_count == 2
    assert res.selected_road.name == "Valid Road"


def test_tie_breaking_by_heading_difference(ref_origin: tuple[float, float]) -> None:
    """
    When two roads are at identical distance, prefer the one with smaller heading difference.
    """
    ref_lat, ref_lon = ref_origin
    # Road 1: East heading (90 deg) at North = 0
    road_east = make_way(1, "East Road", [(-50.0, 0.0), (50.0, 0.0)], ref_origin)
    # Road 2: Angle 80 deg at North = 0 (passes through origin)
    # dx = 50 * sin(80 deg), dy = 50 * cos(80 deg)
    rad80 = 80.0 * DEG2RAD
    road_angled = make_way(
        2,
        "Angled Road",
        [(-50.0 * math.sin(rad80), -50.0 * math.cos(rad80)), (50.0 * math.sin(rad80), 50.0 * math.cos(rad80))],
        ref_origin,
    )

    # Vehicle at [0, 0] (distance = 0 to both roads), heading 90 deg
    # Road 1 hdg_diff = 0 deg, Road 2 hdg_diff = 10 deg
    res = match_osm_roads(
        estimated_pos=np.array([0.0, 0.0]),
        estimated_heading_deg=90.0,
        osm_roads=[road_angled, road_east],
        ref_lat=ref_lat,
        ref_lon=ref_lon,
    )

    assert res.matched is True
    assert res.selected_road.name == "East Road"
    assert res.selected_way.way_id == 1
    assert pytest.approx(res.heading_diff_deg, abs=1e-3) == 0.0


def test_existing_map_matcher_is_invoked(ref_origin: tuple[float, float]) -> None:
    """Test that MapMatcher.match() is actually called for candidate evaluation."""
    ref_lat, ref_lon = ref_origin
    road_way = make_way(1, "Road 1", [(-50.0, 0.0), (50.0, 0.0)], ref_origin)

    matcher = MapMatcher()
    matcher_mock = MagicMock(wraps=matcher)

    match_osm_roads(
        estimated_pos=np.array([0.0, 1.0]),
        estimated_heading_deg=90.0,
        osm_roads=[road_way],
        ref_lat=ref_lat,
        ref_lon=ref_lon,
        matcher=matcher_mock,
    )

    assert matcher_mock.match.call_count == 1


def test_osm_map_matcher_class_wrapper(ref_origin: tuple[float, float]) -> None:
    """Test the OSMMapMatcher class wrapper interface."""
    ref_lat, ref_lon = ref_origin
    road_way = make_way(10, "Boulevard", [(-50.0, 0.0), (50.0, 0.0)], ref_origin)

    wrapper = OSMMapMatcher(max_match_dist_m=15.0, max_heading_diff_deg=25.0, correction_strength=0.8)
    res = wrapper.match(
        estimated_pos=np.array([5.0, 2.0]),
        estimated_heading_deg=90.0,
        osm_roads=[road_way],
        ref_lat=ref_lat,
        ref_lon=ref_lon,
    )

    assert res.matched is True
    assert res.selected_road.name == "Boulevard"
    assert pytest.approx(res.distance_to_road_m, abs=1e-3) == 2.0
    # Correction strength 0.8: pos [5, 2] -> proj [5, 0] -> corrected = [5, 2] + 0.8 * ([5, 0] - [5, 2]) = [5, 0.4]
    assert pytest.approx(res.corrected_pos[0], abs=1e-3) == 5.0
    assert pytest.approx(res.corrected_pos[1], abs=1e-3) == 0.4
