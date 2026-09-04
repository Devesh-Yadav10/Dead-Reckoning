"""
test_osm_map_matcher_realistic.py — Realistic offline synthetic road network tests.

Phase 7 (Part A): Tests multi-candidate OSM map matching under realistic network
conditions including:
  1. Parallel roads (correct nearest lane/road selected).
  2. Perpendicular intersection (road aligned with heading chosen over perpendicular).
  3. T-junction (approaching vehicle matches correct through/branch road).
  4. Curved road (multi-segment polyline with changing headings).
  5. Nearby side road (main road vs close parallel service road).
  6. Two roads with similar heading (distance discriminator).
  7. Two roads with similar distance but different heading (heading discriminator).
  8. Road transition (smooth transition between connected consecutive OSM ways).
  9. Wrong-direction candidate (oneway metadata preserved and observable).
 10. Ambiguous intersection (multiple plausible candidates handled safely and deterministically).
 11. Temporal continuity (soft preference for previous way prevents jumping).
 12. Reset capability (state does not leak across sessions).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from navigate.iekf_tracker import DEG2RAD, EARTH_RADIUS_M
from navigate.map_matching import MapMatcher, MapMatchResult, RoadPolyline
from navigate.osm_map_matcher import (
    OSMMatchResult,
    OSMMapMatcher,
    match_osm_roads,
)
from navigate.osm_road_provider import OSMRoadWay


# ===========================================================================
# Helpers for Creating Synthetic WGS84 OSMRoadWays from Local ENU Metres
# ===========================================================================

REF_LAT = 51.5074
REF_LON = -0.1278


def enu_to_wgs84(east_m: float, north_m: float, ref_lat: float = REF_LAT, ref_lon: float = REF_LON) -> tuple[float, float]:
    """Converts local ENU metres to WGS84 (lat, lon) degrees."""
    d_lat = (north_m / EARTH_RADIUS_M) * (180.0 / math.pi)
    d_lon = (east_m / (EARTH_RADIUS_M * math.cos(ref_lat * DEG2RAD))) * (180.0 / math.pi)
    return ref_lat + d_lat, ref_lon + d_lon


def make_synthetic_way(
    way_id: int,
    name: str,
    enu_points: list[tuple[float, float]],
    highway_type: str = "primary",
    oneway: str | None = None,
    ref_lat: float = REF_LAT,
    ref_lon: float = REF_LON,
) -> OSMRoadWay:
    """Creates an OSMRoadWay from a list of local ENU (East, North) coordinate pairs."""
    coords = [enu_to_wgs84(e, n, ref_lat, ref_lon) for e, n in enu_points]
    tags = {"highway": highway_type, "name": name}
    if oneway is not None:
        tags["oneway"] = oneway
    return OSMRoadWay(
        way_id=way_id,
        highway_type=highway_type,
        coordinates=coords,
        name=name,
        oneway=oneway,
        tags=tags,
    )


# ===========================================================================
# 1. Parallel Roads
# ===========================================================================

def test_scenario_parallel_roads() -> None:
    """
    Two parallel roads running East-West at North = 0m (Road A) and North = 15m (Road B).
    Vehicle is at (East=20m, North=2m), heading East (90 deg).
    Expected: Road A is chosen (dist=2m vs 13m).
    """
    road_a = make_synthetic_way(101, "Main St (South)", [(-100.0, 0.0), (100.0, 0.0)])
    road_b = make_synthetic_way(102, "Main St (North)", [(-100.0, 15.0), (100.0, 15.0)])

    res = match_osm_roads(
        estimated_pos=np.array([20.0, 2.0]),
        estimated_heading_deg=90.0,
        osm_roads=[road_b, road_a],
        ref_lat=REF_LAT,
        ref_lon=REF_LON,
    )

    assert res.matched is True
    assert res.selected_way is not None
    assert res.selected_way.way_id == 101
    assert pytest.approx(res.distance_to_road_m, abs=1e-3) == 2.0


# ===========================================================================
# 2. Perpendicular Intersection
# ===========================================================================

def test_scenario_perpendicular_intersection() -> None:
    """
    Intersection of East-West avenue and North-South boulevard at (0, 0).
    Vehicle is at (East=1.0m, North=5.0m), heading North (0 deg).
    Expected: North-South boulevard is matched; East-West avenue is rejected by heading gate (90 deg diff > 30 deg).
    """
    ew_road = make_synthetic_way(201, "East-West Ave", [(-100.0, 0.0), (100.0, 0.0)])
    ns_road = make_synthetic_way(202, "North-South Blvd", [(0.0, -100.0), (0.0, 100.0)])

    res = match_osm_roads(
        estimated_pos=np.array([1.0, 5.0]),
        estimated_heading_deg=0.0,
        osm_roads=[ew_road, ns_road],
        ref_lat=REF_LAT,
        ref_lon=REF_LON,
    )

    assert res.matched is True
    assert res.selected_way.way_id == 202
    assert res.selected_road.name == "North-South Blvd"
    assert pytest.approx(res.distance_to_road_m, abs=1e-3) == 1.0
    assert res.rejected_heading_count == 1


# ===========================================================================
# 3. T-Junction
# ===========================================================================

def test_scenario_t_junction() -> None:
    """
    T-Junction: Stem runs from (0, -100) up to (0, 0), Top bar runs from (-100, 0) to (100, 0).
    Vehicle driving North along stem at (0.5, -20.0), heading 0 deg.
    Expected: Stem road is matched; top bar is rejected by heading gate.
    """
    stem = make_synthetic_way(301, "Stem Road", [(0.0, -100.0), (0.0, 0.0)])
    cross = make_synthetic_way(302, "Cross Road", [(-100.0, 0.0), (100.0, 0.0)])

    res = match_osm_roads(
        estimated_pos=np.array([0.5, -20.0]),
        estimated_heading_deg=0.0,
        osm_roads=[cross, stem],
        ref_lat=REF_LAT,
        ref_lon=REF_LON,
    )

    assert res.matched is True
    assert res.selected_way.way_id == 301
    assert pytest.approx(res.distance_to_road_m, abs=1e-3) == 0.5


# ===========================================================================
# 4. Curved Road
# ===========================================================================

def test_scenario_curved_road() -> None:
    """
    Curved road with 4 segments bending from East (90 deg) towards North-East (45 deg).
    Waypoints: (0, 0) -> (50, 0) [90 deg] -> (85.35, 35.35) [45 deg] -> (120.7, 70.7) [45 deg]
    Vehicle is at (70.0, 18.0) heading 45 deg.
    Expected: Matches the 45 deg segment with low perpendicular distance.
    """
    curve = make_synthetic_way(
        401,
        "Curving Highway",
        [(0.0, 0.0), (50.0, 0.0), (85.355, 35.355), (120.71, 70.71)],
    )

    res = match_osm_roads(
        estimated_pos=np.array([70.0, 18.0]),
        estimated_heading_deg=45.0,
        osm_roads=[curve],
        ref_lat=REF_LAT,
        ref_lon=REF_LON,
    )

    assert res.matched is True
    assert res.segment_idx == 1
    assert pytest.approx(res.heading_diff_deg, abs=1.0) == 0.0
    assert res.distance_to_road_m < 3.0


# ===========================================================================
# 5. Nearby Side Road
# ===========================================================================

def test_scenario_nearby_side_road() -> None:
    """
    Main road (North=0, heading 90 deg) and parallel Service road (North=6m, heading 90 deg).
    Vehicle is at (30.0, 1.0) heading 90 deg (1m from main, 5m from service).
    Expected: Main road selected with dist=1.0m.
    """
    main_road = make_synthetic_way(501, "Main Arterial", [(-50.0, 0.0), (100.0, 0.0)], highway_type="primary")
    service_road = make_synthetic_way(502, "Service Lane", [(-50.0, 6.0), (100.0, 6.0)], highway_type="service")

    res = match_osm_roads(
        estimated_pos=np.array([30.0, 1.0]),
        estimated_heading_deg=90.0,
        osm_roads=[service_road, main_road],
        ref_lat=REF_LAT,
        ref_lon=REF_LON,
    )

    assert res.matched is True
    assert res.selected_way.way_id == 501
    assert pytest.approx(res.distance_to_road_m, abs=1e-3) == 1.0


# ===========================================================================
# 6. Two Roads with Similar Heading
# ===========================================================================

def test_scenario_similar_heading_different_distance() -> None:
    """
    Road 1 at heading 90 deg (North=0).
    Road 2 at heading 95 deg (North=12).
    Vehicle at (0.0, 1.5), heading 92 deg.
    Expected: Road 1 chosen due to much smaller distance (1.5m vs ~10.5m).
    """
    road1 = make_synthetic_way(601, "Road 1", [(-50.0, 0.0), (50.0, 0.0)])
    # Road 2: Angle 95 deg: dx = 100 * cos(5 deg), dy = -100 * sin(5 deg)
    dx = 50.0 * math.cos(5.0 * DEG2RAD)
    dy = 50.0 * math.sin(5.0 * DEG2RAD)
    road2 = make_synthetic_way(602, "Road 2", [(-dx, 12.0 + dy), (dx, 12.0 - dy)])

    res = match_osm_roads(
        estimated_pos=np.array([0.0, 1.5]),
        estimated_heading_deg=92.0,
        osm_roads=[road2, road1],
        ref_lat=REF_LAT,
        ref_lon=REF_LON,
    )

    assert res.matched is True
    assert res.selected_way.way_id == 601
    assert pytest.approx(res.distance_to_road_m, abs=1e-2) == 1.5


# ===========================================================================
# 7. Two Roads with Similar Distance but Different Heading
# ===========================================================================

def test_scenario_similar_distance_different_heading() -> None:
    """
    Road A heading 90 deg (East), distance 2.0m.
    Road B heading 65 deg, distance 2.0m.
    Vehicle heading 90 deg.
    Expected: Road A selected because heading difference (0 deg) is much better than Road B (25 deg).
    """
    road_a = make_synthetic_way(701, "Straight East", [(-50.0, 2.0), (50.0, 2.0)])
    rad65 = 65.0 * DEG2RAD
    dx = 50.0 * math.sin(rad65)
    dy = 50.0 * math.cos(rad65)
    # Road B passes through (0, 2.0) with heading 65 deg
    road_b = make_synthetic_way(702, "Angled Road", [(-dx, 2.0 - dy), (dx, 2.0 + dy)])

    res = match_osm_roads(
        estimated_pos=np.array([0.0, 0.0]),
        estimated_heading_deg=90.0,
        osm_roads=[road_b, road_a],
        ref_lat=REF_LAT,
        ref_lon=REF_LON,
    )

    assert res.matched is True
    assert res.selected_way.way_id == 701
    assert pytest.approx(res.heading_diff_deg, abs=1e-3) == 0.0


# ===========================================================================
# 8. Road Transition Between Connected Ways
# ===========================================================================

def test_scenario_road_transition() -> None:
    """
    Way 1: [(-100, 0), (0, 0)] (Segment 1)
    Way 2: [(0, 0), (100, 0)] (Segment 2)
    Step 1: Vehicle at (-10, 0.5), heading 90 deg -> Matches Way 1.
    Step 2: Vehicle moves to (10, 0.5), heading 90 deg -> Transitions cleanly to Way 2.
    """
    way1 = make_synthetic_way(801, "Section 1", [(-100.0, 0.0), (0.0, 0.0)])
    way2 = make_synthetic_way(802, "Section 2", [(0.0, 0.0), (100.0, 0.0)])

    matcher = OSMMapMatcher()

    # Step 1
    res1 = matcher.match(
        estimated_pos=np.array([-10.0, 0.5]),
        estimated_heading_deg=90.0,
        osm_roads=[way1, way2],
        ref_lat=REF_LAT,
        ref_lon=REF_LON,
    )
    assert res1.matched is True
    assert res1.selected_way.way_id == 801

    # Step 2: Transition across junction point (0, 0)
    res2 = matcher.match(
        estimated_pos=np.array([10.0, 0.5]),
        estimated_heading_deg=90.0,
        osm_roads=[way1, way2],
        ref_lat=REF_LAT,
        ref_lon=REF_LON,
    )
    assert res2.matched is True
    assert res2.selected_way.way_id == 802
    assert pytest.approx(res2.distance_to_road_m, abs=1e-3) == 0.5


# ===========================================================================
# 9. Wrong-Direction Candidate (One-Way Metadata Preservation)
# ===========================================================================

def test_scenario_oneway_metadata_preserved() -> None:
    """
    Verifies that one-way tags are preserved through the OSMRoadWay and OSMMatchResult
    data pipeline.
    """
    oneway_road = make_synthetic_way(
        901, "One-Way Expressway", [(-50.0, 0.0), (50.0, 0.0)], oneway="yes"
    )

    res = match_osm_roads(
        estimated_pos=np.array([10.0, 1.0]),
        estimated_heading_deg=90.0,
        osm_roads=[oneway_road],
        ref_lat=REF_LAT,
        ref_lon=REF_LON,
    )

    assert res.matched is True
    assert res.selected_way is not None
    assert res.selected_way.oneway == "yes"
    assert res.selected_way.tags.get("oneway") == "yes"


# ===========================================================================
# 10. Ambiguous Fork / Intersection Handling
# ===========================================================================

def test_scenario_ambiguous_fork_deterministic() -> None:
    """
    3 candidate roads diverging from (0, 0) at headings 85 deg, 90 deg, 95 deg.
    Vehicle at (5.0, 0.0) heading 90 deg.
    Expected: Matches 90 deg road deterministically without errors.
    """
    rad85 = 85.0 * DEG2RAD
    rad90 = 90.0 * DEG2RAD
    rad95 = 95.0 * DEG2RAD

    way85 = make_synthetic_way(1001, "Fork 85", [(0.0, 0.0), (50.0 * math.sin(rad85), 50.0 * math.cos(rad85))])
    way90 = make_synthetic_way(1002, "Fork 90", [(0.0, 0.0), (50.0 * math.sin(rad90), 50.0 * math.cos(rad90))])
    way95 = make_synthetic_way(1003, "Fork 95", [(0.0, 0.0), (50.0 * math.sin(rad95), 50.0 * math.cos(rad95))])

    matcher = OSMMapMatcher()
    res = matcher.match(
        estimated_pos=np.array([5.0, 0.0]),
        estimated_heading_deg=90.0,
        osm_roads=[way85, way95, way90],
        ref_lat=REF_LAT,
        ref_lon=REF_LON,
    )

    assert res.matched is True
    assert res.selected_way.way_id == 1002


# ===========================================================================
# 11. Temporal Continuity Preference
# ===========================================================================

def test_scenario_temporal_continuity_prevents_jumping() -> None:
    """
    Two parallel roads:
      - Lane A at North = 0.0m
      - Lane B at North = 3.0m
    Vehicle previously on Lane A at (East=10m, North=0.1m).
    Next step: Vehicle estimate jitters to (East=20m, North=1.45m) (near mid-line, slightly closer to A by 0.1m).
    With continuity bonus enabled (1.0m), matcher should softly stick to Lane A.
    Step 3: Vehicle moves directly onto Lane B at (East=30m, North=3.0m) (dist to B=0m, dist to A=3m).
    Difference (3.0m) > continuity bonus (1.0m) -> cleanly switches to Lane B.
    """
    way_a = make_synthetic_way(1101, "Lane A", [(-50.0, 0.0), (50.0, 0.0)])
    way_b = make_synthetic_way(1102, "Lane B", [(-50.0, 3.0), (50.0, 3.0)])

    matcher = OSMMapMatcher(continuity_weight_m=1.0)

    # Step 1: Match on Lane A
    res1 = matcher.match(
        estimated_pos=np.array([10.0, 0.1]),
        estimated_heading_deg=90.0,
        osm_roads=[way_a, way_b],
        ref_lat=REF_LAT,
        ref_lon=REF_LON,
    )
    assert res1.matched is True
    assert res1.selected_way.way_id == 1101

    # Step 2: Mid-line jitter (dist to A = 1.45m, dist to B = 1.55m)
    # Continuity bonus (1.0m) keeps Lane A (score A = 0.45m vs B = 1.55m)
    res2 = matcher.match(
        estimated_pos=np.array([20.0, 1.45]),
        estimated_heading_deg=90.0,
        osm_roads=[way_a, way_b],
        ref_lat=REF_LAT,
        ref_lon=REF_LON,
    )
    assert res2.matched is True
    assert res2.selected_way.way_id == 1101, "Continuity should prevent minor noise jumping between parallel lanes"

    # Step 3: Legitimate maneuver moving onto Lane B (dist to B = 0.0m, dist to A = 3.0m)
    # Difference (3.0m) > continuity bonus (1.0m) -> should transition cleanly to Lane B
    res3 = matcher.match(
        estimated_pos=np.array([30.0, 3.0]),
        estimated_heading_deg=90.0,
        osm_roads=[way_a, way_b],
        ref_lat=REF_LAT,
        ref_lon=REF_LON,
    )
    assert res3.matched is True
    assert res3.selected_way.way_id == 1102, "Legitimate lane changes should successfully transition"


# ===========================================================================
# 12. Reset Functionality
# ===========================================================================

def test_scenario_matcher_reset() -> None:
    """
    Verifies that calling reset() clears previous way state so sessions do not leak.
    """
    way_a = make_synthetic_way(1201, "Road A", [(-50.0, 0.0), (50.0, 0.0)])
    matcher = OSMMapMatcher(continuity_weight_m=2.0)

    matcher.match(
        estimated_pos=np.array([0.0, 0.0]),
        estimated_heading_deg=90.0,
        osm_roads=[way_a],
        ref_lat=REF_LAT,
        ref_lon=REF_LON,
    )
    assert matcher.previous_way_id == 1201

    matcher.reset()
    assert matcher.previous_way_id is None
