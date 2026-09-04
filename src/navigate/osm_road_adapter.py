"""
osm_road_adapter.py — OpenStreetMap Road to Local ENU RoadPolyline Adapter for NAVIGATE 2.0.

Phase 2: Converts OSMRoadWay instances (with WGS84 lat/lon coordinates) into
RoadPolyline instances (with local ENU [East, North] coordinates in metres)
compatible with MapMatcher and NAVIGATE 2.0 pipelines.

Design Constraints:
- Pure NumPy / standard library.
- Reuses existing lat_lon_to_enu_m() coordinate conversion utility from navigate.ai_iekf_pipeline.
- Preserves original waypoint ordering.
- Preserves OSM road name with deterministic fallback ("osm_way_<way_id>").
- Safely validates coordinates and filters out non-finite or malformed points.
- Rejects geometries with fewer than 2 valid points.
"""

from __future__ import annotations

import logging
import math
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from navigate.ai_iekf_pipeline import lat_lon_to_enu_m
from navigate.map_matching import RoadPolyline
from navigate.osm_road_provider import OSMRoadWay

logger = logging.getLogger("osm_road_adapter")


def _is_valid_coord(coord: Any) -> bool:
    """Check if a coordinate is a valid (lat, lon) pair of finite numbers."""
    if not isinstance(coord, (tuple, list)) or len(coord) != 2:
        return False
    try:
        lat, lon = float(coord[0]), float(coord[1])
        if not (math.isfinite(lat) and math.isfinite(lon)):
            return False
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            return False
        return True
    except (TypeError, ValueError):
        return False


def osm_way_to_road_polyline(
    way: OSMRoadWay,
    ref_lat: float,
    ref_lon: float,
) -> Optional[RoadPolyline]:
    """
    Convert a single OSMRoadWay with WGS84 (lat, lon) coordinates into a
    RoadPolyline in local ENU metres relative to (ref_lat, ref_lon).

    Parameters
    ----------
    way : OSMRoadWay
        The OSM road way to convert.
    ref_lat : float
        Reference origin latitude in degrees [-90.0, 90.0].
    ref_lon : float
        Reference origin longitude in degrees [-180.0, 180.0].

    Returns
    -------
    Optional[RoadPolyline]
        The converted RoadPolyline with shape (N, 2) [East, North] in metres,
        or None if the way has fewer than 2 valid coordinate points or invalid inputs.

    Raises
    ------
    ValueError
        If ref_lat or ref_lon is non-finite or out of valid geographic ranges.
    """
    if not (math.isfinite(ref_lat) and math.isfinite(ref_lon)):
        raise ValueError(f"Reference coordinates must be finite: ({ref_lat}, {ref_lon})")
    if not (-90.0 <= ref_lat <= 90.0 and -180.0 <= ref_lon <= 180.0):
        raise ValueError(f"Reference coordinates out of bounds: lat={ref_lat}, lon={ref_lon}")

    if way is None or not hasattr(way, "coordinates") or way.coordinates is None:
        return None

    # Filter and convert valid coordinates to local ENU
    enu_points: List[Tuple[float, float]] = []
    for pt in way.coordinates:
        if not _is_valid_coord(pt):
            continue
        lat, lon = float(pt[0]), float(pt[1])
        east_m, north_m = lat_lon_to_enu_m(lat, lon, ref_lat, ref_lon)
        enu_points.append((east_m, north_m))

    # RoadPolyline requires at least 2 valid vertices
    if len(enu_points) < 2:
        logger.debug(
            "OSMRoadWay %s has fewer than 2 valid vertices after filtering; skipping.",
            getattr(way, "way_id", "unknown"),
        )
        return None

    # Determine road name
    name_str = getattr(way, "name", None)
    if name_str is not None and str(name_str).strip():
        road_name = str(name_str).strip()
    else:
        way_id = getattr(way, "way_id", "unknown")
        road_name = f"osm_way_{way_id}"

    vertices = np.array(enu_points, dtype=np.float64)
    return RoadPolyline(vertices=vertices, name=road_name)


def osm_ways_to_road_polylines(
    ways: Sequence[OSMRoadWay],
    ref_lat: float,
    ref_lon: float,
) -> List[RoadPolyline]:
    """
    Convert a sequence of OSMRoadWay objects into a list of RoadPolyline objects
    in local ENU coordinates relative to (ref_lat, ref_lon).

    Invalid ways or ways with fewer than 2 valid vertices are skipped.

    Parameters
    ----------
    ways : Sequence[OSMRoadWay]
        Sequence of OSM road ways.
    ref_lat : float
        Reference origin latitude in degrees [-90.0, 90.0].
    ref_lon : float
        Reference origin longitude in degrees [-180.0, 180.0].

    Returns
    -------
    List[RoadPolyline]
        List of converted RoadPolyline objects.
    """
    if ways is None:
        return []

    polylines: List[RoadPolyline] = []
    for way in ways:
        poly = osm_way_to_road_polyline(way, ref_lat, ref_lon)
        if poly is not None:
            polylines.append(poly)
    return polylines

