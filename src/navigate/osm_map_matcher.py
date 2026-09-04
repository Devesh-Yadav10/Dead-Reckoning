"""
osm_map_matcher.py — Multi-Candidate OSM Road Map Matcher for NAVIGATE 2.0.

Phase 4: Integrates a collection of candidate OSMRoadWay objects with the
existing MapMatcher to select the best valid road match for vehicle state estimation.

Design Constraints:
- Reuses existing MapMatcher from navigate.map_matching without modifying it.
- Reuses existing osm_way_to_road_polyline from navigate.osm_road_adapter.
- Evaluates all candidate roads through MapMatcher gating (distance + heading gates).
- Selects the best candidate primarily by smallest distance to road, breaking ties
  by smaller heading difference.
- Returns a clean no-match result when no candidates pass gates or when input is empty.
- Pure NumPy / Python (no network access, no GIS dependencies).
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

from navigate.map_matching import MapMatcher, MapMatchResult, RoadPolyline
from navigate.osm_road_adapter import osm_way_to_road_polyline
from navigate.osm_road_provider import OSMRoadWay

logger = logging.getLogger("osm_map_matcher")


@dataclass
class OSMMatchResult:
    """
    Result of evaluating multiple OSM road candidates against an estimated position.

    Attributes
    ----------
    match_result : MapMatchResult
        The underlying MapMatchResult from the existing MapMatcher for the selected road
        (or a no-match result if no candidates passed).
    selected_road : Optional[RoadPolyline]
        The selected RoadPolyline instance (None if no match).
    selected_way : Optional[OSMRoadWay]
        The original OSMRoadWay corresponding to the selected road (None if no match).
    candidate_count : int
        Total number of OSMRoadWay candidates evaluated.
    valid_candidate_count : int
        Number of candidates that passed all MapMatcher gates.
    rejected_candidate_count : int
        Number of candidates rejected due to invalid geometry or gate failures.
    rejected_distance_count : int
        Number of candidates rejected specifically due to the distance gate.
    rejected_heading_count : int
        Number of candidates rejected specifically due to the heading gate.
    """
    match_result: MapMatchResult
    selected_road: Optional[RoadPolyline] = None
    selected_way: Optional[OSMRoadWay] = None
    candidate_count: int = 0
    valid_candidate_count: int = 0
    rejected_candidate_count: int = 0
    rejected_distance_count: int = 0
    rejected_heading_count: int = 0

    @property
    def matched(self) -> bool:
        """True when a valid road match was found."""
        return self.match_result.matched

    @property
    def projected_pos(self) -> np.ndarray:
        """Nearest point on the matched road segment [East, North] in metres."""
        return self.match_result.projected_pos

    @property
    def corrected_pos(self) -> np.ndarray:
        """Corrected position [East, North] in metres."""
        return self.match_result.corrected_pos

    @property
    def road_heading_deg(self) -> float:
        """Heading of the matched road segment [0, 360) in degrees."""
        return self.match_result.road_heading_deg

    @property
    def segment_idx(self) -> int:
        """Segment index within the matched polyline."""
        return self.match_result.segment_idx

    @property
    def distance_to_road_m(self) -> float:
        """Perpendicular distance to the matched road in metres."""
        return self.match_result.distance_to_road_m

    @property
    def heading_diff_deg(self) -> float:
        """Absolute angular heading difference in degrees."""
        return self.match_result.heading_diff_deg

    @property
    def rejection_reason(self) -> str:
        """Reason for rejection when matched is False."""
        return self.match_result.rejection_reason

    def __repr__(self) -> str:
        if self.matched and self.selected_road is not None:
            return (
                f"OSMMatchResult(matched=True, road='{self.selected_road.name}', "
                f"dist={self.distance_to_road_m:.2f}m, "
                f"hdg_diff={self.heading_diff_deg:.1f}deg, "
                f"candidates={self.valid_candidate_count}/{self.candidate_count})"
            )
        return (
            f"OSMMatchResult(matched=False, reason='{self.rejection_reason}', "
            f"candidates={self.candidate_count})"
        )


def match_osm_roads(
    estimated_pos: np.ndarray,
    estimated_heading_deg: float,
    osm_roads: Sequence[OSMRoadWay],
    ref_lat: float,
    ref_lon: float,
    matcher: Optional[MapMatcher] = None,
    previous_way_id: Optional[int] = None,
    continuity_weight_m: float = 1.5,
    heading_weight_m_per_deg: float = 0.05,
    reject_ambiguous: bool = False,
    ambiguity_margin_m: float = 0.1,
) -> OSMMatchResult:
    """
    Match an estimated ENU position and heading against a collection of OSM road ways.

    Converts each OSMRoadWay to a local ENU RoadPolyline, evaluates it using the
    existing MapMatcher, and selects the best candidate using an interpretable
    composite scoring rule balancing perpendicular distance, heading alignment,
    and temporal continuity with the previously matched road way.

    Parameters
    ----------
    estimated_pos : np.ndarray  shape (2,)
        Estimated [East, North] vehicle position in local ENU metres.
    estimated_heading_deg : float
        Estimated vehicle heading in degrees [0, 360), clockwise from North.
    osm_roads : Sequence[OSMRoadWay]
        Collection of candidate road ways from OpenStreetMap.
    ref_lat : float
        Reference origin latitude in degrees [-90.0, 90.0].
    ref_lon : float
        Reference origin longitude in degrees [-180.0, 180.0].
    matcher : Optional[MapMatcher]
        Existing MapMatcher instance to use for gate checks and projection.
        If None, creates a default MapMatcher().
    previous_way_id : Optional[int]
        ID of the previously matched OSMRoadWay (if any) for temporal continuity.
    continuity_weight_m : float
        Soft distance bonus (metres) applied to candidates matching previous_way_id
        to prevent spurious jumping between adjacent parallel lanes (default 1.5m).
    heading_weight_m_per_deg : float
        Scale factor converting degrees of heading difference into an equivalent
        distance cost (default 0.05 m/deg, so 10 deg difference = 0.50m penalty).
    reject_ambiguous : bool
        If True, rejects match when top-2 candidates have near-identical scores
        within ambiguity_margin_m (default False).
    ambiguity_margin_m : float
        Score threshold (metres) below which candidates are considered ambiguous (default 0.1m).

    Returns
    -------
    OSMMatchResult
        Result containing the best match and candidate statistics.
    """
    if matcher is None:
        matcher = MapMatcher()

    pos = np.asarray(estimated_pos, dtype=np.float64).flatten()[:2]
    total_candidates = len(osm_roads) if osm_roads is not None else 0

    # Handle empty candidates
    if not osm_roads:
        no_match = matcher.match(pos, estimated_heading_deg, None)
        no_match.rejection_reason = "no OSM roads provided"
        return OSMMatchResult(
            match_result=no_match,
            selected_road=None,
            selected_way=None,
            candidate_count=0,
            valid_candidate_count=0,
            rejected_candidate_count=0,
            rejected_distance_count=0,
            rejected_heading_count=0,
        )

    # Evaluate all candidates through MapMatcher
    valid_candidates: List[Tuple[float, MapMatchResult, RoadPolyline, OSMRoadWay]] = []
    rejected_count = 0
    rejected_dist_count = 0
    rejected_hdg_count = 0
    last_rejection_reason = ""

    for way in osm_roads:
        # Convert OSMRoadWay -> local ENU RoadPolyline
        road_poly = osm_way_to_road_polyline(way, ref_lat, ref_lon)
        if road_poly is None:
            rejected_count += 1
            continue

        # Evaluate candidate against existing MapMatcher
        match_res = matcher.match(pos, estimated_heading_deg, road_poly)

        if not match_res.matched:
            rejected_count += 1
            reason_low = match_res.rejection_reason.lower()
            if "heading" in reason_low:
                rejected_hdg_count += 1
            elif "distance" in reason_low:
                rejected_dist_count += 1
            last_rejection_reason = match_res.rejection_reason
            continue

        # Candidate passed all gates: calculate composite matching score (lower is better)
        # Score = d_perp + (w_hdg * hdg_diff) - (continuity_bonus if same way_id)
        d_perp = match_res.distance_to_road_m
        hdg_diff = match_res.heading_diff_deg
        continuity_bonus = (
            continuity_weight_m if (previous_way_id is not None and way.way_id == previous_way_id) else 0.0
        )
        candidate_score = d_perp + (heading_weight_m_per_deg * hdg_diff) - continuity_bonus

        valid_candidates.append((candidate_score, match_res, road_poly, way))

    valid_count = len(valid_candidates)

    if valid_count > 0:
        # Sort by candidate score ascending (best candidate first)
        valid_candidates.sort(key=lambda x: x[0])
        best_score, best_match_res, best_road, best_way = valid_candidates[0]

        # Check ambiguity if requested
        if reject_ambiguous and valid_count > 1:
            second_score = valid_candidates[1][0]
            if (second_score - best_score) < ambiguity_margin_m:
                no_match = matcher.match(pos, estimated_heading_deg, None)
                no_match.rejection_reason = (
                    f"ambiguous candidates (score diff {second_score - best_score:.3f}m < {ambiguity_margin_m}m)"
                )
                return OSMMatchResult(
                    match_result=no_match,
                    selected_road=None,
                    selected_way=None,
                    candidate_count=total_candidates,
                    valid_candidate_count=valid_count,
                    rejected_candidate_count=rejected_count,
                    rejected_distance_count=rejected_dist_count,
                    rejected_heading_count=rejected_hdg_count,
                )

        return OSMMatchResult(
            match_result=best_match_res,
            selected_road=best_road,
            selected_way=best_way,
            candidate_count=total_candidates,
            valid_candidate_count=valid_count,
            rejected_candidate_count=rejected_count,
            rejected_distance_count=rejected_dist_count,
            rejected_heading_count=rejected_hdg_count,
        )

    # No candidate passed gates
    no_match = matcher.match(pos, estimated_heading_deg, None)
    no_match.rejection_reason = (
        last_rejection_reason if last_rejection_reason else "no candidate road passed gates"
    )
    return OSMMatchResult(
        match_result=no_match,
        selected_road=None,
        selected_way=None,
        candidate_count=total_candidates,
        valid_candidate_count=0,
        rejected_candidate_count=rejected_count,
        rejected_distance_count=rejected_dist_count,
        rejected_heading_count=rejected_hdg_count,
    )


class OSMMapMatcher:
    """
    Object-oriented multi-candidate OSM matcher with temporal continuity and ambiguity handling.

    Parameters
    ----------
    matcher : Optional[MapMatcher]
        Underlying MapMatcher instance for geometric projection and gating.
    max_match_dist_m : float
        Distance gate threshold in metres (default 20.0).
    max_heading_diff_deg : float
        Heading gate threshold in degrees (default 30.0).
    correction_strength : float
        Interpolation weight toward road centerline (default 0.5).
    continuity_weight_m : float
        Soft distance bonus (metres) applied to the previously matched road way (default 1.5).
    heading_weight_m_per_deg : float
        Scale factor for heading difference penalty in scoring (default 0.05 m/deg).
    reject_ambiguous : bool
        If True, rejects matches when multiple roads have nearly identical scores (default False).
    ambiguity_margin_m : float
        Ambiguity threshold in metres (default 0.1m).
    """

    def __init__(
        self,
        matcher: Optional[MapMatcher] = None,
        max_match_dist_m: float = 20.0,
        max_heading_diff_deg: float = 30.0,
        correction_strength: float = 0.5,
        continuity_weight_m: float = 1.5,
        heading_weight_m_per_deg: float = 0.05,
        reject_ambiguous: bool = False,
        ambiguity_margin_m: float = 0.1,
    ) -> None:
        if matcher is not None:
            self.matcher = matcher
        else:
            self.matcher = MapMatcher(
                max_match_dist_m=max_match_dist_m,
                max_heading_diff_deg=max_heading_diff_deg,
                correction_strength=correction_strength,
            )
        self.continuity_weight_m = float(continuity_weight_m)
        self.heading_weight_m_per_deg = float(heading_weight_m_per_deg)
        self.reject_ambiguous = bool(reject_ambiguous)
        self.ambiguity_margin_m = float(ambiguity_margin_m)

        # Temporal continuity state
        self.previous_way_id: Optional[int] = None
        self.previous_segment_idx: Optional[int] = None

    def reset(self) -> None:
        """Resets temporal matching state (e.g. at the start of a session or blackout)."""
        self.previous_way_id = None
        self.previous_segment_idx = None

    def match(
        self,
        estimated_pos: np.ndarray,
        estimated_heading_deg: float,
        osm_roads: Sequence[OSMRoadWay],
        ref_lat: float,
        ref_lon: float,
    ) -> OSMMatchResult:
        """
        Match estimated position against candidate OSM roads, updating temporal continuity state.
        """
        res = match_osm_roads(
            estimated_pos=estimated_pos,
            estimated_heading_deg=estimated_heading_deg,
            osm_roads=osm_roads,
            ref_lat=ref_lat,
            ref_lon=ref_lon,
            matcher=self.matcher,
            previous_way_id=self.previous_way_id,
            continuity_weight_m=self.continuity_weight_m,
            heading_weight_m_per_deg=self.heading_weight_m_per_deg,
            reject_ambiguous=self.reject_ambiguous,
            ambiguity_margin_m=self.ambiguity_margin_m,
        )
        if res.matched and res.selected_way is not None:
            self.previous_way_id = res.selected_way.way_id
            self.previous_segment_idx = res.segment_idx
        return res


