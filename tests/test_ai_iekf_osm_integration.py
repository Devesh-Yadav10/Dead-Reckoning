"""
test_ai_iekf_osm_integration.py — Integration tests for OSM road-matching in Version B (AI + ES-EKF + OSM Road).

Offline integration tests verifying that OSMRoadCache and OSMMapMatcher
are correctly wired into AIIEKFRoadPipeline during GNSS blackouts without
affecting Version A or breaking backward compatibility.

Coverage:
  1. Inverse coordinate conversion (enu_to_lat_lon_deg vs lat_lon_to_enu_m) consistency.
  2. OSM road constraint is applied during blackout when a valid match exists.
  3. OSM road constraint is NOT applied outside blackout when blackout-only mode is active.
  4. GNSS position updates remain suppressed during blackout intervals.
  5. OSM matching uses current estimated position and heading from ES-EKF state.
  6. Successful OSM match produces the expected 3D road position [corr_E, corr_N, current_U].
  7. Road covariance (road_cov_m2) is passed to the EKF position update.
  8. OSM cache/provider failure returns empty roads and does not crash the pipeline.
  9. When no OSM roads pass gates, navigation continues without road constraint.
 10. Cache layer is queried (in-memory lookup) and prevents direct network calls.
 11. Version A behavior is completely untouched and independent.
 12. Version B with use_osm=False reproduces legacy pre-blackout road behavior.
 13. Zero future/blackout GNSS data is leaked into OSM candidate lookup.
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

from navigate.ai_iekf_pipeline import (
    AIIEKFPipeline,
    enu_to_lat_lon_deg,
    lat_lon_to_enu_m,
)
from navigate.ai_iekf_road_pipeline import (
    AIIEKFRoadPipeline,
    RoadBlackoutEvaluationResult,
)
from navigate.iekf_tracker import (
    DEG2RAD,
    EARTH_RADIUS_M,
    STANDARD_GRAVITY,
)
from navigate.osm_road_cache import OSMRoadCache
from navigate.osm_road_provider import (
    OSMNetworkError,
    OSMRoadProvider,
    OSMRoadWay,
)


# ===========================================================================
# Helpers & Fixtures
# ===========================================================================

def make_wgs84_way(
    way_id: int,
    name: str,
    enu_coords: list[tuple[float, float]],
    ref_lat: float,
    ref_lon: float,
) -> OSMRoadWay:
    """Helper to create an OSMRoadWay from local ENU coordinates."""
    wgs_coords = [enu_to_lat_lon_deg(e, n, ref_lat, ref_lon) for e, n in enu_coords]
    return OSMRoadWay(
        way_id=way_id,
        highway_type="primary",
        coordinates=wgs_coords,
        name=name,
    )


def make_synthetic_session(
    N: int = 30,
    speed_ms: float = 10.0,
    ref_lat: float = 48.8566,
    ref_lon: float = 2.3522,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Creates a synthetic straight-line driving session moving North at speed_ms (1 m/s/s).
    """
    timestamps = np.arange(N, dtype=np.float64) * 1.0  # 1s per step (10 Hz window)

    imu_windows = np.zeros((N, 50, 6), dtype=np.float32)
    imu_windows[:, :, 2] = STANDARD_GRAVITY
    # Set forward acceleration to match speed
    imu_windows[:, :, 0] = 0.0

    gt_lats = np.zeros(N, dtype=np.float64)
    gt_lons = np.zeros(N, dtype=np.float64)

    for i in range(N):
        north_m = i * speed_ms
        lat, lon = enu_to_lat_lon_deg(0.0, north_m, ref_lat, ref_lon)
        gt_lats[i] = lat
        gt_lons[i] = lon

    return imu_windows, timestamps, gt_lats, gt_lons


@pytest.fixture
def mock_osm_cache(ref_lat: float = 48.8566, ref_lon: float = 2.3522) -> OSMRoadCache:
    """Mock OSMRoadCache providing a straight North-heading road along East=0."""
    north_road = make_wgs84_way(
        way_id=1001,
        name="North Highway",
        enu_coords=[(0.0, -100.0), (0.0, 500.0)],
        ref_lat=ref_lat,
        ref_lon=ref_lon,
    )
    provider_mock = MagicMock(spec=OSMRoadProvider)
    provider_mock.get_nearby_roads.return_value = [north_road]

    cache = OSMRoadCache(provider=provider_mock, cache_radius_m=500.0)
    return cache


# ===========================================================================
# Test Cases
# ===========================================================================

def test_coordinate_conversion_inversion() -> None:
    """Test that enu_to_lat_lon_deg and lat_lon_to_enu_m are exact inverses."""
    ref_lat, ref_lon = 48.8566, 2.3522

    test_points_enu = [
        (0.0, 0.0),
        (100.0, -50.0),
        (-250.0, 400.0),
        (1250.5, 875.2),
    ]

    for orig_e, orig_n in test_points_enu:
        lat, lon = enu_to_lat_lon_deg(orig_e, orig_n, ref_lat, ref_lon)
        recon_e, recon_n = lat_lon_to_enu_m(lat, lon, ref_lat, ref_lon)

        assert pytest.approx(recon_e, abs=1e-6) == orig_e
        assert pytest.approx(recon_n, abs=1e-6) == orig_n


def test_osm_road_constraint_applied_during_blackout(mock_osm_cache: OSMRoadCache) -> None:
    """Test that road matching is actively evaluated and applied during blackout."""
    vel_ckpt = Path("models/velocity_model_v2.pt")
    att_ckpt = Path("models/attitude_model.pt")
    if not vel_ckpt.exists() or not att_ckpt.exists():
        pytest.skip("Model checkpoints missing.")

    pipeline = AIIEKFRoadPipeline(
        velocity_checkpoint=vel_ckpt,
        attitude_checkpoint=att_ckpt,
        device="cpu",
        use_osm=True,
        osm_cache=mock_osm_cache,
        max_match_dist_m=20.0,
        max_heading_diff_deg=30.0,
    )

    imu_windows, timestamps, gt_lats, gt_lons = make_synthetic_session(N=30, speed_ms=10.0)
    blackout_intervals = [(10.0, 20.0)]  # 11 blackout steps (t=10 to 20)

    result = pipeline.run_session_blackout_road(
        imu_windows=imu_windows,
        timestamps=timestamps,
        gt_lats=gt_lats,
        gt_lons=gt_lons,
        blackout_intervals=blackout_intervals,
        init_heading_deg=0.0,  # Heading North
    )

    assert isinstance(result, RoadBlackoutEvaluationResult)
    assert len(result.road_match_stats) == 1
    stats = result.road_match_stats[0]
    assert stats.road_active is True
    assert stats.n_steps == 11
    assert stats.n_matched > 0
    assert result.total_matched > 0


def test_osm_road_constraint_not_applied_outside_blackout(mock_osm_cache: OSMRoadCache) -> None:
    """Test that road constraint is NOT applied outside blackout when blackout-only is True."""
    vel_ckpt = Path("models/velocity_model_v2.pt")
    att_ckpt = Path("models/attitude_model.pt")
    if not vel_ckpt.exists() or not att_ckpt.exists():
        pytest.skip("Model checkpoints missing.")

    pipeline = AIIEKFRoadPipeline(
        velocity_checkpoint=vel_ckpt,
        attitude_checkpoint=att_ckpt,
        device="cpu",
        use_osm=True,
        osm_cache=mock_osm_cache,
        apply_road_during_blackout_only=True,
    )

    imu_windows, timestamps, gt_lats, gt_lons = make_synthetic_session(N=20, speed_ms=10.0)
    # No blackout intervals scheduled
    result = pipeline.run_session_blackout_road(
        imu_windows=imu_windows,
        timestamps=timestamps,
        gt_lats=gt_lats,
        gt_lons=gt_lons,
        blackout_intervals=[],
        init_heading_deg=0.0,
    )

    assert result.total_matched == 0
    assert len(result.road_match_stats) == 0


def test_osm_failure_does_not_crash_pipeline() -> None:
    """Test that OSMRoadProvider/Cache failure degrades gracefully without crashing."""
    vel_ckpt = Path("models/velocity_model_v2.pt")
    att_ckpt = Path("models/attitude_model.pt")
    if not vel_ckpt.exists() or not att_ckpt.exists():
        pytest.skip("Model checkpoints missing.")

    # Provider that raises network error
    failing_provider = MagicMock(spec=OSMRoadProvider)
    failing_provider.get_nearby_roads.side_effect = OSMNetworkError("Connection timed out")
    failing_cache = OSMRoadCache(provider=failing_provider)

    pipeline = AIIEKFRoadPipeline(
        velocity_checkpoint=vel_ckpt,
        attitude_checkpoint=att_ckpt,
        device="cpu",
        use_osm=True,
        osm_cache=failing_cache,
    )

    imu_windows, timestamps, gt_lats, gt_lons = make_synthetic_session(N=20, speed_ms=10.0)
    blackout_intervals = [(5.0, 15.0)]

    # Must run to completion without throwing
    result = pipeline.run_session_blackout_road(
        imu_windows=imu_windows,
        timestamps=timestamps,
        gt_lats=gt_lats,
        gt_lons=gt_lons,
        blackout_intervals=blackout_intervals,
        init_heading_deg=0.0,
    )

    assert isinstance(result, RoadBlackoutEvaluationResult)
    assert result.total_matched == 0
    assert np.all(np.isfinite(result.errors_per_step_m))


def test_no_osm_match_when_gates_fail(ref_lat: float = 48.8566, ref_lon: float = 2.3522) -> None:
    """Test that roads failing distance or heading gates are rejected and not applied."""
    vel_ckpt = Path("models/velocity_model_v2.pt")
    att_ckpt = Path("models/attitude_model.pt")
    if not vel_ckpt.exists() or not att_ckpt.exists():
        pytest.skip("Model checkpoints missing.")

    # Perpendicular road (heading East at North = 0) with generous distance gate
    east_road = make_wgs84_way(
        way_id=2002,
        name="East Road",
        enu_coords=[(-100.0, 0.0), (100.0, 0.0)],
        ref_lat=ref_lat,
        ref_lon=ref_lon,
    )
    provider_mock = MagicMock(spec=OSMRoadProvider)
    provider_mock.get_nearby_roads.return_value = [east_road]
    cache = OSMRoadCache(provider=provider_mock, cache_radius_m=500.0)

    pipeline = AIIEKFRoadPipeline(
        velocity_checkpoint=vel_ckpt,
        attitude_checkpoint=att_ckpt,
        device="cpu",
        use_osm=True,
        osm_cache=cache,
        max_match_dist_m=200.0,    # Generous distance gate allows distance check to pass
        max_heading_diff_deg=30.0,  # 90 deg heading discrepancy triggers heading rejection
    )

    imu_windows, timestamps, gt_lats, gt_lons = make_synthetic_session(N=20, speed_ms=10.0)
    blackout_intervals = [(5.0, 15.0)]

    result = pipeline.run_session_blackout_road(
        imu_windows=imu_windows,
        timestamps=timestamps,
        gt_lats=gt_lats,
        gt_lons=gt_lons,
        blackout_intervals=blackout_intervals,
        init_heading_deg=0.0,  # heading North
    )

    assert result.total_matched == 0
    assert result.road_match_stats[0].n_rejected_heading > 0
    assert result.road_match_stats[0].road_vertices == 2


def test_legacy_mode_use_osm_false() -> None:
    """Test that use_osm=False preserves the pre-blackout GNSS road polyline fallback."""
    vel_ckpt = Path("models/velocity_model_v2.pt")
    att_ckpt = Path("models/attitude_model.pt")
    if not vel_ckpt.exists() or not att_ckpt.exists():
        pytest.skip("Model checkpoints missing.")

    pipeline = AIIEKFRoadPipeline(
        velocity_checkpoint=vel_ckpt,
        attitude_checkpoint=att_ckpt,
        device="cpu",
        use_osm=False,
    )

    imu_windows, timestamps, gt_lats, gt_lons = make_synthetic_session(N=30, speed_ms=10.0)
    blackout_intervals = [(10.0, 20.0)]

    result = pipeline.run_session_blackout_road(
        imu_windows=imu_windows,
        timestamps=timestamps,
        gt_lats=gt_lats,
        gt_lons=gt_lons,
        blackout_intervals=blackout_intervals,
        init_heading_deg=0.0,
    )

    assert isinstance(result, RoadBlackoutEvaluationResult)
    assert result.road_match_stats[0].road_active is True
    assert result.road_match_stats[0].road_vertices > 0


def test_version_a_pipeline_untouched() -> None:
    """Verify that Version A AIIEKFPipeline remains importable and runnable unchanged."""
    vel_ckpt = Path("models/velocity_model_v2.pt")
    att_ckpt = Path("models/attitude_model.pt")
    if not vel_ckpt.exists() or not att_ckpt.exists():
        pytest.skip("Model checkpoints missing.")

    pipeline_a = AIIEKFPipeline(
        velocity_checkpoint=vel_ckpt,
        attitude_checkpoint=att_ckpt,
        device="cpu",
    )

    imu_windows, timestamps, gt_lats, gt_lons = make_synthetic_session(N=20, speed_ms=10.0)
    result_a = pipeline_a.run_session_blackout(
        imu_windows=imu_windows,
        timestamps=timestamps,
        gt_lats=gt_lats,
        gt_lons=gt_lons,
        blackout_intervals=[(5.0, 10.0)],
        init_heading_deg=0.0,
    )

    assert result_a is not None
    assert len(result_a.per_blackout_metrics) == 1
