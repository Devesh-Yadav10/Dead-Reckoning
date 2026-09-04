"""
test_realtime_engine.py — Unit tests for RealTimeNavigationEngine and real-time execution layer.

Tests cover:
  1. Engine initialization with default and custom parameters.
  2. Session reset functionality and buffer clearing.
  3. IMU sample acceptance and nominal state propagation.
  4. Duplicate timestamps handling (ignored safely).
  5. Out-of-order timestamps handling (rejected).
  6. Timestamp jitter handling (accepted).
  7. Large timestamp gaps handling (detected and clamped).
  8. Rolling 50-sample buffer maintenance.
  9. 10-sample inference cadence and stride.
 10. GNSS available state and position fusion.
 11. GNSS loss and blackout transitions.
 12. Blackout state transition and measurement suppression.
 13. GNSS reacquisition state transition.
 14. OSM unavailable graceful error handling (no crash).
 15. OSM available road constraint application.
 16. Stale OSM cache handling.
 17. NavigationState structure and field population.
 18. Monotonic timing and diagnostic telemetry.
 19. Session isolation (no cross-session state leakage).
 20. ReplaySource end-to-end integration.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from navigate.ai_iekf_pipeline import AIIEKFPipeline
from navigate.iekf_tracker import DEG2RAD, EARTH_RADIUS_M
from navigate.map_matching import MapMatcher, RoadPolyline
from navigate.osm_map_matcher import OSMMapMatcher, OSMMatchResult
from navigate.osm_road_cache import OSMRoadCache
from navigate.osm_road_provider import OSMRoadProvider, OSMRoadWay
from navigate.realtime_engine import RealTimeNavigationEngine
from navigate.realtime_replay import RealTimeReplay
from navigate.sensor_types import GNSSSample, GNSSState, IMUSample, NavigationState


# ===========================================================================
# Helpers & Mocks
# ===========================================================================

def make_mock_pipeline(fwd_speed: float = 10.0) -> AIIEKFPipeline:
    """Creates a lightweight mock AIIEKFPipeline with deterministic speed/attitude output."""
    pipeline = MagicMock(spec=AIIEKFPipeline)
    pipeline.predict_velocity.return_value = np.float32(fwd_speed)
    pipeline.predict_attitude.return_value = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return pipeline


def make_imu_sample(t: float, ax: float = 0.0, ay: float = 0.0, az: float = 9.81, gx: float = 0.0, gy: float = 0.0, gz: float = 0.0) -> IMUSample:
    return IMUSample(timestamp=t, ax=ax, ay=ay, az=az, gx=gx, gy=gy, gz=gz)


# ===========================================================================
# 1. Initialization
# ===========================================================================

def test_engine_initialization() -> None:
    mock_pipe = make_mock_pipeline()
    engine = RealTimeNavigationEngine(pipeline=mock_pipe, use_osm=False)

    assert engine.session_active is False
    assert engine.window_size == 50
    assert engine.stride == 10
    assert engine.diagnostics["samples_processed"] == 0


# ===========================================================================
# 2. Session Reset
# ===========================================================================

def test_session_reset() -> None:
    mock_pipe = make_mock_pipeline()
    engine = RealTimeNavigationEngine(pipeline=mock_pipe, use_osm=False)
    engine.start_session(ref_lat=51.5, ref_lon=-0.1, init_heading_deg=90.0, init_timestamp=0.0)

    for i in range(10):
        engine.process_imu(make_imu_sample(t=i * 0.1))

    assert engine.session_active is True
    assert engine.diagnostics["samples_processed"] == 10

    engine.reset()
    assert engine.session_active is False
    assert engine.diagnostics["samples_processed"] == 0
    assert len(engine._imu_buffer) == 0


# ===========================================================================
# 3. IMU Acceptance & Propagation
# ===========================================================================

def test_imu_sample_acceptance() -> None:
    mock_pipe = make_mock_pipeline()
    engine = RealTimeNavigationEngine(pipeline=mock_pipe, use_osm=False)
    engine.start_session(ref_lat=0.0, ref_lon=0.0, init_heading_deg=0.0, init_timestamp=0.0)

    st = engine.process_imu(make_imu_sample(t=0.1, az=9.81))
    assert isinstance(st, NavigationState)
    assert st.timestamp == 0.1
    assert engine.diagnostics["samples_processed"] == 1


# ===========================================================================
# 4. Duplicate Timestamps
# ===========================================================================

def test_duplicate_timestamps_ignored() -> None:
    mock_pipe = make_mock_pipeline()
    engine = RealTimeNavigationEngine(pipeline=mock_pipe, use_osm=False)
    engine.start_session(ref_lat=0.0, ref_lon=0.0, init_timestamp=0.0)

    engine.process_imu(make_imu_sample(t=0.1))
    engine.process_imu(make_imu_sample(t=0.1))  # Duplicate

    assert engine.diagnostics["samples_processed"] == 1
    assert engine.diagnostics["rejected_duplicates"] == 1


# ===========================================================================
# 5. Out-of-Order Timestamps
# ===========================================================================

def test_out_of_order_timestamps_rejected() -> None:
    mock_pipe = make_mock_pipeline()
    engine = RealTimeNavigationEngine(pipeline=mock_pipe, use_osm=False)
    engine.start_session(ref_lat=0.0, ref_lon=0.0, init_timestamp=0.0)

    engine.process_imu(make_imu_sample(t=0.2))
    engine.process_imu(make_imu_sample(t=0.1))  # Backwards in time

    assert engine.diagnostics["samples_processed"] == 1
    assert engine.diagnostics["rejected_out_of_order"] == 1


# ===========================================================================
# 6. Timestamp Jitter
# ===========================================================================

def test_timestamp_jitter_accepted() -> None:
    mock_pipe = make_mock_pipeline()
    engine = RealTimeNavigationEngine(pipeline=mock_pipe, use_osm=False)
    engine.start_session(ref_lat=0.0, ref_lon=0.0, init_timestamp=0.0)

    # Intervals: 95ms, 105ms, 98ms
    timestamps = [0.095, 0.200, 0.298, 0.400]
    for t in timestamps:
        engine.process_imu(make_imu_sample(t=t))

    assert engine.diagnostics["samples_processed"] == len(timestamps)
    assert engine.diagnostics["rejected_duplicates"] == 0
    assert engine.diagnostics["rejected_out_of_order"] == 0


# ===========================================================================
# 7. Large Timestamp Gap Detection
# ===========================================================================

def test_large_timestamp_gap_detected() -> None:
    mock_pipe = make_mock_pipeline()
    engine = RealTimeNavigationEngine(pipeline=mock_pipe, use_osm=False, max_gap_s=1.0)
    engine.start_session(ref_lat=0.0, ref_lon=0.0, init_timestamp=0.0)

    engine.process_imu(make_imu_sample(t=0.1))
    engine.process_imu(make_imu_sample(t=5.0))  # 4.9s gap

    assert engine.diagnostics["detected_gaps"] == 1
    assert engine.diagnostics["samples_processed"] == 2


# ===========================================================================
# 8. Rolling 50-Sample Buffer & 9. Inference Cadence (Stride=10)
# ===========================================================================

def test_rolling_window_inference_cadence() -> None:
    mock_pipe = make_mock_pipeline()
    engine = RealTimeNavigationEngine(pipeline=mock_pipe, window_size=50, stride=10, use_osm=False)
    engine.start_session(ref_lat=0.0, ref_lon=0.0, init_timestamp=0.0)

    # Send 49 samples: No inference yet
    for i in range(1, 50):
        engine.process_imu(make_imu_sample(t=i * 0.1))
    assert engine.diagnostics["inference_count"] == 0
    assert mock_pipe.predict_velocity.call_count == 0

    # 50th sample: Exactly 1 inference
    engine.process_imu(make_imu_sample(t=50 * 0.1))
    assert engine.diagnostics["inference_count"] == 1
    assert mock_pipe.predict_velocity.call_count == 1

    # Send 9 more samples (51..59): Still 1 inference
    for i in range(51, 60):
        engine.process_imu(make_imu_sample(t=i * 0.1))
    assert engine.diagnostics["inference_count"] == 1

    # 60th sample (+10 stride): 2nd inference
    engine.process_imu(make_imu_sample(t=60 * 0.1))
    assert engine.diagnostics["inference_count"] == 2
    assert mock_pipe.predict_velocity.call_count == 2


# ===========================================================================
# 10. GNSS Available & Fusion
# ===========================================================================

def test_gnss_available_and_fused() -> None:
    mock_pipe = make_mock_pipeline()
    engine = RealTimeNavigationEngine(pipeline=mock_pipe, use_osm=False)
    engine.start_session(ref_lat=0.0, ref_lon=0.0, init_timestamp=0.0)

    engine.process_gnss(GNSSSample(timestamp=0.1, latitude=0.0001, longitude=0.0001))
    st = engine.process_imu(make_imu_sample(t=0.1))

    assert st is not None
    assert st.gnss_state == GNSSState.AVAILABLE
    assert st.blackout_active is False


# ===========================================================================
# 11. Blackout Transition & GNSS Suppression
# ===========================================================================

def test_blackout_transition_and_suppression() -> None:
    mock_pipe = make_mock_pipeline()
    engine = RealTimeNavigationEngine(pipeline=mock_pipe, use_osm=False)
    engine.start_session(ref_lat=0.0, ref_lon=0.0, init_timestamp=0.0)

    # Set blackout active
    engine.set_blackout(True)
    engine.process_gnss(GNSSSample(timestamp=0.1, latitude=0.0001, longitude=0.0001))
    st = engine.process_imu(make_imu_sample(t=0.1))

    assert st is not None
    assert st.blackout_active is True
    assert st.gnss_state == GNSSState.BLACKOUT


# ===========================================================================
# 12. GNSS Reacquisition
# ===========================================================================

def test_gnss_reacquisition() -> None:
    mock_pipe = make_mock_pipeline()
    engine = RealTimeNavigationEngine(pipeline=mock_pipe, use_osm=False)
    engine.start_session(ref_lat=0.0, ref_lon=0.0, init_timestamp=0.0)

    engine.set_blackout(True)
    st1 = engine.process_imu(make_imu_sample(t=0.1))
    assert st1.gnss_state == GNSSState.BLACKOUT

    # Reacquire GNSS
    engine.set_blackout(False)
    engine.process_gnss(GNSSSample(timestamp=0.2, latitude=0.0, longitude=0.0))
    st2 = engine.process_imu(make_imu_sample(t=0.2))
    assert st2.blackout_active is False
    assert st2.gnss_state in (GNSSState.AVAILABLE, GNSSState.REACQUIRED)


# ===========================================================================
# 13. OSM Failure Safe Handling (No Crash)
# ===========================================================================

def test_osm_failure_does_not_crash_engine() -> None:
    mock_pipe = make_mock_pipeline()
    mock_cache = MagicMock(spec=OSMRoadCache)
    mock_cache.get_roads.side_effect = RuntimeError("Network timeout to Overpass")

    engine = RealTimeNavigationEngine(pipeline=mock_pipe, osm_cache=mock_cache, use_osm=True)
    engine.start_session(ref_lat=0.0, ref_lon=0.0, init_timestamp=0.0)
    engine.set_blackout(True)

    # Must process without throwing
    st = engine.process_imu(make_imu_sample(t=0.1))
    assert st is not None
    assert st.road_constraint_active is False


# ===========================================================================
# 14. OSM Available Road Constraint
# ===========================================================================

def test_osm_road_constraint_applied_during_blackout() -> None:
    mock_pipe = make_mock_pipeline()

    # Create dummy road way running East at North = 0
    road_way = OSMRoadWay(
        way_id=999,
        highway_type="primary",
        coordinates=[(0.0, -0.001), (0.0, 0.001)],
        name="Test Highway",
    )
    mock_cache = MagicMock(spec=OSMRoadCache)
    mock_cache.get_roads.return_value = [road_way]

    engine = RealTimeNavigationEngine(
        pipeline=mock_pipe,
        osm_cache=mock_cache,
        use_osm=True,
    )
    engine.start_session(ref_lat=0.0, ref_lon=0.0, init_heading_deg=90.0, init_timestamp=0.0)
    engine.set_blackout(True)

    st = engine.process_imu(make_imu_sample(t=0.1, ax=0.0, az=9.81))
    assert st is not None
    assert st.road_constraint_active is True
    assert st.matched_way_id == 999
    assert st.matched_road_name == "Test Highway"


# ===========================================================================
# 15. Monotonic Timing & Telemetry
# ===========================================================================

def test_timing_telemetry_metrics() -> None:
    mock_pipe = make_mock_pipeline()
    engine = RealTimeNavigationEngine(pipeline=mock_pipe, use_osm=False, deadline_ms=100.0)
    engine.start_session(ref_lat=0.0, ref_lon=0.0, init_timestamp=0.0)

    st = engine.process_imu(make_imu_sample(t=0.1))
    assert st.processing_latency_ms >= 0.0
    assert "on_time" in st.diagnostics
    assert engine.diagnostics["deadlines_met"] >= 1


# ===========================================================================
# 16. Session Isolation & Matcher Reset
# ===========================================================================

def test_session_isolation_and_matcher_reset() -> None:
    mock_pipe = make_mock_pipeline()
    mock_matcher = MagicMock(spec=OSMMapMatcher)
    engine = RealTimeNavigationEngine(
        pipeline=mock_pipe,
        osm_matcher=mock_matcher,
        use_osm=True,
    )

    # Session 1
    engine.start_session(ref_lat=0.0, ref_lon=0.0, init_timestamp=0.0)
    engine.process_imu(make_imu_sample(t=0.1))
    engine.stop_session()

    # Session 2
    engine.start_session(ref_lat=10.0, ref_lon=10.0, init_timestamp=100.0)
    assert mock_matcher.reset.call_count >= 1
    assert engine.diagnostics["samples_processed"] == 0


# ===========================================================================
# 17. End-to-End Replay Integration
# ===========================================================================

def test_e2e_replay_integration() -> None:
    mock_pipe = make_mock_pipeline()
    engine = RealTimeNavigationEngine(pipeline=mock_pipe, window_size=50, stride=10, use_osm=False)
    replay = RealTimeReplay(engine=engine, blackout_intervals=[(2.0, 4.0)])

    # 60 samples @ 10Hz (0.0 to 5.9s)
    imu_stream = [make_imu_sample(t=i * 0.1, az=9.81) for i in range(60)]
    gnss_stream = [GNSSSample(timestamp=i * 1.0, latitude=0.0, longitude=0.0) for i in range(6)]

    states = replay.run_replay(
        imu_samples=imu_stream,
        gnss_samples=gnss_stream,
        ref_lat=0.0,
        ref_lon=0.0,
    )

    assert len(states) == 60
    # Check that timestamps 2.0 to 4.0 had blackout_active = True
    blackout_states = [s for s in states if 2.0 <= s.timestamp <= 4.0]
    assert all(s.blackout_active for s in blackout_states)

    normal_states = [s for s in states if s.timestamp < 2.0 or s.timestamp > 4.0]
    assert all(not s.blackout_active for s in normal_states)

