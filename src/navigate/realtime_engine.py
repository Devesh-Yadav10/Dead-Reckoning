"""
realtime_engine.py — Real-Time Navigation Execution Engine for NAVIGATE 2.0.

Orchestrates 10 Hz streaming IMU samples, 5-second rolling AI window inference,
ES-EKF state propagation, GNSS availability tracking, blackout transitions,
and robust OSM road-matching constraints.

Design Principles:
- Platform-independent (no Android/OS-specific dependencies).
- Pure composition: reuses existing ES-EKF, AI models, and OSM map matcher.
- Handles timing anomalies (jitter, gaps, duplicates, out-of-order) safely.
- Measures monotonic latency against real-time 100ms deadlines.
- Clean session lifecycle with zero inter-session leakage.
"""

from __future__ import annotations

import collections
import logging
import math
import time
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from navigate.ai_iekf_pipeline import (
    AIIEKFPipeline,
    enu_to_lat_lon_deg,
    lat_lon_to_enu_m,
)
from navigate.iekf_tracker import (
    DEG2RAD,
    RAD2DEG,
    ErrorStateIEKFTracker,
    GNSSBlackoutSchedule,
    quat_to_heading_deg,
)
from navigate.map_matching import MapMatcher, RoadPolyline
from navigate.osm_map_matcher import OSMMatchResult, OSMMapMatcher
from navigate.osm_road_cache import OSMRoadCache
from navigate.sensor_types import (
    GNSSSample,
    GNSSState,
    IMUSample,
    NavigationState,
)

logger = logging.getLogger("realtime_engine")


class RealTimeNavigationEngine:
    """
    Real-Time Navigation Engine fusing IMU, AI models, ES-EKF, and OSM map matching.

    Parameters
    ----------
    pipeline : Optional[AIIEKFPipeline]
        Pre-instantiated AIIEKFPipeline (reused for VelocityModel and AttitudeModel).
    velocity_checkpoint : Union[str, Path]
        Path to VelocityModel checkpoint (used if pipeline is None).
    attitude_checkpoint : Union[str, Path]
        Path to AttitudeModel checkpoint (used if pipeline is None).
    device : Optional[str]
        Device string ('cpu', 'cuda') for AI inference.
    osm_cache : Optional[OSMRoadCache]
        Spatial cache for OpenStreetMap road queries.
    osm_matcher : Optional[OSMMapMatcher]
        Multi-candidate OSM map matcher with continuity support.
    use_osm : bool
        Enable OpenStreetMap road constraint during GNSS blackouts (default True).
    window_size : int
        Number of IMU samples required for an AI model inference window (default 50 = 5.0s @ 10Hz).
    stride : int
        Sample stride between consecutive AI model inferences (default 10 = 1.0s @ 10Hz).
    target_dt_s : float
        Nominal sampling interval in seconds (default 0.1s = 10Hz).
    max_gap_s : float
        Maximum allowed gap before flagging a timing discontinuity (default 1.0s).
    road_cov_m2 : float
        Position noise variance for road pseudo-measurements (default 5.0 m^2).
    cov_speed : float
        Velocity measurement variance (default 0.25^2 (m/s)^2).
    cov_att_deg : float
        Attitude measurement standard deviation in degrees (default 5.0 deg).
    cov_gnss_pos : float
        GNSS measurement variance (default 0.1^2 m^2).
    deadline_ms : float
        Target deadline for 10 Hz real-time processing tick in ms (default 100.0 ms).
    """

    def __init__(
        self,
        pipeline: Optional[AIIEKFPipeline] = None,
        velocity_checkpoint: Union[str, Path] = "models/velocity_model_v2.pt",
        attitude_checkpoint: Union[str, Path] = "models/attitude_model.pt",
        device: Optional[str] = None,
        osm_cache: Optional[OSMRoadCache] = None,
        osm_matcher: Optional[OSMMapMatcher] = None,
        use_osm: bool = True,
        window_size: int = 50,
        stride: int = 10,
        target_dt_s: float = 0.1,
        max_gap_s: float = 1.0,
        road_cov_m2: float = 5.0,
        cov_speed: float = 0.25 ** 2,
        cov_att_deg: float = 5.0,
        cov_gnss_pos: float = 0.1 ** 2,
        deadline_ms: float = 100.0,
    ) -> None:
        # AI Pipeline
        if pipeline is not None:
            self.pipeline = pipeline
        else:
            self.pipeline = AIIEKFPipeline(
                velocity_checkpoint=velocity_checkpoint,
                attitude_checkpoint=attitude_checkpoint,
                device=device,
            )

        # OSM and Map Matching
        self.use_osm = bool(use_osm)
        self.osm_cache = osm_cache if osm_cache is not None else (OSMRoadCache() if self.use_osm else None)
        self.osm_matcher = osm_matcher if osm_matcher is not None else (OSMMapMatcher() if self.use_osm else None)

        # Configuration Parameters
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.target_dt_s = float(target_dt_s)
        self.max_gap_s = float(max_gap_s)
        self.road_cov_m2 = float(road_cov_m2)
        self.cov_speed = float(cov_speed)
        self.cov_att_deg = float(cov_att_deg)
        self.cov_att_rad = float((cov_att_deg * DEG2RAD) ** 2)
        self.cov_gnss_pos = float(cov_gnss_pos)
        self.deadline_ms = float(deadline_ms)

        # Runtime Engine State
        self.session_active: bool = False
        self.ref_lat: float = 0.0
        self.ref_lon: float = 0.0
        self.tracker: Optional[ErrorStateIEKFTracker] = None

        # Rolling Sample Buffer
        self._imu_buffer: Deque[IMUSample] = collections.deque(maxlen=max(200, window_size * 2))
        self._sample_count: int = 0
        self._last_inference_sample_idx: int = -1
        self._last_timestamp: Optional[float] = None

        # Attitude History for Relative Attitude Updates
        self._attitude_history: List[Tuple[float, np.ndarray]] = []

        # GNSS & Blackout State
        self.gnss_state: GNSSState = GNSSState.AVAILABLE
        self.manual_blackout: bool = False
        self.last_gnss_timestamp: Optional[float] = None
        self._pending_gnss_sample: Optional[GNSSSample] = None

        # Latest Output State
        self._latest_state: Optional[NavigationState] = None

        # Diagnostics & Timing Metrics
        self.diagnostics: Dict[str, Any] = {
            "samples_processed": 0,
            "inference_count": 0,
            "rejected_duplicates": 0,
            "rejected_out_of_order": 0,
            "detected_gaps": 0,
            "deadlines_met": 0,
            "deadlines_missed": 0,
            "last_velocity_ms": 0.0,
            "last_attitude_ms": 0.0,
            "last_ekf_ms": 0.0,
            "last_osm_ms": 0.0,
            "last_total_tick_ms": 0.0,
            "max_tick_ms": 0.0,
            "tick_latencies_ms": [],
        }

    # =======================================================================
    # Session Management
    # =======================================================================

    def start_session(
        self,
        ref_lat: float,
        ref_lon: float,
        init_heading_deg: float = 0.0,
        init_timestamp: Optional[float] = None,
        init_pos_enu: Optional[Sequence[float]] = None,
        init_vel_enu: Optional[Sequence[float]] = None,
    ) -> None:
        """
        Starts a new real-time navigation session with clean state.
        """
        self.reset()
        self.ref_lat = float(ref_lat)
        self.ref_lon = float(ref_lon)
        t0 = float(init_timestamp) if init_timestamp is not None else 0.0

        pos0 = list(init_pos_enu) if init_pos_enu is not None else [0.0, 0.0, 0.0]
        vel0 = list(init_vel_enu) if init_vel_enu is not None else [0.0, 0.0, 0.0]

        # Instantiate ES-EKF Tracker
        self.tracker = ErrorStateIEKFTracker(
            init_pos_enu=pos0,
            init_vel_enu=vel0,
            init_heading_deg=init_heading_deg,
            init_lat=self.ref_lat,
            init_lon=self.ref_lon,
            init_timestamp=t0,
        )

        self._last_timestamp = None
        self._attitude_history = [(t0, self.tracker.get_state()["quat"].copy())]
        self.session_active = True
        logger.info(
            f"Navigation session started: ref=({self.ref_lat:.6f}, {self.ref_lon:.6f}), "
            f"heading={init_heading_deg:.1f}deg, t0={t0:.3f}s"
        )

    def stop_session(self) -> None:
        """Stops the current navigation session."""
        self.session_active = False
        logger.info("Navigation session stopped.")

    def reset(self) -> None:
        """
        Resets all session-specific buffers, tracker, and matcher states.
        Preserves spatial OSMRoadCache so cached tiles remain valid across runs.
        """
        self.session_active = False
        self.tracker = None
        self._imu_buffer.clear()
        self._sample_count = 0
        self._last_inference_sample_idx = -1
        self._last_timestamp = None
        self._attitude_history.clear()
        self.gnss_state = GNSSState.AVAILABLE
        self.manual_blackout = False
        self.last_gnss_timestamp = None
        self._pending_gnss_sample = None
        self._latest_state = None

        if self.osm_matcher is not None:
            self.osm_matcher.reset()

        self.diagnostics = {
            "samples_processed": 0,
            "inference_count": 0,
            "rejected_duplicates": 0,
            "rejected_out_of_order": 0,
            "detected_gaps": 0,
            "deadlines_met": 0,
            "deadlines_missed": 0,
            "last_velocity_ms": 0.0,
            "last_attitude_ms": 0.0,
            "last_ekf_ms": 0.0,
            "last_osm_ms": 0.0,
            "last_total_tick_ms": 0.0,
            "max_tick_ms": 0.0,
            "tick_latencies_ms": [],
        }

    def set_blackout(self, enabled: bool) -> None:
        """Manually trigger or terminate a GNSS blackout condition."""
        prev = self.manual_blackout
        self.manual_blackout = bool(enabled)
        if self.manual_blackout:
            self.gnss_state = GNSSState.BLACKOUT
        elif prev and not self.manual_blackout:
            self.gnss_state = GNSSState.REACQUIRED

    # =======================================================================
    # Sensor Ingestion
    # =======================================================================

    def process_gnss(self, sample: GNSSSample) -> None:
        """
        Buffer an incoming GNSS sample for integration during the next navigation tick.
        """
        self._pending_gnss_sample = sample
        self.last_gnss_timestamp = sample.timestamp
        if not self.manual_blackout:
            self.gnss_state = GNSSState.AVAILABLE

    def process_imu(self, sample: IMUSample) -> Optional[NavigationState]:
        """
        Process a streaming 10 Hz IMU sample through validation, model inference,
        ES-EKF propagation, GNSS fusion, and OSM road matching.

        Parameters
        ----------
        sample : IMUSample
            Incoming 6-axis IMU sample.

        Returns
        -------
        Optional[NavigationState]
            Instantaneous navigation state if sample was validly processed.
        """
        tick_start = time.perf_counter()

        if not self.session_active or self.tracker is None:
            # Auto-initialize session if not started explicitly
            self.start_session(
                ref_lat=0.0,
                ref_lon=0.0,
                init_heading_deg=0.0,
                init_timestamp=sample.timestamp,
            )

        t_curr = float(sample.timestamp)

        # -------------------------------------------------------------------
        # 1. Timestamp Validation & Timing Checks
        # -------------------------------------------------------------------
        if self._last_timestamp is not None:
            dt = t_curr - self._last_timestamp

            # Duplicate timestamp: reject/ignore safely
            if abs(dt) < 1e-6:
                self.diagnostics["rejected_duplicates"] += 1
                logger.debug(f"Duplicate timestamp ignored: {t_curr:.6f}")
                return self._latest_state

            # Out-of-order timestamp: reject
            if dt < 0.0:
                self.diagnostics["rejected_out_of_order"] += 1
                logger.warning(
                    f"Out-of-order timestamp rejected: curr={t_curr:.6f} < prev={self._last_timestamp:.6f}"
                )
                return self._latest_state

            # Timing Gap: detect & clamp dt to prevent filter explosion
            if dt > self.max_gap_s:
                self.diagnostics["detected_gaps"] += 1
                logger.warning(f"Large timing gap detected ({dt:.3f}s > {self.max_gap_s}s); clamping dt.")
                dt = self.target_dt_s
        else:
            dt = self.target_dt_s

        self._last_timestamp = t_curr
        self._imu_buffer.append(sample)
        self._sample_count += 1
        self.diagnostics["samples_processed"] += 1

        # -------------------------------------------------------------------
        # 2. ES-EKF IMU Propagation (Predict Step)
        # -------------------------------------------------------------------
        t_ekf_start = time.perf_counter()
        accel_b = np.array([sample.ax, sample.ay, sample.az], dtype=np.float64)
        gyro_b = np.array([sample.gx, sample.gy, sample.gz], dtype=np.float64)
        self.tracker.predict(dt=dt, accel_b=accel_b, gyro_b=gyro_b)
        self.tracker._state.timestamp = t_curr
        ekf_duration_ms = (time.perf_counter() - t_ekf_start) * 1000.0
        self.diagnostics["last_ekf_ms"] = ekf_duration_ms

        # -------------------------------------------------------------------
        # 3. 5-Second Rolling AI Window Inference (1-Second Cadence)
        # -------------------------------------------------------------------
        vel_latency_ms = 0.0
        att_latency_ms = 0.0

        if len(self._imu_buffer) >= self.window_size:
            should_infer = (
                self._last_inference_sample_idx < 0
                or (self._sample_count - self._last_inference_sample_idx) >= self.stride
            )

            if should_infer:
                # Extract the latest 50 samples
                window_samples = list(self._imu_buffer)[-self.window_size:]
                window_arr = np.array([s.to_array() for s in window_samples], dtype=np.float32)  # [50, 6]

                # A. VelocityModel Inference
                t_vel_start = time.perf_counter()
                fwd_speed_ms = float(self.pipeline.predict_velocity(window_arr))
                vel_latency_ms = (time.perf_counter() - t_vel_start) * 1000.0
                self.diagnostics["last_velocity_ms"] = vel_latency_ms

                # Velocity Update & Non-Holonomic Constraints
                self.tracker.update_velocity(
                    forward_speed_ms=max(0.0, fwd_speed_ms),
                    cov_speed=self.cov_speed,
                )
                self.tracker.update_nhc()

                # B. AttitudeModel Inference
                t_att_start = time.perf_counter()
                q_rel = self.pipeline.predict_attitude(window_arr)
                att_latency_ms = (time.perf_counter() - t_att_start) * 1000.0
                self.diagnostics["last_attitude_ms"] = att_latency_ms

                # Relative Attitude Update (referenced against 5s history)
                target_start_t = t_curr - 5.0
                best_q_start = self._attitude_history[0][1]
                min_dt_hist = abs(self._attitude_history[0][0] - target_start_t)
                for hist_t, hist_q in self._attitude_history:
                    diff_t = abs(hist_t - target_start_t)
                    if diff_t < min_dt_hist:
                        min_dt_hist = diff_t
                        best_q_start = hist_q

                self.tracker.update_relative_attitude(
                    q_rel_network=q_rel,
                    q_start=best_q_start,
                    cov_att_rad=self.cov_att_rad,
                )

                self._last_inference_sample_idx = self._sample_count
                self.diagnostics["inference_count"] += 1

        self._attitude_history.append((t_curr, self.tracker.get_state()["quat"].copy()))
        # Prune old attitude history beyond 10 seconds
        while len(self._attitude_history) > 1 and (t_curr - self._attitude_history[0][0]) > 10.0:
            self._attitude_history.pop(0)

        # -------------------------------------------------------------------
        # 4. GNSS Measurement Fusion & Blackout Handling
        # -------------------------------------------------------------------
        is_bo = self.manual_blackout
        if self._pending_gnss_sample is not None:
            gnss = self._pending_gnss_sample
            self._pending_gnss_sample = None

            if not is_bo:
                # Convert GNSS WGS84 to local ENU
                e, n = lat_lon_to_enu_m(gnss.latitude, gnss.longitude, self.ref_lat, self.ref_lon)
                gnss_pos = [e, n, gnss.altitude]
                self.tracker.update_gnss_position(
                    pos_enu_meas=gnss_pos,
                    cov_pos=gnss.accuracy_m ** 2 if gnss.accuracy_m > 0 else self.cov_gnss_pos,
                    is_blackout=False,
                )
                self.gnss_state = GNSSState.AVAILABLE
            else:
                self.tracker.update_gnss_position(
                    pos_enu_meas=[0.0, 0.0, 0.0],
                    cov_pos=self.cov_gnss_pos,
                    is_blackout=True,
                )
                self.gnss_state = GNSSState.BLACKOUT

        # -------------------------------------------------------------------
        # 5. OpenStreetMap Soft Road Constraint (During Blackout Only)
        # -------------------------------------------------------------------
        road_applied = False
        matched_way_id: Optional[int] = None
        matched_road_name: Optional[str] = None
        t_osm_start = time.perf_counter()

        if is_bo and self.use_osm and self.osm_cache is not None and self.osm_matcher is not None:
            current_state = self.tracker.get_state()
            pos_en = current_state["pos_enu"][:2]
            heading_deg = quat_to_heading_deg(current_state["quat"])

            est_lat, est_lon = enu_to_lat_lon_deg(
                east_m=pos_en[0],
                north_m=pos_en[1],
                ref_lat_deg=self.ref_lat,
                ref_lon_deg=self.ref_lon,
            )

            try:
                # Spatial cache lookup (handles failures gracefully)
                osm_ways = self.osm_cache.get_roads(lat=est_lat, lon=est_lon)
                if osm_ways:
                    osm_match: OSMMatchResult = self.osm_matcher.match(
                        estimated_pos=pos_en,
                        estimated_heading_deg=heading_deg,
                        osm_roads=osm_ways,
                        ref_lat=self.ref_lat,
                        ref_lon=self.ref_lon,
                    )

                    if osm_match.matched:
                        road_applied = True
                        if osm_match.selected_way is not None:
                            matched_way_id = osm_match.selected_way.way_id
                            matched_road_name = osm_match.selected_way.name

                        road_pos_enu = np.array([
                            osm_match.corrected_pos[0],
                            osm_match.corrected_pos[1],
                            current_state["pos_enu"][2],
                        ], dtype=np.float64)

                        self.tracker.update_gnss_position(
                            pos_enu_meas=road_pos_enu,
                            cov_pos=self.road_cov_m2,
                            is_blackout=False,
                        )
            except Exception as e:
                logger.warning(f"OSM matching failed safely: {e}")

        osm_duration_ms = (time.perf_counter() - t_osm_start) * 1000.0
        self.diagnostics["last_osm_ms"] = osm_duration_ms

        # -------------------------------------------------------------------
        # 6. Final State Construction & Latency Accounting
        # -------------------------------------------------------------------
        total_tick_ms = (time.perf_counter() - tick_start) * 1000.0
        self.diagnostics["last_total_tick_ms"] = total_tick_ms
        self.diagnostics["max_tick_ms"] = max(self.diagnostics["max_tick_ms"], total_tick_ms)
        self.diagnostics["tick_latencies_ms"].append(total_tick_ms)

        if total_tick_ms <= self.deadline_ms:
            self.diagnostics["deadlines_met"] += 1
        else:
            self.diagnostics["deadlines_missed"] += 1

        final_st = self.tracker.get_state()
        cur_pos = final_st["pos_enu"]
        cur_lat, cur_lon = enu_to_lat_lon_deg(cur_pos[0], cur_pos[1], self.ref_lat, self.ref_lon)
        cur_hdg = quat_to_heading_deg(final_st["quat"])

        self._latest_state = NavigationState(
            timestamp=t_curr,
            pos_enu=cur_pos.copy(),
            vel_enu=final_st["vel_enu"].copy(),
            quat=final_st["quat"].copy(),
            heading_deg=cur_hdg,
            latitude=cur_lat,
            longitude=cur_lon,
            gnss_state=self.gnss_state,
            blackout_active=is_bo,
            road_constraint_active=road_applied,
            matched_way_id=matched_way_id,
            matched_road_name=matched_road_name,
            processing_latency_ms=total_tick_ms,
            diagnostics={
                "velocity_latency_ms": vel_latency_ms,
                "attitude_latency_ms": att_latency_ms,
                "ekf_latency_ms": ekf_duration_ms,
                "osm_latency_ms": osm_duration_ms,
                "on_time": total_tick_ms <= self.deadline_ms,
            },
        )
        return self._latest_state

    def get_latest_state(self) -> Optional[NavigationState]:
        """Returns the most recent NavigationState solution."""
        return self._latest_state

    def get_diagnostics(self) -> Dict[str, Any]:
        """Returns engine runtime diagnostics and timing telemetry."""
        return dict(self.diagnostics)
