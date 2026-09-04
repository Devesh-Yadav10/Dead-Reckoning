"""
sensor_types.py — Platform-independent sensor data models and navigation state representations.

Defines standardized data interfaces for:
  - IMUSample: 6-axis accelerometer + gyroscope with monotonic timestamps.
  - GNSSSample: WGS84 coordinates with accuracy/uncertainty metrics.
  - GNSSState: Explicit lifecycle states (AVAILABLE, LOST, BLACKOUT, REACQUIRED).
  - NavigationState: Complete instantaneous navigation solution with diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple
import numpy as np


class GNSSState(str, Enum):
    """Lifecycle states for GNSS availability."""
    AVAILABLE = "AVAILABLE"
    LOST = "LOST"
    BLACKOUT = "BLACKOUT"
    REACQUIRED = "REACQUIRED"


@dataclass(frozen=True)
class IMUSample:
    """
    Standardized 6-axis smartphone IMU sample.

    Coordinate & Unit Conventions
    -----------------------------
    - Timestamp: seconds (monotonic or epoch float, e.g. 1700000000.123).
    - Acceleration (ax, ay, az): m/s^2 in phone body frame (Forward-Left-Up FLU or raw sensor frame).
    - Angular Velocity (gx, gy, gz): rad/s in phone body frame.
    """
    timestamp: float
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float

    def to_array(self) -> np.ndarray:
        """Returns [ax, ay, az, gx, gy, gz] as float64 numpy array."""
        return np.array([self.ax, self.ay, self.az, self.gx, self.gy, self.gz], dtype=np.float64)


@dataclass(frozen=True)
class GNSSSample:
    """
    Standardized GNSS position fix.

    Coordinate & Unit Conventions
    -----------------------------
    - Timestamp: seconds.
    - Latitude, Longitude: degrees in WGS84 datum.
    - Altitude: metres above WGS84 ellipsoid (default 0.0).
    - Accuracy (accuracy_m): 1-sigma horizontal position standard deviation in metres (default 2.5m).
    """
    timestamp: float
    latitude: float
    longitude: float
    altitude: float = 0.0
    accuracy_m: float = 2.5


@dataclass
class NavigationState:
    """
    Instantaneous real-time navigation state solution produced by the navigation engine.

    Attributes
    ----------
    timestamp : float
        Current solution timestamp in seconds.
    pos_enu : np.ndarray
        Local ENU position [East, North, Up] in metres.
    vel_enu : np.ndarray
        Local ENU velocity [v_East, v_North, v_Up] in m/s.
    quat : np.ndarray
        Body-to-navigation unit quaternion [qw, qx, qy, qz] (qw >= 0).
    heading_deg : float
        Vehicle heading in degrees [0, 360), clockwise from true North.
    latitude : float
        Estimated WGS84 latitude in degrees.
    longitude : float
        Estimated WGS84 longitude in degrees.
    gnss_state : GNSSState
        Current GNSS availability status.
    blackout_active : bool
        True if operating under GNSS blackout / loss conditions.
    road_constraint_active : bool
        True if an OSM road constraint was successfully applied on this step.
    matched_way_id : Optional[int]
        OSM way ID of matched road candidate (if road constraint applied).
    matched_road_name : Optional[str]
        Human-readable name of matched road candidate (if available).
    processing_latency_ms : float
        Total computation time for this navigation update in milliseconds.
    diagnostics : Dict[str, Any]
        Performance and timing diagnostic metrics.
    """
    timestamp: float
    pos_enu: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    vel_enu: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64))
    heading_deg: float = 0.0
    latitude: float = 0.0
    longitude: float = 0.0
    gnss_state: GNSSState = GNSSState.AVAILABLE
    blackout_active: bool = False
    road_constraint_active: bool = False
    matched_way_id: Optional[int] = None
    matched_road_name: Optional[str] = None
    processing_latency_ms: float = 0.0
    diagnostics: Dict[str, Any] = field(default_factory=dict)

