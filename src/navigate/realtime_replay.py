"""
realtime_replay.py — Real-time replay source for sensor streams and recorded sessions.

Replays recorded IMU and GNSS sequences through RealTimeNavigationEngine to test
real-time execution, blackout handling, and state progression deterministically.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from navigate.realtime_engine import RealTimeNavigationEngine
from navigate.sensor_types import GNSSSample, IMUSample, NavigationState

logger = logging.getLogger("realtime_replay")


class RealTimeReplay:
    """
    Feeds recorded IMU and GNSS streams through RealTimeNavigationEngine.

    Parameters
    ----------
    engine : RealTimeNavigationEngine
        Target real-time navigation engine instance.
    blackout_intervals : Optional[Sequence[Tuple[float, float]]]
        List of (start_s, end_s) timestamps during which GNSS is suppressed.
    """

    def __init__(
        self,
        engine: RealTimeNavigationEngine,
        blackout_intervals: Optional[Sequence[Tuple[float, float]]] = None,
    ) -> None:
        self.engine = engine
        self.blackout_intervals = list(blackout_intervals) if blackout_intervals is not None else []

    def is_blackout(self, t: float) -> bool:
        """Returns True if timestamp t falls inside any blackout interval."""
        for t_start, t_end in self.blackout_intervals:
            if t_start <= t <= t_end:
                return True
        return False

    def run_replay(
        self,
        imu_samples: Sequence[IMUSample],
        gnss_samples: Optional[Sequence[GNSSSample]] = None,
        ref_lat: float = 0.0,
        ref_lon: float = 0.0,
        init_heading_deg: float = 0.0,
    ) -> List[NavigationState]:
        """
        Executes a deterministic replay through the engine.

        Parameters
        ----------
        imu_samples : Sequence[IMUSample]
            Ordered stream of 10 Hz IMU samples.
        gnss_samples : Optional[Sequence[GNSSSample]]
            Optional stream of GNSS position fixes.
        ref_lat : float
            Origin latitude (degrees).
        ref_lon : float
            Origin longitude (degrees).
        init_heading_deg : float
            Initial heading (degrees).

        Returns
        -------
        List[NavigationState]
            Complete recorded sequence of output navigation states.
        """
        if not imu_samples:
            return []

        t0 = imu_samples[0].timestamp
        self.engine.start_session(
            ref_lat=ref_lat,
            ref_lon=ref_lon,
            init_heading_deg=init_heading_deg,
            init_timestamp=t0,
        )

        output_states: List[NavigationState] = []
        gnss_idx = 0
        num_gnss = len(gnss_samples) if gnss_samples is not None else 0

        for imu in imu_samples:
            t = imu.timestamp

            # Check and update blackout condition
            in_bo = self.is_blackout(t)
            self.engine.set_blackout(in_bo)

            # Ingest available GNSS fixes up to current time
            if gnss_samples is not None:
                while gnss_idx < num_gnss and gnss_samples[gnss_idx].timestamp <= t:
                    gnss = gnss_samples[gnss_idx]
                    self.engine.process_gnss(gnss)
                    gnss_idx += 1

            # Process IMU tick
            state = self.engine.process_imu(imu)
            if state is not None:
                output_states.append(state)

        self.engine.stop_session()
        return output_states

