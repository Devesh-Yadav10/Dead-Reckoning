"""
osm_road_cache.py — In-Memory Spatial & Temporal Road Cache for NAVIGATE 2.0.

Phase 3: Provides an in-memory caching layer around OSMRoadProvider to prevent
repeated Overpass API requests during real-time navigation.

Design Constraints:
- Pure Python in-memory storage (no disk/database persistence).
- Evaluates spatial coverage using existing haversine_distance_m().
- Evaluates freshness using configurable max_age_s.
- Configurable cache_radius_m (default 400 m) and query_radius_m (default 500 m).
- Returns List[OSMRoadWay] (keeps data layer separated from RoadPolyline adapter).
- Graceful degradation: falls back to previous stale cache on network failure if
  available, or returns an empty list if no cache exists without crashing.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import logging
import math
import time
from typing import Callable, List, Optional, Sequence

from navigate.dead_reckoning import haversine_distance_m
from navigate.osm_road_provider import (
    OSMRoadProvider,
    OSMRoadWay,
    OSMProviderError,
)

logger = logging.getLogger("osm_road_cache")

# Default in-memory cache configuration parameters
DEFAULT_CACHE_RADIUS_M: float = 400.0
DEFAULT_QUERY_RADIUS_M: float = 500.0
DEFAULT_MAX_AGE_S: float = 3600.0  # 1 hour default expiration


class CacheStatus(str, Enum):
    """Status indicating how road data was resolved during a get_roads() call."""
    HIT_FRESH = "HIT_FRESH"                    # Served from valid, fresh in-memory cache
    HIT_STALE = "HIT_STALE"                    # Served from stale cache (e.g. forced or fallback)
    FETCHED_NEW = "FETCHED_NEW"                # Freshly fetched from provider and stored in cache
    FETCH_FAILED_STALE = "FETCH_FAILED_STALE"  # Provider fetch failed; served previous stale cache
    FETCH_FAILED_EMPTY = "FETCH_FAILED_EMPTY"  # Provider fetch failed and no cache existed; returned []
    EMPTY_RESULT = "EMPTY_RESULT"              # Provider returned 0 roads (cached empty area)


@dataclass
class CacheEntry:
    """
    A single in-memory cached road network region.

    Attributes
    ----------
    center_lat : float
        Latitude origin of the cached query in degrees.
    center_lon : float
        Longitude origin of the cached query in degrees.
    roads : List[OSMRoadWay]
        List of parsed OSM road ways retrieved for this region.
    timestamp : float
        Creation/refresh timestamp in epoch seconds.
    cache_radius_m : float
        Effective spatial radius of cache validity around the center.
    query_radius_m : float
        Original query radius used when requesting data from Overpass.
    """
    center_lat: float
    center_lon: float
    roads: List[OSMRoadWay]
    timestamp: float
    cache_radius_m: float
    query_radius_m: float

    def age(self, current_time: float) -> float:
        """Return age of this cache entry in seconds relative to current_time."""
        return max(0.0, current_time - self.timestamp)

    def is_fresh(self, current_time: float, max_age_s: float) -> bool:
        """Check if cache entry is within max_age_s."""
        return self.age(current_time) <= max_age_s

    def contains(self, lat: float, lon: float) -> bool:
        """Check if given (lat, lon) is within the valid cache_radius_m."""
        dist = haversine_distance_m(lat, lon, self.center_lat, self.center_lon)
        return dist <= self.cache_radius_m


class OSMRoadCache:
    """
    In-memory spatial and temporal cache wrapping OSMRoadProvider.

    Prevents repetitive Overpass API calls by serving roads from memory when
    the vehicle position is within cache_radius_m of the last query origin and
    the cached data is younger than max_age_s.
    """

    def __init__(
        self,
        provider: Optional[OSMRoadProvider] = None,
        cache_radius_m: float = DEFAULT_CACHE_RADIUS_M,
        query_radius_m: float = DEFAULT_QUERY_RADIUS_M,
        max_age_s: float = DEFAULT_MAX_AGE_S,
        time_fn: Callable[[], float] = time.time,
    ) -> None:
        """
        Initialize the in-memory road cache.

        Parameters
        ----------
        provider : Optional[OSMRoadProvider]
            Underlying road provider. If None, instantiates a default OSMRoadProvider.
        cache_radius_m : float
            Radius in metres from cache center within which cached roads remain valid.
        query_radius_m : float
            Search radius in metres requested from Overpass during refresh.
        max_age_s : float
            Maximum age of cached roads in seconds before requiring refresh.
        time_fn : Callable[[], float]
            Time provider returning current timestamp (defaults to time.time).
        """
        if cache_radius_m <= 0.0:
            raise ValueError(f"cache_radius_m must be positive, got {cache_radius_m}")
        if query_radius_m <= 0.0:
            raise ValueError(f"query_radius_m must be positive, got {query_radius_m}")
        if max_age_s <= 0.0:
            raise ValueError(f"max_age_s must be positive, got {max_age_s}")

        self.cache_radius_m = float(cache_radius_m)
        self.query_radius_m = float(query_radius_m)
        self.max_age_s = float(max_age_s)
        self.time_fn = time_fn

        self.provider = provider if provider is not None else OSMRoadProvider(
            default_radius_m=self.query_radius_m
        )

        self._entry: Optional[CacheEntry] = None
        self._last_status: Optional[CacheStatus] = None

    @property
    def entry(self) -> Optional[CacheEntry]:
        """Return the current cache entry, if any."""
        return self._entry

    @property
    def last_status(self) -> Optional[CacheStatus]:
        """Return the status code of the most recent get_roads() call."""
        return self._last_status

    def clear(self) -> None:
        """Clear the cached entry."""
        self._entry = None
        self._last_status = None

    def is_cached(self, lat: float, lon: float) -> bool:
        """
        Check if the specified coordinate is covered by a fresh cache entry.
        """
        if self._entry is None:
            return False
        now = self.time_fn()
        return self._entry.contains(lat, lon) and self._entry.is_fresh(now, self.max_age_s)

    def get_roads(
        self,
        lat: float,
        lon: float,
        force_refresh: bool = False,
    ) -> List[OSMRoadWay]:
        """
        Retrieve nearby road ways for (lat, lon), serving from cache when valid
        or querying OSMRoadProvider when missing, out-of-range, or stale.

        Parameters
        ----------
        lat : float
            Latitude in degrees [-90.0, 90.0].
        lon : float
            Longitude in degrees [-180.0, 180.0].
        force_refresh : bool
            If True, bypasses cache check and forces a provider query.

        Returns
        -------
        List[OSMRoadWay]
            List of road ways near the specified location.

        Raises
        ------
        ValueError
            If lat or lon is non-finite or out of bounds.
        """
        if not (math.isfinite(lat) and math.isfinite(lon)):
            raise ValueError(f"Coordinates must be finite: ({lat}, {lon})")
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            raise ValueError(f"Coordinates out of bounds: lat={lat}, lon={lon}")

        now = self.time_fn()

        # Check existing cache validity
        if self._entry is not None and not force_refresh:
            spatially_valid = self._entry.contains(lat, lon)
            fresh = self._entry.is_fresh(now, self.max_age_s)

            if spatially_valid and fresh:
                self._last_status = CacheStatus.HIT_FRESH
                logger.debug(
                    "Cache HIT (fresh): age=%.1fs, dist=%.1fm",
                    self._entry.age(now),
                    haversine_distance_m(lat, lon, self._entry.center_lat, self._entry.center_lon),
                )
                return self._entry.roads

            if spatially_valid and not fresh:
                logger.info(
                    "Cache STALE: age=%.1fs > max_age=%.1fs. Attempting refresh.",
                    self._entry.age(now),
                    self.max_age_s,
                )

        # Cache is missing, out-of-bounds, stale, or forced refresh
        return self._refresh(lat, lon, now)

    def _refresh(self, lat: float, lon: float, now: float) -> List[OSMRoadWay]:
        """Query the provider and update cache, handling failures gracefully."""
        try:
            roads = self.provider.get_nearby_roads(
                lat=lat,
                lon=lon,
                radius_m=self.query_radius_m,
                raise_on_error=True,
            )

            # Successfully retrieved roads
            self._entry = CacheEntry(
                center_lat=lat,
                center_lon=lon,
                roads=roads,
                timestamp=now,
                cache_radius_m=self.cache_radius_m,
                query_radius_m=self.query_radius_m,
            )

            if len(roads) > 0:
                self._last_status = CacheStatus.FETCHED_NEW
                logger.debug("Successfully refreshed road cache with %d ways.", len(roads))
            else:
                self._last_status = CacheStatus.EMPTY_RESULT
                logger.debug("Refreshed road cache: 0 ways found in region.")

            return roads

        except (OSMProviderError, Exception) as exc:
            logger.warning(
                "OSMRoadProvider query failed for (%.5f, %.5f): %s",
                lat, lon, exc,
            )

            # Fallback to existing cache if available
            if self._entry is not None:
                self._last_status = CacheStatus.FETCH_FAILED_STALE
                logger.info(
                    "Serving stale cache fallback with %d roads (age=%.1fs).",
                    len(self._entry.roads),
                    self._entry.age(now),
                )
                return self._entry.roads

            # No cache available
            self._last_status = CacheStatus.FETCH_FAILED_EMPTY
            logger.error("Provider failed and no cache entry available. Returning empty road list.")
            return []
