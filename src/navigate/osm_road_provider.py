"""
osm_road_provider.py — OpenStreetMap / Overpass API Road Network Provider for NAVIGATE 2.0.

Phase 1: Self-contained module to query nearby drivable roads from OpenStreetMap
via the public Overpass API and parse the response into clean road records with
ordered latitude/longitude coordinates.

Design Constraints:
- Pure Python standard library for networking (urllib.request) and JSON parsing.
- No external mapping or geometry library dependencies (no shapely, no geopy).
- Configurable radius and Overpass endpoint.
- Graceful handling of network timeouts, HTTP errors, malformed JSON, and empty responses.
- Drivable highway filtering (excludes footways, cycleways, pedestrian paths, steps, tracks).
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger("osm_road_provider")

# Default public Overpass API endpoint (no API key required)
DEFAULT_OVERPASS_ENDPOINT = "https://overpass-api.de/api/interpreter"

# Known public Overpass API mirror endpoints
PUBLIC_OVERPASS_ENDPOINTS: Tuple[str, ...] = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
)

# Default query radius around position in metres
DEFAULT_QUERY_RADIUS_M = 500.0

# Default HTTP timeout in seconds
DEFAULT_TIMEOUT_S = 25.0

# Drivable highway types included by default
DEFAULT_DRIVABLE_HIGHWAY_TYPES: Tuple[str, ...] = (
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "residential",
    "unclassified",
    "service",
)

# Explicitly excluded non-drivable road/path types
EXCLUDED_HIGHWAY_TYPES: Set[str] = {
    "footway",
    "cycleway",
    "path",
    "pedestrian",
    "steps",
    "track",
    "bridleway",
    "corridor",
    "proposed",
    "construction",
    "raceway",
}


# ===========================================================================
# Exceptions
# ===========================================================================

class OSMProviderError(Exception):
    """Base exception for OpenStreetMap road provider errors."""
    pass


class OSMNetworkError(OSMProviderError):
    """Raised when an HTTP, network connection, or timeout error occurs."""
    pass


class OSMResponseError(OSMProviderError):
    """Raised when the Overpass API returns invalid JSON or an error payload."""
    pass


# ===========================================================================
# Data Structures
# ===========================================================================

@dataclass
class OSMRoadWay:
    """
    A single drivable road way retrieved from OpenStreetMap.

    Attributes
    ----------
    way_id : int
        Unique OSM way identifier.
    highway_type : str
        OSM highway tag value (e.g., 'primary', 'residential', 'secondary').
    coordinates : List[Tuple[float, float]]
        Ordered sequence of (latitude, longitude) waypoints defining the road geometry.
    name : Optional[str]
        Human-readable road name from OSM tags if available.
    oneway : Optional[str]
        Oneway traffic status from OSM tags if available ('yes', 'no', '-1', etc.).
    tags : Dict[str, str]
        Complete dictionary of raw OSM tags associated with this way.
    """
    way_id: int
    highway_type: str
    coordinates: List[Tuple[float, float]]
    name: Optional[str] = None
    oneway: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)

    @property
    def num_points(self) -> int:
        """Number of coordinate points in this road way."""
        return len(self.coordinates)


# ===========================================================================
# Road Provider Class
# ===========================================================================

class OSMRoadProvider:
    """
    Queries OpenStreetMap Overpass API for nearby drivable road networks.

    Parameters
    ----------
    endpoint : str
        URL of the Overpass API interpreter endpoint.
    default_radius_m : float
        Default search radius in metres around the target coordinate (default: 500m).
    timeout_s : float
        HTTP request timeout in seconds (default: 25s).
    highway_types : Optional[Sequence[str]]
        List or tuple of drivable OSM highway types to include. Defaults to
        ['motorway', 'trunk', 'primary', 'secondary', 'tertiary',
         'residential', 'unclassified', 'service'].
    user_agent : str
        Custom HTTP User-Agent header for Overpass API requests.
    """

    def __init__(
        self,
        endpoint: str = DEFAULT_OVERPASS_ENDPOINT,
        default_radius_m: float = DEFAULT_QUERY_RADIUS_M,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        highway_types: Optional[Sequence[str]] = None,
        user_agent: str = "NAVIGATE_2.0/1.0 (Autonomous Navigation Research)",
    ) -> None:
        if default_radius_m <= 0.0:
            raise ValueError("default_radius_m must be positive.")
        if timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive.")

        self.endpoint = endpoint
        self.default_radius_m = float(default_radius_m)
        self.timeout_s = float(timeout_s)
        self.user_agent = user_agent

        # Filter out any explicitly excluded non-drivable types
        if highway_types is not None:
            self.highway_types = tuple(
                h for h in highway_types if h not in EXCLUDED_HIGHWAY_TYPES
            )
        else:
            self.highway_types = DEFAULT_DRIVABLE_HIGHWAY_TYPES

    # -----------------------------------------------------------------------
    # Query Builder
    # -----------------------------------------------------------------------

    def build_overpass_query(
        self,
        latitude: float,
        longitude: float,
        radius_m: Optional[float] = None,
    ) -> str:
        """
        Constructs an Overpass QL query to fetch nearby drivable road ways with geometry.

        Parameters
        ----------
        latitude : float
            Target latitude in decimal degrees.
        longitude : float
            Target longitude in decimal degrees.
        radius_m : Optional[float]
            Search radius in metres (defaults to self.default_radius_m).

        Returns
        -------
        str
            Formatted Overpass QL query string.
        """
        r = float(radius_m) if radius_m is not None else self.default_radius_m
        if r <= 0.0:
            raise ValueError("Query radius must be positive.")

        # Build regex for matching highway types
        types_regex = "|".join(self.highway_types)

        # Overpass QL query requesting JSON format and full way geometry (qt = quad-tile order for performance)
        query = (
            f"[out:json][timeout:{int(self.timeout_s)}];\n"
            f"(\n"
            f'  way(around:{r:.1f},{latitude:.7f},{longitude:.7f})["highway"~"^({types_regex})$"];\n'
            f");\n"
            f"out geom qt;"
        )
        return query

    # -----------------------------------------------------------------------
    # Response Parser
    # -----------------------------------------------------------------------

    def parse_overpass_response(self, response_json: Dict[str, Any]) -> List[OSMRoadWay]:
        """
        Parses an Overpass API JSON dictionary into a list of OSMRoadWay objects.

        Parameters
        ----------
        response_json : Dict[str, Any]
            Decoded JSON response from the Overpass API.

        Returns
        -------
        List[OSMRoadWay]
            Parsed list of drivable road ways with valid ordered coordinates.
        """
        if not isinstance(response_json, dict):
            raise OSMResponseError(
                f"Expected JSON object in response, got {type(response_json).__name__}"
            )

        elements = response_json.get("elements", [])
        if not isinstance(elements, list):
            raise OSMResponseError(
                f"Expected list in 'elements' field, got {type(elements).__name__}"
            )

        road_ways: List[OSMRoadWay] = []

        for elem in elements:
            if not isinstance(elem, dict):
                continue

            # Only process 'way' elements
            if elem.get("type") != "way":
                continue

            way_id = elem.get("id")
            if way_id is None or not isinstance(way_id, int):
                logger.debug(f"Skipping element without valid way ID: {elem}")
                continue

            tags = elem.get("tags", {})
            if not isinstance(tags, dict):
                tags = {}

            highway = tags.get("highway")
            if not highway or highway in EXCLUDED_HIGHWAY_TYPES:
                continue

            # Extract ordered geometry coordinates
            geom = elem.get("geometry", [])
            if not isinstance(geom, list) or len(geom) < 2:
                logger.debug(
                    f"Way {way_id} has insufficient geometry points ({len(geom) if isinstance(geom, list) else 0}); skipping."
                )
                continue

            coords: List[Tuple[float, float]] = []
            for pt in geom:
                if isinstance(pt, dict) and "lat" in pt and "lon" in pt:
                    try:
                        lat_val = float(pt["lat"])
                        lon_val = float(pt["lon"])
                        coords.append((lat_val, lon_val))
                    except (ValueError, TypeError):
                        continue

            if len(coords) < 2:
                logger.debug(f"Way {way_id} parsed fewer than 2 valid coordinates; skipping.")
                continue

            road_way = OSMRoadWay(
                way_id=way_id,
                highway_type=highway,
                coordinates=coords,
                name=tags.get("name"),
                oneway=tags.get("oneway"),
                tags=tags,
            )
            road_ways.append(road_way)

        return road_ways

    # -----------------------------------------------------------------------
    # Query Execution
    # -----------------------------------------------------------------------

    def get_nearby_roads(
        self,
        lat: float = 0.0,
        lon: float = 0.0,
        radius_m: Optional[float] = None,
        raise_on_error: bool = False,
        **kwargs: Any,
    ) -> List[OSMRoadWay]:
        """
        Queries the Overpass API for drivable road ways near (lat, lon).

        Parameters
        ----------
        lat : float
            Target latitude in degrees.
        lon : float
            Target longitude in degrees.
        radius_m : Optional[float]
            Query radius in metres. If None, uses default_radius_m.
        raise_on_error : bool
            If True, re-raises OSMProviderError subclasses on network/response failure.
            If False (default), logs errors and returns an empty list.

        Returns
        -------
        List[OSMRoadWay]
            Parsed list of drivable road ways.
        """
        if "latitude" in kwargs:
            lat = float(kwargs.pop("latitude"))
        if "longitude" in kwargs:
            lon = float(kwargs.pop("longitude"))
        query_str = self.build_overpass_query(lat, lon, radius_m)
        encoded_data = urllib.parse.urlencode({"data": query_str}).encode("utf-8")

        req = urllib.request.Request(
            self.endpoint,
            data=encoded_data,
            headers={
                "User-Agent": self.user_agent,
                "Content-Type": "application/x-www-form-urlencoded",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as response:
                status_code = response.getcode()
                if status_code != 200:
                    err_msg = f"Overpass API returned HTTP status {status_code}"
                    logger.warning(err_msg)
                    if raise_on_error:
                        raise OSMNetworkError(err_msg)
                    return []

                raw_body = response.read().decode("utf-8")

            try:
                response_json = json.loads(raw_body)
            except json.JSONDecodeError as e:
                err_msg = f"Failed to parse Overpass response as JSON: {e}"
                logger.warning(err_msg)
                if raise_on_error:
                    raise OSMResponseError(err_msg) from e
                return []

            road_ways = self.parse_overpass_response(response_json)
            logger.info(
                f"Retrieved {len(road_ways)} drivable OSM road ways near ({lat:.5f}, {lon:.5f}) within {radius_m or self.default_radius_m}m."
            )
            return road_ways

        except OSMProviderError:
            raise
        except urllib.error.HTTPError as e:
            err_msg = f"Overpass HTTP error: {e.code} {e.reason}"
            logger.warning(err_msg)
            if raise_on_error:
                raise OSMNetworkError(err_msg) from e
            return []
        except urllib.error.URLError as e:
            err_msg = f"Overpass network connection error: {e.reason}"
            logger.warning(err_msg)
            if raise_on_error:
                raise OSMNetworkError(err_msg) from e
            return []
        except TimeoutError as e:
            err_msg = f"Overpass request timed out after {self.timeout_s}s"
            logger.warning(err_msg)
            if raise_on_error:
                raise OSMNetworkError(err_msg) from e
            return []
        except Exception as e:
            err_msg = f"Unexpected error querying Overpass API: {e}"
            logger.warning(err_msg)
            if raise_on_error:
                raise OSMProviderError(err_msg) from e
            return []
