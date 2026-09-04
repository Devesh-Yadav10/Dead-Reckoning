"""
test_osm_connection.py — Standalone live OpenStreetMap Overpass connectivity test.

Performs a live query to the OpenStreetMap Overpass API and reports:
- Number of roads returned
- Way IDs
- Highway types
- Road names (if available)
- Number of geometry coordinates per road

Usage:
  python scripts/test_osm_connection.py
  python scripts/test_osm_connection.py --lat 51.4778 --lon -0.0014 --radius 400
"""

import argparse
import sys
from pathlib import Path

# Add project root and src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from navigate.osm_road_provider import DEFAULT_OVERPASS_ENDPOINT, OSMRoadProvider


def test_live_connection(
    lat: float,
    lon: float,
    radius_m: float,
    endpoint: str = DEFAULT_OVERPASS_ENDPOINT,
) -> None:
    """Queries live Overpass API and displays retrieved drivable road network summary."""
    print("=" * 70)
    print("NAVIGATE 2.0 — Live OpenStreetMap / Overpass Connectivity Test")
    print("=" * 70)
    print(f"Target Coordinate : ({lat:.6f}, {lon:.6f})")
    print(f"Search Radius     : {radius_m:.1f} m")
    print(f"Overpass Endpoint : {endpoint}")
    print("Sending live Overpass query...")

    provider = OSMRoadProvider(
        endpoint=endpoint,
        default_radius_m=radius_m,
        timeout_s=30.0,
    )

    query_str = provider.build_overpass_query(lat, lon, radius_m=radius_m)
    print("\nExact Overpass QL Query:")
    print("-" * 70)
    print(query_str)
    print("-" * 70)
    print("Sending live Overpass query...")

    try:
        roads = provider.get_nearby_roads(lat, lon, radius_m=radius_m, raise_on_error=True)
    except Exception as e:
        print(f"\n[ERROR] Overpass API query failed: {e}")
        print("\nNote: Public Overpass instances may occasionally experience high load or timeouts.")
        print("You can try another public Overpass mirror endpoint via the --endpoint flag, e.g.:")
        print("  --endpoint https://overpass.kumi.systems/api/interpreter")
        print("  --endpoint https://maps.mail.ru/osm/tools/overpass/api/interpreter")
        print("  --endpoint https://overpass.private.coffee/api/interpreter")
        sys.exit(1)

    print("\n" + "-" * 70)
    print(f"RESULTS: {len(roads)} drivable road ways returned")
    print("-" * 70)
    print(f"{'Index':<6} | {'Way ID':<12} | {'Highway Type':<15} | {'Points':<8} | {'Name'}")
    print("-" * 70)

    for idx, road in enumerate(roads, start=1):
        name_str = road.name if road.name else "(unnamed)"
        print(
            f"{idx:<6} | {road.way_id:<12} | {road.highway_type:<15} | "
            f"{road.num_points:<8} | {name_str}"
        )

    print("=" * 70)
    if roads:
        # Print first road sample geometry
        sample_road = roads[0]
        print(f"\nSample Way ID {sample_road.way_id} ({sample_road.highway_type}) - First 3 coordinates:")
        for pt_idx, (pt_lat, pt_lon) in enumerate(sample_road.coordinates[:3], start=1):
            print(f"  Point {pt_idx}: lat={pt_lat:.6f}, lon={pt_lon:.6f}")
    print("=" * 70)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Test live OpenStreetMap Overpass connection for NAVIGATE 2.0"
    )
    p.add_argument("--lat", type=float, default=51.5074, help="Target latitude (default: 51.5074)")
    p.add_argument("--lon", type=float, default=-0.1278, help="Target longitude (default: -0.1278)")
    p.add_argument("--radius", type=float, default=500.0, help="Search radius in metres (default: 500)")
    p.add_argument("--endpoint", type=str, default=DEFAULT_OVERPASS_ENDPOINT, help="Overpass endpoint URL")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    test_live_connection(
        lat=args.lat,
        lon=args.lon,
        radius_m=args.radius,
        endpoint=args.endpoint,
    )
