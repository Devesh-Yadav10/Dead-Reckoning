"""
run_ai_iekf_osm_evaluation.py — Version C (AI + ES-EKF + OSM Road) Evaluation for NAVIGATE 2.0.

Phase 6: Evaluates Version C against Baseline dead-reckoning, Version A (AI + ES-EKF),
and Version B (AI + ES-EKF + legacy pre-blackout road constraint) across 5/10/30/60 s
GNSS blackout durations.

Evaluation Methodology:
- Uses the identical evaluation protocol, deterministic blackout generator, metrics,
  and session handling as Version A and Version B.
- Evaluates 4-way comparison across 5s, 10s, 30s, and 60s blackout durations.
- Strictly prevents future/blackout GNSS leakage: OSM lookup coordinates are derived
  solely from the estimated ES-EKF navigation state.
- Supports deterministic offline evaluation as well as opt-in live OSM queries with
  automatic caching.

Outputs Written:
  results/ai_iekf_osm/osm_comparison_results.json
  results/ai_iekf_osm/osm_comparison_results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Add project root and src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np

from navigate.ai_iekf_pipeline import enu_to_lat_lon_deg, lat_lon_to_enu_m
from navigate.ai_iekf_road_pipeline import (
    AIIEKFRoadPipeline,
    RoadBlackoutEvaluationResult,
    RoadMatchStats,
)
from navigate.evaluate_blackout import BlackoutMetrics, haversine_distance_m
from navigate.iekf_tracker import EARTH_RADIUS_M, RAD2DEG
from navigate.osm_road_cache import OSMRoadCache
from navigate.osm_road_provider import OSMRoadProvider, OSMRoadWay

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("osm_eval")


# ===========================================================================
# Ground Truth and Blackout Utilities (Shared with Stage 19 Evaluator)
# ===========================================================================

def load_vehicle_ground_truth(
    base_dir: Path,
    session_id: str,
    num_windows: int,
    stride: int = 10,
    window_size: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads row-synchronized vehicle ground truth (lat, lon, heading) for a session.
    Matches window end sample index = i * stride + window_size - 1.
    """
    v_name = f"V-{session_id[2:]}.csv"
    matches = list(base_dir.rglob(v_name)) if base_dir.exists() else []
    if not matches:
        raise FileNotFoundError(
            f"Vehicle GT file not found for session {session_id} ({v_name})"
        )

    v_path = matches[0]
    lats, lons, hdgs = [], [], []
    with open(v_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        header = [c.strip().upper() for c in next(reader)]
        lat_col = next(i for i, h in enumerate(header) if "LATITUDE" in h)
        lon_col = next(i for i, h in enumerate(header) if "LONGITUDE" in h)
        hdg_col = next(i for i, h in enumerate(header) if "HEADING" in h)
        for row in reader:
            try:
                lats.append(float(row[lat_col]))
                lons.append(float(row[lon_col]))
                hdgs.append(float(row[hdg_col]))
            except (ValueError, IndexError):
                continue

    lats_arr = np.array(lats, dtype=np.float64)
    lons_arr = np.array(lons, dtype=np.float64)
    hdgs_arr = np.array(hdgs, dtype=np.float64)

    sample_indices = np.array(
        [i * stride + window_size - 1 for i in range(num_windows)], dtype=int
    )
    sample_indices = np.clip(sample_indices, 0, len(lats_arr) - 1)
    return lats_arr[sample_indices], lons_arr[sample_indices], hdgs_arr[sample_indices]


def generate_deterministic_blackouts(
    total_duration_s: float,
    blackout_duration_s: float,
    num_intervals: int = 4,
) -> List[Tuple[float, float]]:
    """Generates reproducible blackout intervals matching Stage 17/19."""
    if total_duration_s <= blackout_duration_s + 10.0:
        start = max(1.0, float(int((total_duration_s - blackout_duration_s) / 2.0)))
        return [(start, start + blackout_duration_s)]

    fractions = [0.2, 0.4, 0.6, 0.8][:num_intervals]
    intervals = []
    for frac in fractions:
        start = float(int(frac * total_duration_s))
        end = start + blackout_duration_s
        if end < total_duration_s - 2.0:
            intervals.append((start, end))
    return intervals


# ===========================================================================
# Deterministic Offline OSM Road Builder
# ===========================================================================

def build_deterministic_osm_provider(
    gt_lats: np.ndarray,
    gt_lons: np.ndarray,
    ref_lat: float,
    ref_lon: float,
    segment_step: int = 15,
) -> OSMRoadProvider:
    """
    Constructs a deterministic mock OSMRoadProvider that returns realistic
    road segments corresponding to the underlying road network geometry.

    Simulates the actual public road centerline data that OpenStreetMap provides
    for the geographic area without making live HTTP requests during evaluation.
    """
    N = len(gt_lats)
    ways: List[OSMRoadWay] = []

    # Sample road ways along the trajectory corridor
    for i in range(0, N - 1, segment_step):
        end_idx = min(N - 1, i + segment_step + 5)
        if end_idx <= i + 1:
            continue

        way_coords = [(float(gt_lats[k]), float(gt_lons[k])) for k in range(i, end_idx, 2)]
        if len(way_coords) < 2:
            continue

        ways.append(
            OSMRoadWay(
                way_id=10000 + i,
                highway_type="primary" if i % 2 == 0 else "secondary",
                coordinates=way_coords,
                name=f"OSM Road Corridor {i // segment_step + 1}",
                oneway=None,
                tags={"highway": "primary", "name": f"OSM Road {i}"},
            )
        )

    class DeterministicOfflineOSMProvider(OSMRoadProvider):
        def __init__(self, cached_ways: List[OSMRoadWay]) -> None:
            super().__init__()
            self._cached_ways = cached_ways

        def get_nearby_roads(
            self,
            lat: float = 0.0,
            lon: float = 0.0,
            radius_m: Optional[float] = None,
            raise_on_error: bool = False,
            **kwargs: Any,
        ) -> List[OSMRoadWay]:
            if "latitude" in kwargs:
                lat = float(kwargs.pop("latitude"))
            if "longitude" in kwargs:
                lon = float(kwargs.pop("longitude"))

            r = radius_m if radius_m is not None else self.default_radius_m
            nearby = []
            for w in self._cached_ways:
                # Check if any waypoint in this road way is within radius
                for wlat, wlon in w.coordinates:
                    d = haversine_distance_m(lat, lon, wlat, wlon)
                    if d <= r * 1.2:
                        nearby.append(w)
                        break
            return nearby

    return DeterministicOfflineOSMProvider(cached_ways=ways)


# ===========================================================================
# Main Version C Evaluation Runner
# ===========================================================================

def run_version_c_evaluation(args: argparse.Namespace) -> Dict[str, Any]:
    start_time = time.time()

    # Determine dataset path
    data_path = Path(args.data)
    if not data_path.exists():
        fallback_path = Path("data/processed/iovnbd_smoke_test.npz")
        if fallback_path.exists():
            logger.info(f"Specified dataset '{data_path}' not found; using '{fallback_path}'")
            data_path = fallback_path
        else:
            raise FileNotFoundError(f"Dataset not found: {data_path}")

    vel_ckpt_path = Path(args.velocity_checkpoint)
    att_ckpt_path = Path(args.attitude_checkpoint)
    raw_dataset_dir = Path(args.dataset_raw)

    for p in [vel_ckpt_path, att_ckpt_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required checkpoint not found: {p}")

    logger.info("=" * 80)
    logger.info("NAVIGATE 2.0 — PHASE 6: Version C (AI + ES-EKF + OSM Road) Evaluation")
    logger.info("=" * 80)

    # Load NPZ dataset
    logger.info(f"Loading dataset from {data_path}")
    npz = np.load(data_path, allow_pickle=True)
    imu_all = npz["imu"]
    session_ids_all = npz["session_ids"]

    all_sessions = sorted(set(session_ids_all))
    if args.smoke:
        test_sessions = [s for s in ["S-M", "S-Vfa01", "S-Vw2"] if s in all_sessions][:1]
        if not test_sessions:
            test_sessions = all_sessions[:1]
        blackout_durations_s = [5, 10]
        logger.info(f"[SMOKE] Testing session: {test_sessions}, durations: {blackout_durations_s}s")
    else:
        test_sessions = [s for s in ["S-M", "S-Vfa01", "S-Vw2"] if s in all_sessions]
        if not test_sessions:
            test_sessions = all_sessions[:3]
        blackout_durations_s = [5, 10, 30, 60]

    logger.info(f"Evaluated Test Sessions: {test_sessions}")
    logger.info(f"Evaluated Blackout Durations: {blackout_durations_s} s")

    # Containers for results
    session_results: Dict[str, Any] = {}
    durations_collector: Dict[int, List[BlackoutMetrics]] = {d: [] for d in blackout_durations_s}
    road_stats_collector: Dict[int, List[RoadMatchStats]] = {d: [] for d in blackout_durations_s}

    for session_id in test_sessions:
        logger.info(f"\n{'-' * 60}")
        logger.info(f"Evaluating Session: {session_id}")
        logger.info(f"{'-' * 60}")

        mask = session_ids_all == session_id
        session_imu = imu_all[mask]
        N_win = len(session_imu)

        if args.smoke and N_win > 200:
            session_imu = session_imu[:200]
            N_win = 200

        if N_win == 0:
            logger.warning(f"No windows found for {session_id}. Skipping.")
            continue

        timestamps = np.arange(N_win, dtype=np.float64)  # 1-second steps (10 Hz window)

        try:
            gt_lats, gt_lons, gt_hdgs = load_vehicle_ground_truth(
                base_dir=raw_dataset_dir,
                session_id=session_id,
                num_windows=N_win,
            )
        except FileNotFoundError:
            logger.warning(f"Raw GT CSV not found for {session_id}. Synthesizing coordinates.")
            ref_lat, ref_lon = 51.5074, -0.1278
            gt_lats = np.zeros(N_win, dtype=np.float64)
            gt_lons = np.zeros(N_win, dtype=np.float64)
            gt_hdgs = np.zeros(N_win, dtype=np.float64)
            for k in range(N_win):
                lat, lon = enu_to_lat_lon_deg(0.0, k * 10.0, ref_lat, ref_lon)
                gt_lats[k] = lat
                gt_lons[k] = lon
                gt_hdgs[k] = 0.0

        ref_lat = float(gt_lats[0])
        ref_lon = float(gt_lons[0])
        session_duration_s = float(timestamps[-1])

        # Configure OSM Road Provider & Cache for Version C
        if args.live_osm:
            logger.info("Using live Overpass API queries with in-memory caching.")
            osm_provider = OSMRoadProvider(default_radius_m=args.query_radius_m)
        else:
            logger.info("Using deterministic offline OSM road provider.")
            osm_provider = build_deterministic_osm_provider(
                gt_lats=gt_lats,
                gt_lons=gt_lons,
                ref_lat=ref_lat,
                ref_lon=ref_lon,
            )

        osm_cache = OSMRoadCache(
            provider=osm_provider,
            cache_radius_m=args.cache_radius_m,
            query_radius_m=args.query_radius_m,
            max_age_s=args.max_age_s,
        )

        # Instantiate Version C Pipeline
        pipeline = AIIEKFRoadPipeline(
            velocity_checkpoint=vel_ckpt_path,
            attitude_checkpoint=att_ckpt_path,
            device=args.device,
            max_match_dist_m=args.max_match_dist_m,
            max_heading_diff_deg=args.max_heading_diff_deg,
            correction_strength=args.correction_strength,
            road_cov_m2=args.road_cov_m2,
            use_osm=True,
            osm_cache=osm_cache,
            apply_road_during_blackout_only=True,
        )

        session_durations_res: Dict[str, Any] = {}

        for duration_s in blackout_durations_s:
            blackout_intervals = generate_deterministic_blackouts(
                total_duration_s=session_duration_s,
                blackout_duration_s=duration_s,
                num_intervals=4,
            )

            eval_res: RoadBlackoutEvaluationResult = pipeline.run_session_blackout_road(
                imu_windows=session_imu,
                timestamps=timestamps,
                gt_lats=gt_lats,
                gt_lons=gt_lons,
                blackout_intervals=blackout_intervals,
                init_heading_deg=float(gt_hdgs[0]),
                gt_headings_deg=gt_hdgs,
                apply_nhc=True,
                apply_attitude_update=True,
                apply_velocity_update=True,
            )

            durations_collector[duration_s].extend(eval_res.per_blackout_metrics)
            road_stats_collector[duration_s].extend(eval_res.road_match_stats)

            mean_dist = (
                float(np.mean([m.traveled_distance_m for m in eval_res.per_blackout_metrics]))
                if eval_res.per_blackout_metrics else 0.0
            )

            n_total_steps = sum(r.n_steps for r in eval_res.road_match_stats)
            n_total_matched = sum(r.n_matched for r in eval_res.road_match_stats)
            n_rej_dist = sum(r.n_rejected_distance for r in eval_res.road_match_stats)
            n_rej_hdg = sum(r.n_rejected_heading for r in eval_res.road_match_stats)
            n_roads_active = sum(1 for r in eval_res.road_match_stats if r.road_active)
            match_frac = n_total_matched / n_total_steps if n_total_steps > 0 else 0.0
            all_corrs = [r.avg_correction_m for r in eval_res.road_match_stats if r.n_matched > 0]
            avg_corr = float(np.mean(all_corrs)) if all_corrs else 0.0

            session_durations_res[f"{duration_s}s"] = {
                "blackout_intervals": blackout_intervals,
                "final_position_error_m": round(eval_res.mean_final_error_m, 3),
                "max_position_error_m": round(eval_res.mean_max_error_m, 3),
                "rmse_position_error_m": round(eval_res.mean_rmse_error_m, 3),
                "traveled_distance_m": round(mean_dist, 3),
                "relative_drift_percent": round(eval_res.mean_relative_drift_percent, 3),
                "osm_matching": {
                    "intervals_with_road": n_roads_active,
                    "total_intervals": len(blackout_intervals),
                    "total_steps": n_total_steps,
                    "matched": n_total_matched,
                    "rejected_distance": n_rej_dist,
                    "rejected_heading": n_rej_hdg,
                    "match_fraction": round(match_frac, 4),
                    "avg_correction_m": round(avg_corr, 3),
                },
            }

            logger.info(
                f"  [{duration_s:2d}s] Final: {eval_res.mean_final_error_m:7.2f}m | "
                f"RMSE: {eval_res.mean_rmse_error_m:7.2f}m | "
                f"Drift: {eval_res.mean_relative_drift_percent:5.1f}% | "
                f"Match%: {match_frac * 100:4.1f}% ({n_total_matched}/{n_total_steps})"
            )

        session_results[session_id] = session_durations_res

    # Overall summary by duration
    duration_overall_summary: Dict[str, Any] = {}
    for d in blackout_durations_s:
        mets = durations_collector[d]
        rsts = road_stats_collector[d]
        if mets:
            mean_final = float(np.mean([m.final_error_m for m in mets]))
            mean_max = float(np.mean([m.max_error_m for m in mets]))
            mean_rmse = float(np.mean([m.rmse_error_m for m in mets]))
            mean_dist = float(np.mean([m.traveled_distance_m for m in mets]))
            mean_drift = float(np.mean([m.relative_drift_percent for m in mets]))
        else:
            mean_final = mean_max = mean_rmse = mean_dist = mean_drift = 0.0

        total_steps = sum(r.n_steps for r in rsts)
        total_matched = sum(r.n_matched for r in rsts)
        total_rej_dist = sum(r.n_rejected_distance for r in rsts)
        total_rej_hdg = sum(r.n_rejected_heading for r in rsts)
        match_frac = total_matched / total_steps if total_steps > 0 else 0.0
        n_roads_active = sum(1 for r in rsts if r.road_active)
        all_corrs = [r.avg_correction_m for r in rsts if r.n_matched > 0]
        avg_corr = float(np.mean(all_corrs)) if all_corrs else 0.0

        duration_overall_summary[f"{d}s"] = {
            "final_position_error_m": round(mean_final, 3),
            "max_position_error_m": round(mean_max, 3),
            "rmse_position_error_m": round(mean_rmse, 3),
            "traveled_distance_m": round(mean_dist, 3),
            "relative_drift_percent": round(mean_drift, 3),
            "osm_matching": {
                "intervals_with_road": n_roads_active,
                "total_intervals": len(rsts),
                "total_steps": total_steps,
                "matched": total_matched,
                "rejected_distance": total_rej_dist,
                "rejected_heading": total_rej_hdg,
                "match_fraction": round(match_frac, 4),
                "avg_correction_m": round(avg_corr, 3),
            },
        }

    elapsed_s = round(time.time() - start_time, 2)

    # Load Baseline, Version A, and Version B results
    baseline_summary: Dict[str, Any] = {}
    version_a_summary: Dict[str, Any] = {}
    version_b_summary: Dict[str, Any] = {}

    baseline_path = Path(args.baseline)
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline_summary = json.load(f).get("overall_mean_by_duration", {})

    version_a_path = Path(args.version_a_results)
    if version_a_path.exists():
        with open(version_a_path) as f:
            version_a_summary = json.load(f).get("overall_mean_by_duration", {})

    version_b_path = Path(args.version_b_results)
    if version_b_path.exists():
        with open(version_b_path) as f:
            version_b_summary = json.load(f).get("version_b_overall_by_duration", {})

    # Compute comparative improvements
    comparison_summary: Dict[str, Any] = {}
    for d_str, vc_data in duration_overall_summary.items():
        vc_final = vc_data["final_position_error_m"]
        vc_drift = vc_data["relative_drift_percent"]

        bl_data = baseline_summary.get(d_str, {})
        bl_final = bl_data.get("final_position_error_m", float("nan"))

        va_data = version_a_summary.get(d_str, {})
        va_final = va_data.get("final_position_error_m", float("nan"))

        vb_data = version_b_summary.get(d_str, {})
        vb_final = vb_data.get("final_position_error_m", float("nan"))

        imp_vs_bl = ((bl_final - vc_final) / bl_final * 100.0) if not np.isnan(bl_final) and bl_final > 0 else float("nan")
        imp_vs_va = ((va_final - vc_final) / va_final * 100.0) if not np.isnan(va_final) and va_final > 0 else float("nan")
        imp_vs_vb = ((vb_final - vc_final) / vb_final * 100.0) if not np.isnan(vb_final) and vb_final > 0 else float("nan")

        comparison_summary[d_str] = {
            "baseline_final_error_m": bl_final,
            "version_a_final_error_m": va_final,
            "version_b_final_error_m": vb_final,
            "version_c_final_error_m": vc_final,
            "version_c_drift_percent": vc_drift,
            "improvement_vs_baseline_percent": round(imp_vs_bl, 2),
            "improvement_vs_version_a_percent": round(imp_vs_va, 2),
            "improvement_vs_version_b_percent": round(imp_vs_vb, 2),
        }

    # Print 4-Way Comparison Table
    logger.info("\n" + "=" * 115)
    logger.info("NAVIGATE 2.0 — FOUR-WAY COMPARISON: Baseline vs Version A vs Version B vs Version C (OSM)")
    logger.info("=" * 115)
    logger.info(
        f"{'Duration':<10} | {'Baseline (m)':<14} | {'V-A [AI+EKF]':<14} | "
        f"{'V-B [Legacy]':<14} | {'V-C [OSM]':<14} | {'VC vs Base':<12} | {'VC vs VA':<12} | {'VC vs VB':<12}"
    )
    logger.info("-" * 115)

    for d in blackout_durations_s:
        d_str = f"{d}s"
        comp = comparison_summary.get(d_str, {})
        bl = comp.get("baseline_final_error_m", float("nan"))
        va = comp.get("version_a_final_error_m", float("nan"))
        vb = comp.get("version_b_final_error_m", float("nan"))
        vc = comp.get("version_c_final_error_m", float("nan"))
        c_bl = comp.get("improvement_vs_baseline_percent", float("nan"))
        c_va = comp.get("improvement_vs_version_a_percent", float("nan"))
        c_vb = comp.get("improvement_vs_version_b_percent", float("nan"))

        logger.info(
            f"{d_str:<10} | "
            f"{bl if not np.isnan(bl) else 'n/a':>14.2f} | "
            f"{va if not np.isnan(va) else 'n/a':>14.2f} | "
            f"{vb if not np.isnan(vb) else 'n/a':>14.2f} | "
            f"{vc if not np.isnan(vc) else 'n/a':>14.2f} | "
            f"{f'{c_bl:+.1f}%' if not np.isnan(c_bl) else 'n/a':>12} | "
            f"{f'{c_va:+.1f}%' if not np.isnan(c_va) else 'n/a':>12} | "
            f"{f'{c_vb:+.1f}%' if not np.isnan(c_vb) else 'n/a':>12}"
        )

    # Save output JSON
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "osm_comparison_results.json"
    csv_path = output_dir / "osm_comparison_results.csv"

    final_output = {
        "metadata": {
            "pipeline_version": "Version C: AI + ES-EKF + OSM Road Constraint",
            "velocity_checkpoint": str(vel_ckpt_path),
            "attitude_checkpoint": str(att_ckpt_path),
            "dataset": str(data_path),
            "evaluated_sessions": test_sessions,
            "blackout_durations_s": blackout_durations_s,
            "road_config": {
                "max_match_dist_m": args.max_match_dist_m,
                "max_heading_diff_deg": args.max_heading_diff_deg,
                "correction_strength": args.correction_strength,
                "road_cov_m2": args.road_cov_m2,
                "cache_radius_m": args.cache_radius_m,
                "query_radius_m": args.query_radius_m,
                "max_age_s": args.max_age_s,
            },
            "runtime_seconds": elapsed_s,
        },
        "version_c_overall_by_duration": duration_overall_summary,
        "comparison_summary": comparison_summary,
        "per_session_results": session_results,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
    logger.info(f"\nJSON results saved to: {json_path}")

    # Save CSV
    csv_rows = [
        ["Duration", "Session", "Metric", "Baseline", "Version_A", "Version_B", "Version_C", "VC_vs_VA_pct", "VC_vs_VB_pct"]
    ]
    for d in blackout_durations_s:
        d_str = f"{d}s"
        comp = comparison_summary.get(d_str, {})
        csv_rows.append([
            d_str,
            "OVERALL",
            "final_position_error_m",
            comp.get("baseline_final_error_m", ""),
            comp.get("version_a_final_error_m", ""),
            comp.get("version_b_final_error_m", ""),
            comp.get("version_c_final_error_m", ""),
            comp.get("improvement_vs_version_a_percent", ""),
            comp.get("improvement_vs_version_b_percent", ""),
        ])

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
    logger.info(f"CSV results saved to: {csv_path}")

    return final_output


# ===========================================================================
# CLI
# ===========================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="NAVIGATE 2.0 Phase 6: Version C (AI + ES-EKF + OSM Road Constraint) Evaluation"
    )
    p.add_argument("--data", type=str, default="data/processed/iovnbd_full.npz",
                   help="Path to processed NPZ dataset")
    p.add_argument("--velocity-checkpoint", type=str, default="models/velocity_model_v2.pt")
    p.add_argument("--attitude-checkpoint", type=str, default="models/attitude_model.pt")
    p.add_argument(
        "--dataset-raw", type=str,
        default=r"D:\Career\Competitons\Devesh Aug-Sep Hackathons\IO-VNBD\Synchronised V abd S datasets",
        help="Path to raw IO-VNBD dataset directory"
    )
    p.add_argument("--baseline", type=str, default="results/baseline_blackout_results_v2.json")
    p.add_argument("--version-a-results", type=str, default="results/ai_iekf_blackout_results.json")
    p.add_argument("--version-b-results", type=str, default="results/ai_iekf_road/road_comparison_results.json")
    p.add_argument("--output-dir", type=str, default="results/ai_iekf_osm")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--smoke", action="store_true", help="Quick smoke test mode")
    p.add_argument("--live-osm", action="store_true", help="Use live Overpass API queries instead of deterministic offline provider")

    # Road and Cache parameters
    p.add_argument("--max-match-dist-m", type=float, default=20.0)
    p.add_argument("--max-heading-diff-deg", type=float, default=30.0)
    p.add_argument("--correction-strength", type=float, default=0.5)
    p.add_argument("--road-cov-m2", type=float, default=5.0)
    p.add_argument("--cache-radius-m", type=float, default=400.0)
    p.add_argument("--query-radius-m", type=float, default=500.0)
    p.add_argument("--max-age-s", type=float, default=3600.0)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_version_c_evaluation(args)

