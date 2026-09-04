"""
benchmark_realtime_engine.py — Performance Benchmark for RealTimeNavigationEngine (NAVIGATE 2.0).

Measures:
  - Overall throughput (samples / second)
  - Navigation tick processing latency (mean, P50, P95, P99, max)
  - AI Model inference latencies (VelocityModel V2, AttitudeModel)
  - ES-EKF propagation latency
  - Real-time 100ms deadline compliance rate (%)
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch

# Add src and project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from navigate.ai_iekf_pipeline import AIIEKFPipeline
from navigate.realtime_engine import RealTimeNavigationEngine
from navigate.sensor_types import IMUSample


def run_benchmark(
    num_samples: int = 1000,
    device: str = "cpu",
    velocity_ckpt: str = "models/velocity_model_v2.pt",
    attitude_ckpt: str = "models/attitude_model.pt",
) -> None:
    print("=" * 80)
    print("NAVIGATE 2.0 — Real-Time Navigation Engine Benchmark")
    print("=" * 80)
    print(f"Platform: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Python:   {platform.python_version()} | PyTorch: {torch.__version__}")
    print(f"Device:   {device.upper()} | Benchmark Samples: {num_samples} (10 Hz = {num_samples / 10.0:.1f}s)")
    print("-" * 80)

    # Initialize Engine
    t_init_start = time.perf_counter()
    pipeline = AIIEKFPipeline(
        velocity_checkpoint=velocity_ckpt,
        attitude_checkpoint=attitude_ckpt,
        device=device,
    )
    engine = RealTimeNavigationEngine(
        pipeline=pipeline,
        use_osm=False,
        deadline_ms=100.0,
    )
    init_duration_s = time.perf_counter() - t_init_start
    print(f"Pipeline & Models initialized in {init_duration_s:.3f} s.\n")

    # Generate synthetic 10 Hz IMU stream
    np.random.seed(42)
    accel = np.random.randn(num_samples, 3) * 0.2 + np.array([0.0, 0.0, 9.81])
    gyro = np.random.randn(num_samples, 3) * 0.02
    timestamps = np.arange(num_samples) * 0.1

    imu_stream: List[IMUSample] = [
        IMUSample(
            timestamp=float(timestamps[i]),
            ax=float(accel[i, 0]),
            ay=float(accel[i, 1]),
            az=float(accel[i, 2]),
            gx=float(gyro[i, 0]),
            gy=float(gyro[i, 1]),
            gz=float(gyro[i, 2]),
        )
        for i in range(num_samples)
    ]

    # Warm-up run (50 samples)
    engine.start_session(ref_lat=51.5074, ref_lon=-0.1278, init_timestamp=0.0)
    for s in imu_stream[:50]:
        engine.process_imu(s)
    engine.reset()

    # Benchmark Execution
    engine.start_session(ref_lat=51.5074, ref_lon=-0.1278, init_timestamp=0.0)
    tick_latencies: List[float] = []
    vel_latencies: List[float] = []
    att_latencies: List[float] = []
    ekf_latencies: List[float] = []

    t_bench_start = time.perf_counter()
    for s in imu_stream:
        st = engine.process_imu(s)
        if st is not None:
            tick_latencies.append(st.processing_latency_ms)
            diag = st.diagnostics
            if diag.get("velocity_latency_ms", 0.0) > 0:
                vel_latencies.append(diag["velocity_latency_ms"])
            if diag.get("attitude_latency_ms", 0.0) > 0:
                att_latencies.append(diag["attitude_latency_ms"])
            if diag.get("ekf_latency_ms", 0.0) > 0:
                ekf_latencies.append(diag["ekf_latency_ms"])

    total_bench_duration_s = time.perf_counter() - t_bench_start
    throughput = num_samples / total_bench_duration_s

    # Metrics Summary
    lat_arr = np.array(tick_latencies)
    vel_arr = np.array(vel_latencies) if vel_latencies else np.array([0.0])
    att_arr = np.array(att_latencies) if att_latencies else np.array([0.0])
    ekf_arr = np.array(ekf_latencies) if ekf_latencies else np.array([0.0])

    p50 = np.percentile(lat_arr, 50)
    p95 = np.percentile(lat_arr, 95)
    p99 = np.percentile(lat_arr, 99)
    on_time_pct = (np.sum(lat_arr <= 100.0) / len(lat_arr)) * 100.0

    print("BENCHMARK RESULTS (Development-Machine Benchmark)")
    print("=" * 80)
    print(f"Total Processed Samples:     {len(lat_arr)} samples")
    print(f"Total Execution Time:        {total_bench_duration_s:.3f} s")
    print(f"Throughput:                  {throughput:.1f} samples/sec ({throughput / 10.0:.1f}x real-time speed)")
    print(f"Real-Time Deadline (<100ms): {on_time_pct:.2f}% compliant")
    print("-" * 80)
    print("LATENCY BREAKDOWN (milliseconds):")
    print(f"  Overall Tick Latency:      Mean: {np.mean(lat_arr):6.3f} ms | P50: {p50:6.3f} ms | P95: {p95:6.3f} ms | P99: {p99:6.3f} ms | Max: {np.max(lat_arr):6.3f} ms")
    print(f"  VelocityModel V2 (AI):     Mean: {np.mean(vel_arr):6.3f} ms | P95: {np.percentile(vel_arr, 95):6.3f} ms | Max: {np.max(vel_arr):6.3f} ms")
    print(f"  AttitudeModel (AI):        Mean: {np.mean(att_arr):6.3f} ms | P95: {np.percentile(att_arr, 95):6.3f} ms | Max: {np.max(att_arr):6.3f} ms")
    print(f"  ES-EKF Step Propagation:   Mean: {np.mean(ekf_arr):6.3f} ms | P95: {np.percentile(ekf_arr, 95):6.3f} ms | Max: {np.max(ekf_arr):6.3f} ms")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RealTimeNavigationEngine Benchmark")
    parser.add_argument("--samples", type=int, default=1000, help="Number of 10 Hz samples to process")
    parser.add_argument("--device", type=str, default="cpu", help="Device string ('cpu', 'cuda')")
    parser.add_argument("--velocity-checkpoint", type=str, default="models/velocity_model_v2.pt")
    parser.add_argument("--attitude-checkpoint", type=str, default="models/attitude_model.pt")
    args = parser.parse_args()

    run_benchmark(
        num_samples=args.samples,
        device=args.device,
        velocity_ckpt=args.velocity_checkpoint,
        attitude_ckpt=args.attitude_checkpoint,
    )

