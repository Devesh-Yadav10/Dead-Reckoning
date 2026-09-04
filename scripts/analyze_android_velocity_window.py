"""
analyze_android_velocity_window.py — Forensic diagnostic tool for Android IMU input windows.

Analyzes raw 50x6 IMU windows captured from Android (or provided via CSV/JSON),
evaluates them under PyTorch VelocityModel and ONNX Runtime, and compares their statistics
against synthetic stationary windows and real IO-VNBD dataset stationary windows.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import onnxruntime as ort
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

import sys
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from navigate.models.velocity_model import VelocityModel


def load_model_and_metadata() -> Tuple[VelocityModel, ort.InferenceSession, Dict[str, Any]]:
    """Loads PyTorch model, ONNX Runtime session, and metadata."""
    meta_path = MODELS_DIR / "model_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)["velocity_model"]

    # Load PyTorch
    ckpt_path = MODELS_DIR / "velocity_model_v2.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    pt_model = VelocityModel(in_channels=6, hidden_size=128, window_size=50)
    pt_model.load_state_dict(ckpt["model_state_dict"])
    pt_model.eval()

    # Load ONNX
    onnx_path = MODELS_DIR / "velocity_model_v2.onnx"
    ort_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    return pt_model, ort_session, meta


def compute_channel_stats(data: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Computes min, max, mean, and std for 6 IMU channels across (N, 50, 6) or (N, 6) data."""
    flat = data.reshape(-1, 6)
    channels = ["ax", "ay", "az", "gx", "gy", "gz"]
    stats = {}
    for i, ch in enumerate(channels):
        stats[ch] = {
            "min": float(flat[:, i].min()),
            "max": float(flat[:, i].max()),
            "mean": float(flat[:, i].mean()),
            "std": float(flat[:, i].std()),
        }
    return stats


def print_stats_table(raw_stats: Dict[str, Dict[str, float]], norm_stats: Dict[str, Dict[str, float]], title: str) -> None:
    """Prints a formatted summary table of raw and normalized channel statistics."""
    print("=" * 80)
    print(f"{title}")
    print("=" * 80)
    print(f"{'Channel':<8} | {'Raw Mean':>10} {'Raw Std':>10} {'Raw Min':>10} {'Raw Max':>10} | {'Norm Mean':>10} {'Norm Std':>10}")
    print("-" * 80)
    for ch in ["ax", "ay", "az", "gx", "gy", "gz"]:
        r = raw_stats[ch]
        n = norm_stats[ch]
        print(f"{ch:<8} | {r['mean']:>10.4f} {r['std']:>10.4f} {r['min']:>10.4f} {r['max']:>10.4f} | {n['mean']:>10.4f} {n['std']:>10.4f}")
    print("-" * 80)


def run_inference_on_windows(
    windows: np.ndarray,
    pt_model: VelocityModel,
    ort_session: ort.InferenceSession,
    meta: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs PyTorch and ONNX inference on a batch of (N, 50, 6) windows.
    Returns (normalized_windows, pt_speeds_ms, onnx_speeds_ms).
    """
    imu_mean = np.array(meta["imu_mean"], dtype=np.float32)
    imu_std = np.array(meta["imu_std"], dtype=np.float32)
    target_mean = float(meta["target_mean"])
    target_std = float(meta["target_std"])

    norm_windows = (windows - imu_mean) / imu_std

    pt_speeds = []
    onnx_speeds = []

    with torch.no_grad():
        for i in range(len(windows)):
            w_norm = norm_windows[i : i + 1]
            
            # PyTorch
            pt_out, _ = pt_model(torch.from_numpy(w_norm))
            pt_speed = float(pt_out[0, 0].item() * target_std + target_mean)
            pt_speeds.append(pt_speed)

            # ONNX
            ort_out = ort_session.run(["velocity_output"], {"imu_input": w_norm})[0]
            ort_speed = float(ort_out[0, 0] * target_std + target_mean)
            onnx_speeds.append(ort_speed)

    return norm_windows, np.array(pt_speeds), np.array(onnx_speeds)


def parse_capture_file(file_path: Path) -> np.ndarray:
    """
    Parses captured Android IMU data from JSON or CSV.
    Expected shapes:
      - JSON: List of 6-element float arrays, or list of objects, or 50x6 array.
      - CSV: Columns ax,ay,az,gx,gy,gz or raw samples.
    """
    if file_path.suffix.lower() == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "samples" in data:
            data = data["samples"]
        arr = np.array(data, dtype=np.float32)
    elif file_path.suffix.lower() == ".csv":
        rows = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("timestamp") or line.startswith("ax"):
                    continue
                parts = [float(x.strip()) for x in line.split(",") if x.strip()]
                if len(parts) >= 6:
                    if len(parts) == 6:
                        rows.append(parts)
                    elif len(parts) == 7:
                        rows.append(parts[1:7])
                    else:
                        rows.append(parts[:6])
        arr = np.array(rows, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

    # Ensure shape (N, 50, 6)
    if arr.ndim == 2:
        if arr.shape[1] != 6:
            raise ValueError(f"Expected 6 channels, got {arr.shape[1]}")
        if len(arr) < 50:
            raise ValueError(f"Need at least 50 samples, got {len(arr)}")
        num_windows = (len(arr) - 50) // 10 + 1
        windows = np.zeros((num_windows, 50, 6), dtype=np.float32)
        for i in range(num_windows):
            windows[i] = arr[i * 10 : i * 10 + 50]
        return windows
    elif arr.ndim == 3 and arr.shape[1:] == (50, 6):
        return arr
    else:
        raise ValueError(f"Unrecognized data array shape: {arr.shape}")


def generate_synthetic_stationary_window(tilt_pitch_deg: float = 0.0, tilt_roll_deg: float = 0.0) -> np.ndarray:
    """Generates a 50-sample stationary window with gravity and optional tilt."""
    pitch = np.radians(tilt_pitch_deg)
    roll = np.radians(tilt_roll_deg)
    
    g = 9.81
    ax = -g * np.sin(roll) * np.cos(pitch)
    ay = -g * np.sin(pitch)
    az = g * np.cos(pitch) * np.cos(roll)

    w = np.zeros((1, 50, 6), dtype=np.float32)
    w[:, :, 0] = ax
    w[:, :, 1] = ay
    w[:, :, 2] = az
    return w


def load_iovnbd_stationary_windows(limit: int = 500) -> Optional[np.ndarray]:
    """Loads ground-truth stationary windows from processed IO-VNBD dataset if available."""
    npz_path = DATA_DIR / "processed" / "iovnbd_smoke_test.npz"
    if not npz_path.exists():
        return None
    npz = np.load(npz_path, allow_pickle=True)
    imu = npz["imu"]
    vel = npz["velocity"]
    stat_mask = vel < 0.1
    stat_imu = imu[stat_mask]
    if len(stat_imu) > limit:
        stat_imu = stat_imu[:limit]
    return stat_imu


def main() -> None:
    parser = argparse.ArgumentParser(description="Forensic Analysis of Android Velocity Model Input Windows")
    parser.add_argument("--capture-file", type=str, default=None, help="Path to captured Android IMU JSON or CSV file")
    parser.add_argument("--pitch-deg", type=float, default=0.0, help="Simulate synthetic phone pitch tilt in degrees")
    parser.add_argument("--roll-deg", type=float, default=0.0, help="Simulate synthetic phone roll tilt in degrees")
    args = parser.parse_args()

    pt_model, ort_session, meta = load_model_and_metadata()

    # 1. Benchmark: Synthetic Flat Stationary Window
    synth_flat = generate_synthetic_stationary_window(0.0, 0.0)
    _, pt_flat, ort_flat = run_inference_on_windows(synth_flat, pt_model, ort_session, meta)
    stats_flat_raw = compute_channel_stats(synth_flat)
    norm_flat, _, _ = run_inference_on_windows(synth_flat, pt_model, ort_session, meta)
    stats_flat_norm = compute_channel_stats(norm_flat)

    print("\n" + "=" * 80)
    print("NAVIGATE 2.0 — VELOCITY MODEL INPUT DOMAIN & ORIENTATION DIAGNOSTIC")
    print("=" * 80)
    print(f"Synthetic Flat Stationary Prediction: PyTorch = {pt_flat[0]:.4f} m/s ({pt_flat[0]*3.6:.2f} km/h), ONNX = {ort_flat[0]:.4f} m/s ({ort_flat[0]*3.6:.2f} km/h)")

    # 2. IO-VNBD Real Stationary Windows Benchmark
    iov_stat = load_iovnbd_stationary_windows(limit=500)
    if iov_stat is not None:
        norm_iov, pt_iov, ort_iov = run_inference_on_windows(iov_stat, pt_model, ort_session, meta)
        stats_iov_raw = compute_channel_stats(iov_stat)
        stats_iov_norm = compute_channel_stats(norm_iov)
        print(f"IO-VNBD Ground-Truth Stationary (N={len(iov_stat)}): Mean ONNX Speed = {ort_iov.mean():.4f} m/s ({ort_iov.mean()*3.6:.2f} km/h), Median = {np.median(ort_iov):.4f} m/s, Std = {ort_iov.std():.4f} m/s")

    # 3. Captured Android Data or Simulated Tilt
    if args.capture_file:
        cap_path = Path(args.capture_file)
        android_windows = parse_capture_file(cap_path)
        source_name = f"Android Capture ({cap_path.name})"
    else:
        android_windows = generate_synthetic_stationary_window(args.pitch_deg, args.roll_deg)
        source_name = f"Simulated Android Window (Pitch={args.pitch_deg}°, Roll={args.roll_deg}°)"

    norm_and, pt_and, ort_and = run_inference_on_windows(android_windows, pt_model, ort_session, meta)
    stats_and_raw = compute_channel_stats(android_windows)
    stats_and_norm = compute_channel_stats(norm_and)

    print_stats_table(stats_and_raw, stats_and_norm, f"{source_name} Channel Statistics")

    print("\n" + "=" * 80)
    print("PREDICTIONS SUMMARY:")
    print("=" * 80)
    print(f"{'Window #':<10} | {'PyTorch (m/s)':>15} | {'PyTorch (km/h)':>15} | {'ONNX (m/s)':>15} | {'ONNX (km/h)':>15}")
    print("-" * 80)
    for idx in range(min(15, len(ort_and))):
        print(f"{idx+1:<10} | {pt_and[idx]:>15.4f} | {pt_and[idx]*3.6:>15.2f} | {ort_and[idx]:>15.4f} | {ort_and[idx]*3.6:>15.2f}")
    if len(ort_and) > 15:
        print(f"... ({len(ort_and) - 15} additional windows omitted)")
    print("-" * 80)
    print(f"Average Predicted Speed: {ort_and.mean():.4f} m/s ({ort_and.mean()*3.6:.2f} km/h)")
    print(f"Min / Max Speed:         [{ort_and.min():.4f} .. {ort_and.max():.4f}] m/s ([{ort_and.min()*3.6:.2f} .. {ort_and.max()*3.6:.2f}] km/h)")
    print("=" * 80 + "\n")

    # 4. Comparative Cross-Source Table
    if iov_stat is not None:
        print("=" * 80)
        print("COMPARATIVE THREE-SOURCE CHANNEL STATISTICS TABLE:")
        print("=" * 80)
        print(f"{'Metric':<18} | {'Synthetic Flat':>18} | {'IO-VNBD Stationary':>18} | {source_name:>22}")
        print("-" * 80)
        for ch in ["ax", "ay", "az", "gx", "gy", "gz"]:
            print(f"{ch + ' mean':<18} | {stats_flat_raw[ch]['mean']:>18.4f} | {stats_iov_raw[ch]['mean']:>18.4f} | {stats_and_raw[ch]['mean']:>22.4f}")
            print(f"{ch + ' std':<18} | {stats_flat_raw[ch]['std']:>18.4f} | {stats_iov_raw[ch]['std']:>18.4f} | {stats_and_raw[ch]['std']:>22.4f}")
        print("-" * 80)
        print(f"{'Mean AI Speed':<18} | {ort_flat[0]:>14.2f} m/s | {ort_iov.mean():>14.2f} m/s | {ort_and.mean():>18.2f} m/s")
        print(f"{'Mean AI Speed (km/h)':<18} | {ort_flat[0]*3.6:>12.2f} km/h | {ort_iov.mean()*3.6:>12.2f} km/h | {ort_and.mean()*3.6:>16.2f} km/h")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

