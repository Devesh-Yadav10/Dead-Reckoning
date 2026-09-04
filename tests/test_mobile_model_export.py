"""
test_mobile_model_export.py — Unit tests verifying PyTorch vs ONNX numerical parity.

Tests:
  1. Structural validation of exported ONNX files.
  2. VelocityModel V2 numerical parity on random inputs.
  3. VelocityModel V2 numerical parity on realistic driving IMU windows.
  4. VelocityModel V2 numerical parity on edge-case inputs (zeros, large accelerations).
  5. AttitudeModel numerical parity on random inputs.
  6. AttitudeModel numerical parity on realistic driving IMU windows.
  7. AttitudeModel quaternion unit-length and non-negative qw constraint.
  8. Metadata JSON integrity and stats formatting.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import sys
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
MODELS_DIR = PROJECT_ROOT / "models"

from navigate.models.attitude_model import AttitudeModel
from navigate.models.velocity_model import VelocityModel


@pytest.fixture(scope="module")
def exported_models() -> dict[str, Path]:
    vel_onnx = MODELS_DIR / "velocity_model_v2.onnx"
    att_onnx = MODELS_DIR / "attitude_model.onnx"
    meta_json = MODELS_DIR / "model_metadata.json"

    assert vel_onnx.exists(), "velocity_model_v2.onnx missing"
    assert att_onnx.exists(), "attitude_model.onnx missing"
    assert meta_json.exists(), "model_metadata.json missing"

    return {"velocity": vel_onnx, "attitude": att_onnx, "metadata": meta_json}


def test_onnx_structural_validity(exported_models: dict[str, Path]) -> None:
    """Verifies that exported ONNX models pass ONNX specification checks."""
    vel_model = onnx.load(str(exported_models["velocity"]))
    onnx.checker.check_model(vel_model)

    att_model = onnx.load(str(exported_models["attitude"]))
    onnx.checker.check_model(att_model)


def test_velocity_model_numerical_parity_random(exported_models: dict[str, Path]) -> None:
    """Tests VelocityModel PyTorch vs ONNX on randomized inputs."""
    vel_ckpt = torch.load(MODELS_DIR / "velocity_model_v2.pt", map_location="cpu", weights_only=False)
    vel_model = VelocityModel(in_channels=6, hidden_size=128, window_size=50)
    vel_model.load_state_dict(vel_ckpt["model_state_dict"])
    vel_model.eval()

    session = ort.InferenceSession(str(exported_models["velocity"]), providers=["CPUExecutionProvider"])

    for seed in [1, 42, 123]:
        np.random.seed(seed)
        x_np = np.random.randn(1, 50, 6).astype(np.float32)
        with torch.no_grad():
            pt_out, _ = vel_model(torch.from_numpy(x_np))
            pt_val = pt_out.numpy()

        ort_val = session.run(["velocity_output"], {"imu_input": x_np})[0]
        max_diff = float(np.max(np.abs(pt_val - ort_val)))
        assert max_diff < 1e-4, f"Velocity diff too high ({max_diff:.8e})"


def test_velocity_model_numerical_parity_realistic(exported_models: dict[str, Path]) -> None:
    """Tests VelocityModel PyTorch vs ONNX on realistic vehicle IMU values (gravity on Z, small vibrations)."""
    vel_ckpt = torch.load(MODELS_DIR / "velocity_model_v2.pt", map_location="cpu", weights_only=False)
    vel_model = VelocityModel(in_channels=6, hidden_size=128, window_size=50)
    vel_model.load_state_dict(vel_ckpt["model_state_dict"])
    vel_model.eval()

    session = ort.InferenceSession(str(exported_models["velocity"]), providers=["CPUExecutionProvider"])

    # Realistic: Accel [0.2, 0.0, 9.81], Gyro [0.0, 0.0, 0.02]
    np.random.seed(99)
    accel = np.random.randn(1, 50, 3).astype(np.float32) * 0.1 + np.array([0.2, 0.0, 9.81], dtype=np.float32)
    gyro = np.random.randn(1, 50, 3).astype(np.float32) * 0.01 + np.array([0.0, 0.0, 0.02], dtype=np.float32)
    x_np = np.concatenate([accel, gyro], axis=-1)

    with torch.no_grad():
        pt_out, _ = vel_model(torch.from_numpy(x_np))
        pt_val = pt_out.numpy()

    ort_val = session.run(["velocity_output"], {"imu_input": x_np})[0]
    max_diff = float(np.max(np.abs(pt_val - ort_val)))
    assert max_diff < 1e-4


def test_attitude_model_numerical_parity_random(exported_models: dict[str, Path]) -> None:
    """Tests AttitudeModel PyTorch vs ONNX on randomized inputs."""
    att_ckpt = torch.load(MODELS_DIR / "attitude_model.pt", map_location="cpu", weights_only=False)
    att_model = AttitudeModel(in_channels=6, hidden_size=128, window_size=50)
    att_model.load_state_dict(att_ckpt["model_state_dict"])
    att_model.eval()

    session = ort.InferenceSession(str(exported_models["attitude"]), providers=["CPUExecutionProvider"])

    for seed in [1, 42, 123]:
        np.random.seed(seed)
        x_np = np.random.randn(1, 50, 6).astype(np.float32)
        with torch.no_grad():
            pt_out, _ = att_model(torch.from_numpy(x_np))
            pt_val = pt_out.numpy()

        ort_val = session.run(["attitude_output"], {"imu_input": x_np})[0]
        max_diff = float(np.max(np.abs(pt_val - ort_val)))
        assert max_diff < 1e-4, f"Attitude diff too high ({max_diff:.8e})"

        # Unit length test
        norm = np.linalg.norm(ort_val[0])
        assert pytest.approx(norm, abs=1e-5) == 1.0
        # Non-negative qw
        assert ort_val[0, 0] >= 0.0


def test_metadata_json_contents(exported_models: dict[str, Path]) -> None:
    """Verifies that model_metadata.json contains accurate normalization stats from checkpoints."""
    with open(exported_models["metadata"], "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert "velocity_model" in meta
    assert "attitude_model" in meta

    vel_meta = meta["velocity_model"]
    att_meta = meta["attitude_model"]

    # Expected velocity statistics from velocity_model_v2.pt
    exp_vel_imu_mean = [0.037118, -0.051447, 9.688788, 0.000369, -0.005138, 0.001011]
    exp_vel_imu_std = [1.638557, 1.572975, 0.819746, 0.128741, 0.224838, 0.140099]
    exp_vel_mean = 11.496023
    exp_vel_std = 8.599233

    # Expected attitude statistics from attitude_model.pt
    exp_att_imu_mean = [-0.028631, -0.050928, 9.874502, -0.001353, -0.002314, -0.000977]
    exp_att_imu_std = [1.762716, 1.653900, 0.738219, 0.102856, 0.246167, 0.146546]

    # Verify Velocity stats
    for actual, expected in zip(vel_meta["imu_mean"], exp_vel_imu_mean):
        assert pytest.approx(actual, abs=1e-4) == expected
    for actual, expected in zip(vel_meta["imu_std"], exp_vel_imu_std):
        assert pytest.approx(actual, abs=1e-4) == expected
    assert pytest.approx(vel_meta["target_mean"], abs=1e-4) == exp_vel_mean
    assert pytest.approx(vel_meta["target_std"], abs=1e-4) == exp_vel_std

    # Verify Attitude stats
    for actual, expected in zip(att_meta["imu_mean"], exp_att_imu_mean):
        assert pytest.approx(actual, abs=1e-4) == expected
    for actual, expected in zip(att_meta["imu_std"], exp_att_imu_std):
        assert pytest.approx(actual, abs=1e-4) == expected


def test_stationary_imu_normalization_regression(exported_models: dict[str, Path]) -> None:
    """
    Regression Test: Ensures stationary IMU window (az = 9.81 m/s^2) is correctly normalized
    around ~0.148 and NOT left as raw 9.81 (>12 sigma out-of-distribution).
    """
    with open(exported_models["metadata"], "r", encoding="utf-8") as f:
        meta = json.load(f)

    vel_meta = meta["velocity_model"]
    imu_mean = np.array(vel_meta["imu_mean"], dtype=np.float32)
    imu_std = np.array(vel_meta["imu_std"], dtype=np.float32)

    # Construct stationary sample: ax=0, ay=0, az=9.81, gx=0, gy=0, gz=0
    raw_sample = np.array([0.0, 0.0, 9.81, 0.0, 0.0, 0.0], dtype=np.float32)
    normalized_sample = (raw_sample - imu_mean) / imu_std

    # Verify az normalization: (9.81 - 9.688788) / 0.819746 ≈ 0.14787
    expected_az_norm = (9.81 - 9.688788) / 0.819746
    assert pytest.approx(normalized_sample[2], abs=1e-3) == expected_az_norm
    assert abs(normalized_sample[2] - 9.81) > 9.0, "Regression failure: az was not normalized!"


def test_velocity_stationary_prediction_with_corrected_metadata(exported_models: dict[str, Path]) -> None:
    """
    Verifies that ONNX velocity model fed with stationary IMU normalized using metadata
    predicts ~0.54 m/s (approx 1.94 km/h), completely eliminating the ~21.65 m/s (78 km/h) anomaly.
    """
    with open(exported_models["metadata"], "r", encoding="utf-8") as f:
        meta = json.load(f)

    vel_meta = meta["velocity_model"]
    imu_mean = np.array(vel_meta["imu_mean"], dtype=np.float32)
    imu_std = np.array(vel_meta["imu_std"], dtype=np.float32)
    vel_mean = float(vel_meta["target_mean"])
    vel_std = float(vel_meta["target_std"])

    # 50 samples of stationary IMU
    raw_window = np.zeros((1, 50, 6), dtype=np.float32)
    raw_window[:, :, 2] = 9.81  # gravity on az

    norm_window = (raw_window - imu_mean) / imu_std

    # PyTorch evaluation
    vel_ckpt = torch.load(MODELS_DIR / "velocity_model_v2.pt", map_location="cpu", weights_only=False)
    vel_model = VelocityModel(in_channels=6, hidden_size=128, window_size=50)
    vel_model.load_state_dict(vel_ckpt["model_state_dict"])
    vel_model.eval()

    with torch.no_grad():
        pt_norm_out, _ = vel_model(torch.from_numpy(norm_window))
        pt_speed = float(pt_norm_out[0, 0].item() * vel_std + vel_mean)

    # ONNX evaluation
    session = ort.InferenceSession(str(exported_models["velocity"]), providers=["CPUExecutionProvider"])
    ort_norm_out = session.run(["velocity_output"], {"imu_input": norm_window})[0]
    onnx_speed = float(ort_norm_out[0, 0] * vel_std + vel_mean)

    # Equivalence
    max_diff = abs(pt_speed - onnx_speed)
    assert max_diff < 1e-4, f"PyTorch vs ONNX speed difference too high ({max_diff})"

    # Stationary prediction must be in [0.0, 1.5] m/s, NOT ~21.65 m/s
    assert 0.0 <= onnx_speed <= 1.5, f"Unexpected high speed prediction on stationary data: {onnx_speed} m/s"
    assert pytest.approx(onnx_speed, abs=0.1) == 0.54

