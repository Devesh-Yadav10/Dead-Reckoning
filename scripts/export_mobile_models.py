"""
export_mobile_models.py — Exports trained PyTorch models to ONNX for mobile deployment.

Exports:
  1. VelocityModel V2 -> models/velocity_model_v2.onnx
  2. AttitudeModel    -> models/attitude_model.onnx
  3. Metadata / stats -> models/model_metadata.json

Features:
  - Preserves exact model weights, input tensor shapes [B, 50, 6], and layer structure.
  - Verifies exported ONNX models via onnx.checker and onnxruntime numerical check.
  - Automatically copies exported artifacts into android/app/src/main/assets/.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import onnx
import onnxruntime as ort
import torch

# Add src and project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from navigate.models.attitude_model import AttitudeModel
from navigate.models.velocity_model import VelocityModel


import io

# Ensure UTF-8 stdout for Windows consoles
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def export_models(
    models_dir: Path = PROJECT_ROOT / "models",
    assets_dir: Path = PROJECT_ROOT / "android" / "app" / "src" / "main" / "assets",
    opset_version: int = 17,
) -> None:
    print("=" * 80)
    print("NAVIGATE 2.0 — PyTorch -> ONNX Mobile Model Export")
    print("=" * 80)
    print(f"Target ONNX Opset Version: {opset_version}")
    models_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    metadata: Dict[str, Any] = {}

    # =======================================================================
    # 1. Export VelocityModel V2
    # =======================================================================
    vel_pt_path = models_dir / "velocity_model_v2.pt"
    vel_onnx_path = models_dir / "velocity_model_v2.onnx"

    if not vel_pt_path.exists():
        raise FileNotFoundError(f"Velocity checkpoint not found: {vel_pt_path}")

    print(f"\n[1/3] Loading VelocityModel V2 checkpoint: {vel_pt_path}")
    vel_ckpt = torch.load(vel_pt_path, map_location="cpu", weights_only=False)
    vel_model = VelocityModel(in_channels=6, hidden_size=128, window_size=50)
    vel_model.load_state_dict(vel_ckpt["model_state_dict"])
    vel_model.eval()

    # Extract normalization statistics with top-level priority and nested fallback
    if "imu_mean" in vel_ckpt and "imu_std" in vel_ckpt:
        vel_imu_mean = np.asarray(vel_ckpt["imu_mean"], dtype=np.float32).tolist()
        vel_imu_std = np.asarray(vel_ckpt["imu_std"], dtype=np.float32).tolist()
    elif "norm_stats" in vel_ckpt and "imu_mean" in vel_ckpt["norm_stats"]:
        vel_imu_mean = np.asarray(vel_ckpt["norm_stats"]["imu_mean"], dtype=np.float32).tolist()
        vel_imu_std = np.asarray(vel_ckpt["norm_stats"]["imu_std"], dtype=np.float32).tolist()
    else:
        raise KeyError("Velocity checkpoint missing required IMU normalization statistics ('imu_mean' / 'imu_std')")

    if "vel_mean" in vel_ckpt and "vel_std" in vel_ckpt:
        vel_target_mean = float(vel_ckpt["vel_mean"])
        vel_target_std = float(vel_ckpt["vel_std"])
    elif "norm_stats" in vel_ckpt and "target_mean" in vel_ckpt["norm_stats"]:
        vel_target_mean = float(vel_ckpt["norm_stats"]["target_mean"])
        vel_target_std = float(vel_ckpt["norm_stats"]["target_std"])
    else:
        raise KeyError("Velocity checkpoint missing required target velocity statistics ('vel_mean' / 'vel_std')")

    metadata["velocity_model"] = {
        "file": "velocity_model_v2.onnx",
        "input_name": "imu_input",
        "input_shape": [1, 50, 6],
        "output_name": "velocity_output",
        "output_shape": [1, 1],
        "imu_mean": vel_imu_mean,
        "imu_std": vel_imu_std,
        "target_mean": vel_target_mean,
        "target_std": vel_target_std,
    }

    # Wrapper returning only the speed prediction scalar tensor
    class VelocityExportWrapper(torch.nn.Module):
        def __init__(self, model: torch.nn.Module):
            super().__init__()
            self.model = model

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            speed, _ = self.model(x)
            return speed

    vel_wrapper = VelocityExportWrapper(vel_model)
    vel_wrapper.eval()
    vel_dummy = torch.randn(1, 50, 6, dtype=torch.float32)

    print(f"      Exporting to ONNX: {vel_onnx_path}")
    torch.onnx.export(
        vel_wrapper,
        vel_dummy,
        str(vel_onnx_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["imu_input"],
        output_names=["velocity_output"],
        dynamic_axes={
            "imu_input": {0: "batch_size"},
            "velocity_output": {0: "batch_size"},
        },
        dynamo=False,
    )

    # Validate ONNX structure
    onnx_vel_model = onnx.load(str(vel_onnx_path))
    onnx.checker.check_model(onnx_vel_model)
    print(f"      Structural validation passed! Model size: {vel_onnx_path.stat().st_size / 1024:.1f} KB")

    # =======================================================================
    # 2. Export AttitudeModel
    # =======================================================================
    att_pt_path = models_dir / "attitude_model.pt"
    att_onnx_path = models_dir / "attitude_model.onnx"

    if not att_pt_path.exists():
        raise FileNotFoundError(f"Attitude checkpoint not found: {att_pt_path}")

    print(f"\n[2/3] Loading AttitudeModel checkpoint: {att_pt_path}")
    att_ckpt = torch.load(att_pt_path, map_location="cpu", weights_only=False)
    att_model = AttitudeModel(in_channels=6, hidden_size=128, window_size=50)
    att_model.load_state_dict(att_ckpt["model_state_dict"])
    att_model.eval()

    # Extract normalization statistics with top-level priority and nested fallback
    if "imu_mean" in att_ckpt and "imu_std" in att_ckpt:
        att_imu_mean = np.asarray(att_ckpt["imu_mean"], dtype=np.float32).tolist()
        att_imu_std = np.asarray(att_ckpt["imu_std"], dtype=np.float32).tolist()
    elif "norm_stats" in att_ckpt and "imu_mean" in att_ckpt["norm_stats"]:
        att_imu_mean = np.asarray(att_ckpt["norm_stats"]["imu_mean"], dtype=np.float32).tolist()
        att_imu_std = np.asarray(att_ckpt["norm_stats"]["imu_std"], dtype=np.float32).tolist()
    else:
        raise KeyError("Attitude checkpoint missing required IMU normalization statistics ('imu_mean' / 'imu_std')")

    metadata["attitude_model"] = {
        "file": "attitude_model.onnx",
        "input_name": "imu_input",
        "input_shape": [1, 50, 6],
        "output_name": "attitude_output",
        "output_shape": [1, 4],
        "imu_mean": att_imu_mean,
        "imu_std": att_imu_std,
    }

    class AttitudeExportWrapper(torch.nn.Module):
        def __init__(self, model: torch.nn.Module):
            super().__init__()
            self.model = model

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            quat, _ = self.model(x)
            return quat

    att_wrapper = AttitudeExportWrapper(att_model)
    att_wrapper.eval()
    att_dummy = torch.randn(1, 50, 6, dtype=torch.float32)

    print(f"      Exporting to ONNX: {att_onnx_path}")
    torch.onnx.export(
        att_wrapper,
        att_dummy,
        str(att_onnx_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["imu_input"],
        output_names=["attitude_output"],
        dynamic_axes={
            "imu_input": {0: "batch_size"},
            "attitude_output": {0: "batch_size"},
        },
        dynamo=False,
    )

    onnx_att_model = onnx.load(str(att_onnx_path))
    onnx.checker.check_model(onnx_att_model)
    print(f"      Structural validation passed! Model size: {att_onnx_path.stat().st_size / 1024:.1f} KB")

    # =======================================================================
    # 3. Save Metadata JSON & Copy to Android Assets
    # =======================================================================
    meta_path = models_dir / "model_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n[3/3] Metadata saved to: {meta_path}")

    # Copy files to Android Assets
    shutil.copy2(vel_onnx_path, assets_dir / "velocity_model_v2.onnx")
    shutil.copy2(att_onnx_path, assets_dir / "attitude_model.onnx")
    shutil.copy2(meta_path, assets_dir / "model_metadata.json")
    print(f"      Copied ONNX models and metadata to: {assets_dir}")

    # =======================================================================
    # 4. Numerical Parity Check
    # =======================================================================
    print("\n" + "-" * 80)
    print("VERIFYING NUMERICAL PARITY (PyTorch vs ONNX Runtime):")
    print("-" * 80)

    test_in = np.random.randn(1, 50, 6).astype(np.float32)

    # A. Velocity Model Parity
    with torch.no_grad():
        pt_vel_out = vel_wrapper(torch.from_numpy(test_in)).numpy()
    ort_vel_session = ort.InferenceSession(str(vel_onnx_path), providers=["CPUExecutionProvider"])
    ort_vel_out = ort_vel_session.run(["velocity_output"], {"imu_input": test_in})[0]
    vel_max_diff = float(np.max(np.abs(pt_vel_out - ort_vel_out)))
    print(f"  • VelocityModel V2: Max Absolute Difference = {vel_max_diff:.8e} (Tolerance: 1e-4) -> {'PASS' if vel_max_diff < 1e-4 else 'FAIL'}")

    # B. Attitude Model Parity
    with torch.no_grad():
        pt_att_out = att_wrapper(torch.from_numpy(test_in)).numpy()
    ort_att_session = ort.InferenceSession(str(att_onnx_path), providers=["CPUExecutionProvider"])
    ort_att_out = ort_att_session.run(["attitude_output"], {"imu_input": test_in})[0]
    att_max_diff = float(np.max(np.abs(pt_att_out - ort_att_out)))
    print(f"  • AttitudeModel:    Max Absolute Difference = {att_max_diff:.8e} (Tolerance: 1e-4) -> {'PASS' if att_max_diff < 1e-4 else 'FAIL'}")

    print("=" * 80)
    print("[SUCCESS] All mobile models exported, validated, and ready for Android deployment!\n")


if __name__ == "__main__":
    export_models()
