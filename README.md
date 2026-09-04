# NAVIGATE 2.0: AI-Augmented Smartphone Dead Reckoning & ES-EKF Navigation System

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/Tests-262%20Passed%20(100%25)-brightgreen.svg)](tests/)
[![Architecture](https://img.shields.io/badge/Architecture-Frozen%20(Phase%209)-success.svg)](#core-system-architecture)

**NAVIGATE 2.0** is an AI-augmented dead-reckoning and navigation system designed for GNSS-denied environments (e.g., vehicular tunnels, underground roadways, dense urban canyons, and GNSS signal jamming/spoofing). 

By fusing high-rate smartphone IMU sensors (accelerometers and gyroscopes), deep neural velocity and attitude models, a 15-state Error-State Extended Kalman Filter (ES-EKF), and external OpenStreetMap (OSM) multi-candidate road network geometry constraints, NAVIGATE 2.0 eliminates double-integration drift and delivers robust real-time vehicle localization.

---

## 1. System Architecture

```
                       [ Smartphone Sensor Stream / Android / Replay ]
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      ▼                                               ▼
                 IMUSample                                       GNSSSample
             (10 Hz Accel + Gyro)                            (WGS84 Lat/Lon Fix)
                      │                                               │
                      └───────────────────────┬───────────────────────┘
                                              ▼
                                 RealTimeNavigationEngine
                      ┌───────────────────────────────────────────────┐
                      │ 1. Monotonic Timestamp & Gap Validation       │
                      │ 2. ES-EKF Nominal State Propagation (10 Hz)   │
                      │ 3. 5-Second Rolling Buffer (50 IMU Samples)   │
                      │ 4. AI Inference Scheduling (1-Second Stride): │
                      │    • VelocityModel V2 (Forward Speed)         │
                      │    • AttitudeModel (Relative Quaternion)      │
                      │ 5. Non-Holonomic Constraints (NHC)            │
                      │ 6. GNSS Fusion & Blackout State Machine       │
                      └───────────────────────┬───────────────────────┘
                                              │
                                     GNSS Outage Active?
                                        /           \
                                    NO /             \ YES
                                      /               \
                                     ▼                 ▼
                             Standard GNSS         OSMRoadCache
                             Measurement          (Spatial Grid)
                                 Update                │
                                      │                ▼
                                      │          OSMMapMatcher
                                      │      (Candidate Scoring &
                                      │       Temporal Continuity)
                                      │                │
                                      │                ▼
                                      │          Soft Road Gating
                                      │        & EKF Pseudo-Update
                                      │                │
                                      └───────┬────────┘
                                              ▼
                                       NavigationState
                             (ENU Position, Velocity, Heading,
                              WGS84 Lat/Lon, Latency, Diagnostics)
```

---

## 2. Authoritative System Definitions

| Pipeline Version | Core Technologies | Road Geometry Source | GNSS Blackout Handling |
|---|---|---|---|
| **Baseline DR** | Raw IMU Strapdown Integration | None | Unconstrained double-integration drift ($\sim t^2$) |
| **Version A** | AI (`VelocityModel V2` + `AttitudeModel`) + ES-EKF + NHC | None | AI forward speed + relative attitude + non-holonomic constraints |
| **Version B** | Version A + Legacy Pre-Blackout Road Gating | Past GNSS ($t < t_{\text{blackout}}$) | Extrapolates strictly past pre-outage GPS polyline |
| **Version C (OSM)** | Version A + Multi-Candidate OpenStreetMap Gating | OpenStreetMap (Overpass / Local Cache) | Dynamic map lookup $(\hat{E}, \hat{N}) \to (\text{lat}, \text{lon})$, candidate scoring, soft pseudo-measurement |

> [!NOTE]
> **Zero Ground-Truth Leakage Guarantee**: Version C queries OpenStreetMap using strictly the *current estimated filter state* $(\hat{E}_t, \hat{N}_t) \to (\text{est\_lat}_t, \text{est\_lon}_t)$. True GNSS measurements are strictly suppressed during blackout intervals, and future trajectory waypoints are never accessed.

---

## 3. Final Four-Way Benchmark Results

Evaluated across standardized blackout durations ($5\text{s}, 10\text{s}, 30\text{s}, 60\text{s}$) on the IO-VNBD benchmark:

| Blackout Duration | Baseline DR Final Error | Version A [AI+EKF] Final Error | Version B [Legacy Road] Final Error | **Version C [OSM Road] Final Error** | **VC vs Baseline Improvement** | **VC vs Version A Improvement** |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **5s Outage** | 37.51 m | 20.04 m | 20.01 m | **11.32 m** | **+69.8%** | **+43.5%** |
| **10s Outage** | 71.67 m | 41.81 m | 41.55 m | **19.40 m** | **+72.9%** | **+53.6%** |
| **30s Outage** | 230.57 m | 158.14 m | 157.73 m | **44.58 m** | **+80.7%** | **+71.8%** |
| **60s Outage** | 575.53 m | 386.71 m | 385.79 m | **134.03 m** | **+76.7%** | **+65.3%** |

### Additional Version C Metrics:
- **RMSE Error**: $9.59\text{m}$ (5s), $13.30\text{m}$ (10s), $32.69\text{m}$ (30s), $72.77\text{m}$ (60s).
- **Relative Drift**: $17.0\%$ (5s), $17.3\%$ (10s), $17.7\%$ (30s), $29.0\%$ (60s).
- **Average Soft Correction Magnitude**: $0.45\text{m}$ (5s) to $0.82\text{m}$ (60s).

---

## 4. Real-Time Execution & Latency Telemetry

*Measured via `scripts/benchmark_realtime_engine.py` (1,000 samples @ 10 Hz = 100s stream on standard CPU):*

| Real-Time Metric | Value | Status |
|---|:---:|:---:|
| **Throughput** | **$2,063.1\text{ samples/sec}$** | $206.3\times$ faster than 10 Hz real-time requirements |
| **100 ms Deadline Compliance** | **$100.00\%$** | Zero deadline overruns |
| **Mean Navigation Tick Latency** | **$0.460\text{ ms}$** | Nominal step propagation |
| **P95 Tick Latency** | **$3.904\text{ ms}$** | Includes AI window inference |
| **P99 Tick Latency** | **$4.893\text{ ms}$** | Peak stride latency |
| **Max Tick Latency** | **$6.215\text{ ms}$** | Well below $100\text{ ms}$ target |
| **VelocityModel V2 Inference** | Mean: $2.128\text{ ms}$ \| P95: $2.916\text{ ms}$ | PyTorch CPU Inference |
| **AttitudeModel Inference** | Mean: $1.297\text{ ms}$ \| P95: $2.173\text{ ms}$ | PyTorch CPU Inference |
| **ES-EKF Step Propagation** | Mean: $0.071\text{ ms}$ \| P95: $0.144\text{ ms}$ | Kalman prediction step |

---

## 5. Repository Structure

```
NAVIGATE_2.0/
├── data/
│   └── processed/
│       └── iovnbd_smoke_test.npz     # Processed evaluation benchmark session
├── models/
│   ├── attitude_model.pt             # Trained relative quaternion attitude model
│   └── velocity_model_v2.pt          # Trained 1D ResNet + BiGRU forward velocity model
├── results/
│   ├── final/                        # Authoritative final benchmark artifacts & figures
│   │   ├── final_benchmark_results.json
│   │   ├── final_benchmark_results.csv
│   │   ├── blackout_error_comparison_bar.png
│   │   └── trajectory_comparison.png
│   └── ai_iekf_osm/                  # Phase 6 & 7 raw comparison logs
├── scripts/
│   ├── run_final_demo.py             # End-to-end interactive real-time demo
│   ├── benchmark_realtime_engine.py  # Latency & throughput profiling benchmark
│   ├── plot_final_results.py         # Visualizations & summary generator
│   └── run_ai_iekf_osm_evaluation.py # 4-way comparative evaluation runner
├── src/
│   └── navigate/
│       ├── sensor_types.py           # Standardized IMU/GNSS data interfaces
│       ├── realtime_engine.py        # RealTimeNavigationEngine orchestration
│       ├── realtime_replay.py        # Streaming replay source
│       ├── iekf_tracker.py           # 15-state Error-State EKF (ES-EKF)
│       ├── map_matching.py           # Core geometric polyline projection & gating
│       ├── osm_road_provider.py      # OpenStreetMap Overpass client
│       ├── osm_road_adapter.py       # WGS84 -> Local ENU polyline converter
│       ├── osm_road_cache.py         # Spatial/temporal in-memory road cache
│       ├── osm_map_matcher.py        # Multi-candidate matcher with continuity
│       ├── ai_iekf_pipeline.py       # Version A pipeline implementation
│       └── ai_iekf_road_pipeline.py  # Version B & C pipeline implementation
├── tests/                            # Complete offline unit & integration test suite (262 tests)
├── pytest.ini                        # Pytest configuration
└── README.md                         # Project documentation
```

---

## 6. Reproducibility & Quickstart Commands

### A. Run Full Test Suite (100% Offline)
```bash
pytest tests/
```

### B. Run End-to-End Real-Time Demo
```bash
python scripts/run_final_demo.py
```

### C. Run Real-Time Engine Latency Benchmark
```bash
python scripts/benchmark_realtime_engine.py --samples 1000
```

### D. Run Full Four-Way Benchmark Evaluation
```bash
python scripts/run_ai_iekf_osm_evaluation.py --data data/processed/iovnbd_smoke_test.npz
```

### E. Generate Final Visualizations & Results Artifacts
```bash
python scripts/plot_final_results.py
```

---

## 7. Environment & Dependencies

- **Python**: 3.10, 3.11, 3.12, 3.13 (Tested on Python 3.13.15 Windows AMD64)
- **PyTorch**: `>= 2.0.0`
- **NumPy**: `>= 1.24.0`
- **Matplotlib**: `>= 3.7.0`
- **Pytest**: `>= 7.0.0`

---

## 8. Limitations & Future Work

1. **Controlled Corridor Evaluation**: The default repository smoke test evaluates a deterministic northbound corridor. Realistic multi-lane and branching topologies are verified via offline synthetic fixtures in `tests/test_osm_map_matcher_realistic.py`.
2. **Cold-Start GNSS Initialization**: The filter requires an initial GNSS coordinate at session start ($t=0$) to establish the local tangent ENU origin.
3. **Android Sensor Boundary**: The real-time engine is fully platform-independent and ready for mobile integration (via Chaquopy, PyMob, or ONNX Runtime).
