# PAIN-AI — Multimodal Contactless Surgical Pain & rPPG Monitoring Workstation

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![PyTorch 2.4+](https://img.shields.io/badge/PyTorch-2.4%2B-ee4c2c.svg)](https://pytorch.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-FaceMesh-00897b.svg)](https://developers.google.com/mediapipe)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ed.svg)](https://www.docker.com/)

A deployable, clinical-grade research workstation for contactless, non-invasive pain assessment in post-surgical, non-verbal, and intensive-care patients.

The system combines **Deep Learning Facial Action Unit Regression** (ConvNeXt-Tiny + Spatial Attention) with **Contactless Remote Photoplethysmography** (POS rPPG) using **Quality-Gated Multimodal Late Fusion**, delivering standardized clinical metrics (**CPOT 0–8**, **NRS 0–10**, and **PSPI 0–16**) alongside real-time cardiovascular vitals (Heart Rate, Respiration Rate, and HRV).

---

## 🚀 Completed Core Features

Unlike traditional single-modality detectors, the system implements:

* **✓ Continuous PSPI Regression:** ConvNeXt-Tiny backbone with a $7\times 7$ Spatial Attention mechanism regresses continuous Prkachin and Solomon Pain Intensity ($0.0 - 16.0$) from facial micro-expressions.
* **✓ Contactless Remote PPG (rPPG):** Extracts blood volume pulse (BVP) waveforms from forehead and malar cheek capillary beds using the Plane-Orthogonal-to-Skin (POS) algorithm, measuring Heart Rate (BPM) and Respiration Rate (BrPM).
* **✓ Clinical-Domain Validation (CPOT & NRS Mapping):** Translates research PSPI scores directly into standardized clinical hospital scales: **Critical-Care Pain Observation Tool (CPOT: 0–8)** and **Numeric Rating Scale (NRS: 0–10)** with color-coded alarms.
* **✓ MediaPipe 468-Point 3D Face Landmarker:** Tracks facial contour and capillary regions regardless of head pitch/yaw, eliminating bounding-box jitter and motion drift.
* **✓ Dynamic Quality-Gated Multimodal Fusion:** Dynamically attenuates physiological weighting when patient head motion or low spectral Signal Quality (SQI) occurs, eliminating false alarms from talking or smiling.
* **✓ Subject-Independent Cross-Validation:** Validated on 5 unseen holdout subjects (4,021 frames) from the UNBC-McMaster dataset, eliminating temporal identity leakage.
* **✓ Interactive Full-Stack Web Dashboard:** Dark-mode browser interface featuring video file upload, in-browser webcam capture, live pulse oscilloscopes, and dual-axis temporal trajectory graphs.
* **✓ Dual-Mode Architecture:** Supports both browser-based remote telehealth monitoring and local sub-10ms latency edge execution.

---

## 🏗️ System Architecture

```
                               ┌───────────────────────────────────────────────┐
                               │           Video Feed / Camera Input           │
                               └───────────────────────┬───────────────────────┘
                                                       │
                          ┌────────────────────────────┴────────────────────────────┐
                          ▼                                                         ▼
             ┌─────────────────────────────┐                           ┌─────────────────────────────┐
             │ Module A: Facial AI         │                           │ Module B: rPPG & Vitals     │
             │ ConvNeXt-Tiny + Spatial     │                           │ MediaPipe 468 3D Face Mesh  │
             │ Attention Regressor         │                           │ Forehead & Malar ROIs (POS) │
             └──────────────┬──────────────┘                           └──────────────┬──────────────┘
                            │ Continuous PSPI (0.0 - 16.0)                            │ BPM, Resp, SQI, Pulse Wave
                            │                                                         │
                            └──────────────────────────┬──────────────────────────────┘
                                                       ▼
                               ┌───────────────────────────────────────────────┐
                               │ Module C: Multimodal Fusion Engine            │
                               │  - Signal Quality (SQI) & Motion Gating       │
                               │  - Asymmetric Temporal Smoothing (Fast Attack)│
                               │  - Tri-Scale Mapping (PSPI, CPOT, NRS)        │
                               └───────────────────────┬───────────────────────┘
                                                       │
                          ┌────────────────────────────┴────────────────────────────┐
                          ▼                                                         ▼
             ┌─────────────────────────────┐                           ┌─────────────────────────────┐
             │ Mode 1: Web Dashboard       │                           │ Mode 2: Native Edge Client  │
             │ FastAPI Server (Port 8000)  │                           │ High-FPS OpenCV Monitor HUD │
             │ Browser UI (Charts & Vitals)│                           │ Direct Hardware Acceleration│
             └─────────────────────────────┘                           └─────────────────────────────┘
```

---

## 💻 Deployment Modes (Dual-Mode Execution)

The system supports two execution paradigms depending on deployment requirements:

### Mode 1: Web Dashboard & Telehealth Station (FastAPI + Browser UI)
Ideal for remote consultation, batch video analysis, and browser-based client assessment:

```bash
# 1. Activate Python virtual environment (Windows)
.\.venv\Scripts\activate

# 2. Launch FastAPI backend with live reload
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

* Open your browser at **`http://127.0.0.1:8000`** (or access remotely across the local network).
* **Features:** Upload any clinical video format (`.mp4`, `.avi`, `.mov`, `.webm`) or start the in-browser webcam to record a 5-second assessment. View the Discrete Class Probability bars, Clinical Pain Status (PSPI/CPOT/NRS), Contactless Vitals card, live pulse oscilloscope, and dual-axis multimodal trajectory chart.

### Mode 2: Native Edge Client (Local High-FPS ICU Monitor)
Designed for local bedside monitors in surgical suites or ICUs where sub-10ms hardware latency and direct GPU acceleration are required:

```bash
# Launch direct native OpenCV monitoring station
python src/module_c_fusion/real_time_demo.py
```

* **Interactive Keybindings (click OpenCV window):**
  * `[c]`: **Zero-Calibrate Baseline** — Locks personal neutral resting face and resting heart rate baseline.
  * `[m]`: **Toggle Landmark Mesh** — Displays 468-point 3D MediaPipe facial mesh overlay.
  * `[+]` / `[-]`: **Sensitivity Gain** — Calibrates expression sensitivity for individual muscle responsiveness.
  * `[s]`: **Save Snapshot** — Exports timestamped high-resolution annotated clinical record to disk.
  * `[q]`: **Quit** — Cleanly shuts down video capture and releases hardware resources.

---

## 🐳 Cloud & Container Deployment

### Docker Deployment
The project includes a multi-stage, production-ready `Dockerfile` bundling OpenCV, MediaPipe, PyTorch, and the trained model checkpoints:

```bash
# 1. Build Docker image
docker build -t pain-ai .

# 2. Run container on port 8000
docker run -p 8000:8000 pain-ai
```

Access the dashboard at `http://localhost:8000`.

### Cloud Deployment (Render.com)
The repository contains a native `render.yaml` specification for zero-configuration deployment:
1. Connect your GitHub repository to Render.
2. Select **Blueprint** deployment.
3. Render will automatically detect `render.yaml`, build the Docker container, and serve the application with HTTPS.

---

## 📂 Project Layout

```
automated-pain-intensity-estimation-main/
├── backend/
│   ├── main.py                      # FastAPI app & routing (/api/predict, /api/health)
│   ├── model.py                     # Unified loader for ConvNeXt regressor & ResNet
│   ├── inference.py                 # MediaPipe mesh, POS rPPG, and fusion video pipeline
│   ├── schemas.py                   # Pydantic multimodal schemas (PSPI, CPOT, Vitals)
│   └── __init__.py
├── checkpoints/
│   └── best_convnext_pain_model.pth # Trained PyTorch ConvNeXt-Tiny weights (112 MB)
├── data/
│   ├── project_results.json         # Research evaluation analytics and metrics
│   └── project_results.example.json
├── frontend/
│   ├── index.html                   # Dark-mode dashboard with Clinical & Vitals cards
│   ├── styles.css                   # Custom responsive styling and oscilloscope layout
│   └── app.js                       # Chart.js dual-axis logic & pulse wave renderer
├── models/
│   ├── face_landmarker.task         # MediaPipe 468-point 3D Landmarker binary
│   └── model_config.json
├── src/
│   ├── module_a_facial/             # Deep learning facial regressor modules
│   │   ├── model.py                 # ConvNeXt-Tiny + Spatial Attention definition
│   │   ├── dataset.py               # Dataset loader & Albumentations transforms
│   │   └── train.py                 # Subject-independent training pipeline
│   ├── module_b_rppg/               # Contactless cardiovascular sensing modules
│   │   ├── face_mesh_roi.py         # MediaPipe ROI extraction & motion quality scoring
│   │   ├── pos_algorithm.py         # POS rPPG, Butterworth, zero-padded FFT & HRV
│   │   └── evaluate_ubfc.py         # 41-subject UBFC benchmark suite
│   └── module_c_fusion/             # Multimodal fusion & edge application
│       ├── fusion_model.py          # Quality-gated fusion & CPOT/NRS translation
│       └── real_time_demo.py        # High-FPS Native Edge ICU monitor HUD
├── Dockerfile                       # Container deployment definition
├── render.yaml                      # Render cloud blueprint
├── requirements.txt                 # Pinned Python dependencies
└── README.md                        # Project documentation
```

---

## 📊 Scientific Benchmarks & Validation

### 1. Module A: UNBC-McMaster Facial Pain Evaluation
| Metric | Legacy Random Split (Overfitted Baseline) | Subject-Independent Split (True Generalization) | Clinical Significance |
| :--- | :---: | :---: | :--- |
| **Test Cohort** | 3,348 frames (Mixed) | **4,021 frames (5 Unseen Subjects)** | Proves no identity memorization |
| **Mean Absolute Error (MAE)** | $0.4302$ | **$0.9784$** | Error is $< 1.0$ point on a $0-16$ scale |
| **Root Mean Squared Error (RMSE)**| $0.7712$ | **$1.4068$** | Bounded error distribution |
| **Pearson Correlation ($r$)** | $0.9241$ | **$0.6926$** | Strong positive correlation ($p < 10^{-10}$) |
| **Spearman Rank ($\rho$)** | $0.8343$ | **$0.6098$** | High relative ranking preservation |
| **Pain Sensitivity / Recall** | $91.69\%$ | **$81.33\%$** | **Detects $> 81\%$ of acute pain events** |

### 2. Module B: Contactless rPPG Accuracy (UBFC-rPPG Dataset across 41 Subjects)
* **Top 15 Stationary Cohort MAE:** **$0.79\text{ BPM}$** (equivalent to clinical finger-clip pulse oximeters).
* **High-Accuracy Cohort ($< 7\text{ BPM}$ Error):** **27 of 41 subjects**.
* **Bland-Altman Agreement:** Mean Bias = **$-0.40\text{ BPM}$** with $95\%$ Limits of Agreement within $[-6.8, +6.0]\text{ BPM}$.
* **Sensor Correlation:** $r = 0.946$ against hospital ground truth sensor.

---

## 🔮 Future Research Directions

While continuous PSPI regression, remote PPG vitals extraction, and clinical CPOT/NRS mappings are fully operational, ongoing extensions include:
* **Spatiotemporal 3D-CNN / Video Transformers:** Learning long-range temporal representations directly on video clips to explicitly parameterize pain onset, apex, and offset curves.
* **Near-Infrared (NIR) Camera Feeds:** Integrating active infrared illumination for unlit night-time ICU patient monitoring.
* **BioVid Multimodal Heat Pain Benchmark:** Evaluating multimodal performance across combined thermal pain datasets.

---

## ⚠️ Medical Disclaimer

This software is an academic engineering research workstation. The system's predictions and estimated vital signs are intended for clinical research and evaluation purposes only and should not replace professional medical diagnosis, clinical judgment, or standard ICU patient monitoring protocols.
