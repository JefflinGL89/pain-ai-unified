"""
Real-Time Multimodal Surgical Pain & rPPG Contactless Monitoring Station
Fuses Module A (ConvNeXt-Tiny Facial Regressor), Module B (MediaPipe Face Mesh + POS rPPG),
and Module C (Quality-Gated Fusion Engine) into an interactive clinical monitor.
"""

from collections import deque
from datetime import datetime
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

# Add src directories to module search path dynamically
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR / "src" / "module_a_facial"))
sys.path.append(str(ROOT_DIR / "src" / "module_b_rppg"))
sys.path.append(str(ROOT_DIR / "src" / "module_c_fusion"))

from model import ConvNeXtPainRegressor
from face_mesh_roi import FaceMeshTracker
from pos_algorithm import POSrPPGProcessor
from fusion_model import MultimodalPainFusionEngine

# CLAHE for ambient lighting equalization
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def preprocess_face_crop(face_crop, device):
    """Applies CLAHE equalization in LAB color space and standard ImageNet normalization."""
    if face_crop is None or face_crop.size == 0:
        return None

    face_resized = cv2.resize(face_crop, (224, 224), interpolation=cv2.INTER_AREA)

    # Lighting equalization on luminance channel
    lab = cv2.cvtColor(face_resized, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq = clahe.apply(l)
    face_eq = cv2.cvtColor(cv2.merge((l_eq, a, b)), cv2.COLOR_LAB2RGB)

    img_norm = face_eq.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_norm = (img_norm - mean) / std

    tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"  CONTACTLESS MULTIMODAL SURGICAL PAIN MONITOR")
    print(f"  Execution Device: {device}")
    print(f"=======================================================\n")

    # 1. Initialize Module A (Facial Regressor)
    checkpoint_path = ROOT_DIR / "checkpoints" / "best_convnext_pain_model.pth"
    if not checkpoint_path.exists():
        print(f"[Error] Checkpoint not found at: {checkpoint_path}")
        return

    model = ConvNeXtPainRegressor(pretrained=False).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"[Module A] Loaded ConvNeXt-Tiny weights from {checkpoint_path.name}")

    # 2. Initialize Module B (Face Mesh Tracker & POS Processor)
    face_tracker = FaceMeshTracker()
    rppg_processor = POSrPPGProcessor(target_fs=30.0, window_sec=10.0)

    # 3. Initialize Module C (Fusion Engine)
    fusion_engine = MultimodalPainFusionEngine(base_phys_weight=0.20)

    # Buffers & UI variables
    pulse_graph_buffer = deque([0.0] * 140, maxlen=140)
    calibration_samples = []
    raw_pain_val = 0.0
    sensitivity_gain = 2.8
    show_mesh = False

    # Connect to camera
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[Error] Could not open webcam at index 0. Testing index 1...")
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("[Fatal] No video capture device found.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("\n-------------------------------------------------------")
    print("  KEYBOARD SHORTCUTS (Focus the OpenCV window):")
    print("  [c] - Zero-Calibrate Baseline (Keep relaxed face!)")
    print("  [m] - Toggle Facial Landmark Mesh Overlay")
    print("  [+] - Increase Pain Sensitivity Gain")
    print("  [-] - Decrease Pain Sensitivity Gain")
    print("  [s] - Save Clinical Snapshot / PDF Report")
    print("  [q] - Quit System")
    print("-------------------------------------------------------\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        now = time.perf_counter()
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # --- Step 1: Face Mesh Tracking & ROI Extraction ---
        tracking = face_tracker.process_frame(frame)

        if tracking["face_detected"] and tracking["bbox"] is not None:
            x1, y1, x2, y2 = tracking["bbox"]
            face_crop = frame[y1:y2, x1:x2]

            # --- Step 2: Module A Facial Inference ---
            tensor = preprocess_face_crop(face_crop, device)
            if tensor is not None:
                with torch.no_grad():
                    # Model outputs [0.0, 1.0], scaled to 0-16 PSPI
                    raw_pain_val = model(tensor).item() * 16.0

                # Auto-calibrate neutral baseline over initial 25 frames
                if not fusion_engine.neutral_face_offset:
                    calibration_samples.append(raw_pain_val)
                    if len(calibration_samples) >= 25:
                        neutral_median = float(np.median(calibration_samples))
                        fusion_engine.set_neutral_face(neutral_median)
                        print(f"[Auto-Calibration] Neutral Face Baseline set: {neutral_median:.2f}")

            # --- Step 3: Module B POS rPPG Sample Feed ---
            if tracking["skin_rgb_mean"] is not None:
                rppg_processor.add_sample(tracking["skin_rgb_mean"], now)

            # Draw Face Bounding Box & Forehead Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 230, 100), 2)
            if tracking["forehead_box"] is not None:
                fx1, fy1, fx2, fy2 = tracking["forehead_box"]
                cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (255, 200, 0), 1)

            # Optional landmark mesh overlay
            if show_mesh and tracking["landmarks_px"] is not None:
                for pt in tracking["landmarks_px"][::3]:  # Subsample for rendering speed
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 1, (0, 255, 255), -1)

        # --- Step 4: Compute rPPG Metrics ---
        rppg_data = rppg_processor.compute_metrics()
        current_bpm = None
        current_resp = None
        current_sqi = 0.0
        current_hrv = None

        if rppg_data is not None:
            pulse_graph_buffer.append(rppg_data["wave_val"])
            if rppg_data["bpm"] is not None:
                current_bpm = rppg_data["bpm"]
                current_resp = rppg_data["resp_rate"]
                current_sqi = rppg_data["sqi"]
                current_hrv = rppg_data["hrv_sdnn"]

                # Lock baseline resting heart rate
                if fusion_engine.baseline_bpm is None and rppg_data["is_stable"]:
                    fusion_engine.set_baseline_vitals(current_bpm, current_resp)
                    print(f"[Auto-Calibration] Resting Heart Rate Baseline locked: {current_bpm:.1f} BPM")

        # --- Step 5: Module C Multimodal Quality-Aware Fusion ---
        motion = tracking.get("motion_score", 0.0)
        fusion_out = fusion_engine.process(
            raw_face_score=raw_pain_val,
            sensitivity_gain=sensitivity_gain,
            bpm=current_bpm,
            resp_rate=current_resp,
            sqi=current_sqi,
            motion_score=motion,
            hrv_sdnn=current_hrv
        )

        final_pspi = fusion_out["pspi_score"]
        status_text = fusion_out["status_text"]
        status_color = fusion_out["status_color_bgr"]
        cpot_score = fusion_out["cpot_score"]
        nrs_score = fusion_out["nrs_score"]

        # --- Step 6: Render Clinical HUD ---
        # Dark HUD background panel
        hud = frame.copy()
        cv2.rectangle(hud, (15, 15), (460, 275), (18, 18, 18), -1)
        frame = cv2.addWeighted(hud, 0.85, frame, 0.15, 0)

        # Header Title
        cv2.putText(frame, "SURGICAL PAIN & rPPG MONITOR", (28, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # Status Badge
        cv2.putText(frame, f"STATUS: {status_text}", (28, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, status_color, 2)

        # Clinical Scores Banner
        scores_line = f"PSPI: {final_pspi:.2f}/16 | CPOT: {cpot_score}/8 | NRS: {nrs_score}/10"
        cv2.putText(frame, scores_line, (28, 102),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 230, 230), 1)

        # Visual Pain Meter Bar
        bar_fill = int(np.clip(final_pspi / 16.0, 0.0, 1.0) * 410)
        cv2.rectangle(frame, (28, 115), (438, 132), (50, 50, 50), -1)
        cv2.rectangle(frame, (28, 115), (28 + bar_fill, 132), status_color, -1)

        # Model Diagnostics Subtitle
        offset_str = f"{fusion_engine.neutral_face_offset:.2f}" if fusion_engine.neutral_face_offset else "Calibrating..."
        cv2.putText(frame, f"Raw AI: {raw_pain_val:.2f} | Neutral: {offset_str} | Gain: {sensitivity_gain:.1f}x",
                    (28, 152), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 180, 180), 1)

        # Vitals Readout
        if current_bpm is not None:
            base_hr_str = f"{fusion_engine.baseline_bpm:.0f}" if fusion_engine.baseline_bpm else "Locking..."
            cv2.putText(frame, f"Heart Rate: {current_bpm:.1f} BPM (Resting: {base_hr_str})", (28, 182),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.54, (0, 220, 255), 2)

            resp_str = f"{current_resp:.1f} BrPM" if current_resp else "--"
            hrv_str = f"{current_hrv:.1f} ms" if current_hrv else "--"
            sqi_str = f"{int(current_sqi * 100)}%"
            cv2.putText(frame, f"Resp: {resp_str} | HRV: {hrv_str} | SQI: {sqi_str}", (28, 208),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (180, 255, 180), 1)
        else:
            prog_val = 0
            if rppg_data:
                prog_val = int(rppg_data.get("calibrating_progress", 0.0) * 100)
            cv2.putText(frame, f"Calibrating rPPG Pulse... {prog_val}%", (28, 182),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 200, 255), 2)
            cv2.putText(frame, "Remain stationary for cardiac baseline lock", (28, 208),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)

        # Motion & Quality Indicator
        mot_str = f"Motion: {motion:.1f} px/f"
        w_str = f"Weights: Face {fusion_out['face_weight']*100:.0f}% / Phys {fusion_out['phys_weight']*100:.0f}%"
        cv2.putText(frame, f"{mot_str} | {w_str}", (28, 232),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 140), 1)

        # Keyboard Controls legend
        cv2.putText(frame, "[c] Zero Baseline | [m] Mesh | [+/-] Gain | [s] Snapshot | [q] Quit",
                    (28, 258), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (140, 200, 255), 1)

        # --- Step 7: Live Oscilloscope Waveform ---
        gx, gy, gw, gh = w - 300, h - 105, 280, 85
        cv2.rectangle(frame, (gx, gy), (gx + gw, gy + gh), (18, 18, 18), -1)
        cv2.rectangle(frame, (gx, gy), (gx + gw, gy + gh), (60, 60, 60), 1)
        cv2.putText(frame, "Live rPPG Blood Volume Pulse", (gx + 8, gy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)

        mid_y = gy + (gh // 2)
        n_pts = len(pulse_graph_buffer)
        for i in range(1, n_pts):
            x_prev = gx + int((i - 1) * (gw / n_pts))
            x_curr = gx + int(i * (gw / n_pts))
            y_prev = int(mid_y - pulse_graph_buffer[i - 1] * (gh * 0.42))
            y_curr = int(mid_y - pulse_graph_buffer[i] * (gh * 0.42))
            cv2.line(frame, (x_prev, y_prev), (x_curr, y_curr), (0, 80, 255), 2)

        cv2.imshow("Contactless Surgical Pain & rPPG Monitor", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("c"):
            fusion_engine.set_neutral_face(raw_pain_val)
            if current_bpm is not None:
                fusion_engine.set_baseline_vitals(current_bpm, current_resp)
            print(f"[Manual Calibration] Neutral Offset: {raw_pain_val:.2f} | Resting HR: {current_bpm}")
        elif key == ord("m"):
            show_mesh = not show_mesh
        elif key in (ord("+"), ord("=")):
            sensitivity_gain = min(10.0, sensitivity_gain + 0.5)
        elif key in (ord("-"), ord("_")):
            sensitivity_gain = max(0.5, sensitivity_gain - 0.5)
        elif key == ord("s"):
            snap_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            snap_path = ROOT_DIR / f"clinical_snapshot_{snap_time}.png"
            cv2.imwrite(str(snap_path), frame)
            print(f"[Snapshot Saved] -> {snap_path.resolve()}")

    cap.release()
    cv2.destroyAllWindows()
    print("Monitor closed cleanly.")


if __name__ == "__main__":
    main()
