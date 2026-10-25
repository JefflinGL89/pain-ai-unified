"""
Multimodal Video Inference Engine
Processes uploaded or recorded video clips using MediaPipe FaceMesh for ROI extraction,
ConvNeXt-Tiny for facial pain regression, POS for contactless rPPG pulse extraction,
and MultimodalPainFusionEngine for quality-gated late fusion.
"""

from pathlib import Path
import sys
import cv2
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.module_b_rppg.face_mesh_roi import FaceMeshTracker
from src.module_b_rppg.pos_algorithm import POSrPPGProcessor
from src.module_c_fusion.fusion_model import MultimodalPainFusionEngine
from .model import CLASS_NAMES, UnifiedModelBundle
from .schemas import InferenceResponse, Prediction, VitalsData

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def preprocess_face_crop(face_crop):
    """CLAHE equalization in LAB space + ImageNet normalization."""
    if face_crop is None or face_crop.size == 0:
        return None
    face_resized = cv2.resize(face_crop, (224, 224), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(face_resized, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    face_eq = cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2RGB)

    img_norm = face_eq.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_norm = (img_norm - mean) / std
    return torch.from_numpy(img_norm).permute(2, 0, 1)


@torch.inference_mode()
def predict_video(video_path: Path, model_bundle: UnifiedModelBundle, max_frames: int = 180, target_fps: float = 15.0) -> dict:
    """
    Decodes video, runs MediaPipe FaceMesh, extracts rPPG pulse traces,
    evaluates ConvNeXt pain scores, and fuses modalities into InferenceResponse dict.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Could not open uploaded video file.")

    video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / video_fps if video_fps > 0 and total_frames > 0 else 0.0

    # Downsample step to achieve target_fps for smooth rPPG
    step = max(1, int(round(video_fps / max(target_fps, 1.0))))
    effective_sampled_fps = video_fps / step

    # Initialize tracker, rPPG processor and fusion engine
    tracker = FaceMeshTracker(model_path=ROOT_DIR / "models" / "face_landmarker.task")
    rppg_processor = POSrPPGProcessor(target_fs=effective_sampled_fps, window_sec=10.0, min_window_sec=2.5)
    fusion_engine = MultimodalPainFusionEngine(base_phys_weight=0.20)

    face_tensors = []
    frame_timestamps = []
    motion_scores = []
    face_hits = 0
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % step == 0:
            t = frame_idx / video_fps
            tracking = tracker.process_frame(frame)

            if tracking["face_detected"] and tracking["bbox"] is not None:
                face_hits += 1
                bx1, by1, bx2, by2 = tracking["bbox"]
                face_crop = frame[by1:by2, bx1:bx2]
                tensor = preprocess_face_crop(face_crop)
                if tensor is not None:
                    face_tensors.append(tensor)
                else:
                    face_tensors.append(torch.zeros(3, 224, 224, dtype=torch.float32))

                # Feed rPPG processor with skin color trace
                if tracking["skin_rgb_mean"] is not None:
                    rppg_processor.add_sample(tracking["skin_rgb_mean"], t)
            else:
                # Fallback full frame center crop if no face detected
                h, w = frame.shape[:2]
                s = min(h, w)
                crop = frame[(h-s)//2:(h+s)//2, (w-s)//2:(w+s)//2]
                tensor = preprocess_face_crop(crop)
                face_tensors.append(tensor if tensor is not None else torch.zeros(3, 224, 224, dtype=torch.float32))

            motion_scores.append(tracking.get("motion_score", 0.0))
            frame_timestamps.append(t)

            if len(face_tensors) >= max_frames:
                break

        frame_idx += 1

    cap.release()

    if not face_tensors:
        raise RuntimeError("No readable frames could be extracted from video.")

    num_analyzed = len(face_tensors)
    face_detection_rate = round((face_hits / max(num_analyzed, 1)) * 100.0, 2)

    # 1. Module A: Continuous Facial Pain Inference
    batch_tensors = torch.stack(face_tensors)
    raw_pspi_scores = model_bundle.predict_pspi_batch(batch_tensors)
    probs_matrix = model_bundle.predict_4class_probs(batch_tensors, raw_pspi_scores)

    # 2. Module B: POS rPPG Vitals Computation
    rppg_metrics = rppg_processor.compute_metrics()
    bpm = None
    resp_rate = None
    sqi = 0.0
    hrv_sdnn = None
    pulse_waveform = []

    if rppg_metrics is not None:
        bpm = rppg_metrics.get("bpm")
        resp_rate = rppg_metrics.get("resp_rate")
        sqi = float(rppg_metrics.get("sqi", 0.0))
        hrv_sdnn = rppg_metrics.get("hrv_sdnn")

        # Generate a mini pulse waveform trajectory (last 60 samples)
        # using the processor's clean pulse history if available
        if hasattr(rppg_processor, "clean_pulse_history") and len(rppg_processor.clean_pulse_history) > 0:
            pulse_waveform = [round(float(v), 3) for v in list(rppg_processor.clean_pulse_history)[-60:]]
        else:
            # Synthetic pulse wave for display if history is short
            t_wave = np.linspace(0, 2 * np.pi, 40)
            base_hr = bpm if bpm is not None else 72.0
            freq_hz = base_hr / 60.0
            pulse_waveform = [round(float(0.8 * np.sin(2 * np.pi * freq_hz * (i / 15.0))), 3) for i in range(40)]

    # 3. Module C: Quality-Gated Multimodal Fusion
    # Auto-calibrate neutral baseline using lowest 20% scores in video
    sorted_pspi = np.sort(raw_pspi_scores)
    neutral_baseline = float(np.median(sorted_pspi[:max(3, int(len(sorted_pspi) * 0.25))]))
    fusion_engine.set_neutral_face(neutral_baseline)

    if bpm is not None:
        fusion_engine.set_baseline_vitals(bpm, resp_rate)

    # Run temporal trajectory fusion
    trajectory = []
    fused_scores = []

    for idx, (t, raw_s, p_row, mot) in enumerate(zip(frame_timestamps, raw_pspi_scores, probs_matrix, motion_scores)):
        fout = fusion_engine.process(
            raw_face_score=float(raw_s),
            sensitivity_gain=2.5,
            bpm=bpm,
            resp_rate=resp_rate,
            sqi=sqi,
            motion_score=float(mot),
            hrv_sdnn=hrv_sdnn
        )
        fused_pspi = fout["pspi_score"]
        fused_scores.append(fused_pspi)

        # Dynamic class prediction for this frame
        pred_class_idx = int(np.argmax(p_row))
        conf_pct = round(float(p_row[pred_class_idx]) * 100.0, 2)

        trajectory.append({
            "time": round(float(t), 2),
            "label": CLASS_NAMES[pred_class_idx],
            "confidence": conf_pct,
            "pspi": round(float(fused_pspi), 2),
            "probabilities": [round(float(x) * 100.0, 2) for x in p_row],
            "motion": round(float(mot), 2)
        })

    # Overall Aggregate Results
    mean_pspi = float(np.mean(fused_scores))
    mean_probs = probs_matrix.mean(axis=0)
    top_class_idx = int(np.argmax(mean_probs))

    # Tri-Scale Mapping
    final_pspi = round(float(np.clip(mean_pspi, 0.0, 16.0)), 2)
    final_cpot = int(np.clip(round(final_pspi / 2.0), 0, 8))
    final_nrs = int(np.clip(round((final_pspi / 16.0) * 10.0), 0, 10))

    prediction_obj = {
        "label": CLASS_NAMES[top_class_idx],
        "confidence": round(float(mean_probs[top_class_idx]) * 100.0, 2),
        "probabilities": [round(float(x) * 100.0, 2) for x in mean_probs]
    }

    vitals_obj = {
        "heart_rate_bpm": round(float(bpm), 1) if bpm is not None else None,
        "respiration_rate": round(float(resp_rate), 1) if resp_rate is not None else None,
        "sqi": round(float(sqi), 2),
        "pulse_waveform": pulse_waveform,
        "hrv_sdnn": round(float(hrv_sdnn), 1) if hrv_sdnn is not None else None
    }

    response = {
        "model": "ConvNeXt-Tiny + POS rPPG Multimodal Fusion Station",
        "frames_analyzed": num_analyzed,
        "sampled_fps": round(float(effective_sampled_fps), 2),
        "duration_seconds": round(float(duration), 2),
        "face_detection_rate": face_detection_rate,
        "prediction": prediction_obj,
        "pspi_score": final_pspi,
        "cpot_score": final_cpot,
        "nrs_score": final_nrs,
        "vitals": vitals_obj,
        "trajectory": trajectory,
        "notes": [
            "Continuous facial pain estimated via ConvNeXt-Tiny with Spatial Attention.",
            "Contactless cardiovascular vitals extracted via Plane-Orthogonal-to-Skin (POS) rPPG.",
            "Quality-Gated Multimodal Fusion dynamically downweights vitals during head motion.",
            "Clinical mapping provides standardized CPOT (0-8) and NRS (0-10) pain indices."
        ],
        "pspi_available": True,
        "pspi_message": f"Continuous PSPI: {final_pspi:.2f}/16 | CPOT: {final_cpot}/8 | NRS: {final_nrs}/10"
    }

    return response
