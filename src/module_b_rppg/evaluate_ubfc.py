"""
UBFC-rPPG Dataset Evaluation Pipeline
Processes uncompressed facial videos (vid.avi) and pulse oximeter ground truth (ground_truth.txt),
evaluating POS rPPG heart rate estimation against medical sensor data across all subjects.
"""

from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from scipy import signal
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]

# Cascade fallback for video frames
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)


def extract_rgb_traces(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, 30.0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    rgb_traces = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        if len(faces) > 0:
            faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
            x, y, w, h = faces[0]

            # Forehead crop
            fh_y1, fh_y2 = y, y + int(h * 0.22)
            fh_x1, fh_x2 = x + int(w * 0.25), x + int(w * 0.75)
            forehead = frame[max(0, fh_y1):fh_y2, max(0, fh_x1):fh_x2]

            if forehead.size > 0:
                b, g, r = cv2.split(forehead)
                rgb_traces.append([np.mean(r), np.mean(g), np.mean(b)])
            elif len(rgb_traces) > 0:
                rgb_traces.append(rgb_traces[-1])
        elif len(rgb_traces) > 0:
            rgb_traces.append(rgb_traces[-1])

    cap.release()
    return np.array(rgb_traces) if len(rgb_traces) > 0 else None, fps


def apply_pos_algorithm(rgb_traces, fps):
    mean_rgb = np.mean(rgb_traces, axis=0) + 1e-6
    norm_traces = rgb_traces / mean_rgb
    X = norm_traces[:, 1] - norm_traces[:, 2]
    Y = norm_traces[:, 1] + norm_traces[:, 2] - 2 * norm_traces[:, 0]

    window_length = int(fps * 1.5)
    H = np.zeros(len(rgb_traces))
    for t in range(window_length, len(rgb_traces)):
        win_X, win_Y = X[t - window_length:t], Y[t - window_length:t]
        alpha = np.std(win_X) / (np.std(win_Y) + 1e-6)
        H[t] = win_X[-1] + alpha * win_Y[-1]
    return H


def calculate_heart_rate(pulse_signal, fps):
    nyq = 0.5 * fps
    b, a = signal.butter(3, [0.75 / nyq, 2.80 / nyq], btype="bandpass")
    filtered = signal.filtfilt(b, a, pulse_signal)

    n_fft = max(2048, int(2 ** np.ceil(np.log2(len(filtered)))))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / fps)
    fft_mag = np.abs(np.fft.rfft(filtered, n=n_fft))

    valid_idx = np.where((freqs >= 0.75) & (freqs <= 2.80))[0]
    if len(valid_idx) == 0:
        return 72.0

    peak_local = np.argmax(fft_mag[valid_idx])
    best_freq = freqs[valid_idx[peak_local]]
    return float(best_freq * 60.0)


def evaluate_all():
    unified_dir = ROOT_DIR / "dataset" / "ubfc_rppg_unified"
    results = []

    print(f"Scanning directory: {unified_dir}")
    if not unified_dir.exists():
        print(f"Notice: Directory not found: {unified_dir}")
        return

    subject_folders = sorted([f for f in unified_dir.iterdir() if f.is_dir()])
    print(f"Found {len(subject_folders)} subjects to process.\n")

    for i, subject_dir in enumerate(subject_folders, 1):
        subject_id = subject_dir.name
        vid_path = subject_dir / "vid.avi"
        gt_path = subject_dir / "ground_truth.txt"

        if not vid_path.exists() or not gt_path.exists():
            continue

        try:
            with open(gt_path, "r") as f:
                lines = f.readlines()
            gt_bpm_array = np.array([float(x) for x in lines[1].split()])
            gt_mean = float(np.mean(gt_bpm_array))
            gt_min = float(np.min(gt_bpm_array))
            gt_max = float(np.max(gt_bpm_array))

            rgb_data, fps_val = extract_rgb_traces(vid_path)
            if rgb_data is None:
                continue

            raw_pulse = apply_pos_algorithm(rgb_data, fps_val)
            estimated_hr = calculate_heart_rate(raw_pulse, fps_val)

            error = abs(gt_mean - estimated_hr)
            results.append({
                "Subject": subject_id,
                "GT_Mean_HR": round(gt_mean, 1),
                "GT_Min_HR": round(gt_min, 1),
                "GT_Max_HR": round(gt_max, 1),
                "Estimated_HR": round(estimated_hr, 1),
                "Absolute_Error": round(error, 1),
            })
            print(f"[{i}/{len(subject_folders)}] {subject_id}: GT={gt_mean:.1f} | Est={estimated_hr:.1f} | Err={error:.1f} BPM")
        except Exception as e:
            print(f"[{i}/{len(subject_folders)}] {subject_id} failed: {e}")

    df = pd.DataFrame(results)
    out_csv = ROOT_DIR / "dataset" / "rppg_full_analysis.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved analysis to: {out_csv}")
    print(f"Average Error across {len(df)} subjects: {df['Absolute_Error'].mean():.2f} BPM")


if __name__ == "__main__":
    evaluate_all()
