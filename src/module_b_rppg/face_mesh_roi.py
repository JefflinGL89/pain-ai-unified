"""
Face Mesh ROI Extraction and Landmark Tracking Module
Provides high-precision facial landmark tracking, region-of-interest (ROI)
extraction for rPPG (forehead and cheeks), and head motion quality scoring.
Uses MediaPipe FaceLandmarker (Tasks API) with an automatic fallback to
OpenCV Haar Cascade.
"""

from pathlib import Path
import cv2
import numpy as np

# Key landmark indices in the MediaPipe 468-point topology
FOREHEAD_LANDMARKS = [10, 67, 109, 297, 338, 108, 337, 151, 9, 8, 107, 336]
LEFT_CHEEK_LANDMARKS = [117, 118, 101, 205, 50, 123, 147, 187]
RIGHT_CHEEK_LANDMARKS = [346, 347, 330, 425, 280, 352, 376, 411]
KEY_STABILITY_LANDMARKS = [1, 10, 33, 263, 61, 291, 152]  # Nose, Forehead, Eyes, Mouth corners, Chin


class FaceMeshTracker:
    def __init__(self, model_path=None, use_mediapipe=True):
        """
        Initializes the face tracker.
        :param model_path: Path to face_landmarker.task model.
        :param use_mediapipe: Boolean to enable MediaPipe Tasks API.
        """
        self.use_mediapipe = use_mediapipe
        self.detector = None
        self.prev_landmarks = None
        self.smooth_box = None
        self.BOX_EMA_ALPHA = 0.35

        # Locate model file
        if model_path is None:
            base_dir = Path(__file__).resolve().parents[2]
            model_path = base_dir / "models" / "face_landmarker.task"

        self.model_path = Path(model_path)

        if self.use_mediapipe and self.model_path.exists():
            try:
                import mediapipe as mp
                from mediapipe.tasks import python
                from mediapipe.tasks.python import vision

                base_options = python.BaseOptions(model_asset_path=str(self.model_path))
                options = vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    output_face_blendshapes=False,
                    output_facial_transformation_matrixes=False,
                    num_faces=1,
                    min_face_detection_confidence=0.5,
                    min_face_presence_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                self.detector = vision.FaceLandmarker.create_from_options(options)
                self.mp_image_module = mp.Image
                self.mp_format = mp.ImageFormat.SRGB
                print("[FaceMeshTracker] Initialized MediaPipe FaceLandmarker successfully.")
            except Exception as e:
                print(f"[FaceMeshTracker] MediaPipe initialization error: {e}. Falling back to Haar Cascade.")
                self.detector = None

        # Fallback Haar Cascade
        if self.detector is None:
            cascade_file = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.cascade = cv2.CascadeClassifier(cascade_file)
            print("[FaceMeshTracker] Initialized OpenCV Haar Cascade fallback.")
        else:
            self.cascade = None

    def process_frame(self, frame_bgr):
        """
        Processes a BGR camera frame and returns tracked landmarks,
        bounding box, ROIs, and motion quality score.
        """
        h, w = frame_bgr.shape[:2]
        result = {
            "face_detected": False,
            "bbox": None,          # (x1, y1, x2, y2) with padding
            "landmarks_px": None,  # (N, 2) array of pixel coords
            "motion_score": 0.0,   # Pixel displacement between frames
            "skin_rgb_mean": None, # [R, G, B] for rPPG
            "forehead_box": None,  # (fx1, fy1, fx2, fy2)
            "is_frontal": True
        }

        if self.detector is not None:
            # MediaPipe Tasks API
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = self.mp_image_module(image_format=self.mp_format, data=frame_rgb)
            detection = self.detector.detect(mp_image)

            if detection.face_landmarks and len(detection.face_landmarks) > 0:
                raw_landmarks = detection.face_landmarks[0]
                pts = np.array([[lm.x * w, lm.y * h] for lm in raw_landmarks], dtype=np.float32)
                result["face_detected"] = True
                result["landmarks_px"] = pts

                # Motion Metric (displacement of key landmarks)
                if self.prev_landmarks is not None:
                    curr_pts = pts[KEY_STABILITY_LANDMARKS]
                    prev_pts = self.prev_landmarks[KEY_STABILITY_LANDMARKS]
                    disp = np.linalg.norm(curr_pts - prev_pts, axis=1)
                    result["motion_score"] = float(np.mean(disp))
                self.prev_landmarks = pts.copy()

                # Frontal pose check (yaw angle approximation)
                nose_x = pts[1][0]
                left_eye_x = pts[33][0]
                right_eye_x = pts[263][0]
                eye_dist = abs(right_eye_x - left_eye_x) + 1e-6
                eye_mid = (left_eye_x + right_eye_x) / 2.0
                yaw_ratio = abs(nose_x - eye_mid) / eye_dist
                result["is_frontal"] = bool(yaw_ratio < 0.22)

                # Bounding box with 15% margin
                min_x, min_y = np.min(pts, axis=0)
                max_x, max_y = np.max(pts, axis=0)
                fw, fh = max_x - min_x, max_y - min_y
                pad_x = int(fw * 0.15)
                pad_y = int(fh * 0.15)
                raw_box = np.array([
                    max(0, min_x - pad_x),
                    max(0, min_y - pad_y),
                    min(w, max_x + pad_x),
                    min(h, max_y + pad_y)
                ], dtype=np.float32)

                # Smooth bounding box with EMA
                if self.smooth_box is None:
                    self.smooth_box = raw_box
                else:
                    self.smooth_box = (self.BOX_EMA_ALPHA * raw_box) + ((1.0 - self.BOX_EMA_ALPHA) * self.smooth_box)

                bx1, by1, bx2, by2 = self.smooth_box.astype(int)
                result["bbox"] = (bx1, by1, bx2, by2)

                # Extract Forehead Polygon and Cheek ROIs for rPPG
                fh_pts = pts[FOREHEAD_LANDMARKS].astype(np.int32)
                fh_min_x, fh_min_y = np.min(fh_pts, axis=0)
                fh_max_x, fh_max_y = np.max(fh_pts, axis=0)
                fh_min_x, fh_min_y = max(0, fh_min_x), max(0, fh_min_y)
                fh_max_x, fh_max_y = min(w, fh_max_x), min(h, fh_max_y)
                result["forehead_box"] = (fh_min_x, fh_min_y, fh_max_x, fh_max_y)

                # Forehead Mask
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillConvexPoly(mask, fh_pts, 255)

                # Also add Cheek Patches
                l_cheek_pts = pts[LEFT_CHEEK_LANDMARKS].astype(np.int32)
                r_cheek_pts = pts[RIGHT_CHEEK_LANDMARKS].astype(np.int32)
                cv2.fillConvexPoly(mask, l_cheek_pts, 255)
                cv2.fillConvexPoly(mask, r_cheek_pts, 255)

                skin_pixels = frame_rgb[mask > 0]
                if len(skin_pixels) > 50:
                    mean_r = float(np.mean(skin_pixels[:, 0]))
                    mean_g = float(np.mean(skin_pixels[:, 1]))
                    mean_b = float(np.mean(skin_pixels[:, 2]))
                    result["skin_rgb_mean"] = [mean_r, mean_g, mean_b]

                return result

        # Fallback: Haar Cascade
        if self.cascade is not None:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            faces = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
            if len(faces) > 0:
                faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                fx, fy, fw, fh = faces[0]
                pad_x = int(fw * 0.15)
                pad_y = int(fh * 0.15)
                raw_box = np.array([
                    max(0, fx - pad_x),
                    max(0, fy - pad_y),
                    min(w, fx + fw + pad_x),
                    min(h, fy + fh + pad_y)
                ], dtype=np.float32)

                if self.smooth_box is None:
                    self.smooth_box = raw_box
                else:
                    self.smooth_box = (self.BOX_EMA_ALPHA * raw_box) + ((1.0 - self.BOX_EMA_ALPHA) * self.smooth_box)

                bx1, by1, bx2, by2 = self.smooth_box.astype(int)
                result["face_detected"] = True
                result["bbox"] = (bx1, by1, bx2, by2)

                # Forehead crop estimation
                fh_w = bx2 - bx1
                fh_h = by2 - by1
                fh_x1 = bx1 + int(fh_w * 0.28)
                fh_x2 = bx1 + int(fh_w * 0.72)
                fh_y1 = by1 + int(fh_h * 0.08)
                fh_y2 = by1 + int(fh_h * 0.24)
                result["forehead_box"] = (fh_x1, fh_y1, fh_x2, fh_y2)

                forehead_crop = frame_bgr[max(0, fh_y1):min(h, fh_y2), max(0, fh_x1):min(w, fh_x2)]
                if forehead_crop.size > 0:
                    b, g, r = cv2.split(forehead_crop)
                    result["skin_rgb_mean"] = [float(np.mean(r)), float(np.mean(g)), float(np.mean(b))]

        return result
