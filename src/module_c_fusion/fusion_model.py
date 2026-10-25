"""
Multimodal Pain Fusion Engine
Combines Facial Expression Deep Learning (Module A) with Contactless
Physiological rPPG Stress Metrics (Module B) using dynamic, quality-gated
late fusion. Translates research PSPI scores into standardized clinical
scales (CPOT 0-8 and NRS 0-10).
"""

from typing import Dict, Optional
import numpy as np


class MultimodalPainFusionEngine:
    def __init__(self, base_phys_weight=0.20, deadzone=0.10):
        """
        :param base_phys_weight: Default weighting assigned to physiological rPPG stress when SQI is high (0.15 to 0.25).
        :param deadzone: Facial difference threshold below which micro-movements are suppressed as neutral.
        """
        self.base_phys_weight = float(base_phys_weight)
        self.deadzone = float(deadzone)

        # Asymmetric smoothing: fast attack on acute pain, smooth decay
        self.smoothed_score = 0.0
        self.ALPHA_RISE = 0.40
        self.ALPHA_FALL = 0.15

        # Baseline storage
        self.neutral_face_offset: Optional[float] = None
        self.baseline_bpm: Optional[float] = None
        self.baseline_resp: Optional[float] = None

    def set_neutral_face(self, raw_val: float):
        """Locks the user's neutral resting face baseline score."""
        self.neutral_face_offset = float(raw_val)
        self.smoothed_score = 0.0

    def set_baseline_vitals(self, bpm: float, resp: Optional[float] = None):
        """Locks resting vital baselines."""
        self.baseline_bpm = float(bpm)
        if resp is not None:
            self.baseline_resp = float(resp)

    def process(
        self,
        raw_face_score: float,
        sensitivity_gain: float = 2.5,
        bpm: Optional[float] = None,
        resp_rate: Optional[float] = None,
        sqi: float = 0.0,
        motion_score: float = 0.0,
        hrv_sdnn: Optional[float] = None,
    ) -> Dict:
        """
        Computes fused pain assessment across facial and physiological domains.
        """
        # 1. Calibrated Facial Score (Module A)
        if self.neutral_face_offset is None:
            face_calibrated = 0.0
        else:
            delta = abs(raw_face_score - self.neutral_face_offset)
            effective_delta = max(0.0, delta - self.deadzone)
            face_calibrated = float(np.clip(effective_delta * sensitivity_gain, 0.0, 16.0))

        # 2. Physiological Autonomic Stress Score (Module B)
        phys_stress = 0.0
        has_phys = False

        if bpm is not None and self.baseline_bpm is not None:
            has_phys = True
            hr_elev = max(0.0, bpm - self.baseline_bpm)
            # 20 BPM elevation corresponds to maximum sympathetic pain arousal
            hr_component = np.clip(hr_elev / 20.0, 0.0, 1.0) * 16.0

            # Optional Respiration component
            if resp_rate is not None and self.baseline_resp is not None:
                rr_elev = max(0.0, resp_rate - self.baseline_resp)
                rr_component = np.clip(rr_elev / 8.0, 0.0, 1.0) * 16.0
                phys_stress = 0.75 * hr_component + 0.25 * rr_component
            else:
                phys_stress = hr_component

        # 3. Dynamic Quality-Aware Fusion Weighting
        if has_phys:
            # Attenuate physiological weight if SQI is poor (< 0.25)
            sqi_factor = float(np.clip((sqi - 0.20) / 0.40, 0.0, 1.0))
            # Attenuate physiological weight if head motion is detected (> 3.5 px displacement)
            motion_factor = float(np.clip(1.0 - (motion_score / 4.0), 0.0, 1.0))

            effective_phys_weight = self.base_phys_weight * sqi_factor * motion_factor
        else:
            effective_phys_weight = 0.0

        effective_face_weight = 1.0 - effective_phys_weight
        fused_target = (effective_face_weight * face_calibrated) + (effective_phys_weight * phys_stress)

        # 4. Asymmetric Temporal Smoothing (Fast onset, smooth release)
        if fused_target > self.smoothed_score:
            self.smoothed_score = (self.ALPHA_RISE * fused_target) + ((1.0 - self.ALPHA_RISE) * self.smoothed_score)
        else:
            self.smoothed_score = (self.ALPHA_FALL * fused_target) + ((1.0 - self.ALPHA_FALL) * self.smoothed_score)

        final_pspi = float(np.clip(self.smoothed_score, 0.0, 16.0))

        # 5. Clinical Scale Translations
        # Critical-Care Pain Observation Tool (CPOT: 0 to 8)
        # 0-2: Relaxed / No Pain | 3-4: Mild/Moderate | 5-6: Moderate | 7-8: Severe
        cpot_score = int(np.clip(round(final_pspi / 2.0), 0, 8))

        # Numeric Rating Scale (NRS: 0 to 10)
        nrs_score = int(np.clip(round((final_pspi / 16.0) * 10.0), 0, 10))

        # Clinical Tiers
        if final_pspi < 1.2:
            status_text = "NO PAIN (RELAXED)"
            status_color_bgr = (0, 220, 0)      # Green
            alert_level = 0
        elif final_pspi < 3.5:
            status_text = "MILD PAIN"
            status_color_bgr = (0, 220, 255)    # Yellow
            alert_level = 1
        elif final_pspi < 6.5:
            status_text = "MODERATE PAIN"
            status_color_bgr = (0, 140, 255)    # Orange
            alert_level = 2
        else:
            status_text = "SEVERE PAIN (CRITICAL)"
            status_color_bgr = (0, 0, 255)      # Red
            alert_level = 3

        return {
            "pspi_score": final_pspi,
            "cpot_score": cpot_score,
            "nrs_score": nrs_score,
            "face_calibrated": face_calibrated,
            "phys_stress": phys_stress,
            "face_weight": effective_face_weight,
            "phys_weight": effective_phys_weight,
            "status_text": status_text,
            "status_color_bgr": status_color_bgr,
            "alert_level": alert_level,
            "is_calibrated": bool(self.neutral_face_offset is not None),
        }
