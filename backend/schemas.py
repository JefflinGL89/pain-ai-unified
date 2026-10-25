from typing import List, Optional
from pydantic import BaseModel, Field


class Prediction(BaseModel):
    label: str
    confidence: float
    probabilities: List[float]


class VitalsData(BaseModel):
    heart_rate_bpm: Optional[float] = Field(None, description="Contactless estimated heart rate in BPM")
    respiration_rate: Optional[float] = Field(None, description="Estimated respiration rate in BrPM")
    sqi: float = Field(0.0, description="Signal Quality Index (0.0 to 1.0)")
    pulse_waveform: List[float] = Field(default_factory=list, description="Recent normalized rPPG blood volume pulse samples")
    hrv_sdnn: Optional[float] = Field(None, description="Heart rate variability (SDNN in ms) if available")


class InferenceResponse(BaseModel):
    model: str
    frames_analyzed: int
    sampled_fps: float
    duration_seconds: float
    face_detection_rate: Optional[float] = Field(None, description="Percentage of frames with detected face")
    prediction: Prediction

    # Tri-Scale Pain Outputs
    pspi_score: float = Field(..., description="Continuous Prkachin & Solomon Pain Intensity score (0.0 - 16.0)")
    cpot_score: int = Field(..., description="Critical-Care Pain Observation Tool score (0 - 8)")
    nrs_score: int = Field(..., description="Numeric Rating Scale score (0 - 10)")

    # Multimodal Physiological Vitals
    vitals: VitalsData

    # Temporal Trajectory & Metadata
    trajectory: List[dict]
    notes: List[str]
    pspi_available: bool = True
    pspi_message: str = "PSPI continuously estimated via ConvNeXt-Tiny + Spatial Attention."
