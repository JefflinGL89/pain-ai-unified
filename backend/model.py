"""
Unified Model Loader
Constructs and initializes both the ConvNeXt-Tiny Continuous Pain Regressor (Module A)
and the legacy ResNet-18 discrete classifier with robust fallback.
"""

from pathlib import Path
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.module_a_facial.model import ConvNeXtPainRegressor

CLASS_NAMES = ["No Pain", "Low Pain", "Moderate Pain", "High Pain"]


def build_resnet_classifier(num_classes=4):
    """Builds teammate's ResNet18 4-class classifier."""
    model = resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, num_classes)
    )
    return model


def _extract_state_dict(obj):
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model", "weights"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        if obj and all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
    raise RuntimeError("Unsupported checkpoint format. Expected a state_dict or a training checkpoint dictionary.")


class UnifiedModelBundle:
    """
    Encapsulates both the continuous ConvNeXt-Tiny pain regressor
    and the discrete 4-class ResNet classifier.
    """
    def __init__(self, convnext_model=None, resnet_model=None, device=None, messages=None):
        self.convnext = convnext_model
        self.resnet = resnet_model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.messages = messages or []

    @property
    def has_convnext(self) -> bool:
        return self.convnext is not None

    @property
    def has_resnet(self) -> bool:
        return self.resnet is not None

    def predict_pspi_batch(self, tensors: torch.Tensor) -> np.ndarray:
        """
        Runs batch through ConvNeXtPainRegressor.
        Returns continuous PSPI scores in [0.0, 16.0].
        """
        if self.convnext is None:
            return np.zeros(len(tensors), dtype=np.float32)

        self.convnext.eval()
        with torch.no_grad():
            tensors = tensors.to(self.device)
            # Regressor outputs [0.0, 1.0], scale to 0-16 PSPI
            raw_scores = self.convnext(tensors).cpu().numpy().flatten() * 16.0
        return raw_scores

    def predict_4class_probs(self, tensors: torch.Tensor, pspi_scores: np.ndarray = None) -> np.ndarray:
        """
        Returns (N, 4) probability array across ["No Pain", "Low Pain", "Moderate Pain", "High Pain"].
        If ResNet checkpoint is loaded, evaluates ResNet logits.
        Otherwise, maps continuous PSPI scores via calibrated clinical membership functions.
        """
        if self.resnet is not None:
            self.resnet.eval()
            with torch.no_grad():
                tensors = tensors.to(self.device)
                logits = self.resnet(tensors)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
            return probs

        # Calibrated membership mapping from continuous PSPI (0.0 to 16.0)
        # No Pain: center 0.0, Low Pain: center 2.0, Moderate: center 5.0, High: center 9.0+
        if pspi_scores is None:
            pspi_scores = self.predict_pspi_batch(tensors)

        probs = []
        for s in pspi_scores:
            s_val = float(np.clip(s, 0.0, 16.0))
            # Gaussian response curves centered at clinical thresholds
            w0 = np.exp(-((s_val - 0.0) ** 2) / (2 * (1.2 ** 2)))
            w1 = np.exp(-((s_val - 2.2) ** 2) / (2 * (1.4 ** 2)))
            w2 = np.exp(-((s_val - 4.8) ** 2) / (2 * (1.8 ** 2)))
            w3 = 1.0 / (1.0 + np.exp(-(s_val - 6.5) / 1.2))  # Sigmoidal high pain saturation

            vec = np.array([w0, w1, w2, w3], dtype=np.float32)
            denom = np.sum(vec) + 1e-6
            probs.append(vec / denom)

        return np.array(probs, dtype=np.float32)


def load_unified_models(root_dir=None, device=None):
    """
    Constructs unified model bundle loading both ConvNeXt and ResNet checkpoints.
    """
    if root_dir is None:
        root_dir = ROOT_DIR

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    messages = []

    # 1. Load ConvNeXt-Tiny Pain Regressor
    convnext_ckpt = root_dir / "checkpoints" / "best_convnext_pain_model.pth"
    convnext_model = None

    if convnext_ckpt.exists():
        try:
            convnext = ConvNeXtPainRegressor(pretrained=False).to(device)
            state = torch.load(convnext_ckpt, map_location=device, weights_only=False)
            convnext.load_state_dict(state)
            convnext.eval()
            convnext_model = convnext
            msg = f"ConvNeXt-Tiny pain regressor loaded successfully ({convnext_ckpt.name})."
            messages.append(msg)
            print(f"[ModelLoader] {msg}")
        except Exception as e:
            err = f"Failed to load ConvNeXt checkpoint: {type(e).__name__}: {e}"
            messages.append(err)
            print(f"[ModelLoader Error] {err}")
    else:
        err = f"ConvNeXt checkpoint not found at {convnext_ckpt}."
        messages.append(err)
        print(f"[ModelLoader Warning] {err}")

    # 2. Load Teammate's ResNet-18 Classifier (if available)
    resnet_path_env = os.getenv("MODEL_CHECKPOINT", "models/resnet18_stage09b_weights.pth")
    resnet_ckpt = root_dir / resnet_path_env
    resnet_model = None

    if resnet_ckpt.exists():
        try:
            resnet = build_resnet_classifier().to(device)
            ckpt = torch.load(resnet_ckpt, map_location=device, weights_only=False)
            state = _extract_state_dict(ckpt)
            cleaned = {}
            for k, v in state.items():
                k = k.replace("module.", "")
                if k.startswith("backbone."):
                    k = k[len("backbone."):]
                cleaned[k] = v
            resnet.load_state_dict(cleaned, strict=False)
            resnet.eval()
            resnet_model = resnet
            msg = f"ResNet-18 classifier loaded successfully ({resnet_ckpt.name})."
            messages.append(msg)
            print(f"[ModelLoader] {msg}")
        except Exception as e:
            err = f"ResNet-18 load failed: {type(e).__name__}: {e}"
            messages.append(err)
            print(f"[ModelLoader Warning] {err}")
    else:
        msg = f"ResNet-18 weights not found at {resnet_ckpt}. Using calibrated continuous PSPI-to-class projection."
        messages.append(msg)
        print(f"[ModelLoader Notice] {msg}")

    bundle = UnifiedModelBundle(
        convnext_model=convnext_model,
        resnet_model=resnet_model,
        device=device,
        messages=messages
    )

    combined_status = " | ".join(messages)
    return bundle, combined_status
