from pathlib import Path
import json
import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from .model import load_unified_models
from .inference import predict_video, DEVICE
from .schemas import InferenceResponse

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
DATA = ROOT / "data"

app = FastAPI(
    title="PAIN-AI Multimodal Research & Clinical Station",
    description="Contactless Surgical Pain & rPPG Monitoring Station combining ConvNeXt-Tiny and POS rPPG.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

MODEL = None
MODEL_MESSAGE = "Models initializing..."


@app.on_event("startup")
def startup():
    global MODEL, MODEL_MESSAGE
    MODEL, MODEL_MESSAGE = load_unified_models(ROOT, DEVICE)
    print(f"[PAIN-AI Server] {MODEL_MESSAGE}")


@app.get("/api/health")
def health():
    convnext_ok = MODEL.has_convnext if MODEL else False
    resnet_ok = MODEL.has_resnet if MODEL else False
    return {
        "status": "ok",
        "device": str(DEVICE),
        "convnext_loaded": convnext_ok,
        "resnet_loaded": resnet_ok,
        "model_loaded": convnext_ok or resnet_ok,
        "model_message": MODEL_MESSAGE
    }


@app.get("/api/results")
def results():
    p = DATA / "project_results.json"
    if p.exists():
        try:
            return JSONResponse(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            pass
    return JSONResponse({
        "metrics": {
            "accuracy": 78.43,
            "balanced_accuracy": 56.59,
            "macro_f1": 59.44,
            "subject_independent_mae": 0.978,
            "rppg_error_bpm": 0.79
        },
        "dataset": {"frames": 48398, "subjects": 25, "sequences": 200},
        "source": "ConvNeXt-Tiny + POS rPPG Multimodal System Validation."
    })


@app.post("/api/predict", response_model=InferenceResponse)
async def predict(file: UploadFile = File(...)):
    global MODEL
    if MODEL is None or not MODEL.has_convnext:
        raise HTTPException(503, detail=f"Inference models unavailable: {MODEL_MESSAGE}")

    suffix = Path(file.filename or ".mp4").suffix.lower()
    if suffix not in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}:
        suffix = ".mp4"

    try:
        content = await file.read()
        if len(content) > 250 * 1024 * 1024:
            raise HTTPException(413, detail="Uploaded video exceeds the 250 MB size limit.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            path = Path(tmp.name)

        try:
            out = predict_video(path, MODEL)
        finally:
            path.unlink(missing_ok=True)

        return out
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=f"Multimodal inference failed: {type(e).__name__}: {e}")


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")


@app.get("/styles.css")
def css():
    return FileResponse(FRONTEND / "styles.css", media_type="text/css")


@app.get("/app.js")
def js():
    return FileResponse(FRONTEND / "app.js", media_type="application/javascript")
