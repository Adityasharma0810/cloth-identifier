import pathlib

# Fix for model trained on Linux/Colab and loaded on Windows
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

from io import BytesIO
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

# -----------------------------------
# Paths
# -----------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "cloth-image-classifier.pkl"

# -----------------------------------
# FastAPI App
# -----------------------------------

app = FastAPI(
    title="Cloth Classifier API",
    version="1.0",
    description="Upload a cloth image and get prediction result"
)

# Global model variable
model = None


# -----------------------------------
# Response Model
# -----------------------------------

class PredictionResponse(BaseModel):
    filename: str
    predicted_class: str
    confidence: float


# -----------------------------------
# Load Model
# -----------------------------------

def load_model():
    global model

    if model is None:
        try:
            from fastai.vision.all import load_learner
        except Exception as e:
            raise RuntimeError(
                f"FastAI loading error: {str(e)}"
            )

        if not MODEL_PATH.exists():
            raise RuntimeError(
                f"Model file not found:\n{MODEL_PATH}"
            )

        print("Loading model...")
        model = load_learner(MODEL_PATH)
        print("Model loaded successfully")

    return model


# -----------------------------------
# Startup Event
# -----------------------------------

@app.on_event("startup")
def startup():
    import torch

    print("Checking GPU status...")

    print("CUDA Available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU Name:", torch.cuda.get_device_name(0))
    else:
        print("Running on CPU only")

    load_model()


# -----------------------------------
# Home API
# -----------------------------------

@app.get("/")
def home():
    return {
        "message": "Cloth Classifier API is running"
    }


# -----------------------------------
# Health API
# -----------------------------------

@app.get("/health")
def health():
    load_model()
    return {
        "status": "success",
        "model": MODEL_PATH.name
    }


# -----------------------------------
# Prediction API
# -----------------------------------

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    print("\n========== NEW REQUEST ==========")
    print("Step 1: Request received")

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Please upload an image file"
        )

    contents = await file.read()
    print("Step 2: File read complete")

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty"
        )

    # Open image safely
    try:
        image = Image.open(BytesIO(contents)).convert("RGB")

        # Resize image for faster prediction
        image = image.resize((224, 224))

        print("Step 3: Image opened + resized")
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file"
        )

    try:
        from fastai.vision.all import PILImage
    except Exception as e:
        raise RuntimeError(
            f"FastAI import error: {str(e)}"
        )

    loaded_model = load_model()
    print("Step 4: Model loaded")

    fastai_image = PILImage.create(image)
    print("Step 5: FastAI image created")

    print("Step 6: Starting prediction...")

    pred_class, pred_idx, probs = loaded_model.predict(fastai_image)

    print("Step 7: Prediction complete")

    confidence = float(probs[pred_idx])

    print("Predicted Class:", pred_class)
    print("Confidence:", round(confidence, 4))
    print("=================================\n")

    return PredictionResponse(
        filename=file.filename,
        predicted_class=str(pred_class),
        confidence=round(confidence, 4)
    )


# -----------------------------------
# Runtime Error Handler
# -----------------------------------

@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Any, exc: RuntimeError):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)}
    )


# -----------------------------------
# Run Server
# -----------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "fastapi-cloth-classifier:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )