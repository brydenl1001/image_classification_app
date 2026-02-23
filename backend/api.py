from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn
import io
import os
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms


app = FastAPI(title="Waste Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Paths
ROOT_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(ROOT_DIR, "waste_model.pth")
DATA_DIR = os.path.join(ROOT_DIR, "waste_dataset")


# Load class names from training folders
def load_class_names():
    train_dir = os.path.join(DATA_DIR, "train")
    if not os.path.isdir(train_dir):
        return []
    classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    classes = sorted(classes)
    return classes


CLASS_NAMES = load_class_names()


# Build model architecture to match training (resnet18)
def build_model(num_classes: int):
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


# Image transforms (match train.py)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# Load model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = None


def load_model():
    global MODEL
    if MODEL is not None:
        return MODEL

    if not CLASS_NAMES:
        raise RuntimeError("No classes found in waste_dataset/train. Ensure dataset exists on the server.")

    num_classes = len(CLASS_NAMES)
    model = build_model(num_classes)

    if not os.path.isfile(MODEL_PATH):
        raise RuntimeError(f"Model file not found at {MODEL_PATH}")

    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    MODEL = model
    return MODEL


def heuristic_recyclable(label: str) -> bool:
    l = label.lower()
    recyclable_keywords = [
        "bottle",
        "plastic",
        "can",
        "aluminum",
        "glass",
        "paper",
        "cardboard",
        "magazine",
        "newspaper",
        "office_paper",
        "steel",
    ]
    non_recyclable_keywords = [
        "food",
        "eggshell",
        "coffee",
        "tea",
        "clothing",
        "shoes",
        "styrofoam",
        "egg",
        "ground",
    ]

    for k in non_recyclable_keywords:
        if k in l:
            return False
    for k in recyclable_keywords:
        if k in l:
            return True

    # default: False (conservative)
    return False


class PredictionResponse(BaseModel):
    predicted: str
    confidence: float
    recyclable: bool
    details: dict


@app.on_event("startup")
def on_startup():
    try:
        load_model()
        print("Model loaded, classes:", CLASS_NAMES)
    except Exception as e:
        print("Warning: failed to load model on startup:", e)


@app.get("/")
def root():
    return {"status": "ok", "classes": CLASS_NAMES}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    model = None
    try:
        model = load_model()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    try:
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            top_prob, top_idx = torch.max(probs, dim=1)

        predicted = CLASS_NAMES[top_idx.item()] if CLASS_NAMES else str(top_idx.item())
        confidence = float(top_prob.item())
        recyclable = heuristic_recyclable(predicted)

        # Prepare minimal details
        details = {
            "all_classes": CLASS_NAMES,
            "raw_scores": outputs.squeeze(0).cpu().tolist(),
        }

        return PredictionResponse(predicted=predicted, confidence=confidence, recyclable=recyclable, details=details)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
