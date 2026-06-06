from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import pandas as pd
import pickle
import numpy as np

from utils.ocr import extract_text
from utils.parser import parse_text
from utils.features import compute_features


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load both stage models
with open("model/model_stage1.pkl", "rb") as f:
    model_stage1 = pickle.load(f)

with open("model/model_stage2.pkl", "rb") as f:
    model_stage2 = pickle.load(f)


def to_python(obj):
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_python(x) for x in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def build_stage1_input(feats):
    """S1-S4 + derived features → predicts S5, S6"""
    return pd.DataFrame([{
        'S1_SGPA': feats.get('S1_SGPA', 0),
        'S2_SGPA': feats.get('S2_SGPA', 0),
        'S3_SGPA': feats.get('S3_SGPA', 0),
        'S4_SGPA': feats.get('S4_SGPA', 0),
        'trend_4': feats.get('sgpa_trend', 0),
        'avg_4':   feats.get('avg_4', 0),
        'max_4':   feats.get('max_4', 0),
        'min_4':   feats.get('min_4', 0),
        'num_FF':  feats.get('num_FF', 0),
        'num_AA':  feats.get('num_AA', 0),
    }])


def build_stage2_input(feats, s5, s6):
    """S1-S6 + derived features → predicts S7, S8, CGPA"""
    sgpas_6 = np.array([
        feats.get('S1_SGPA', 0),
        feats.get('S2_SGPA', 0),
        feats.get('S3_SGPA', 0),
        feats.get('S4_SGPA', 0),
        s5, s6
    ])
    x = np.arange(1, 7)
    trend_6 = np.polyfit(x, sgpas_6, 1)[0]

    return pd.DataFrame([{
        'S1_SGPA':  feats.get('S1_SGPA', 0),
        'S2_SGPA':  feats.get('S2_SGPA', 0),
        'S3_SGPA':  feats.get('S3_SGPA', 0),
        'S4_SGPA':  feats.get('S4_SGPA', 0),
        'S5_SGPA':  s5,
        'S6_SGPA':  s6,
        'trend_6':  trend_6,
        'avg_6':    float(np.mean(sgpas_6)),
        'max_6':    float(np.max(sgpas_6)),
        'min_6':    float(np.min(sgpas_6)),
        'num_FF':   feats.get('num_FF', 0),
        'num_AA':   feats.get('num_AA', 0),
    }])


def run_pipeline(feats):
    """Run two-stage prediction and return full results."""
    # Stage 1: predict S5, S6
    stage1_input = build_stage1_input(feats)
    stage1_pred = model_stage1.predict(stage1_input)[0]
    s5_pred = round(float(stage1_pred[0]), 2)
    s6_pred = round(float(stage1_pred[1]), 2)

    # Stage 2: predict S7, S8, CGPA
    stage2_input = build_stage2_input(feats, s5_pred, s6_pred)
    stage2_pred = model_stage2.predict(stage2_input)[0]
    s7_pred   = round(float(stage2_pred[0]), 2)
    s8_pred   = round(float(stage2_pred[1]), 2)
    cgpa_pred = round(float(stage2_pred[2]), 2)

    return {
        "predicted_cgpa": cgpa_pred,
        "predicted_sgpa": {
            "S5": s5_pred,
            "S6": s6_pred,
            "S7": s7_pred,
            "S8": s8_pred,
        }
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        content = await file.read()

        if not file.filename.lower().endswith((".pdf", ".png", ".jpg", ".jpeg")):
            return {"error": "Unsupported file type. Upload PDF or image."}

        text = extract_text(content)
        print("\n========== OCR TEXT ==========\n", text, "\n==============================\n")

        parsed = parse_text(text)
        print("\n======= PARSED DATA ==========\n", parsed, "\n==============================\n")

        feats = compute_features(parsed)
        result = run_pipeline(feats)
        result["extracted_data"] = to_python(feats)

        return result

    except Exception as e:
        print("\n ERROR:\n", str(e))
        return {"error": str(e)}


class ManualInput(BaseModel):
    S1_SGPA: float
    S2_SGPA: float
    S3_SGPA: float
    S4_SGPA: float
    num_FF: int
    num_AA: int
    avg_grade_math: float = 0.0
    avg_grade_hard_subjects: float = 0.0


@app.post("/predict_manual")
async def predict_manual(data: ManualInput):
    try:
        feats = compute_features(data.dict())
        result = run_pipeline(feats)
        result["mode"] = "manual"
        result["extracted_data"] = to_python(feats)

        return result

    except Exception as e:
        return {"error": str(e)}