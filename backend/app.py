from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd
import pickle
import numpy as np

# OCR-only pipeline
from utils.ocr import extract_text
from utils.parser import parse_text
from utils.features import compute_features


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # dev-friendly
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)


def to_python(obj):
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_python(x) for x in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        
        content = await file.read()

        if not file.filename.lower().endswith((".pdf", ".png", ".jpg", ".jpeg")):
            return {"error": "Unsupported file type. Upload PDF or image."}

        
        text = extract_text(content)

        print("\n========== OCR TEXT ==========\n")
        print(text)  # print partial to avoid flooding logs
        print("\n==============================\n")

        
        parsed = parse_text(text)

        print("\n======= PARSED DATA ==========\n")
        print(parsed)
        print("\n==============================\n")

        
        feats = compute_features(parsed)

        
        input_df = pd.DataFrame([{
            'S1_SGPA': feats.get('S1_SGPA', 0),
            'S2_SGPA': feats.get('S2_SGPA', 0),
            'S3_SGPA': feats.get('S3_SGPA', 0),
            'S4_SGPA': feats.get('S4_SGPA', 0),
            'sgpa_trend': feats.get('sgpa_trend', 0),
            'avg_4': feats.get('avg_4', 0),
            'max_4': feats.get('max_4', 0),
            'min_4': feats.get('min_4', 0),
            'num_FF': feats.get('num_FF', 0),
            'num_AA': feats.get('num_AA', 0),
            'avg_grade_math': feats.get('avg_grade_math', 0),
            'avg_grade_hard_subjects': feats.get('avg_grade_hard_subjects', 0),
        }])

       
        pred = model.predict(input_df)

        
        clean_feats = to_python(feats)

        return {
            "predicted_cgpa": round(float(pred[0]), 2),
            "extracted_data": clean_feats
        }

    except Exception as e:
        print("\n❌ ERROR OCCURRED:\n", str(e))
        return {"error": str(e)}

from pydantic import BaseModel


class ManualInput(BaseModel):
    S1_SGPA: float
    S2_SGPA: float
    S3_SGPA: float
    S4_SGPA: float
    num_FF: int
    num_AA: int
    avg_grade_math: float
    avg_grade_hard_subjects: float



@app.post("/predict_manual")
async def predict_manual(data: ManualInput):
    try:
        parsed = data.dict()

        # same pipeline as OCR
        feats = compute_features(parsed)

        input_df = pd.DataFrame([{
            'S1_SGPA': feats.get('S1_SGPA', 0),
            'S2_SGPA': feats.get('S2_SGPA', 0),
            'S3_SGPA': feats.get('S3_SGPA', 0),
            'S4_SGPA': feats.get('S4_SGPA', 0),
            'sgpa_trend': feats.get('sgpa_trend', 0),
            'avg_4': feats.get('avg_4', 0),
            'max_4': feats.get('max_4', 0),
            'min_4': feats.get('min_4', 0),
            'num_FF': feats.get('num_FF', 0),
            'num_AA': feats.get('num_AA', 0),
            'avg_grade_math': feats.get('avg_grade_math', 0),
            'avg_grade_hard_subjects': feats.get('avg_grade_hard_subjects', 0),
        }])

        pred = model.predict(input_df)

        clean_feats = to_python(feats)

        return {
            "predicted_cgpa": round(float(pred[0]), 2),
            "extracted_data": clean_feats,
            "mode": "manual"
        }

    except Exception as e:
        return {"error": str(e)}