from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import warnings
from fastapi import FastAPI
from pydantic import BaseModel
from scipy.special import expit
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names")

MODELS_DIR = Path(__file__).parent.parent / "models"

app = FastAPI()

imputer = joblib.load(MODELS_DIR / "knn_imputer.pkl")
scaler   = joblib.load(MODELS_DIR / "scaler_with_feature_engineering.pkl")
models   = {
    "Logistic Regression": joblib.load(MODELS_DIR / "logistic_regression_with_feature_engineering.pkl"),
    "Decision Tree":       joblib.load(MODELS_DIR / "decision_tree_with_feature_engineering.pkl"),
    "Linear SVM":          joblib.load(MODELS_DIR / "linear_svm_with_feature_engineering.pkl"),
    "Random Forest":       joblib.load(MODELS_DIR / "random_forest_with_feature_engineering.pkl"),
    "XGBoost":             joblib.load(MODELS_DIR / "xgboost_with_feature_engineering.pkl"),
    "LightGBM":            joblib.load(MODELS_DIR / "lightgbm_with_feature_engineering.pkl"),
}


class Input(BaseModel):
    age_years:   float
    gender:      int
    height:      float
    weight:      float
    smoke:       int
    alco:        int
    active:      int
    ap_hi:       Optional[float] = None
    ap_lo:       Optional[float] = None
    cholesterol: Optional[int]   = None
    gluc:        Optional[int]   = None


@app.post("/predict")
def predict(body: Input):
    bmi = body.weight / (body.height / 100) ** 2
    row = pd.DataFrame([{
        "gender": body.gender, "ap_hi": body.ap_hi, "ap_lo": body.ap_lo,
        "cholesterol": body.cholesterol, "gluc": body.gluc,
        "smoke": body.smoke, "alco": body.alco, "active": body.active,
        "bmi": bmi, "age_years": body.age_years,
    }])

    imputed = imputer.transform(row)
    imputed[0, 3] = round(imputed[0, 3])
    imputed[0, 4] = round(imputed[0, 4])
    scaled = scaler.transform(imputed)

    probs = {}
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            probs[name] = round(float(model.predict_proba(scaled)[0][1]), 4)
        else:
            probs[name] = round(float(expit(model.decision_function(scaled)[0])), 4)

    mean = float(np.mean(list(probs.values())))
    label = "Low" if mean < 0.30 else "Moderate" if mean < 0.60 else "High"

    estimated = {}
    if body.ap_hi       is None: estimated["ap_hi"]       = round(float(imputed[0, 1]), 1)
    if body.ap_lo       is None: estimated["ap_lo"]       = round(float(imputed[0, 2]), 1)
    if body.cholesterol is None: estimated["cholesterol"] = int(imputed[0, 3])
    if body.gluc        is None: estimated["gluc"]        = int(imputed[0, 4])

    return {"risk_percent": round(mean * 100, 1), "risk_label": label, "models": probs, "estimated": estimated}
