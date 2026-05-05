import sys
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.pipeline import models, scaler, process_and_predict, generate_recommendations, generate_all_explanations

_STATIC = Path(__file__).resolve().parents[1] / "frontend" / "static"
_OUTPUT = Path(__file__).resolve().parents[1] / "output"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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


def _to_form(body: Input) -> dict:
    return body.model_dump()


@app.get("/models")
def list_models():
    return {"models": list(models.keys())}


@app.post("/predict")
def predict(body: Input):
    try:
        probs, mean_prob, imputed_row = process_and_predict(_to_form(body))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    label = "Low" if mean_prob < 0.30 else "Moderate" if mean_prob < 0.60 else "High"

    estimated = {}
    if body.ap_hi       is None: estimated["ap_hi"]       = round(float(imputed_row[1]), 1)
    if body.ap_lo       is None: estimated["ap_lo"]       = round(float(imputed_row[2]), 1)
    if body.cholesterol is None: estimated["cholesterol"] = int(imputed_row[3])
    if body.gluc        is None: estimated["gluc"]        = int(imputed_row[4])

    return {
        "risk_percent": round(float(mean_prob) * 100, 1),
        "risk_label":   label,
        "models":       {k: round(float(v), 4) for k, v in probs.items()},
        "estimated":    estimated,
    }


@app.post("/recommendations")
def recommendations(body: Input):
    try:
        form = _to_form(body)
        _, mean_prob, _ = process_and_predict(form)
        recs = generate_recommendations(form, mean_prob, process_and_predict)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return {"recommendations": [
        {"action": r["action"], "new_prob": round(r["new_prob"], 4), "reduction": round(r["reduction"], 4)}
        for r in recs
    ]}


@app.post("/explain")
def explain(body: Input):
    try:
        _, _, imputed_row = process_and_predict(_to_form(body))
        scaled = scaler.transform([imputed_row])
        contributions = generate_all_explanations(models, scaled)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    serialisable = {
        model: [[feat, round(float(impact), 4)] for feat, impact in pairs]
        for model, pairs in contributions.items()
    }
    return {"explanations": serialisable}

app.mount("/output", StaticFiles(directory=_OUTPUT), name="output")
app.mount("/", StaticFiles(directory=_STATIC, html=True), name="static")
