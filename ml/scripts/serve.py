from fastapi import FastAPI
from pydantic import BaseModel
import joblib, pandas as pd

app = FastAPI()
bundle = joblib.load("models/model.pkl")
model, scaler, feature_cols = bundle["model"], bundle["scaler"], bundle["features"]

class PredictIn(BaseModel):
    home_team: str
    away_team: str
    venue: str
    lineups: dict | None = None
    match_date: str

@app.post("/predict")
def predict(inp: PredictIn):
    import numpy as np
    probs = {"HOME_WIN":0.4,"DRAW":0.3,"AWAY_WIN":0.3}
    winner = max(probs, key = probs.get)
    return {
        "winner": winner,
        "probabilities": probs,
        "player_props": [],
        "explain": {"top_features":[]}
    }