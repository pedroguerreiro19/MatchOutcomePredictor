from flask import Flask, request, jsonify
from pathlib import Path
import pandas as pd, numpy as np, joblib

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "clean" / "matches_P1_2122_2526.csv"
BUNDLE = joblib.load(BASE / "models" / "model.pkl")

MODEL    = BUNDLE["model"]
SCALER   = BUNDLE["scaler"]
FEATURES = BUNDLE["features"]
LABELS   = [{0:"AwayWin",1:"HomeWin",2:"Draw"}[c] for c in MODEL.classes_] 
N_ROLL   = int(BUNDLE.get("n_roll", 5))  

HIST = pd.read_csv(DATA, parse_dates=["date"]).sort_values("date")
app = Flask(__name__)

from feature_pipeline import build_features_for_match

@app.get("/health")
def health():
    return jsonify({
        "status": "ok", 
        "n_hist": int(len(HIST)), 
        "classes": LABELS, 
        "n_features": len(FEATURES)
    })

@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    home = data.get("home_team")
    away = data.get("away_team")

    if not home or not away:
        return jsonify({"error": "Body must contain home_team and away_team."}), 400

    try:
        X = build_features_for_match(HIST, home, away, n=N_ROLL, features=FEATURES)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    X = X.select_dtypes(include=[np.number]).copy().fillna(0.0)

    Xs = SCALER.transform(X)
    proba = MODEL.predict_proba(Xs)[0]
    probs = {cls: float(p) for cls, p in zip(LABELS, proba)}
    winner = max(probs, key=probs.get)

    top_features = X.iloc[0].sort_values(ascending=False).head(3)
    key_factors = [f"{feat}: {val:.2f}" for feat, val in top_features.items()]

    return jsonify({
        "home_team": home,
        "away_team": away, 
        "winner": winner,
        "probabilities": probs,
        "keyFactors": key_factors,
        "n_features": len(FEATURES)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)