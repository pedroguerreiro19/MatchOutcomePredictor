from flask import Flask, request, jsonify
from pathlib import Path
import pandas as pd, numpy as np, joblib
import shap
from feature_pipeline import build_features_for_match, build_features

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "clean" / "matches_P1_2122_2526.csv"
BUNDLE = joblib.load(BASE / "models" / "model.pkl")

MODEL    = BUNDLE["model"]
RAW_MODEL = BUNDLE.get("raw_model") 
SCALER   = BUNDLE["scaler"]
FEATURES = BUNDLE["features"]
LABELS   = BUNDLE["labels"]
N_ROLL   = int(BUNDLE.get("n_roll", 5))

HIST = pd.read_csv(DATA, parse_dates=["date"]).sort_values("date")
app = Flask(__name__)


X_bg, _ = build_features(HIST, n=N_ROLL, mode="train")
X_bg = X_bg.loc[:, ~X_bg.columns.duplicated()]
X_bg = X_bg.reindex(columns=FEATURES).fillna(0.0)
background = SCALER.transform(X_bg.sample(min(100, len(X_bg)), random_state=42))

if RAW_MODEL is not None:
    EXPLAINER = shap.TreeExplainer(RAW_MODEL)
else:
    EXPLAINER = shap.KernelExplainer(MODEL.predict_proba, background)


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

    X = X.loc[:, ~X.columns.duplicated()]
    X = X.reindex(columns=FEATURES).fillna(0.0)
    Xs = SCALER.transform(X)

    proba = MODEL.predict_proba(Xs)[0]
    probs = {cls: float(p) for cls, p in zip(LABELS, proba)}
    winner = max(probs, key=probs.get)

    shap_values = EXPLAINER.shap_values(Xs)

    pred_class_idx = LABELS.index(winner)
    values = shap_values[0, :, pred_class_idx]

    feature_importance = sorted(
        zip(FEATURES, values),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:3]

    key_factors = [f"{feat}: {val:+.3f}" for feat, val in feature_importance]   

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