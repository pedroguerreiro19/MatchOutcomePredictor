from flask import Flask, request, jsonify
from pathlib import Path
import pandas as pd, numpy as np, joblib
import shap
from feature_pipeline import build_features_for_match, build_features

FEATURE_NAMES = {
    "elo_diff": "Elo Rating Difference",
    "rank_diff": "League Rank Difference",
    "gf_diff_r10": "Avg Goals Scored (last 10)",
    "ga_diff_r10": "Avg Goals Conceded (last 10)",
    "win_diff_r10": "Avg Wins (last 10)",
    "draw_diff_r10": "Avg Draws (last 10)",
    "loss_diff_r10": "Avg Losses (last 10)",
    "points_diff_r10": "Avg Points (last 10)",
    "shots_diff_r10": "Avg Shots (last 10)",
    "shots_ot_diff_r10": "Avg Shots on Target (last 10)",
    "fouls_diff_r10": "Avg Fouls (last 10)",
    "yellow_diff_r10": "Avg Yellow Cards (last 10)",
    "red_diff_r10": "Avg Red Cards (last 10)",
    "corners_diff_r10": "Avg Corners (last 10)"
}

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "clean" / "matches_P1_1011_2526.csv"
BUNDLE = joblib.load(BASE / "models" / "model_xgb.pkl")

MODEL    = BUNDLE["model"]
FEATURES = BUNDLE["features"]
LABELS   = BUNDLE["labels"]
N_ROLL   = int(BUNDLE.get("n_roll", 5))

HIST = pd.read_csv(DATA, parse_dates=["date"]).sort_values("date")
app = Flask(__name__)


X_bg, _ = build_features(HIST, n=N_ROLL, mode="train")
X_bg = X_bg.loc[:, ~X_bg.columns.duplicated()]
X_bg = X_bg.reindex(columns=FEATURES).fillna(0.0)

if hasattr(MODEL, "calibrated_classifiers_"):
    base_model = MODEL.calibrated_classifiers_[0].estimator
else:
    base_model = MODEL

EXPLAINER = shap.TreeExplainer(base_model, feature_perturbation="interventional")

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

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=FEATURES)

    proba = MODEL.predict_proba(X)[0]
    probs = {cls: float(p) for cls, p in zip(LABELS, proba)}
    winner = max(probs, key=probs.get)

    shap_values = EXPLAINER.shap_values(X, check_additivity=False)

    if isinstance(shap_values, list):  
        shap_array = shap_values[0]  
    else:
        shap_array = shap_values  

    pred_class_idx = LABELS.index(winner)
    values = shap_array[0, :, pred_class_idx]  

    values = np.array(values).astype(float).flatten()

    feature_importance = sorted(
        [{"feature": FEATURE_NAMES.get(f, f), "impact": round(float(v), 4)} for f, v in zip(FEATURES, values)],
        key=lambda x: abs(x["impact"]),
        reverse=True
    )[:5]

    return jsonify({
        "home_team": home,
        "away_team": away,
        "winner": winner,
        "probabilities": probs,
        "keyFactors": feature_importance if feature_importance else ["No significant factors identified."],
        "n_features": len(FEATURES)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)