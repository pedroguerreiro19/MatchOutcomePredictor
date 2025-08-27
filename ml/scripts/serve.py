from flask import Flask, request, jsonify
from pathlib import Path
import pandas as pd, numpy as np, joblib

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "clean" / "matches_P1_2122_2526.csv"
BUNDLE = joblib.load(BASE / "models" / "model.pkl")

MODEL    = BUNDLE["model"]
SCALER   = BUNDLE["scaler"]
FEATURES = BUNDLE["features"]
LABELS   = list(getattr(MODEL, "classes_", BUNDLE.get("labels", ["AWAY_WIN","DRAW","HOME_WIN"])))
N_ROLL   = int(BUNDLE.get("n_roll", 5))  

HIST = pd.read_csv(DATA, parse_dates=["date"]).sort_values("date")
app = Flask(__name__)

def _team_view(df, side):
    import numpy as np
    opp = "away" if side == "home" else "home"
    return pd.DataFrame({
        "date": df["date"],
        "team": df[f"{side}_team"],
        "opp":  df[f"{opp}_team"],
        "is_home": 1 if side == "home" else 0,
        "gf": df[f"{side}_goals"],
        "ga": df[f"{opp}_goals"],
        "points": df[f"{side}_points"] if f"{side}_points" in df.columns else np.nan,
        "shots": df.get(f"{side}_shots", 0),
        "shots_ot": df.get(f"{side}_shots_ot", 0),
        "corners": df.get(f"{side}_corners", 0),
        "yellow": df.get(f"{side}_yellow", 0),
        "red": df.get(f"{side}_red", 0),
        "ht_gf": df.get(f"ht_{side}_goals", np.nan),
        "ht_ga": df.get(f"ht_{opp}_goals", np.nan),
    })


def _roll_team(g, n=5):
    g = g.sort_values("date").copy()
    g["win"]  = (g["gf"] > g["ga"]).astype(int)
    g["draw"] = (g["gf"] == g["ga"]).astype(int)
    g["loss"] = (g["gf"] < g["ga"]).astype(int)

    g["goal_diff"]      = g["gf"] - g["ga"]
    g["goals_total"]    = g["gf"] + g["ga"]
    g["ht_goal_diff"]   = g["ht_gf"] - g["ht_ga"]
    g["ht_goals_total"] = g["ht_gf"] + g["ht_ga"]

    cols = [
        "points","gf","ga","win","draw","loss",
        "shots","shots_ot","corners","yellow","red","ht_gf","ht_ga",
        "goal_diff","goals_total","ht_goal_diff","ht_goals_total"
    ]
    for c in cols:
        g[f"{c}_r{n}"] = g[c].rolling(n, min_periods=1).mean().shift(1)
    return g



def build_features_for_match(home: str, away: str, when_pd: pd.Timestamp, n=N_ROLL) -> pd.DataFrame:
    hist = HIST[HIST["date"] < when_pd].copy()
    if hist.empty:
        raise ValueError("No history before this date.")
    have_home = (hist["home_team"].eq(home) | hist["away_team"].eq(home)).any()
    have_away = (hist["home_team"].eq(away) | hist["away_team"].eq(away)).any()
    if not (have_home and have_away):
        raise ValueError("Insufficient history for one or both teams.")

    stack = pd.concat([_team_view(hist, "home"), _team_view(hist, "away")], ignore_index=True)
    rolled = stack.groupby("team", group_keys=False).apply(lambda df: _roll_team(df, n=n))

    H = rolled[rolled["team"] == home].sort_values("date").tail(1).add_prefix("home_")
    A = rolled[rolled["team"] == away].sort_values("date").tail(1).add_prefix("away_")
    if H.empty or A.empty:
        raise ValueError()

    X = pd.concat([H.reset_index(drop=True), A.reset_index(drop=True)], axis=1)

    X = X.reindex(columns=FEATURES, fill_value=0.0).fillna(0.0)
    return X

@app.get("/health")
def health():
    return jsonify({"status": "ok", "n_hist": int(len(HIST)), "classes": LABELS, "n_features": len(FEATURES)})

@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    home = data.get("home_team")
    away = data.get("away_team")
    date = data.get("match_date")

    if not home or not away or not date:
        return jsonify({"error": "Body must contain home_team, away_team, match_date (YYYY-MM-DD)."}), 400

    try:
        when = pd.to_datetime(date)
    except Exception:
        return jsonify({"error": "wrong match_date format. Use YYYY-MM-DD."}), 400

    try:
        X = build_features_for_match(home, away, when, n=N_ROLL)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    Xs = SCALER.transform(X.values)
    proba = MODEL.predict_proba(Xs)[0]
    probs = {cls: float(p) for cls, p in zip(LABELS, proba)}
    winner = max(probs, key=probs.get)

    return jsonify({
        "home_team": home, "away_team": away, "match_date": date,
        "winner": winner, "probabilities": probs,
        "n_features": len(FEATURES)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)