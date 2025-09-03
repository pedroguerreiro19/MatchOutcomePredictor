import joblib
import pandas as pd
from feature_pipeline import build_features_for_match
import numpy as np

def predict_match(model_path, data_path, home_team, away_team):
    """
    Predict match outcome with improved pipeline
    """
    
    # Load model bundle
    bundle = joblib.load(model_path)
    model = bundle["model"]
    scaler = bundle["scaler"]  # NEW: Load scaler
    features = bundle["features"] 
    labels = bundle["labels"]
    n_roll = bundle["n_roll"]
    model_type = bundle.get("model_type", "base")
    
    print(f"Loaded {model_type} model with {len(features)} features")
    
    # Load historical data
    df = pd.read_csv(data_path, parse_dates=["date"]).sort_values("date")
    
    # Build features for the match
    try:
        X = build_features_for_match(df, home_team, away_team, n=n_roll, features=features)
    except Exception as e:
        return {"error": f"Failed to build features: {str(e)}"}
    
    # CRITICAL FIX: Apply the same scaling used in training
    X_scaled = pd.DataFrame(
        scaler.transform(X), 
        columns=X.columns,
        index=X.index
    )
    
    print(f"\n=== SCALED FEATURES CHECK ===")
    print("Scaled feature ranges:")
    for col in X_scaled.columns:
        if abs(X_scaled[col].iloc[0]) > 5:  # Flag extreme values
            print(f"WARNING: {col} = {X_scaled[col].iloc[0]:.3f} (very high)")
        elif abs(X_scaled[col].iloc[0]) > 0.01:
            print(f"{col}: {X_scaled[col].iloc[0]:.3f}")
    print("===============================\n")
    
    # Get predictions from scaled features
    probabilities = model.predict_proba(X_scaled)[0]
    prediction = model.predict(X_scaled)[0]
    
    # Convert to more intuitive format
    prob_dict = {labels[i]: float(probabilities[i]) for i in range(len(labels))}
    winner = labels[prediction]
    
    # Sanity check: warn about extreme predictions
    max_prob = max(probabilities)
    if max_prob > 0.95:
        print(f"WARNING: Very high confidence prediction ({max_prob:.3f})")
        print("This might indicate a model calibration issue.")
    
    # Feature importance analysis (for interpretability)
    if hasattr(model, 'feature_importances_'):
        # For base XGBoost model
        importances = model.feature_importances_
    elif hasattr(model, 'calibrated_classifiers_'):
        # For calibrated model (correct attribute)
        importances = model.calibrated_classifiers_[0].base_estimator.feature_importances_
    else:
        importances = [0] * len(features)
    
    # Get top contributing features with their actual values
    feature_contributions = []
    for i, feature in enumerate(features):
        contribution = abs(X_scaled[feature].iloc[0]) * importances[i]
        feature_contributions.append({
            "feature": _readable_feature_name(feature),
            "impact": float(contribution),
            "raw_value": float(X.iloc[0][feature]),
            "scaled_value": float(X_scaled.iloc[0][feature])
        })
    
    # Sort by impact and take top 5
    feature_contributions.sort(key=lambda x: x["impact"], reverse=True)
    top_features = feature_contributions[:5]
    
    return {
        "home_team": home_team,
        "away_team": away_team,
        "winner": winner,
        "probabilities": prob_dict,
        "keyFactors": [{"feature": f["feature"], "impact": f["impact"]} for f in top_features],
        "n_features": len(features),
        "model_type": model_type,
        "confidence_level": _get_confidence_level(max_prob)
    }

def _get_confidence_level(max_prob):
    """Categorize prediction confidence"""
    if max_prob > 0.8:
        return "Very High"
    elif max_prob > 0.6:
        return "High"
    elif max_prob > 0.45:
        return "Medium"
    else:
        return "Low"

def _readable_feature_name(feature):
    """Convert technical feature names to readable ones"""
    name_mapping = {
        "elo_diff": "Elo Rating Difference",
        "rank_diff": "League Rank Difference", 
        "gf_diff_r10": "Avg Goals Scored (last 10)",
        "ga_diff_r10": "Avg Goals Conceded (last 10)",
        "win_diff_r10": "Avg Wins (last 10)",
        "points_diff_r10": "Avg Points (last 10)",
        "shots_diff_r10": "Avg Shots (last 10)",
        "shots_ot_diff_r10": "Avg Shots on Target (last 10)",
        "corners_diff_r10": "Avg Corners (last 10)",
        "fouls_diff_r10": "Avg Fouls (last 10)",
        "yellow_diff_r10": "Avg Yellow Cards (last 10)",
        "h2h_win_home": "Head-to-Head Home Wins",
        "h2h_win_away": "Head-to-Head Away Wins"
    }
    return name_mapping.get(feature, feature.replace("_", " ").title())

# Example usage and testing
if __name__ == "__main__":
    model_path = "models/model_xgb.pkl"
    data_path = "data/clean/matches_P1_1011_2526.csv"
    
    # Test problematic prediction
    print("=== TESTING FAMALICAO VS GUIMARAES ===")
    result = predict_match(model_path, data_path, "Famalicao", "Guimaraes")
    print(f"Prediction: {result}")
    
    print("\n=== TESTING REVERSE MATCH ===")
    result2 = predict_match(model_path, data_path, "Guimaraes", "Famalicao")
    print(f"Prediction: {result2}")
    
    # Sanity check: probabilities should be somewhat different but not extreme
    prob1_home = result["probabilities"]["HomeWin"]
    prob2_away = result2["probabilities"]["AwayWin"] 
    
    print(f"\nSanity Check:")
    print(f"Famalicao home win prob: {prob1_home:.3f}")
    print(f"Famalicao away win prob: {prob2_away:.3f}")
    print(f"Difference: {abs(prob1_home - prob2_away):.3f}")
    
    if abs(prob1_home - prob2_away) > 0.4:
        print("WARNING: Large difference suggests home advantage is overweighted")
    else:
        print("OK: Reasonable difference accounting for home advantage")