from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import warnings

from sklearn.metrics import accuracy_score, f1_score, classification_report, log_loss
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

from feature_pipeline import build_features

warnings.filterwarnings("ignore", category=UserWarning)
RND = 42

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "clean" / "matches_P1_2122_2526.csv"
MODEL = BASE / "models" / "model.pkl"
MODEL.parent.mkdir(parents=True, exist_ok=True)

N_ROLL = 10

print("Loading:", DATA)
df = pd.read_csv(DATA, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

# Split train/test
cut = int(len(df) * 0.8)
df_train, df_test = df.iloc[:cut], df.iloc[cut:]

# Build features
X_train, y_train = build_features(df_train, n=N_ROLL, mode="train")
X_test, y_test   = build_features(df_test,  n=N_ROLL, mode="train")

# Remove columns que não devem estar no modelo
drop_cols = ["home_goals", "away_goals", "Unnamed: 0"]
X_train = X_train.drop(columns=[c for c in drop_cols if c in X_train], errors="ignore")
X_test  = X_test.drop(columns=[c for c in drop_cols if c in X_test], errors="ignore")

# Garantir apenas numéricas
X_train = X_train.select_dtypes(include=[np.number]).copy().fillna(0.0)
X_test  = X_test.select_dtypes(include=[np.number]).copy().fillna(0.0)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("First rows train:\n", X_train.head())

# Scaler
scaler = StandardScaler(with_mean=True)
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# Modelo base
print("\n=== HistGradientBoosting + Calibration (3 classes) ===")
base = HistGradientBoostingClassifier(random_state=RND)
base.fit(X_train_s, y_train)

# Calibrado
calibrated = CalibratedClassifierCV(estimator=base, cv=3, method="sigmoid")
calibrated.fit(X_train_s, y_train)

# Avaliação
pred = calibrated.predict(X_test_s)
proba = calibrated.predict_proba(X_test_s)

acc = accuracy_score(y_test, pred)
f1 = f1_score(y_test, pred, average="macro")
print(f"acc={acc:.3f} | f1_macro={f1:.3f} | log_loss={log_loss(y_test, proba):.3f}")
print(classification_report(y_test, pred, target_names=["AwayWin","HomeWin","Draw"]))

# Bundle: calibrado p/ previsões, raw_model p/ SHAP
bundle = {
    "model": calibrated,            # usado no serve.py p/ probs
    "raw_model": base,              # usado no serve.py p/ SHAP
    "scaler": scaler,
    "features": list(X_train.columns),  # só features válidas
    "labels": ["AwayWin","HomeWin","Draw"], 
    "n_roll": N_ROLL
}
joblib.dump(bundle, MODEL)
print("Saved:", MODEL)