from pathlib import Path
import pandas as pd
import joblib
import warnings
import numpy as np

from sklearn.metrics import accuracy_score, f1_score, classification_report, log_loss, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from feature_pipeline import build_features

warnings.filterwarnings("ignore", category=UserWarning)
RND = 42

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "clean" / "matches_P1_1011_2526.csv"
MODEL = BASE / "models" / "model_xgb.pkl"
MODEL.parent.mkdir(parents=True, exist_ok=True)

N_ROLL = 10

print("Loading:", DATA)
df = pd.read_csv(DATA, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

cut = int(len(df) * 0.8)
df_train, df_test = df.iloc[:cut], df.iloc[cut:]

X_train, y_train = build_features(df_train, n=N_ROLL, mode="train")
X_test, y_test   = build_features(df_test,  n=N_ROLL, mode="train")

print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# FIX 1: Feature Consistency Check
print("\n=== FEATURE CONSISTENCY CHECK ===")
print("Training features range:")
print(X_train.describe())
print("\nTest features range:")
print(X_test.describe())

# Check for extreme values that might cause overfitting
for col in X_train.columns:
    train_range = X_train[col].max() - X_train[col].min()
    test_range = X_test[col].max() - X_test[col].min()
    if train_range > 0 and abs(train_range - test_range) / train_range > 0.5:
        print(f"WARNING: {col} has very different ranges between train and test")

# FIX 2: Feature Scaling (Optional but recommended)
print("\n=== APPLYING FEATURE SCALING ===")
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train), 
    columns=X_train.columns, 
    index=X_train.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test), 
    columns=X_test.columns, 
    index=X_test.index
)

print("Scaled features summary:")
print(f"Train mean: {X_train_scaled.mean().abs().mean():.3f}")
print(f"Train std: {X_train_scaled.std().mean():.3f}")

# FIX 3: Model with Better Regularization
print("\n=== XGBoost (3 classes) with Improved Regularization ===")
base_model = XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    random_state=RND,
    n_estimators=300,  # Reduced from 500
    learning_rate=0.03,  # Reduced from 0.05  
    max_depth=4,  # Reduced from 6
    subsample=0.8,
    colsample_bytree=0.7,  # Reduced from 0.8
    reg_lambda=2.0,  # Increased from 1.0
    reg_alpha=1.0,  # Increased from 0.0
    tree_method="hist",
    min_child_weight=3,  # Added
    gamma=0.1  # Added
)

# FIX 4: Probability Calibration
print("Training base model...")
base_model.fit(X_train_scaled, y_train)

print("Applying probability calibration...")
calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
calibrated_model.fit(X_train_scaled, y_train)

# Evaluate both models
print("\n=== MODEL COMPARISON ===")
base_pred = base_model.predict(X_test_scaled)
base_proba = base_model.predict_proba(X_test_scaled)
cal_pred = calibrated_model.predict(X_test_scaled)
cal_proba = calibrated_model.predict_proba(X_test_scaled)

print("\nBase Model:")
acc_base = accuracy_score(y_test, base_pred)
f1_base = f1_score(y_test, base_pred, average="macro")
print(f"acc={acc_base:.3f} | f1_macro={f1_base:.3f} | log_loss={log_loss(y_test, base_proba):.3f}")

print("\nCalibrated Model:")
acc_cal = accuracy_score(y_test, cal_pred)
f1_cal = f1_score(y_test, cal_pred, average="macro")
print(f"acc={acc_cal:.3f} | f1_macro={f1_cal:.3f} | log_loss={log_loss(y_test, cal_proba):.3f}")

# FIX 5: Probability Distribution Analysis
print("\n=== PROBABILITY DISTRIBUTION ANALYSIS ===")
max_probas_base = base_proba.max(axis=1)
max_probas_cal = cal_proba.max(axis=1)

print(f"Base model - Avg max probability: {max_probas_base.mean():.3f}")
print(f"Base model - % predictions > 90%: {(max_probas_base > 0.9).mean()*100:.1f}%")
print(f"Base model - % predictions > 80%: {(max_probas_base > 0.8).mean()*100:.1f}%")

print(f"Calibrated model - Avg max probability: {max_probas_cal.mean():.3f}")
print(f"Calibrated model - % predictions > 90%: {(max_probas_cal > 0.9).mean()*100:.1f}%")
print(f"Calibrated model - % predictions > 80%: {(max_probas_cal > 0.8).mean()*100:.1f}%")

# FIX 6: ELO Difference Sanity Check
print("\n=== ELO DIFFERENCE ANALYSIS ===")
elo_diffs = X_test_scaled['elo_diff'] * scaler.scale_[X_train.columns.get_loc('elo_diff')] + scaler.mean_[X_train.columns.get_loc('elo_diff')]
prob_home_wins = cal_proba[:, 1]  # HomeWin class

# Check correlation between ELO difference and predicted probabilities
correlation = np.corrcoef(elo_diffs, prob_home_wins)[0, 1]
print(f"Correlation between ELO diff and home win probability: {correlation:.3f}")

# Sample some predictions to check sanity
print("\nSample ELO vs Prediction analysis:")
for i in range(min(5, len(elo_diffs))):
    print(f"ELO diff: {elo_diffs.iloc[i]:+.1f} -> Home win prob: {prob_home_wins[i]:.3f}")

print(classification_report(y_test, cal_pred, target_names=["AwayWin","HomeWin","Draw"]))

# Choose best model (prefer calibrated unless accuracy drops significantly)
if acc_cal >= acc_base - 0.02:  # Allow 2% accuracy drop for better calibration
    final_model = calibrated_model
    model_type = "calibrated"
    print(f"\nUsing CALIBRATED model (accuracy: {acc_cal:.3f})")
else:
    final_model = base_model
    model_type = "base"
    print(f"\nUsing BASE model (accuracy: {acc_base:.3f})")

bundle = {
    "model": final_model,
    "scaler": scaler,  # IMPORTANT: Save scaler for predictions
    "features": list(X_train.columns),
    "labels": ["AwayWin","HomeWin","Draw"],
    "n_roll": N_ROLL,
    "model_type": model_type
}
joblib.dump(bundle, MODEL)
print("Saved:", MODEL)