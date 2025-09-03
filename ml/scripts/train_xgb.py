from pathlib import Path
import pandas as pd
import joblib
import warnings

from sklearn.metrics import accuracy_score, f1_score, classification_report, log_loss
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

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\n=== XGBoost (3 classes) ===")
model = XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    random_state=RND,
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=0.0,
    tree_method="hist"
)

model.fit(X_train, y_train)

pred = model.predict(X_test)
proba = model.predict_proba(X_test)

acc = accuracy_score(y_test, pred)
f1 = f1_score(y_test, pred, average="macro")
print(f"acc={acc:.3f} | f1_macro={f1:.3f} | log_loss={log_loss(y_test, proba):.3f}")
print(classification_report(y_test, pred, target_names=["AwayWin","HomeWin","Draw"]))


bundle = {
    "model": model,
    "features": list(build_features(df_train, n=N_ROLL, mode='train')[0].columns),
    "labels": ["AwayWin","HomeWin","Draw"],
    "n_roll": N_ROLL,
    "scaler": scaler
}
joblib.dump(bundle, MODEL)
print("Saved:", MODEL)