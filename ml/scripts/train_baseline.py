from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import warnings

from sklearn.metrics import accuracy_score, f1_score, classification_report, log_loss
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

try:
    from imblearn.over_sampling import RandomOverSampler
    HAVE_IMB = True
except Exception:
    HAVE_IMB = False

try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False

from feature_pipeline import build_features

warnings.filterwarnings("ignore", category=UserWarning)
RND = 42

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "clean" / "matches_P1_2122_2526.csv"
MODEL = BASE / "models" / "model.pkl"
MODEL.parent.mkdir(parents=True, exist_ok=True)

N_ROLL = 5

print("Loading:", DATA)
df = pd.read_csv(DATA, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

cut = int(len(df) * 0.8)
df_train, df_test = df.iloc[:cut], df.iloc[cut:]

X_train, y_train = build_features(df_train, n=N_ROLL, mode="train")
X_train = X_train.select_dtypes(include=[np.number]).copy().fillna(0.0)

X_test, y_test = build_features(df_test, n=N_ROLL, mode="train")
X_test = X_test.select_dtypes(include=[np.number]).copy().fillna(0.0)


if HAVE_IMB:
    ros = RandomOverSampler(random_state=RND)
    X_train, y_train = ros.fit_resample(X_train, y_train)
    print("After oversample:", dict(zip([0, 1], np.bincount(y_train))))
else:
    print("imblearn not found.")

scaler = StandardScaler(with_mean=True)
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

models = {
    "HistGradientBoosting": CalibratedClassifierCV(
        HistGradientBoostingClassifier(random_state=RND),
        cv=5, method="isotonic"
    ),
}
if HAVE_XGB:
    models["XGBoost"] = CalibratedClassifierCV(
        XGBClassifier(
            n_estimators=800,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RND,
            n_jobs=-1
        ),
        cv=5, method="isotonic"
    )

best = None
for name, clf in models.items():
    print(f"\n=== {name} ===")
    if name == "XGBoost":
        clf.fit(X_train_s, y_train)
        pred = clf.predict(X_test_s)
        proba = clf.predict_proba(X_test_s)
    else:
        clf.fit(X_train_s, y_train)
        pred = clf.predict(X_test_s)
        proba = clf.predict_proba(X_test_s) if hasattr(clf, "predict_proba") else None

    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="binary")
    print(f"acc={acc:.3f} | f1={f1:.3f}")
    print(classification_report(y_test, pred, target_names=["AwayWin/Draw", "HomeWin"]))

    if proba is not None:
        print("log_loss:", f"{log_loss(y_test, proba):.3f}")

    if best is None or f1 > best["f1"]:
        best = {"name": name, "clf": clf, "acc": acc, "f1": f1}

print(f"\n>>> Best model: {best['name']}  acc={best['acc']:.3f}  f1={best['f1']:.3f}")

bundle = {
    "model": best["clf"],
    "model_name": best["name"],
    "scaler": scaler,             
    "features": list(X_train.columns),
    "labels": ["AwayWin/Draw", "HomeWin"],
    "n_roll": N_ROLL
}
joblib.dump(bundle, MODEL)
print("Saved:", MODEL)