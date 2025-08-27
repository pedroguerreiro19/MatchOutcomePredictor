from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, log_loss
from sklearn.ensemble import GradientBoostingClassifier

try:
    from imblearn.over_sampling import RandomOverSampler
    HAVE_IMB = True
except Exception:
    HAVE_IMB = False

HAVE_XGB = False
try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except Exception:
    pass

from feature_pipeline import build_features  

warnings.filterwarnings("ignore", category=UserWarning)
RND = 42

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "clean" / "matches_P1_2122_2526.csv"
MODEL = BASE / "models" / "model.pkl"
MODEL.parent.mkdir(parents=True, exist_ok=True)

N_ROLL = 5  

print("Loading:", DATA)
df = pd.read_csv(DATA, parse_dates=["date"])
Xy, target = build_features(df, n=N_ROLL)

FORBIDDEN = {
    "home_goals","away_goals","goal_diff","goals_total",
    "ht_home_goals","ht_away_goals","ht_goal_diff","ht_goals_total",
    "home_points","away_points"
}

num_cols = [c for c in Xy.select_dtypes(include=[np.number]).columns if c not in FORBIDDEN]
X = Xy[num_cols].copy().fillna(0.0)

print("Samples:", len(X))
print("Target dist:", dict(zip([0,1], np.bincount(target))))
print("N features:", X.shape[1])

cut = int(len(X) * 0.8)
X_train, X_test = X.iloc[:cut], X.iloc[cut:]
y_train, y_test = target[:cut], target[cut:]

if HAVE_IMB:
    ros = RandomOverSampler(random_state=RND)
    X_train, y_train = ros.fit_resample(X_train, y_train)
    print("After oversample:", dict(zip([0,1], np.bincount(y_train))))
else:
    print("imblearn not found.")

scaler = StandardScaler(with_mean=True)
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

models = {
    "GradientBoosting": GradientBoostingClassifier(random_state=RND),
}
if HAVE_XGB:
    models["XGBoost"] = XGBClassifier(
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
    )

best = None
for name, clf in models.items():
    print(f"\n=== {name} ===")
    Xtr, Xte = (X_train_s, X_test_s)
    clf.fit(Xtr, y_train)
    pred = clf.predict(Xte)
    acc = accuracy_score(y_test, pred)
    f1  = f1_score(y_test, pred, average="binary")
    print(f"acc={acc:.3f} | f1={f1:.3f}")
    print(classification_report(y_test, pred, target_names=["AwayWin/Draw","HomeWin"]))
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(Xte)
        print("log_loss:", f"{log_loss(y_test, proba):.3f}")

    if best is None or f1 > best["f1"]:
        best = {"name": name, "clf": clf, "acc": acc, "f1": f1}

print(f"\n>>> Best model: {best['name']}  acc={best['acc']:.3f}  f1={best['f1']:.3f}")

bundle = {
    "model": best["clf"],
    "model_name": best["name"],
    "scaler": scaler,
    "features": list(X.columns),
    "labels": ["AwayWin/Draw", "HomeWin"],
    "n_roll": N_ROLL
}
joblib.dump(bundle, MODEL)
print("Saved:", MODEL)