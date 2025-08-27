from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import warnings

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

try:
    from imblearn.over_sampling import RandomOverSampler
    HAVE_IMB = True
except Exception:
    HAVE_IMB = False


HAVE_XGB = HAVE_LGB = False
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


def ensure_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
    seen = {}
    new_cols = []
    for c in df.columns:
        if c not in seen:
            seen[c] = 0
            new_cols.append(c)
        else:
            seen[c] += 1
            new_cols.append(f"{c}__dup{seen[c]}")
    out = df.copy()
    out.columns = new_cols
    return out

def add_home_away_deltas(X: pd.DataFrame) -> pd.DataFrame:
    X = ensure_unique_columns(X)
    home_suffixes = {c[len("home_"):] for c in X.columns if c.startswith("home_")}
    to_add = {}
    for s in home_suffixes:
        h = f"home_{s}"
        a = f"away_{s}"
        if h in X.columns and a in X.columns:
            delta_name = f"delta_{s}"
            to_add[delta_name] = X[h].to_numpy() - X[a].to_numpy()
    if to_add:
        X = X.assign(**to_add)
    X = X.loc[:, ~X.columns.duplicated()]
    return X

def evaluate(clf, Xtr, ytr, Xte, yte, name: str, class_names: list[str]):
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc = accuracy_score(yte, pred)
    f1  = f1_score(yte, pred, average="macro")
    print(f"\n=== {name} ===")
    print(f"acc={acc:.3f} | f1_macro={f1:.3f}")
    try:
        print(classification_report(yte, pred, target_names=class_names, digits=2))
    except Exception:
        print("classification_report indisponível.")
    if hasattr(clf, "predict_proba"):
        try:
            proba = clf.predict_proba(Xte)
            print("log_loss:", f"{log_loss(yte, proba):.3f}")
        except Exception:
            pass
    return f1, acc


print("Loading:", DATA)
df = pd.read_csv(DATA, parse_dates=["date"])
Xy = build_features(df, n=N_ROLL)

le = LabelEncoder()
y  = le.fit_transform(Xy["target"])
CLASS_NAMES = le.classes_.tolist()  

FORBIDDEN = {
    "home_goals","away_goals","goal_diff","goals_total",
    "ht_home_goals","ht_away_goals","ht_goal_diff","ht_goals_total",
    "home_points","away_points"
}

num_cols = [c for c in Xy.select_dtypes(include=[np.number]).columns if c not in FORBIDDEN]
X = Xy[num_cols].copy().fillna(0.0)

X = add_home_away_deltas(X)

print("Samples:", len(X))
print("Target dist:", dict(zip(CLASS_NAMES, np.bincount(y))))
print("N features:", X.shape[1])

cut = int(len(X) * 0.8)
X_train, X_test = X.iloc[:cut], X.iloc[cut:]
y_train, y_test = y[:cut], y[cut:]

if HAVE_IMB:
    ros = RandomOverSampler(random_state=RND)
    X_train, y_train = ros.fit_resample(X_train, y_train)
    print("After oversample:", dict(zip(CLASS_NAMES, np.bincount(y_train))))
else:
    print("imblearn não disponível — segue sem oversampling.")

scaler = StandardScaler(with_mean=True)
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

models: dict[str, tuple[str, object]] = {
    "LogisticRegression": ("scaled", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=RND)),
    "RandomForest":      ("raw",    RandomForestClassifier(n_estimators=700, random_state=RND, class_weight="balanced_subsample")),
    "GradientBoosting":  ("raw",    GradientBoostingClassifier(random_state=RND)),
    "NaiveBayes":        ("raw",    GaussianNB()),
    "SVC_RBF":           ("scaled", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RND)),
}
if HAVE_XGB:
    models["XGBoost"] = ("raw", XGBClassifier(
        n_estimators=800, learning_rate=0.05, max_depth=5,
        subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0,
        objective="multi:softprob", eval_metric="mlogloss", random_state=RND
    ))


best = None
for name, (kind, clf) in models.items():
    Xtr, Xte = (X_train_s, X_test_s) if kind == "scaled" else (X_train, X_test)
    f1, acc = evaluate(clf, Xtr, y_train, Xte, y_test, name, CLASS_NAMES)
    if (best is None) or (f1 > best["f1"]):
        best = {"name": name, "clf": clf, "acc": acc, "f1": f1, "kind": kind}

print(f"\n>>> Best model: {best['name']}  acc={best['acc']:.3f}  f1={best['f1']:.3f}")

bundle = {
    "model": best["clf"],
    "model_name": best["name"],
    "scaler": scaler,                 
    "features": list(X.columns),      
    "labels": list(y.unique()),            
    "n_roll": N_ROLL
}
joblib.dump(bundle, MODEL)
print("Saved:", MODEL)