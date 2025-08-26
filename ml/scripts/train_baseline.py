from pathlib import Path
import pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from feature_pipeline import build_features

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "clean" / "matches_P1_2122_2526.csv"   
MODEL = BASE / "models" / "model.pkl"


df = pd.read_csv(DATA, parse_dates=["date"])
Xy = build_features(df, n=5)

y = Xy["target"]

forbidden = ["home_goals","away_goals","goal_diff","goals_total",
             "ht_home_goals","ht_away_goals","ht_goal_diff","ht_goals_total",
             "home_points","away_points"]

num_cols = [c for c in Xy.select_dtypes(include=["number"]).columns if c not in forbidden]
X = Xy[num_cols]

print("Samples:", len(X))
print("Distribution:", y.value_counts().to_dict())


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, stratify=y, random_state=42
)


scaler = StandardScaler(with_mean=False)
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

clf = LogisticRegression(max_iter=3000, class_weight="balanced")
clf.fit(X_train_s, y_train)

pred = clf.predict(X_test_s)
print("acc:", f"{accuracy_score(y_test, pred):.3f}",
      "| f1_macro:", f"{f1_score(y_test, pred, average='macro'):.3f}")
print(classification_report(y_test, pred))


joblib.dump({"model": clf, "scaler": scaler, "features": num_cols, "labels": sorted(y.unique().tolist())}, MODEL)
print("Saved:", MODEL)