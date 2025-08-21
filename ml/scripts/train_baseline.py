import pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from feature_pipeline import build_features

df = pd.read_csv("data/raw/matches.csv", parse_dates=["date"])
Xy = build_features(df).dropna()

y = Xy["target"]
X = Xy.select_dtypes(include=["number"]).drop(columns=[], errors="ignore")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2, shuffle = False)

scaler = StandardScaler(with_mean = False)  
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

models = {
    "logreg": LogisticRegression(max_iter = 1000),
    "xgb": XGBClassifier(eval_metric = "mlogloss")
}
best_name, best_model, best_acc = None, None, 0.0

for name, model in models.items():
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)
    acc = accuracy_score(y_test, preds)
    print(name, "acc", acc, "f1", f1_score(y_test, preds, average = "macro"))
    if acc > best_acc:
        best_acc, best_name, best_model = acc, name, model

joblib.dump({"model": best_model, "scaler": scaler, "features": X.columns.tolist()}, "models/model.pkl")
print("Saved best:", best_name, "acc", best_acc)