import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_PATH = BASE_DIR / "data" / "raw" / "P1_2526.csv"
OUT_PATH = BASE_DIR / "data" / "clean" / "cleanP1_2526.csv"

df = pd.read_csv(RAW_PATH)

df = df[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]].copy()

df["Date"] = pd.to_datetime(df["Date"], dayfirst = True)

def calc_points(hg, ag):
    if hg > ag: return (3,0)
    if hg < ag: return (0,3)
    else: return (1,1)

df[["home_points", "away_points"]] = df.apply(lambda row: calc_points(row["FTHG"], row["FTAG"]), axis = 1, result_type = "expand")

df = df.rename(columns={
    "Date": "date",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "home_goals",
    "FTAG": "away_goals"
})


df.to_csv(OUT_PATH, index=False)
print(f"matches.csv created in {OUT_PATH}, with {len(df)} matches")