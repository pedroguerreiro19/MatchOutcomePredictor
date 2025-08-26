from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
CLEAN_PATH = BASE / "data" / "clean"
OUT_PATH = BASE / "data" / "clean" / "matches_P1_2122_2526.csv"

dfs = []
for p in sorted(CLEAN_PATH.glob("cleanP1_*.csv")):
    df = pd.read_csv(p, parse_dates = ["date"])
    df["season_file"] = p.stem
    dfs.append(df)

full = pd.concat(dfs, ignore_index = True)

full = full.sort_values("date").drop_duplicates(subset = ["date", "home_team", "away_team"], keep = "first")

OUT_PATH.parent.mkdir(parents = True, exist_ok = True)
full.to_csv(OUT_PATH, index = True)

