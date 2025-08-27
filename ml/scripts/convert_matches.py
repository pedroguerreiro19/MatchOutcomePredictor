import argparse, sys, glob
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parents[1]
RAW_PATH   = BASE / "data" / "raw"
CLEAN_PATH = BASE / "data" / "clean"


RENAME = {
    "Div": "league_code",
    "Date": "date",
    "Time": "time",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "home_goals", "HG": "home_goals",
    "FTAG": "away_goals", "AG": "away_goals",
    "FTR":  "ft_result", "Res": "ft_result",
    "HTHG": "ht_home_goals",
    "HTAG": "ht_away_goals",
    "HTR":  "ht_result",

    "Attendance": "attendance",
    "Referee": "referee",
    "HS": "home_shots", "AS": "away_shots",
    "HST": "home_shots_ot", "AST": "away_shots_ot",
    "HHW": "home_hit_woodwork", "AHW": "away_hit_woodwork",
    "HC": "home_corners", "AC": "away_corners",
    "HF": "home_fouls", "AF": "away_fouls",
    "HFKC": "home_freekicks_conceded", "AFKC": "away_freekicks_conceded",
    "HO": "home_offsides", "AO": "away_offsides",
    "HY": "home_yellow", "AY": "away_yellow",
    "HR": "home_red",    "AR": "away_red",
    "HBP": "home_bookings_pts", "ABP": "away_bookings_pts",
}

def _to_num(s: pd.Series): return pd.to_numeric(s, errors="coerce")

def _points(hg, ag):
    if pd.isna(hg) or pd.isna(ag): return np.nan, np.nan
    if hg > ag: return 3, 0
    if hg < ag: return 0, 3
    return 1, 1

def _result_from_goals(hg, ag):
    if pd.isna(hg) or pd.isna(ag): return pd.NA
    if hg > ag: return "H"
    if hg < ag: return "A"
    return "D"

def process_one(in_path: Path) -> Path:
    df = pd.read_csv(in_path, low_memory=False)

    cols_present = [c for c in RENAME if c in df.columns]

    out = pd.DataFrame()
    for src in cols_present:
        tgt = RENAME[src]
        if src == "Date":
            out[tgt] = pd.to_datetime(df[src], dayfirst=True, errors="coerce")
        elif src in {
            "FTHG","FTAG","HTHG","HTAG","HS","AS","HST","AST","HHW","AHW",
            "HC","AC","HF","AF","HFKC","AFKC","HO","AO","HY","AY","HR","AR","Attendance"
        }:
            out[tgt] = _to_num(df[src])
        else:
            out[tgt] = df[src].astype("string")

    out["hour"] = out["time"].str.extract(r"^(\d{1,2})", expand=False).astype("Int64") if "time" in out.columns else pd.Series([pd.NA]*len(out))
    out["weekday"] = out["date"].dt.dayofweek

    if "ft_result_raw" not in out.columns and {"home_goals","away_goals"}.issubset(out.columns):
        out["ft_result_raw"] = [_result_from_goals(hg, ag) for hg, ag in zip(out["home_goals"], out["away_goals"])]
    
    if {"home_goals","away_goals"}.issubset(out.columns):
        hp, ap = zip(*[_points(hg, ag) for hg, ag in zip(out["home_goals"], out["away_goals"])])
        out["home_points"], out["away_points"] = hp, ap
        out["goal_diff"]   = out["home_goals"] - out["away_goals"]
        out["goals_total"] = out["home_goals"] + out["away_goals"]

    if "ht_result_raw" not in out.columns and {"ht_home_goals","ht_away_goals"}.issubset(out.columns):
        out["ht_result_raw"] = [_result_from_goals(hg, ag) for hg, ag in zip(out["ht_home_goals"], out["ht_away_goals"])]
    if {"ht_home_goals","ht_away_goals"}.issubset(out.columns):
        out["ht_goal_diff"]   = out["ht_home_goals"] - out["ht_away_goals"]
        out["ht_goals_total"] = out["ht_home_goals"] + out["ht_away_goals"]

    if "home_fouls" not in out.columns and "home_freekicks_conceded" in out.columns:
        out["home_fouls_conceded"] = out["home_freekicks_conceded"]
    if "away_fouls" not in out.columns and "away_freekicks_conceded" in out.columns:
        out["away_fouls_conceded"] = out["away_freekicks_conceded"]

    if "home_bookings_pts" not in out.columns and {"home_yellow","home_red"}.issubset(out.columns):
        out["home_bookings_pts"] = 10*_to_num(out["home_yellow"]) + 25*_to_num(out["home_red"])
    if "away_bookings_pts" not in out.columns and {"away_yellow","away_red"}.issubset(out.columns):
        out["away_bookings_pts"] = 10*_to_num(out["away_yellow"]) + 25*_to_num(out["away_red"])

    out = out.dropna(subset=["date"]).reset_index(drop=True)

    for c in out.select_dtypes(include="string").columns:
        out[c] = out[c].str.strip()

    out_path = CLEAN_PATH / f"clean{in_path.stem}.csv"
    out.to_csv(out_path, index=False)
    print(f" {in_path.name} -> {out_path} ({len(out)} lines, {len(out.columns)} cols)")
    return out_path

def main():
    ap = argparse.ArgumentParser(description="Clean CSVs")
    ap.add_argument("inputs", nargs="+", help="Raw CSVs")
    args = ap.parse_args()

    files = []
    for pat in args.inputs:
        hits = glob.glob(pat)
        if not hits and Path(pat).exists():
            hits = [pat]
        files.extend(hits)

    if not files:
        print("No file found.", file=sys.stderr)
        sys.exit(1)

    for f in files:
        try:
            process_one(Path(f))
        except Exception as e:
            print(f"{f}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()