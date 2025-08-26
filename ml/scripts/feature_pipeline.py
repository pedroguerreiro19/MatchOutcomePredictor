import pandas as pd
import numpy as np

NUM_FEATS_AVAILABLE = [
    "home_points","away_points",
    "home_goals","away_goals",
    "home_shots","away_shots",
    "home_shots_ot","away_shots_ot",
    "home_corners","away_corners",
    "home_yellow","away_yellow",
    "home_red","away_red",
    "ht_home_goals","ht_away_goals",
    "goal_diff","goals_total",
    "ht_goal_diff","ht_goals_total",
]

def _col(df, name, default=0.0):
    return df[name] if name in df.columns else default

def _team_view(m, side):
    opp = "away" if side == "home" else "home"
    out = pd.DataFrame({
        "date": m["date"],
        "team": m[f"{side}_team"],
        "opp":  m[f"{opp}_team"],
        "is_home": 1 if side=="home" else 0,
        "gf": m[f"{side}_goals"],
        "ga": m[f"{opp}_goals"],
        "points": m[f"{side}_points"] if f"{side}_points" in m.columns else np.nan,
        "shots": _col(m, f"{side}_shots", 0),
        "shots_ot": _col(m, f"{side}_shots_ot", 0),
        "corners": _col(m, f"{side}_corners", 0),
        "yellow": _col(m, f"{side}_yellow", 0),
        "red": _col(m, f"{side}_red", 0),
        "ht_gf": _col(m, f"ht_{side}_goals", np.nan),
        "ht_ga": _col(m, f"ht_{opp}_goals", np.nan),
    })
    return out

def _rolling_team_stats(team_games: pd.DataFrame, n=5):
    g = team_games.sort_values("date").copy()
    g["win"]  = (g["gf"] > g["ga"]).astype(int)
    g["draw"] = (g["gf"] == g["ga"]).astype(int)
    g["loss"] = (g["gf"] < g["ga"]).astype(int)

    roll_cols = ["points","gf","ga","win","draw","loss",
                 "shots","shots_ot","corners","yellow","red","ht_gf","ht_ga"]
    for c in roll_cols:
        g[f"{c}_r{n}"] = g[c].rolling(n, min_periods=1).mean().shift(1)
    return g

def build_features(matches: pd.DataFrame, n=5) -> pd.DataFrame:
    m = matches.copy()
    m = m.dropna(subset=["date","home_team","away_team","home_goals","away_goals"]).sort_values("date")

    target = np.select(
        [m["home_goals"] > m["away_goals"], m["home_goals"] == m["away_goals"]],
        ["HOME_WIN","DRAW"], default="AWAY_WIN"
    )

    home = _team_view(m, "home")
    away = _team_view(m, "away")

    team_stack = pd.concat([home, away], ignore_index=True)
    rolled = (team_stack.groupby("team", group_keys=False)
                        .apply(lambda df: _rolling_team_stats(df, n=n)))

    H = rolled[rolled["is_home"]==1].add_prefix("home_").reset_index(drop=True)
    A = rolled[rolled["is_home"]==0].add_prefix("away_").reset_index(drop=True)

    X = pd.concat([m.reset_index(drop=True), H, A], axis=1)
    X["target"] = target


    return X.dropna(subset=["home_gf_r5","away_gf_r5"], how="any")