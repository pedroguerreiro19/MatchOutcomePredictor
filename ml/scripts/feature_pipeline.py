import pandas as pd
import numpy as np


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


def compute_elo(matches: pd.DataFrame, k=20.0, home_adv=60.0, start_elo=1500.0) -> pd.DataFrame:
 
    m = matches.sort_values("date").reset_index(drop=True).copy()
    ratings = {}
    home_pre = []
    away_pre = []

    for i, row in m.iterrows():
        h, a = row["home_team"], row["away_team"]
        rh = ratings.get(h, start_elo)
        ra = ratings.get(a, start_elo)

        home_pre.append(rh)
        away_pre.append(ra)

        hg, ag = row["home_goals"], row["away_goals"]
        if pd.isna(hg) or pd.isna(ag):
            continue

        if hg > ag: s_h, s_a = 1.0, 0.0
        elif hg < ag: s_h, s_a = 0.0, 1.0
        else: s_h, s_a = 0.5, 0.5

        exp_h = 1.0 / (1.0 + 10 ** (-( (rh + home_adv) - ra ) / 400.0))
        exp_a = 1.0 - exp_h

        ratings[h] = rh + k * (s_h - exp_h)
        ratings[a] = ra + k * (s_a - exp_a)

    m["home_elo_pre"] = pd.Series(home_pre, index=m.index)
    m["away_elo_pre"] = pd.Series(away_pre, index=m.index)
    m["home_elo_diff"] = m["home_elo_pre"] - m["away_elo_pre"]
    return m

def _rolling_team_stats(team_games: pd.DataFrame, n=5):
    g = team_games.sort_values("date").copy()
    g["win"]  = (g["gf"] > g["ga"]).astype(int)
    g["draw"] = (g["gf"] == g["ga"]).astype(int)
    g["loss"] = (g["gf"] < g["ga"]).astype(int)

    g["unbeaten"] = ((g["win"] == 1) | (g["draw"] == 1)).astype(int)
    g["scored"]   = (g["gf"] > 0).astype(int)
    g["clean"]    = (g["ga"] == 0).astype(int)

    roll_cols = ["points","gf","ga","win","draw","loss",
                 "shots","shots_ot","corners","yellow","red","ht_gf","ht_ga",
                 "unbeaten","scored","clean"]

    for c in roll_cols:
        g[f"{c}_r{n}"]   = g[c].rolling(n, min_periods=1).mean().shift(1)
        g[f"{c}_ewm{n}"] = g[c].ewm(span=n, min_periods=1).mean().shift(1)
        g[f"{c}_std{n}"] = g[c].rolling(n, min_periods=1).std().shift(1)

    return g

def build_features(matches: pd.DataFrame, n=5) -> tuple[pd.DataFrame, np.ndarray]:
    m = matches.copy()
    m = m.dropna(subset=["date","home_team","away_team","home_goals","away_goals"]).sort_values("date")
    m = compute_elo(m)

    target = np.where(m["home_goals"] > m["away_goals"], 1, 0)

    home = _team_view(m, "home")
    away = _team_view(m, "away")

    team_stack = pd.concat([home, away], ignore_index=True)
    rolled = (team_stack.groupby("team", group_keys=False)
                        .apply(lambda df: _rolling_team_stats(df, n=n)))

    H = rolled[rolled["is_home"]==1].add_prefix("home_").reset_index(drop=True)
    A = rolled[rolled["is_home"]==0].add_prefix("away_").reset_index(drop=True)

    X = pd.concat([m.reset_index(drop=True), H, A], axis=1)

    mask = X[[f"home_gf_r{n}", f"away_gf_r{n}"]].notna().all(axis=1)
    X = X.loc[mask].reset_index(drop=True)
    target = target[mask.to_numpy()]

    return X, target


def build_features_for_match(hist: pd.DataFrame, home: str, away: str, when_pd: pd.Timestamp, n=5) -> pd.DataFrame:
    past = hist[hist["date"] < when_pd].copy().sort_values("date")
    if past.empty:
        raise ValueError()

    past = compute_elo(past)
    stack = pd.concat([_team_view(past, "home"), _team_view(past, "away")], ignore_index=True)
    rolled = stack.groupby("team", group_keys=False).apply(lambda df: _rolling_team_stats(df, n=n))

    H = rolled[rolled["team"] == home].sort_values("date").tail(1).add_prefix("home_")
    A = rolled[rolled["team"] == away].sort_values("date").tail(1).add_prefix("away_")
    if H.empty or A.empty:
        raise ValueError()

    h_elo = past[(past["home_team"]==home) | (past["away_team"]==home)].sort_values("date").tail(1)
    a_elo = past[(past["home_team"]==away) | (past["away_team"]==away)].sort_values("date").tail(1)
    home_elo_pre = float(h_elo["home_elo_pre"].iloc[0]) if "home_elo_pre" in h_elo else np.nan
    away_elo_pre = float(a_elo["home_elo_pre"].iloc[0]) if "home_elo_pre" in a_elo else np.nan
    elo_df = pd.DataFrame({"home_elo_pre":[home_elo_pre], "away_elo_pre":[away_elo_pre], "home_elo_diff":[home_elo_pre-away_elo_pre]})

    X = pd.concat([elo_df.reset_index(drop=True), H.reset_index(drop=True), A.reset_index(drop=True)], axis=1)
    h2h = _h2h_features(past, home, away, n=5)
    X = pd.concat([elo_df.reset_index(drop=True),
               H.reset_index(drop=True),
               A.reset_index(drop=True),
               h2h.reset_index(drop=True)], axis=1)
    return X

def _h2h_features(hist: pd.DataFrame, home: str, away: str, n=5):
    h2h = hist[((hist["home_team"] == home) & (hist["away_team"] == away)) |
               ((hist["home_team"] == away) & (hist["away_team"] == home))] \
              .sort_values("date").tail(n)

    if h2h.empty:
        return pd.DataFrame([{
            "h2h_games": 0,
            "h2h_home_win_rate": 0.0,
            "h2h_goal_diff_avg": 0.0
        }])

    gd = np.where(h2h["home_team"].eq(home),
                  h2h["home_goals"] - h2h["away_goals"],
                  h2h["away_goals"] - h2h["home_goals"])

    hw = (gd > 0).mean()

    return pd.DataFrame([{
        "h2h_games": len(h2h),
        "h2h_home_win_rate": float(hw),
        "h2h_goal_diff_avg": float(np.mean(gd))
    }])