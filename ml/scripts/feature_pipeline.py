import pandas as pd
import numpy as np 

def _team_view(m, side):
    opp = "away" if side == "home" else "home"
    return pd.DataFrame({
        "date": m["date"],
        "team": m[f"{side}_team"],
        "opp":  m[f"{opp}_team"],
        "is_home": 1 if side == "home" else 0,
        "gf": m[f"{side}_goals"],
        "ga": m[f"{opp}_goals"],
        "points": m[f"{side}_points"] if f"{side}_points" in m.columns else np.nan
    })


def compute_elo(matches: pd.DataFrame, k=20.0, home_adv=50.0, start_elo=1500.0) -> pd.DataFrame:
    m = matches.sort_values("date").reset_index(drop=True).copy()
    ratings = {}
    home_pre, away_pre = [], []

    for _, row in m.iterrows():
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

        exp_h = 1.0 / (1.0 + 10 ** (-((rh + home_adv) - ra) / 400.0))
        exp_a = 1.0 - exp_h

        ratings[h] = rh + k * (s_h - exp_h)
        ratings[a] = ra + k * (s_a - exp_a)

    m["home_elo_pre"] = home_pre
    m["away_elo_pre"] = away_pre
    m["elo_diff"] = m["home_elo_pre"] - m["away_elo_pre"]
    return m


def _rolling_team_stats(team_games: pd.DataFrame, n=5):
    g = team_games.sort_values("date").copy()
    g["win"]  = (g["gf"] > g["ga"]).astype(int)
    g["draw"] = (g["gf"] == g["ga"]).astype(int)
    g["loss"] = (g["gf"] < g["ga"]).astype(int)

    for c in ["gf","ga","win","draw","loss","points"]:
        g[f"{c}_r{n}"] = g[c].rolling(n, min_periods=1).mean().shift(1)

    return g


def _ranking_at_date(df: pd.DataFrame, date: pd.Timestamp) -> dict:
    past = df[df["date"] < date].copy()
    if past.empty:
        return {}

    standings = past.groupby("home_team").agg(
        pts=("home_points", "sum"),
        gf=("home_goals", "sum"),
        ga=("away_goals", "sum")
    ).reset_index()

    standings["gd"] = standings["gf"] - standings["ga"]
    standings = standings.sort_values(["pts","gd","gf"], ascending=False)
    standings["rank"] = range(1, len(standings) + 1)

    return dict(zip(standings["home_team"], standings["rank"]))


def build_features(matches: pd.DataFrame, n=5, mode="train") -> tuple[pd.DataFrame, np.ndarray]:
    m = matches.dropna(subset=["date","home_team","away_team"])\
               .sort_values("date").copy()

    target = None
    if mode == "train":
        m = m.dropna(subset=["home_goals","away_goals"])
        target = np.where(m["home_goals"] > m["away_goals"], 1, 0)

    m = compute_elo(m)

    home = _team_view(m, "home")
    away = _team_view(m, "away")
    stack = pd.concat([home, away], ignore_index=True)

    rolled = stack.groupby("team", group_keys=False)\
                  .apply(lambda df: _rolling_team_stats(df, n=n))

    H = rolled[rolled["is_home"] == 1].add_prefix("home_").reset_index(drop=True)
    A = rolled[rolled["is_home"] == 0].add_prefix("away_").reset_index(drop=True)

    X = pd.concat([m.reset_index(drop=True), H, A], axis=1)

    ranks_list = []
    for _, row in m.iterrows():
        ranks = _ranking_at_date(m, row["date"])
        ranks_list.append({
            "home_rank": ranks.get(row["home_team"], np.nan),
            "away_rank": ranks.get(row["away_team"], np.nan)
        })
    ranks_df = pd.DataFrame(ranks_list)
    ranks_df["rank_diff"] = ranks_df["away_rank"] - ranks_df["home_rank"]

    X = pd.concat([X.reset_index(drop=True), ranks_df.reset_index(drop=True)], axis=1)

    valid = X[[f"home_gf_r{n}", f"away_gf_r{n}"]].notna().all(axis=1)
    X = X.loc[valid].reset_index(drop=True)
    if target is not None:
        target = target[valid.to_numpy()]

    return X, target