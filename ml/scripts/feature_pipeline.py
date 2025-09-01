import pandas as pd
import numpy as np

def _team_view(m: pd.DataFrame, side: str) -> pd.DataFrame:
    opp = "away" if side == "home" else "home"
    return pd.DataFrame({
        "date": m["date"],
        "team": m[f"{side}_team"],
        "opp":  m[f"{opp}_team"],
        "gf": m[f"{side}_goals"],
        "ga": m[f"{opp}_goals"],
        "points": np.where(
            m[f"{side}_goals"].notna() & m[f"{opp}_goals"].notna(),
            np.where(m[f"{side}_goals"] > m[f"{opp}_goals"], 3,
            np.where(m[f"{side}_goals"] == m[f"{opp}_goals"], 1, 0)),
            np.nan
        )
    })


def compute_elo(matches: pd.DataFrame, k=20.0, home_adv=0.0, start_elo=1500.0) -> pd.DataFrame:
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

    pts_home = np.where(past["home_goals"] > past["away_goals"], 3,
                 np.where(past["home_goals"] == past["away_goals"], 1, 0))
    pts_away = np.where(past["away_goals"] > past["home_goals"], 3,
                 np.where(past["away_goals"] == past["home_goals"], 1, 0))

    standings = (
        pd.concat([
            pd.DataFrame({"team": past["home_team"], "pts": pts_home, "gf": past["home_goals"], "ga": past["away_goals"]}),
            pd.DataFrame({"team": past["away_team"], "pts": pts_away, "gf": past["away_goals"], "ga": past["home_goals"]})
        ])
        .groupby("team", as_index=False)
        .sum()
    )

    standings["gd"] = standings["gf"] - standings["ga"]
    standings = standings.sort_values(["pts","gd","gf"], ascending=False)
    standings["rank"] = range(1, len(standings) + 1)

    return dict(zip(standings["team"], standings["rank"]))


def build_features(matches: pd.DataFrame, n=5, mode="train") -> tuple[pd.DataFrame, np.ndarray]:
    m = matches.dropna(subset=["date","home_team","away_team"]).sort_values("date").copy()

    target = None
    if mode == "train":
        m = m.dropna(subset=["home_goals","away_goals"]).copy()
        target = np.where(
            m["home_goals"] > m["away_goals"], 1,
            np.where(m["home_goals"] < m["away_goals"], 0, 2) 
        )

    m = compute_elo(m)

    home = _team_view(m, "home")
    away = _team_view(m, "away")
    stack = pd.concat([home, away], ignore_index=True)
    rolled = stack.groupby("team", group_keys=False).apply(lambda df: _rolling_team_stats(df, n=n))

    H = rolled[rolled["team"].isin(m["home_team"])].add_prefix("home_").reset_index(drop=True)
    A = rolled[rolled["team"].isin(m["away_team"])].add_prefix("away_").reset_index(drop=True)

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

    feats = {}
    for c in ["gf","ga","win","draw","loss","points"]:
        feats[f"{c}_diff_r{n}"] = X[f"home_{c}_r{n}"] - X[f"away_{c}_r{n}"]

    feats["elo_diff"] = X["elo_diff"]
    feats["rank_diff"] = X["rank_diff"]

    Xf = pd.DataFrame(feats).fillna(0.0)

    if target is not None:
        target = target[:len(Xf)]
        return Xf.reset_index(drop=True), target
    else:
        return Xf.reset_index(drop=True), None
        

def build_features_for_match(hist: pd.DataFrame, home: str, away: str, n=5, features=None) -> pd.DataFrame:
    past = hist.copy()
    if past.empty:
        raise ValueError("No history available.")

    past = compute_elo(past)

    home_view = _team_view(past, "home")
    away_view = _team_view(past, "away")
    stack = pd.concat([home_view, away_view], ignore_index=True)

    rolled = stack.groupby("team", group_keys=False).apply(lambda df: _rolling_team_stats(df, n=n))

    H = rolled[rolled["team"] == home].sort_values("date").tail(1).add_prefix("home_")
    A = rolled[rolled["team"] == away].sort_values("date").tail(1).add_prefix("away_")
    if H.empty or A.empty:
        raise ValueError(f"Not enough history for {home} or {away}.")

    base = pd.concat([H.reset_index(drop=True), A.reset_index(drop=True)], axis=1)

    last_date = past["date"].max()
    ranks = _ranking_at_date(past, last_date + pd.Timedelta(days=1))
    rank_diff = (ranks.get(away, 0) - ranks.get(home, 0))

    feats = {}
    for c in ["gf","ga","win","draw","loss","points"]:
        feats[f"{c}_diff_r{n}"] = base[f"home_{c}_r{n}"].values[0] - base[f"away_{c}_r{n}"].values[0]

    feats["elo_diff"] = (base.get("home_elo_pre", pd.Series([0])).values[0] - base.get("away_elo_pre", pd.Series([0])).values[0])
    feats["rank_diff"] = rank_diff

    X = pd.DataFrame([feats])

    if features is not None:
        X = X.reindex(columns=features, fill_value=0.0)

    return X
