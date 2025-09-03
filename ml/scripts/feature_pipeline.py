import pandas as pd
import numpy as np


def _team_view(m: pd.DataFrame, side: str) -> pd.DataFrame:
    opp = "away" if side == "home" else "home"
    return pd.DataFrame({
        "date": m["date"],
        "team": m[f"{side}_team"],
        "opp": m[f"{opp}_team"],
        "gf": m[f"{side}_goals"],
        "ga": m[f"{opp}_goals"],
        "shots": m.get(f"{side}_shots", np.nan),
        "shots_ot": m.get(f"{side}_shots_ot", np.nan),
        "fouls": m.get(f"{side}_fouls", np.nan),
        "yellow": m.get(f"{side}_yellow", np.nan),
        "red": m.get(f"{side}_red", np.nan),
        "corners": m.get(f"{side}_corners", np.nan),
        "points": np.where(
            m[f"{side}_goals"].notna() & m[f"{opp}_goals"].notna(),
            np.where(m[f"{side}_goals"] > m[f"{opp}_goals"], 3,
            np.where(m[f"{side}_goals"] == m[f"{opp}_goals"], 1, 0)),
            np.nan
        ),
        "is_home": 1 if side == "home" else 0
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


def _rolling_team_stats(team_games: pd.DataFrame, n=5, shift_values=True):
    g = team_games.sort_values("date").copy()
    print(f"Team stats input: {len(g)} games, team: {g['team'].iloc[0] if len(g) > 0 else 'EMPTY'}")
    
    g["win"]  = (g["gf"] > g["ga"]).astype(int)
    g["draw"] = (g["gf"] == g["ga"]).astype(int) 
    g["loss"] = (g["gf"] < g["ga"]).astype(int)

    cols = ["gf","ga","win","draw","loss","points","shots","shots_ot","fouls","yellow","red","corners"]
    for c in cols:
        if c in g:
            roll = g[c].rolling(n, min_periods=1).mean()
            g[f"{c}_r{n}"] = roll.shift(1) if shift_values else roll
            print(f"  {c}_r{n}: min={g[f'{c}_r{n}'].min():.2f}, max={g[f'{c}_r{n}'].max():.2f}")

    return g


def _season_year(date: pd.Timestamp) -> int:
    return date.year if date.month >= 8 else date.year - 1

def _ranking_at_date(df: pd.DataFrame, date: pd.Timestamp, n_teams: int = 18) -> dict:
    season_year = _season_year(date)
    season = df[df["date"].apply(_season_year) == season_year]
    past = season[season["date"] < date].copy()
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
        .groupby("team", as_index=False).sum()
    )

    standings["gd"] = standings["gf"] - standings["ga"]
    standings = standings.sort_values(["pts", "gd", "gf"], ascending=False)
    standings["rank"] = range(1, len(standings) + 1)

    rank_map = dict(zip(standings["team"], standings["rank"]))
    for team in set(df["home_team"]) | set(df["away_team"]):
        if team not in rank_map:
            rank_map[team] = n_teams  
    return rank_map

def _h2h_stats(matches: pd.DataFrame, home: str, away: str, n=5):
    h2h = matches[((matches["home_team"] == home) & (matches["away_team"] == away)) |
                  ((matches["home_team"] == away) & (matches["away_team"] == home))].sort_values("date").tail(n)

    if h2h.empty:
        return {"h2h_win_home": 0, "h2h_win_away": 0, "h2h_draw": 0,
                "h2h_gf_home": 0, "h2h_gf_away": 0}

    home_wins = np.sum((h2h["home_team"] == home) & (h2h["home_goals"] > h2h["away_goals"]))
    away_wins = np.sum((h2h["away_team"] == away) & (h2h["away_goals"] > h2h["home_goals"]))
    draws = np.sum(h2h["home_goals"] == h2h["away_goals"])

    gf_home = np.sum(np.where(h2h["home_team"] == home, h2h["home_goals"], h2h["away_goals"]))
    gf_away = np.sum(np.where(h2h["away_team"] == away, h2h["away_goals"], h2h["home_goals"]))

    return {
        "h2h_win_home": home_wins,
        "h2h_win_away": away_wins,
        "h2h_draw": draws,
        "h2h_gf_home": gf_home,
        "h2h_gf_away": gf_away
    }


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

    # FIXED: Proper alignment of features with matches
    home_features = []
    away_features = []
    
    for idx, row in m.iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]
        match_date = row["date"]
        
        # Get the most recent stats for each team before this match
        home_stats = rolled[
            (rolled["team"] == home_team) & 
            (rolled["date"] <= match_date)
        ].sort_values("date").tail(1)
        
        away_stats = rolled[
            (rolled["team"] == away_team) & 
            (rolled["date"] <= match_date)
        ].sort_values("date").tail(1)
        
        # Create feature row for this match
        home_row = {}
        away_row = {}
        
        stat_cols = [f"{c}_r{n}" for c in ["gf","ga","win","draw","loss","points","shots","shots_ot","fouls","yellow","red","corners"]]
        
        for col in stat_cols:
            if len(home_stats) > 0 and col in home_stats.columns:
                home_row[f"home_{col}"] = home_stats[col].iloc[0]
            else:
                home_row[f"home_{col}"] = 0.0
                
            if len(away_stats) > 0 and col in away_stats.columns:
                away_row[f"away_{col}"] = away_stats[col].iloc[0]
            else:
                away_row[f"away_{col}"] = 0.0
        
        home_features.append(home_row)
        away_features.append(away_row)
    
    # Convert to DataFrames
    H = pd.DataFrame(home_features)
    A = pd.DataFrame(away_features)
    
    # Combine with match data
    X = pd.concat([m.reset_index(drop=True), H.reset_index(drop=True), A.reset_index(drop=True)], axis=1)

    # Add rankings
    ranks_list = []
    for _, row in m.iterrows():
        ranks = _ranking_at_date(m, row["date"])
        ranks_list.append({
            "home_rank": ranks.get(row["home_team"], np.nan),
            "away_rank": ranks.get(row["away_team"], np.nan)
        })
    ranks_df = pd.DataFrame(ranks_list)
    ranks_df["rank_diff"] = ranks_df["home_rank"] - ranks_df["away_rank"]
    X = pd.concat([X.reset_index(drop=True), ranks_df.reset_index(drop=True)], axis=1)

    # Calculate feature differences
    feats = {}
    for c in ["gf","ga","win","draw","loss","points","shots","shots_ot","fouls","yellow","red","corners"]:
        home_col = f"home_{c}_r{n}"
        away_col = f"away_{c}_r{n}"
        if home_col in X.columns and away_col in X.columns:
            feats[f"{c}_diff_r{n}"] = X[home_col] - X[away_col]
        else:
            feats[f"{c}_diff_r{n}"] = 0.0

    feats["elo_diff"] = X["elo_diff"]
    feats["rank_diff"] = X["rank_diff"]

    print(f"\n=== SAMPLE FEATURE CALCULATION DEBUG ===")
    if len(feats) > 0:
        sample_idx = 0 if len(m) == 0 else min(5, len(m)-1)
        print(f"Match {sample_idx}: {m.iloc[sample_idx]['home_team']} vs {m.iloc[sample_idx]['away_team']}")
        for feat, values in feats.items():
            if hasattr(values, 'iloc'):
                val = values.iloc[sample_idx] if len(values) > sample_idx else 0
                if abs(val) > 0.001:  
                    print(f"  {feat}: {val:.3f}")
    print("==========================================\n")

    Xf = pd.DataFrame(feats).fillna(0.0)
    if target is not None:
        min_len = min(len(Xf), len(target))
        return Xf.iloc[:min_len].reset_index(drop=True), target[:min_len]
    else:
        return Xf.reset_index(drop=True), None


def build_features_for_match(hist: pd.DataFrame, home: str, away: str, n=5, features=None) -> pd.DataFrame:
    past = hist.copy()
    if past.empty:
        raise ValueError("No history available.")

    print(f"\n=== DEBUG: Building features for {home} vs {away} ===")
    print(f"Historical data: {len(past)} matches from {past['date'].min()} to {past['date'].max()}")
    
    home_matches = past[(past['home_team'] == home) | (past['away_team'] == home)]
    away_matches = past[(past['home_team'] == away) | (past['away_team'] == away)]
    print(f"{home} matches in history: {len(home_matches)}")
    print(f"{away} matches in history: {len(away_matches)}")
    
    if len(home_matches) == 0:
        print(f"ERROR: No historical data found for {home}")
        print(f"Available home teams: {sorted(past['home_team'].unique())[:10]}...")
    if len(away_matches) == 0:
        print(f"ERROR: No historical data found for {away}")
        print(f"Available away teams: {sorted(past['away_team'].unique())[:10]}...")

    past = compute_elo(past)
    
    final_elos = {}
    for _, row in past.iterrows():
        final_elos[row['home_team']] = row['home_elo_pre']
        final_elos[row['away_team']] = row['away_elo_pre']
    
    print(f"Final ELO - {home}: {final_elos.get(home, 'NOT FOUND')}")
    print(f"Final ELO - {away}: {final_elos.get(away, 'NOT FOUND')}")

    home_view = _team_view(past, "home")
    away_view = _team_view(past, "away")
    stack = pd.concat([home_view, away_view], ignore_index=True)
    rolled = stack.groupby("team", group_keys=False).apply(lambda df: _rolling_team_stats(df, n=n, shift_values=False))

    print(f"\nTeams in rolled stats: {sorted(rolled['team'].unique())}")
    
    H = rolled[rolled["team"] == home].sort_values("date").tail(1).add_prefix("home_")
    A = rolled[rolled["team"] == away].sort_values("date").tail(1).add_prefix("away_")
    
    print(f"\nHome team ({home}) stats shape: {H.shape}")
    print(f"Away team ({away}) stats shape: {A.shape}")
    
    if H.empty:
        print(f"ERROR: No rolled stats found for home team {home}")
        available_home_teams = rolled['team'].unique()
        print(f"Available teams in rolled stats: {sorted(available_home_teams)}")
        raise ValueError(f"Not enough history for home team {home}.")
        
    if A.empty:
        print(f"ERROR: No rolled stats found for away team {away}")
        available_away_teams = rolled['team'].unique()
        print(f"Available teams in rolled stats: {sorted(available_away_teams)}")
        raise ValueError(f"Not enough history for away team {away}.")

    if not H.empty:
        print(f"\n{home} recent form (last {n} games):")
        print(f"  Goals scored: {H[f'home_gf_r{n}'].values[0]:.2f}")
        print(f"  Goals conceded: {H[f'home_ga_r{n}'].values[0]:.2f}")
        print(f"  Wins: {H[f'home_win_r{n}'].values[0]:.2f}")
        print(f"  Points: {H[f'home_points_r{n}'].values[0]:.2f}")
    
    if not A.empty:
        print(f"\n{away} recent form (last {n} games):")
        print(f"  Goals scored: {A[f'away_gf_r{n}'].values[0]:.2f}")
        print(f"  Goals conceded: {A[f'away_ga_r{n}'].values[0]:.2f}")
        print(f"  Wins: {A[f'away_win_r{n}'].values[0]:.2f}")
        print(f"  Points: {A[f'away_points_r{n}'].values[0]:.2f}")

    base = pd.concat([H.reset_index(drop=True), A.reset_index(drop=True)], axis=1)

    last_date = past["date"].max()
    ranks = _ranking_at_date(past, last_date + pd.Timedelta(days=1))
    
    print(f"\nCurrent rankings:")
    print(f"  {home}: {ranks.get(home, 'NOT FOUND')}")
    print(f"  {away}: {ranks.get(away, 'NOT FOUND')}")
    
    rank_diff = (ranks.get(home, 18) - ranks.get(away, 18))  

    feats = {}
    for c in ["gf","ga","win","draw","loss","points","shots","shots_ot","fouls","yellow","red","corners"]:
        if f"home_{c}_r{n}" in base.columns and f"away_{c}_r{n}" in base.columns:
            feats[f"{c}_diff_r{n}"] = base[f"home_{c}_r{n}"].values[0] - base[f"away_{c}_r{n}"].values[0]
        else:
            feats[f"{c}_diff_r{n}"] = 0.0

    home_elo = final_elos.get(home, 1500.0)
    away_elo = final_elos.get(away, 1500.0)
    feats["elo_diff"] = home_elo - away_elo
    feats["rank_diff"] = rank_diff

    print(f"\nCalculated ELO diff: {home_elo:.1f} - {away_elo:.1f} = {feats['elo_diff']:.1f}")
    print(f"Calculated rank diff: {ranks.get(home, 18)} - {ranks.get(away, 18)} = {feats['rank_diff']}")

    h2h = _h2h_stats(past, home, away, n=n)
    feats.update(h2h)

    X = pd.DataFrame([feats])
    if features is not None:
        X = X.reindex(columns=features, fill_value=0.0)

    print(f"\n=== Final Features for {home} vs {away} ===")
    for col, val in X.iloc[0].items():
        if abs(val) > 0.001:  
            print(f"{col}: {val}")
    print("===================================\n")

    
    return X