import pandas as pd
import numpy as np

def rolling_form(df, team_col, goals_for_col, goals_against_col, n=5):
    df = df.sort_values("date")
    res = []
    for team, g in df.groupby(team_col):
        g = g.copy()
        g["win"] = (g[goals_for_col] > g[goals_against_col]).astype(int)
        g["draw"] = (g[goals_for_col] == g[goals_against_col]).astype(int)
        g["loss"] = (g[goals_for_col] < g[goals_against_col]).astype(int)
        g["form_w_" + str(n)] = g["win"].rolling(n, min_periods=1).sum().shift(1)
        g["form_d_" + str(n)] = g["draw"].rolling(n, min_periods=1).sum().shift(1)
        g["form_l_" + str(n)] = g["loss"].rolling(n, min_periods=1).sum().shift(1)
        g["gf_" + str(n)] = g[goals_for_col].rolling(n, min_periods=1).mean().shift(1)
        g["ga_" + str(n)] = g[goals_against_col].rolling(n, min_periods=1).mean().shift(1)
        res.append(g)
    return pd.concat(res)

def build_features(matches: pd.DataFrame) -> pd.DataFrame:
    m = matches.copy()

    home = m.rename(columns={
        "home_team":"team","away_team":"opp",
        "home_goals":"gf","away_goals":"ga"
    })
    home["is_home"] = 1
    home = rolling_form(home, "team", "gf", "ga", 5)


    away = m.rename(columns={
        "away_team":"team","home_team":"opp",
        "away_goals":"gf","home_goals":"ga"
    })
    away["is_home"] = 0
    away = rolling_form(away, "team", "gf", "ga", 5)

    features = home.copy()

    features["target"] = np.select(
        [m["home_goals"]>m["away_goals"], m["home_goals"] == m["away_goals"]],
        ["HOME_WIN","DRAW"],
        default="AWAY_WIN"
    )
    return features