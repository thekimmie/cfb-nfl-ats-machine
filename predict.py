# predict.py
"""
Contract:
- Expose: def predict_games(games: list[dict]) -> list[dict]
- Each input game contains: league, game_time, home_team, away_team,
  sportsbook_spread, sportsbook_total, sportsbook_ml_odds_home, sportsbook_ml_odds_away
- We return those join fields + model fields: model_ml_prob_home/away, model_pick, etc.

NOTE: This wires in your RandomForestClassifier saved as final_ats_model.pkl.
For now, we only map 'spread' from the live data. The remaining features are
set to 0 as a temporary fallback — replace the TODO section to compute real values.
"""

import os
import pickle
import math
import numpy as np

# --- Where to find the model ---
PICKLE_PATH = os.getenv("MODEL_WEIGHTS_PATH", "final_ats_model.pkl")

# --- Your training-time features ---
FEATURES = [
    "spread",
    "qb_passing_yards",
    "qb_rushing_yards",
    "qb_total_epa",
    "qb_turnovers",
    "opponent_def_epa_allowed",
    "opponent_def_pass_yards_allowed",
]

# Load the trained estimator once (kept warm by the FastAPI module cache)
with open(PICKLE_PATH, "rb") as f:
    model = pickle.load(f)

def _implied_prob_from_moneyline(odds):
    if odds is None: return None
    try:
        o = float(odds)
    except Exception:
        return None
    if o < 0:
        return (-o) / ((-o) + 100.0)
    return 100.0 / (o + 100.0)

def _feature_row_from_game(g):
    """
    Map the incoming game dict to your model's FEATURES order.
    TEMP: only 'spread' is live; others are placeholders (0.0).
    Replace the TODO lines to compute real values.
    """
    spread = g.get("sportsbook_spread")

    # -------- TODO: replace these placeholders with real feature engineering --------
    qb_passing_yards = 0.0
    qb_rushing_yards = 0.0
    qb_total_epa = 0.0
    qb_turnovers = 0.0
    opponent_def_epa_allowed = 0.0
    opponent_def_pass_yards_allowed = 0.0
    # -----------------------------------------------------------------------------

    vals = {
        "spread": float(spread) if spread is not None else 0.0,
        "qb_passing_yards": float(qb_passing_yards),
        "qb_rushing_yards": float(qb_rushing_yards),
        "qb_total_epa": float(qb_total_epa),
        "qb_turnovers": float(qb_turnovers),
        "opponent_def_epa_allowed": float(opponent_def_epa_allowed),
        "opponent_def_pass_yards_allowed": float(opponent_def_pass_yards_allowed),
    }
    return np.array([vals[f] for f in FEATURES], dtype=float)

def predict_games(games):
    """
    Returns one dict per input, keyed so main.py can join:
    league, game_time, home_team, away_team + model fields.
    """
    out = []
    if not isinstance(games, list):
        return out

    for g in games:
        X = _feature_row_from_game(g).reshape(1, -1)

        # RandomForestClassifier: we can use predict_proba
        # Assuming y=1 means "home team covers" in your training data (adjust if reversed!)
        try:
            proba = float(model.predict_proba(X)[0, 1])
        except Exception:
            # Fallback if the model lacks predict_proba (shouldn't happen for RF)
            pred = int(model.predict(X)[0])
            proba = 0.75 if pred == 1 else 0.25

        p_home = proba
        p_away = 1.0 - p_home

        # Choose a pick
        home_team = g.get("home_team")
        away_team = g.get("away_team")
        model_pick = home_team if p_home >= 0.5 else away_team
        confidence_pct = round(100.0 * max(p_home, p_away), 1)

        # Optional: compare to moneyline implied prob to make a basic value flag
        book_p_home = _implied_prob_from_moneyline(g.get("sportsbook_ml_odds_home"))
        ml_value_flag = False
        if book_p_home is not None:
            # Value if model prob exceeds the book implied by a threshold
            ml_value_flag = (p_home - book_p_home) >= 0.05

        out.append({
            # join keys
            "league": (g.get("league") or "").upper(),
            "game_time": g.get("game_time"),
            "home_team": home_team,
            "away_team": away_team,

            # model outputs we currently support
            "model_ml_prob_home": p_home,
            "model_ml_prob_away": p_away,
            "model_pick": model_pick,
            "confidence_pct": confidence_pct,
            "ml_value_flag": ml_value_flag,

            # leave spread/total model fields out for now (or set None)
            "model_spread": None,
            "model_total": None,
            "edge_vs_line": None,
            "ou_pick": None,
            "ou_confidence_pct": None,
            "total_edge": None,

            # optional signals
            "trap_alert": False,
            "sharp_flag": False,
            "volatility_score": 0.20,
        })

    return out
