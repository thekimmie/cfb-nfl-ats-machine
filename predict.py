# predict.py
import os
import pickle
from pathlib import Path
from typing import List, Dict, Any

# Where to find the trained model (.pkl)
MODEL_WEIGHTS_PATH = os.getenv("MODEL_WEIGHTS_PATH", "final_ats_model.pkl")

# Features your model expects (order matters). Adjust to your trained set.
FEATURES = [
    "spread",
    "qb_passing_yards",
    "qb_rushing_yards",
    "qb_total_epa",
    "qb_turnovers",
    "opponent_def_epa_allowed",
    "opponent_def_pass_yards_allowed",
]

_model = None
_model_error = None

def _load_model():
    global _model, _model_error
    if _model is not None or _model_error is not None:
        return
    try:
        p = Path(MODEL_WEIGHTS_PATH)
        if not p.exists():
            _model_error = f"Model file not found at {p.resolve()}"
            return
        with p.open("rb") as f:
            _model = pickle.load(f)
    except Exception as e:
        _model_error = f"Failed to load model: {e}"

def _moneyline_to_prob(ml: float):
    """Convert American moneyline to implied probability (0..1)."""
    try:
        ml = float(ml)
    except Exception:
        return None
    if ml >= 0:
        return 100.0 / (ml + 100.0)
    else:
        return (-ml) / (-ml + 100.0)

def _row_to_features(g: Dict[str, Any]) -> list:
    """
    Map your incoming 'games' dict to the trained feature vector.
    We use reasonable defaults if missing so this never crashes.
    """
    # Basic spread proxy from the API: home spread w.r.t. sportsbook point
    spread = g.get("sportsbook_spread")
    # If your training used "spread = home_line", use it directly. Default 0 if unknown.
    try:
        spread = float(spread) if spread is not None else 0.0
    except Exception:
        spread = 0.0

    # The rest are (for now) placeholders unless you wire real signals in main.py
    qb_passing_yards = float(g.get("qb_passing_yards") or 0.0)
    qb_rushing_yards = float(g.get("qb_rushing_yards") or 0.0)
    qb_total_epa     = float(g.get("qb_total_epa") or 0.0)
    qb_turnovers     = float(g.get("qb_turnovers") or 0.0)
    opp_def_epa      = float(g.get("opponent_def_epa_allowed") or 0.0)
    opp_def_pass_y   = float(g.get("opponent_def_pass_yards_allowed") or 0.0)

    return [
        spread,
        qb_passing_yards,
        qb_rushing_yards,
        qb_total_epa,
        qb_turnovers,
        opp_def_epa,
        opp_def_pass_y,
    ]

def predict_games(games: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Input: a list of game dicts with at least:
      league, game_time, home_team, away_team,
      sportsbook_spread, sportsbook_total, sportsbook_ml_odds_home, sportsbook_ml_odds_away
    Output: list of dicts with model_* fields merged.
    """
    _load_model()

    out = []
    for g in games:
        # Default probabilities if model isn't loaded: fall back to implied ML
        p_home_ml = _moneyline_to_prob(g.get("sportsbook_ml_odds_home"))
        p_away_ml = _moneyline_to_prob(g.get("sportsbook_ml_odds_away"))

        if _model is not None:
            try:
                feat = _row_to_features(g)
                # Most sklearn classifiers: predict_proba -> [ [p(class0), p(class1)] ]
                # Assume class1 = "home covers/wins". Adjust if your training label differs.
                import numpy as np
                proba = _model.predict_proba([feat])[0]
                p_home = float(proba[1]) if len(proba) > 1 else float(proba[0])
                p_away = 1.0 - p_home
            except Exception:
                # If model call fails, fall back to ML implied if available, else 0.5/0.5
                if p_home_ml is not None and p_away_ml is not None:
                    s = p_home_ml + p_away_ml
                    p_home = p_home_ml / s if s else 0.5
                else:
                    p_home = 0.5
                p_away = 1.0 - p_home
        else:
            # No model loaded -> implied prob from ML odds or coin flip
            if p_home_ml is not None and p_away_ml is not None:
                s = p_home_ml + p_away_ml
                p_home = p_home_ml / s if s else 0.5
            else:
                p_home = 0.5
            p_away = 1.0 - p_home

        # Simple value flag: if model prob beats implied by >= 3% (tune this)
        value_flag = None
        if p_home_ml is not None:
            implied_home = p_home_ml / (p_home_ml + (p_away_ml or 0.000001))
            value_flag = abs(p_home - implied_home) >= 0.03

        pick_team = g.get("home_team") if p_home >= p_away else g.get("away_team")
        conf_pct  = round(max(p_home, p_away) * 100.0, 2)

        out.append({
            **g,
            "model_ml_prob_home": round(p_home, 4),
            "model_ml_prob_away": round(p_away, 4),
            "model_pick": pick_team,
            "confidence_pct": conf_pct,
            "ml_value_flag": bool(value_flag),
            # Optional placeholders (fill if your model produces them)
            "model_spread": None,
            "model_total": None,
            "edge_vs_line": None,
            "ou_pick": None,
            "ou_confidence_pct": None,
            "total_edge": None,
            "trap_alert": False,
            "sharp_flag": False,
            "volatility_score": 0.2,
            # Debug: surface model load issues
            "model_error": _model_error,
        })
    return out
