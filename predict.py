# predict.py — simple drop-in model for /api/model-data
# - Uses sportsbook moneylines to derive "model" win probs as a placeholder.
# - Exposes predict_games(games) exactly as main.py expects.
# - When Tommy's real model is ready, replace score_with_tommy_model().

from typing import List, Dict, Any, Optional

VALUE_MARGIN = 0.05  # 5% edge above implied to call it a "value" play

def _amer_to_prob(odds: Optional[float]) -> Optional[float]:
    try:
        o = float(odds)
    except Exception:
        return None
    if o == 0: 
        return None
    # American odds -> implied probability
    if o > 0:
        return 100.0 / (o + 100.0)
    else:
        return (-o) / ((-o) + 100.0)

def _safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

def score_with_moneylines(game: Dict[str, Any]) -> Dict[str, Any]:
    """Fallback scoring: derive probabilities from offered moneylines."""
    ph = _amer_to_prob(game.get("sportsbook_ml_odds_home"))
    pa = _amer_to_prob(game.get("sportsbook_ml_odds_away"))

    # If one side missing, infer the other; if both missing, give 0.5/0.5
    if ph is None and pa is None:
        ph, pa = 0.5, 0.5
    elif ph is None and pa is not None:
        ph, pa = 1.0 - pa, pa
    elif ph is not None and pa is None:
        ph, pa = ph, 1.0 - ph

    # Normalize lightly in case the book has hold
    total = (ph or 0) + (pa or 0)
    if total > 0:
        ph, pa = (ph or 0)/total, (pa or 0)/total
    else:
        ph, pa = 0.5, 0.5

    # Pick the side with higher prob
    home_name = game.get("home_team")
    away_name = game.get("away_team")
    if ph >= pa:
        pick = home_name
        conf = ph
        implied = _amer_to_prob(game.get("sportsbook_ml_odds_home")) or ph
    else:
        pick = away_name
        conf = pa
        implied = _amer_to_prob(game.get("sportsbook_ml_odds_away")) or pa

    ml_value_flag = bool(conf - implied >= VALUE_MARGIN)

    # You can optionally compute spread/total “edges” if your script does that.
    # We'll leave them None so main.py can compute if you later provide model_spread/total.
    return {
        **game,
        "model_ml_prob_home": float(ph),
        "model_ml_prob_away": float(pa),
        "model_pick": pick,
        "confidence_pct": round(float(conf) * 100.0, 1),
        "ml_value_flag": ml_value_flag,
        # Optional extras you might fill later:
        "model_spread": None,
        "model_total": None,
        "edge_vs_line": None,
        "ou_pick": None,
        "ou_confidence_pct": None,
        "total_edge": None,
        # Some harmless defaults so your UI has values:
        "trap_alert": False,
        "sharp_flag": False,
        "volatility_score": 0.2,
    }

# ===== Hook for Tommy's real model =====
# When we get Tommy's files:
#  - load his .pkl or call his code in this function
#  - keep input/output keys identical to score_with_moneylines
def score_with_tommy_model(game: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # Placeholder: return None to fall back to moneyline scoring.
    # After we wire the real model, return the same dict shape as score_with_moneylines().
    return None

# ===== Public API expected by main.py =====
def predict_games(games: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for g in games or []:
        scored = score_with_tommy_model(g)
        if not scored:
            scored = score_with_moneylines(g)
        out.append(scored)
    return out
