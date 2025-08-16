# predict.py — inference-only, robust fill
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import os, json, math

try:
    import joblib
except Exception:
    joblib = None
import pickle

# --------- Lazy globals ---------
_MODEL = None
_METADATA = None
_FEATURES_USED: Optional[list] = None  # e.g. ["sportsbook_spread","sportsbook_total"]
_POSITIVE_CLASS: Optional[str] = None  # "home" or "away"

def _model_path() -> str:
    return os.getenv("MODEL_WEIGHTS_PATH", "final_ats_model.pkl")

def _metadata_path() -> str:
    return os.getenv("MODEL_METADATA_PATH", "model_metadata.txt")

def _load_metadata_once():
    global _METADATA, _FEATURES_USED, _POSITIVE_CLASS
    if _METADATA is not None:
        return
    _METADATA = {}
    try:
        p = _metadata_path()
        if os.path.exists(p):
            with open(p, "r") as f:
                _METADATA = json.load(f) or {}
    except Exception:
        _METADATA = {}
    _FEATURES_USED = _METADATA.get("features_used")
    _POSITIVE_CLASS = (_METADATA.get("positive_class") or "home").lower()

def _load_model_once():
    global _MODEL
    if _MODEL is not None:
        return
    path = _model_path()
    if not os.path.exists(path):
        _MODEL = None
        return
    try:
        if joblib is not None:
            _MODEL = joblib.load(path)
        else:
            with open(path, "rb") as f:
                _MODEL = pickle.load(f)
    except Exception:
        _MODEL = None

# --------- Helpers ---------
def american_to_prob(odds: Optional[float]) -> Optional[float]:
    if odds is None:
        return None
    try:
        o = float(odds)
    except Exception:
        return None
    if o > 0:
        return 100.0 / (o + 100.0)
    else:
        return (-o) / ((-o) + 100.0)

def normalize_two(ph: Optional[float], pa: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
    if ph is None and pa is None:
        return None, None
    if ph is None:
        return None, 1.0 if pa is not None else None
    if pa is None:
        return 1.0 if ph is not None else None, None
    s = ph + pa
    if s <= 0:
        return None, None
    return ph / s, pa / s

def league_spread_scale(league: str) -> float:
    lg = (league or "").upper()
    if lg == "NFL": return 18.0
    if lg in ("CFB","NCAAF"): return 22.0
    return 20.0

def prob_to_home_spread(p_home: float, league: str) -> float:
    # negative spread => home favored
    scale = league_spread_scale(league)
    return - (p_home - 0.5) * 2.0 * scale

def spread_to_prob_home(spread_home: float, league: str) -> float:
    # invert of prob_to_home_spread; if home spread is -3, p_home > 0.5
    scale = league_spread_scale(league)
    return 0.5 - (spread_home / (2.0 * scale))

def confidence_from_edge(edge_vs_line: Optional[float], league: str, p_home: Optional[float]) -> float:
    # If we have an edge, scale confidence by its magnitude; else fallback to distance from 0.5
    if edge_vs_line is not None:
        denom = 8.0 if (league or "").upper() == "NFL" else 10.0
        base = min(abs(edge_vs_line) / denom, 0.48)  # cap near 48 points → 98%
        return round((0.5 + base) * 100.0, 1)
    if p_home is not None:
        return round((0.5 + min(abs(p_home - 0.5) * 2.0, 0.48)) * 100.0, 1)
    return 50.0

def value_flag_from_odds(p_home: Optional[float], ml_home: Optional[float], ml_away: Optional[float]) -> bool:
    # Value if model disagree with market by >= 3 percentage points
    if p_home is None: 
        return False
    ph_implied = american_to_prob(ml_home)
    pa_implied = american_to_prob(ml_away)
    ph_implied, pa_implied = normalize_two(ph_implied, pa_implied)
    if ph_implied is None:
        return False
    return abs(p_home - ph_implied) >= 0.03

def _vector_from_game(game: Dict[str, Any], features: Optional[list]) -> Optional[list]:
    if not features:
        return None
    vec = []
    for f in features:
        v = game.get(f)
        if v is None:
            return None
        try:
            vec.append(float(v))
        except Exception:
            return None
    return vec

# --------- Main entry ---------
def predict_games(games: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Returns one dict per input game with:
      model_ml_prob_home, model_ml_prob_away, model_pick, confidence_pct,
      ml_value_flag, model_spread, edge_vs_line, (ou fields left None unless totals available).
    """
    _load_metadata_once()
    _load_model_once()

    out: List[Dict[str, Any]] = []

    for g in games:
        league     = g.get("league") or ""
        home       = g.get("home_team")
        away       = g.get("away_team")
        game_time  = g.get("game_time")

        # sportsbook info (may be None)
        spread_h   = g.get("sportsbook_spread")   # home perspective
        total_pts  = g.get("sportsbook_total")
        ml_home    = g.get("sportsbook_ml_odds_home")
        ml_away    = g.get("sportsbook_ml_odds_away")

        # --- 1) Try model pickle first if we have all needed features
        p_home = None
        if _MODEL is not None and _FEATURES_USED:
            vec = _vector_from_game(g, _FEATURES_USED)
            if vec is not None:
                try:
                    import numpy as np
                    X = np.array(vec, dtype=float, ndmin=2)
                    # Assume positive class = "home" unless metadata says otherwise
                    proba = getattr(_MODEL, "predict_proba", None)
                    if callable(proba):
                        # Most scikit models return proba for class 1 at index 1
                        p = proba(X)[0, 1]
                        if (_POSITIVE_CLASS or "home") == "home":
                            p_home = float(p)
                        else:
                            p_home = float(1.0 - p)
                except Exception:
                    p_home = None

        # --- 2) Fallback: use moneyline odds implied probabilities
        if p_home is None:
            ph_imp = american_to_prob(ml_home)
            pa_imp = american_to_prob(ml_away)
            ph_imp, pa_imp = normalize_two(ph_imp, pa_imp)
            if ph_imp is not None:
                p_home = ph_imp

        # --- 3) Fallback: infer from spread if still unknown
        if p_home is None and spread_h is not None:
            try:
                p_home = spread_to_prob_home(float(spread_h), league)
                p_home = max(0.01, min(0.99, p_home))
            except Exception:
                p_home = None

        # If still None, neutral prior
        if p_home is None:
            p_home = 0.50
        p_away = 1.0 - p_home

        # Model pick from p_home
        model_pick = home if p_home >= 0.5 else away

        # Model-derived spread for home (negative => home favored)
        model_spread = prob_to_home_spread(p_home, league)

        # Edge if we have a sportsbook spread
        edge_vs_line = None
        try:
            if spread_h is not None:
                edge_vs_line = float(model_spread) - float(spread_h)
        except Exception:
            edge_vs_line = None

        # Confidence
        confidence_pct = confidence_from_edge(edge_vs_line, league, p_home)

        # Simple value flag vs ML prices
        ml_value_flag = value_flag_from_odds(p_home, ml_home, ml_away)

        # OU fields — we don’t fabricate a model_total; leave None unless total exists AND you later add a total model
        model_total = None
        ou_pick = None
        ou_confidence_pct = None
        total_edge = None
        if total_pts is not None and model_total is not None:
            try:
                total_edge = float(model_total) - float(total_pts)
                if total_edge > 0: ou_pick = "Over"
                if total_edge < 0: ou_pick = "Under"
                if ou_pick:
                    ou_confidence_pct = round(min(abs(total_edge) / 8.0, 0.48) * 100 + 50, 1)
            except Exception:
                pass

        # Conservative defaults for extra flags
        trap_alert = False
        sharp_flag = False
        volatility_score = round(min(abs(edge_vs_line or 0.0) / 10.0, 0.5), 2)

        out.append({
            "league": league, "game_time": game_time,
            "home_team": home, "away_team": away,

            "model_ml_prob_home": round(float(p_home), 4),
            "model_ml_prob_away": round(float(p_away), 4),
            "model_pick": model_pick,
            "confidence_pct": float(confidence_pct),
            "ml_value_flag": bool(ml_value_flag),

            "model_spread": float(model_spread),
            "edge_vs_line": edge_vs_line if edge_vs_line is None else float(edge_vs_line),

            "model_total": model_total,
            "ou_pick": ou_pick,
            "ou_confidence_pct": ou_confidence_pct,
            "total_edge": total_edge,

            "trap_alert": trap_alert,
            "sharp_flag": sharp_flag,
            "volatility_score": float(volatility_score),
        })

    return out
