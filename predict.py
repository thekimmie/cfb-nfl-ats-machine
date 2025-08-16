# predict.py  — drop-in inference module (ATS/ML + Totals)
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import os, json, math

try:
    import joblib
except Exception:
    joblib = None
import pickle

# --------- Lazy-loaded globals ---------
_MODEL = None                   # classifier for home ML prob (uses sportsbook_* features)
_MODEL_PATH = None
_METADATA = None                # optional classifier metadata (features_used, etc.)

_TOTALS_MODEL = None            # regressor that predicts residual over total
_TOTALS_META = None             # expects {"features_used_total": [...], "residual_std": float}
_TOTALS_PATH = None

# --------- Odds helpers ---------
def american_to_prob(odds: Optional[float]) -> Optional[float]:
    if odds is None: return None
    try: o = float(odds)
    except: return None
    return 100.0/(o+100.0) if o>0 else (-o)/((-o)+100.0)

def normalize_no_vig(ph: Optional[float], pa: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
    if ph is None and pa is None: return None, None
    if ph is None: return None, 1.0
    if pa is None: return 1.0, None
    s = ph + pa
    if s<=0: return None, None
    return ph/s, pa/s

def league_spread_scale(league: str) -> float:
    lg = (league or "").upper()
    if lg == "NFL": return 18.0
    if lg in ("CFB","NCAAF","NCAAFB","NCAA"): return 22.0
    return 20.0

def model_prob_to_spread(p_home: float, league: str) -> float:
    scale = league_spread_scale(league)
    return -(p_home - 0.5) * 2.0 * scale  # negative => home favored

def compute_confidence_from_edge(edge_vs_line: float, league: str) -> float:
    denom = 8.0 if (league or "").upper() in ("NFL") else 10.0
    base = 0.5 + min(abs(edge_vs_line)/(2.0*denom), 0.48)  # cap ~98%
    return round(base*100.0,1)

# --------- Model loaders ---------
def _get_model_path() -> str:
    return os.getenv("MODEL_WEIGHTS_PATH", "final_ats_model.pkl")

def _get_totals_path() -> str:
    return os.getenv("MODEL_TOTALS_PATH", "final_totals_model.pkl")

def _load_pickle(path: str):
    if not os.path.exists(path): return None
    try:
        if joblib is not None and path.lower().endswith((".pkl",".joblib")):
            return joblib.load(path)
        with open(path,"rb") as f:
            return pickle.load(f)
    except Exception:
        return None

def _load_json(path: str):
    if not os.path.exists(path): return None
    try:
        with open(path,"r") as f: return json.load(f)
    except Exception:
        return None

def _ensure_classifier_loaded():
    global _MODEL, _MODEL_PATH, _METADATA
    if _MODEL is not None: return
    _MODEL_PATH = _get_model_path()
    _MODEL = _load_pickle(_MODEL_PATH)
    # optional metadata for classifier (not strictly required)
    _METADATA = _load_json(os.getenv("MODEL_METADATA_PATH","model_metadata.txt")) or {}

def _ensure_totals_loaded():
    global _TOTALS_MODEL, _TOTALS_META, _TOTALS_PATH
    if _TOTALS_MODEL is not None: return
    _TOTALS_PATH = _get_totals_path()
    _TOTALS_MODEL = _load_pickle(_TOTALS_PATH)
    _TOTALS_META = _load_json(os.getenv("TOTALS_METADATA_PATH","totals_metadata.json")) or {}

# --------- Runtime helpers ---------
def _clf_predict_proba(game: dict) -> Tuple[Optional[float], Optional[float]]:
    """Return (p_home, p_away) from classifier; falls back to market if needed."""
    # Features: try to follow training features if present; else a simple [spread,total].
    spread = game.get("sportsbook_spread")
    total  = game.get("sportsbook_total")
    if _MODEL is not None:
        try:
            # Try to respect metadata features_used (if present)
            feats = _METADATA.get("features_used") or []
            if feats:
                featmap = {
                    "sportsbook_spread": spread,
                    "sportsbook_total": total
                }
                X = [[featmap.get(k) for k in feats]]
            else:
                X = [[spread, total]]
            proba = _MODEL.predict_proba(X)[0]  # [p_away, p_home] or vice versa
            # Try to infer which column is home. Many scikit models use alphabetical classes_
            # We assume binary classes {0,1} where 1 = home_win if possible.
            if hasattr(_MODEL, "classes_"):
                classes = list(_MODEL.classes_)
                # heuristic: if 1 present, map proba[class==1] as p_home
                if len(classes)==2 and 1 in classes:
                    idx_home = classes.index(1)
                    p_home = float(proba[idx_home])
                    return p_home, 1.0 - p_home
            # fallback: guess the “second” column corresponds to home
            p_home = float(proba[-1])
            return p_home, 1.0 - p_home
        except Exception:
            pass

    # Fallback to market ML if classifier missing
    ph = american_to_prob(game.get("sportsbook_ml_odds_home"))
    pa = american_to_prob(game.get("sportsbook_ml_odds_away"))
    ph_novig, pa_novig = normalize_no_vig(ph, pa)
    return (ph_novig, pa_novig)

def _totals_predict(game: dict) -> Tuple[Optional[float], Optional[str], Optional[float], Optional[float]]:
    """
    Returns: (model_total, ou_pick, ou_conf_pct, total_edge)
    Uses residual model: model_total = sportsbook_total + f(total, spread)
    """
    if _TOTALS_MODEL is None:
        return None, None, None, None
    total = game.get("sportsbook_total")
    spread = game.get("sportsbook_spread")
    if total is None or spread is None:
        return None, None, None, None

    # Respect metadata feature order
    feats = (_TOTALS_META or {}).get("features_used_total") or ["total_close","spread_close"]
    featmap = {
        "total_close": total,
        "spread_close": spread,
        # (aliases in case your metadata used sportsbook_* naming)
        "sportsbook_total": total,
        "sportsbook_spread": spread,
    }
    try:
        X = [[featmap.get(k) for k in feats]]
        pred_resid = float(_TOTALS_MODEL.predict(X)[0])
    except Exception:
        return None, None, None, None

    model_total = float(total) + pred_resid
    total_edge = model_total - float(total)

    # Confidence via Normal(diff, sigma) around the edge
    sigma = float((_TOTALS_META or {}).get("residual_std") or 7.0)
    # Optional deadzone to avoid coin-flip OU: env TOTALS_TIE_MARGIN (default 0.25)
    tie = float(os.getenv("TOTALS_TIE_MARGIN","0.25"))
    if abs(total_edge) <= tie:
        ou_pick = None
        ou_conf = None
    else:
        # P(Over) = 1 - Φ(0 | μ=total_edge, σ=sigma)
        try:
            # erf approximation for normal CDF
            z = total_edge / (sigma * (2**0.5))
            cdf0 = 0.5*(1.0 + math.erf(-z))  # Φ(0; μ=total_edge) = Φ(-z)
            p_over = 1.0 - cdf0
            ou_pick = "Over" if p_over >= 0.5 else "Under"
            ou_conf = round(max(p_over, 1.0-p_over)*100.0, 1)
        except Exception:
            ou_pick = "Over" if total_edge>0 else "Under"
            ou_conf = round(min(98.0, 50.0 + min(abs(total_edge), 14.0)*3.0),1)

    return round(model_total,4), ou_pick, ou_conf, round(total_edge,4)

# --------- Public API ---------
def predict_games(games: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    _ensure_classifier_loaded()
    _ensure_totals_loaded()

    out: List[Dict[str, Any]] = []
    for g in games:
        league = g.get("league") or ""
        home = g.get("home_team")
        away = g.get("away_team")

        # Home win prob
        p_home, p_away = _clf_predict_proba(g)
        # If still None (no odds & no model), default 0.5/0.5
        if p_home is None or p_away is None:
            p_home, p_away = 0.5, 0.5

        # Derived model spread from p_home
        m_spread = model_prob_to_spread(p_home, league)

        # Edge vs current line (home perspective; negative => home favored)
        line_spread = g.get("sportsbook_spread")
        edge_vs_line = None
        conf_pct = None
        if line_spread is not None:
            edge_vs_line = float(m_spread) - float(line_spread)
            conf_pct = compute_confidence_from_edge(edge_vs_line, league)

        # Model pick (ML/ATS direction)
        model_pick = home if p_home >= 0.5 else away

        # Totals (if totals model available + line present)
        model_total, ou_pick, ou_conf, total_edge = _totals_predict(g)

        # Value flag for ML: compare implied vs probs when both odds present
        ml_value_flag = False
        ph = american_to_prob(g.get("sportsbook_ml_odds_home"))
        pa = american_to_prob(g.get("sportsbook_ml_odds_away"))
        ph_novig, pa_novig = normalize_no_vig(ph, pa)
        try:
            # value if model prob exceeds market implied by >= 2.5 ppt
            if ph_novig is not None and pa_novig is not None:
                exp = p_home if model_pick == home else p_away
                mkt = ph_novig if model_pick == home else pa_novig
                ml_value_flag = (exp - mkt) >= 0.025
        except Exception:
            pass

        out.append({
            "league": g.get("league"),
            "game_time": g.get("game_time"),
            "home_team": home,
            "away_team": away,
            "model_ml_prob_home": round(p_home,4),
            "model_ml_prob_away": round(p_away,4),
            "model_spread": round(m_spread,6),
            "model_total": model_total,                  # <-- now populated
            "model_pick": model_pick,
            "confidence_pct": conf_pct,
            "edge_vs_line": None if edge_vs_line is None else round(edge_vs_line,6),
            "ou_pick": ou_pick,
            "ou_confidence_pct": ou_conf,
            "total_edge": total_edge,
            "ml_value_flag": bool(ml_value_flag),
            # passthroughs that your UI expects (keep if present)
            "trap_alert": g.get("trap_alert") or False,
            "sharp_flag": g.get("sharp_flag") or False,
            "volatility_score": g.get("volatility_score") or 0.0,
        })
    return out
