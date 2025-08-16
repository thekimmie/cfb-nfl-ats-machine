# predict.py
# ----------------------------
# Drop-in inference-only module for NFL/CFB ATS / ML predictions.
# - Loads a trained pickle from env MODEL_WEIGHTS_PATH (default: final_ats_model.pkl)
# - Safe import: model is loaded lazily on first call to predict_games()
# - No training happens here.
#
# I/O CONTRACT
# def predict_games(games: list[dict]) -> list[dict]:
#   INPUT (each game dict must include at least):
#     league: "NFL" | "NCAA" | ...
#     game_time: ISO8601 ("...Z")
#     home_team: str
#     away_team: str
#   Optional sportsbook fields:
#     sportsbook_ml_odds_home: int  # American odds, e.g. -150, +135
#     sportsbook_ml_odds_away: int
#     sportsbook_spread: float      # home perspective; negative => home favored
#     sportsbook_total: float
#   Optional features for your trained model (if it needs them):
#     features: Dict[str, float]    # must match metadata['features_used'] order if provided
#
#   OUTPUT (one dict per input) includes keys:
#     league, game_time, home_team, away_team
#     model_ml_prob_home: float (0..1)
#     model_ml_prob_away: float (0..1)
#     model_pick: str (exactly one of the input team names)
#     confidence_pct: float (0..100)
#     ml_value_flag: bool
#   Optional extras (if data present or we can infer):
#     model_spread: float
#     model_total: float
#     edge_vs_line: float
#     ou_pick: str | None
#     ou_confidence_pct: float | None
#     total_edge: float | None
#     trap_alert: bool
#     sharp_flag: bool
#     volatility_score: float

from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import os, sys, json

try:
    import joblib  # preferred for sklearn models
except Exception:
    joblib = None
import pickle  # fallback loader

# --------- Lazy-loaded globals ---------
_MODEL = None                 # trained estimator or pipeline with predict_proba
_MODEL_PATH = None
_METADATA = None              # optional dict from model_metadata.(json|txt)
_FEATURES_USED: Optional[list] = None
_POSITIVE_CLASS: Optional[str] = None  # "home" or "away" if provided in metadata

# --------- Utility functions ---------
def _get_model_path() -> str:
    return os.getenv("MODEL_WEIGHTS_PATH", "final_ats_model.pkl")

def _maybe_load_metadata() -> None:
    """Loads model metadata if present (features_used, positive_class)."""
    # Try JSON first
    global _METADATA, _FEATURES_USED, _POSITIVE_CLASS
    if _METADATA is not None:
        return
    for meta_path in [os.getenv("MODEL_METADATA_PATH", "model_metadata.json"),
                      os.getenv("MODEL_METADATA_PATH", "model_metadata.txt")]:
        try:
            if not os.path.exists(meta_path):
                continue
            with open(meta_path, "r") as f:
                txt = f.read().strip()
                # If it's JSON, parse; otherwise try to parse laxly
                try:
                    md = json.loads(txt)
                except Exception:
                    # lax: try to coerce to JSON-ish
                    txt2 = txt.replace("'", '"')
                    md = json.loads(txt2)
            if isinstance(md, dict):
                _METADATA = md
                if isinstance(md.get("features_used"), list):
                    _FEATURES_USED = md["features_used"]
                pc = md.get("positive_class")
                if isinstance(pc, str) and pc.lower() in ("home", "away"):
                    _POSITIVE_CLASS = pc.lower()
                return
        except Exception:
            continue
    _METADATA = {}

def _load_model_once() -> None:
    """Lazily loads the trained model from pickle."""
    global _MODEL, _MODEL_PATH
    if _MODEL is not None:
        return
    _maybe_load_metadata()
    _MODEL_PATH = _get_model_path()
    if not os.path.exists(_MODEL_PATH):
        _MODEL = None
        return
    try:
        if joblib is not None:
            _MODEL = joblib.load(_MODEL_PATH)
        else:
            with open(_MODEL_PATH, "rb") as f:
                _MODEL = pickle.load(f)
    except Exception:
        _MODEL = None  # fallback math will take over

def american_to_prob(odds: Optional[float]) -> Optional[float]:
    """Convert American odds to implied probability (with vig)."""
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

def normalize_no_vig(ph: Optional[float], pa: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
    """Remove vig by normalizing two implied probs to sum to 1."""
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
    """Heuristic mapping from ML probability deltas to spread points."""
    lg = (league or "").upper()
    if lg == "NFL":
        return 18.0
    if lg in ("CFB", "NCAA", "NCAAF", "NCAAFB"):
        return 22.0
    return 20.0

def prob_to_home_spread(p_home: float, league: str) -> float:
    """Convert home ML prob into a model home spread (negative => home favored)."""
    scale = league_spread_scale(league)
    return - (p_home - 0.5) * 2.0 * scale

def spread_to_prob_home(home_spread: float, league: str) -> float:
    """Inverse of prob_to_home_spread (approx)."""
    scale = league_spread_scale(league)
    return 0.5 - (float(home_spread) / (2.0 * scale))

def compute_confidence_from_edge(edge_vs_line: Optional[float], league: str) -> float:
    """Map spread edge to a confidence percentage (caps below 100)."""
    if edge_vs_line is None:
        return 50.0
    denom = 8.0 if (league or "").upper() == "NFL" else 10.0
    # Edge of denom points ~ 75% confidence; capped around 98%
    pct = 50.0 + min(abs(edge_vs_line) / denom * 25.0, 48.0)
    return round(pct, 1)

def pick_side_from_probs(home: str, away: str, p_home: float) -> str:
    return home if p_home >= 0.5 else away

def value_flag_from_probs_and_odds(
    p_home: float,
    odds_home: Optional[float],
    odds_away: Optional[float],
    threshold_pp: float = 0.03
) -> bool:
    """Flag 'value' if model probability beats the implied by >= threshold_pp on chosen side."""
    # Decide side first
    pick_home = p_home >= 0.5
    implied_home = american_to_prob(odds_home)
    implied_away = american_to_prob(odds_away)
    ih, ia = normalize_no_vig(implied_home, implied_away)
    if pick_home and ih is not None:
        return (p_home - ih) >= threshold_pp
    if (not pick_home) and ia is not None:
        return ((1.0 - p_home) - ia) >= threshold_pp
    return False

def _extract_feature_vector(game: Dict[str, Any]) -> Optional[list]:
    """
    If metadata['features_used'] exists, try to assemble X in that order.
    - Prefer game['features'][name], else game[name], else 0.0
    """
    if not _FEATURES_USED:
        return None
    feats = game.get("features") or {}
    x = []
    for f in _FEATURES_USED:
        val = feats.get(f)
        if val is None:
            val = game.get(f)
        try:
            x.append(float(val))
        except Exception:
            x.append(0.0)
    return x

def _model_probs_for_game(game: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """Return (p_home, p_away) from trained model if possible, else None."""
    if _MODEL is None:
        return None
    # Need features (either vectorizable from metadata, or not possible)
    x = _extract_feature_vector(game)
    if x is None:
        return None
    try:
        # scikit-learn predict_proba
        if hasattr(_MODEL, "predict_proba"):
            proba = _MODEL.predict_proba([x])[0]
            # Map to home prob using classes_ if helpful
            if hasattr(_MODEL, "classes_"):
                classes = list(_MODEL.classes_)
                # common patterns: ["away","home"] or ["home","away"] or [0,1]
                if "home" in classes and "away" in classes:
                    p_home = float(proba[classes.index("home")])
                else:
                    # assume class '1' == home if binary numeric
                    p_home = float(proba[-1])
            else:
                p_home = float(proba[-1])
            p_home = min(max(p_home, 0.0), 1.0)
            return p_home, 1.0 - p_home
        # Fallback: decision_function -> sigmoid
        if hasattr(_MODEL, "decision_function"):
            from math import exp
            z = float(_MODEL.decision_function([x])[0])
            p_home = 1.0 / (1.0 + pow(2.718281828459045, -z))
            p_home = min(max(p_home, 0.0), 1.0)
            return p_home, 1.0 - p_home
    except Exception:
        return None
    return None

def _fallback_probs(game: Dict[str, Any]) -> Tuple[float, float]:
    """
    Use sportsbook ML odds (preferred) or spread to derive probabilities when no model.
    """
    league = game.get("league") or ""
    oh = game.get("sportsbook_ml_odds_home")
    oa = game.get("sportsbook_ml_odds_away")
    ph_raw = american_to_prob(oh)
    pa_raw = american_to_prob(oa)
    ph, pa = normalize_no_vig(ph_raw, pa_raw)
    if ph is not None and pa is not None:
        return float(ph), float(pa)

    # Try spread
    sp = game.get("sportsbook_spread")
    try:
        if sp is not None:
            p_home = spread_to_prob_home(float(sp), league)
            p_home = min(max(p_home, 0.0), 1.0)
            return p_home, 1.0 - p_home
    except Exception:
        pass

    # Dead fallback
    return 0.5, 0.5

def _build_output_for_game(game: Dict[str, Any]) -> Dict[str, Any]:
    league = (game.get("league") or "").strip()
    home = game.get("home_team")
    away = game.get("away_team")
    gtime = game.get("game_time")

    # 1) Try trained model
    p_pair = _model_probs_for_game(game)
    if p_pair is None:
        p_home, p_away = _fallback_probs(game)
    else:
        p_home, p_away = p_pair

    # 2) Derive model_spread from p_home
    model_spread = prob_to_home_spread(p_home, league)  # negative => home favored

    # 3) Edge vs sportsbook line (if we have it)
    edge_vs_line = None
    if game.get("sportsbook_spread") is not None:
        try:
            edge_vs_line = float(model_spread) - float(game.get("sportsbook_spread"))
        except Exception:
            edge_vs_line = None

    # 4) Confidence (spread-based if possible)
    confidence_pct = compute_confidence_from_edge(edge_vs_line, league) if edge_vs_line is not None else round(50.0 + abs(p_home - 0.5) * 90.0, 1)

    # 5) Pick + value flag
    pick = pick_side_from_probs(home, away, p_home)
    ml_value_flag = value_flag_from_probs_and_odds(
        p_home,
        game.get("sportsbook_ml_odds_home"),
        game.get("sportsbook_ml_odds_away"),
        threshold_pp=0.03,  # 3 percentage points
    )

    # Optional OU fields: we don't have a totals model here; leave None unless you add one.
    ou_pick = None
    ou_confidence_pct = None
    total_edge = None
    model_total = None

    # Simple volatility proxy: closer to coinflip => higher volatility
    volatility_score = round(2.0 * (0.5 - abs(p_home - 0.5)), 3)  # 0..1-ish

    out = {
        # Join keys (MUST echo back exactly for merging)
        "league": league,
        "game_time": gtime,
        "home_team": home,
        "away_team": away,

        # Model outputs
        "model_ml_prob_home": float(p_home),
        "model_ml_prob_away": float(p_away),
        "model_pick": pick,
        "confidence_pct": float(confidence_pct),
        "ml_value_flag": bool(ml_value_flag),

        # Nice-to-haves
        "model_spread": float(model_spread),
        "edge_vs_line": float(edge_vs_line) if edge_vs_line is not None else None,
        "model_total": model_total,
        "ou_pick": ou_pick,
        "ou_confidence_pct": ou_confidence_pct,
        "total_edge": total_edge,

        # Signals (defaults; tweak if you later add logic)
        "trap_alert": False,
        "sharp_flag": False,
        "volatility_score": float(volatility_score),
    }
    return out

# --------- Public API ---------
def predict_games(games: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Main entry. Accepts a list of games; returns a list of prediction dicts.
    Safe to call multiple times; model is cached after first load.
    """
    _load_model_once()
    _maybe_load_metadata()

    if not isinstance(games, list):
        raise TypeError("predict_games expects a list of dicts")

    out: List[Dict[str, Any]] = []
    for g in games:
        try:
            out.append(_build_output_for_game(g or {}))
        except Exception as e:
            # Always return join keys so the caller can still merge
            out.append({
                "league": (g or {}).get("league"),
                "game_time": (g or {}).get("game_time"),
                "home_team": (g or {}).get("home_team"),
                "away_team": (g or {}).get("away_team"),
                "error": str(e),
            })
    return out

# --------- CLI smoke test ---------
if __name__ == "__main__":
    """
    Usage:
      cat sample_games.json | python predict.py
      or
      python predict.py < sample_games.json
    """
    data = sys.stdin.read().strip()
    if not data:
        print("[]")
        sys.exit(0)
    try:
        games = json.loads(data)
    except Exception as e:
        print(json.dumps({"error": f"Invalid JSON on stdin: {e}"}))
        sys.exit(1)
    try:
        res = predict_games(games)
        print(json.dumps(res, ensure_ascii=False))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
