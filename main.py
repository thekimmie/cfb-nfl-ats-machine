# main.py — Betting Machine API (FastAPI on Render)

import os, re, io, csv, json, time, logging, asyncio, importlib.util, sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta, timezone

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# -------------------- Config / Env --------------------
ODDS_API_KEY   = os.getenv("ODDS_API_KEY")                  # The Odds API
APISPORTS_KEY  = os.getenv("APISPORTS_KEY")                 # API-SPORTS (RapidAPI) for slates/injuries
CFBD_KEY       = os.getenv("CFBD_KEY")                      # CollegeFootballData (CFB venues)
TIO_API_KEY    = os.getenv("TIO_API_KEY")                   # Tomorrow.io Timeline
CACHE_TTL      = int(os.getenv("CACHE_TTL", "600"))         # seconds
CORS_ORIGIN    = [o.strip() for o in os.getenv("CORS_ORIGIN", "*").split(",")]
LOG_LEVEL      = os.getenv("LOG_LEVEL", "INFO")

# Optional: historical data passthrough (CSV/JSON)
HIST_SOURCE = os.getenv("HIST_SOURCE", "none").lower()      # "file" | "url" | "none"
HIST_PATH   = os.getenv("HIST_PATH")                        # e.g. data/history.csv
HIST_URL    = os.getenv("HIST_URL")                         # e.g. https://raw.githubusercontent.com/.../history.csv

# Optional: model output (CSV/JSON) to merge into rows
MODEL_SOURCE = os.getenv("MODEL_SOURCE", "none").lower()    # "file" | "url" | "script" | "none"
MODEL_PATH   = os.getenv("MODEL_PATH")                      # e.g. data/model_output.csv / .json OR ./predict.py for script mode
MODEL_URL    = os.getenv("MODEL_URL")                       # e.g. https://raw.githubusercontent.com/.../model_output.json
MODEL_WEIGHTS_PATH = os.getenv("MODEL_WEIGHTS_PATH")        # path to .pkl if your script uses it

# Odds window
DAYS_BACK  = int(os.getenv("DAYS_BACK",  "1"))
DAYS_AHEAD = int(os.getenv("DAYS_AHEAD", "14"))

# API bases
ODDS_BASE       = "https://api.the-odds-api.com/v4"
APISPORTS_HOST  = "api-american-football.p.rapidapi.com"
APISPORTS_BASE  = f"https://{APISPORTS_HOST}"
CFBD_BASE       = "https://api.collegefootballdata.com"
TIO_URL         = "https://api.tomorrow.io/v4/timelines"

SPORT_KEYS = ["americanfootball_nfl", "americanfootball_ncaaf"]

# Minimal NFL stadium map (home team -> lat/lon) for weather fallback
NFL_STADIUMS = {
    "Arizona Cardinals": {"lat": 33.5276, "lon": -112.2626},
    "Atlanta Falcons": {"lat": 33.7554, "lon": -84.4008},
    "Baltimore Ravens": {"lat": 39.2780, "lon": -76.6227},
    "Buffalo Bills": {"lat": 42.7738, "lon": -78.7869},
    "Carolina Panthers": {"lat": 35.2251, "lon": -80.8526},
    "Chicago Bears": {"lat": 41.8625, "lon": -87.6166},
    "Cincinnati Bengals": {"lat": 39.0954, "lon": -84.5160},
    "Cleveland Browns": {"lat": 41.5061, "lon": -81.6995},
    "Dallas Cowboys": {"lat": 32.7473, "lon": -97.0945},
    "Denver Broncos": {"lat": 39.7439, "lon": -105.0201},
    "Detroit Lions": {"lat": 42.3400, "lon": -83.0456},
    "Green Bay Packers": {"lat": 44.5013, "lon": -88.0622},
    "Houston Texans": {"lat": 29.6847, "lon": -95.4107},
    "Indianapolis Colts": {"lat": 39.7601, "lon": -86.1639},
    "Jacksonville Jaguars": {"lat": 30.3239, "lon": -81.6373},
    "Kansas City Chiefs": {"lat": 39.0489, "lon": -94.4839},
    "Las Vegas Raiders": {"lat": 36.0909, "lon": -115.1833},
    "Los Angeles Chargers": {"lat": 33.9530, "lon": -118.3390},
    "Los Angeles Rams": {"lat": 33.9530, "lon": -118.3390},
    "Miami Dolphins": {"lat": 25.9580, "lon": -80.2389},
    "Minnesota Vikings": {"lat": 44.9736, "lon": -93.2575},
    "New England Patriots": {"lat": 42.0909, "lon": -71.2643},
    "New Orleans Saints": {"lat": 29.9509, "lon": -90.0815},
    "New York Giants": {"lat": 40.8135, "lon": -74.0745},
    "New York Jets": {"lat": 40.8135, "lon": -74.0745},
    "Philadelphia Eagles": {"lat": 39.9008, "lon": -75.1675},
    "Pittsburgh Steelers": {"lat": 40.4468, "lon": -80.0158},
    "San Francisco 49ers": {"lat": 37.4030, "lon": -121.9700},
    "Seattle Seahawks": {"lat": 47.5952, "lon": -122.3316},
    "Tampa Bay Buccaneers": {"lat": 27.9759, "lon": -82.5033},
    "Tennessee Titans": {"lat": 36.1665, "lon": -86.7713},
    "Washington Commanders": {"lat": 38.9078, "lon": -76.8646},
}

# -------------------- App / CORS / Logging --------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGIN if CORS_ORIGIN != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=getattr(logging, LOG_LEVEL.upper(), logging.INFO))
log = logging.getLogger("betting-machine")

def _redact_api_key(url_str: str) -> str:
    return re.sub(r"(apiKey=)[^&]+", r"\1REDACTED", url_str or "")

# -------------------- Simple in-memory cache --------------------
_cache: Dict[str, Dict[str, Any]] = {}  # key -> {"ts": epoch, "data": object}

def cache_get(key: str):
    item = _cache.get(key)
    if not item: return None
    if time.time() - item["ts"] > CACHE_TTL: return None
    return item["data"]

def cache_set(key: str, data: Any):
    _cache[key] = {"ts": time.time(), "data": data}
    return data

def cache_clear(*keys: str):
    for k in keys:
        _cache.pop(k, None)

# -------------------- Helpers --------------------
def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")

def parse_iso(s: str) -> datetime:
    return datetime.fromisoformat((s or "").replace("Z", "+00:00")).astimezone(timezone.utc)

def current_window():
    now = datetime.utcnow().replace(tzinfo=timezone.utc)
    return iso(now - timedelta(days=DAYS_BACK)), iso(now + timedelta(days=DAYS_AHEAD))

def nearest_hour_point(points: List[Dict[str, Any]], kickoff_iso: str) -> Optional[Dict[str, Any]]:
    if not points: return None
    k = parse_iso(kickoff_iso)
    best, best_delta = None, None
    for p in points:
        ts = parse_iso(p["startTime"])
        d = abs((ts - k).total_seconds())
        if best is None or d < best_delta:
            best, best_delta = p, d
    return best

def compute_weather_alert(values: Dict[str, Any]) -> Optional[bool]:
    if not values: return None
    gust = float(values.get("windGust") or 0)
    precipType = int(values.get("precipitationType") or 0)  # 0=none
    precipInt = float(values.get("precipitationIntensity") or 0)
    temp = float(values.get("temperature") or 60)
    return bool(gust >= 25 or (precipType > 0 and precipInt >= 0.1) or temp <= 20 or temp >= 95)

def league_label_from_odds_key(sport_key: str) -> Optional[str]:
    sk = (sport_key or "").lower()
    if "nfl" in sk: return "NFL"
    if "ncaaf" in sk: return "NCAA"
    return None

def join_key(league: Optional[str], dt_iso: Optional[str], away: Optional[str], home: Optional[str]) -> str:
    lg = (league or "").strip().upper()
    iso_dt = ""
    if dt_iso:
        try:
            iso_dt = parse_iso(dt_iso).replace(minute=0, second=0, microsecond=0).isoformat()
        except Exception:
            iso_dt = dt_iso
    a = (away or "").strip().lower().replace(" ", "")
    h = (home or "").strip().lower().replace(" ", "")
    return f"{lg}|{iso_dt}|{a}@{h}"

def synth_row_key(league: Optional[str], game_time: Optional[str], away: Optional[str], home: Optional[str]) -> str:
    lg  = (league or "").upper().strip()
    iso_dt = ""
    if game_time:
        try:
            iso_dt = parse_iso(game_time).isoformat()
        except Exception:
            iso_dt = str(game_time)
    a = (away or "").lower().replace(" ", "")
    h = (home or "").lower().replace(" ", "")
    return f"{lg}|{iso_dt}|{a}@{h}"

# ---- Type coercion for model fields ----
NUM_FIELDS = [
    "model_spread","model_total","confidence_pct","ou_confidence_pct",
    "model_ml_prob_home","model_ml_prob_away","volatility_score",
    "edge_vs_line","total_edge"
]
BOOL_FIELDS = ["ml_value_flag","trap_alert","sharp_flag","weather_alert"]

def _to_boolish(v):
    if isinstance(v, bool): return v
    if v is None: return None
    s = str(v).strip().lower()
    if s in ("true","1","yes","y"): return True
    if s in ("false","0","no","n"): return False
    return None

def coerce_types(row: Dict[str, Any]):
    for f in NUM_FIELDS:
        if f in row and row[f] is not None:
            try: row[f] = float(row[f])
            except: pass
    for f in BOOL_FIELDS:
        if f in row:
            row[f] = _to_boolish(row[f])

# -------------------- History loader (optional) --------------------
def _csv_to_dicts(text: str) -> List[Dict[str, Any]]:
    return [row for row in csv.DictReader(io.StringIO(text))]

def _hist_load_from_file(path: str):
    try:
        raw = Path(path).read_text(encoding="utf-8", errors="ignore")
        if path.lower().endswith(".csv"):
            return _csv_to_dicts(raw)
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    except Exception:
        return []

async def _hist_load_from_url(url: str):
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.get(url)
            r.raise_for_status()
            ct = (r.headers.get("content-type","") or "").lower()
            if "text/csv" in ct or url.lower().endswith(".csv"):
                return _csv_to_dicts(r.text)
            data = r.json()
            return data if isinstance(data, list) else []
    except Exception:
        return []

async def load_history_rows():
    cached = cache_get("history_rows")
    if cached is not None:
        return cached
    if   HIST_SOURCE == "file" and HIST_PATH:
        rows = _hist_load_from_file(HIST_PATH)
    elif HIST_SOURCE == "url"  and HIST_URL:
        rows = await _hist_load_from_url(HIST_URL)
    else:
        rows = []
    cache_set("history_rows", rows)
    return rows

# -------------------- Model loader (CSV/JSON merge mode) --------------------
async def _model_load_from_url(url: str):
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.get(url)
            r.raise_for_status()
            ct = (r.headers.get("content-type","") or "").lower()
            if "text/csv" in ct or url.lower().endswith(".csv"):
                return _csv_to_dicts(r.text)
            data = r.json()
            return data if isinstance(data, list) else []
    except Exception:
        return []

def _model_load_from_file(path: str):
    try:
        raw = Path(path).read_text(encoding="utf-8", errors="ignore")
        if path.lower().endswith(".csv"):
            return _csv_to_dicts(raw)
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    except Exception:
        return []

def _norm_league(s: Optional[str]) -> str:
    s = (s or "").strip().upper()
    if s in ("NCAAF", "CFB", "NCAA FOOTBALL", "NCAA"): return "NCAA"
    return s

def _norm_team(s: Optional[str]) -> str:
    return (s or "").strip()

def _coerce_iso(s: Optional[str]) -> Optional[str]:
    if not s: return None
    try:
        return parse_iso(s).isoformat().replace("+00:00","Z")
    except Exception:
        try:
            return datetime.fromisoformat(s).astimezone(timezone.utc).isoformat().replace("+00:00","Z")
        except Exception:
            return s

async def load_model_rows():
    """Load precomputed model outputs (CSV/JSON) if MODEL_SOURCE is file/url."""
    cached = cache_get("model_rows")
    if cached is not None:
        return cached

    if   MODEL_SOURCE == "file" and MODEL_PATH:
        rows = _model_load_from_file(MODEL_PATH)
    elif MODEL_SOURCE == "url"  and MODEL_URL:
        rows = await _model_load_from_url(MODEL_URL)
    else:
        rows = []

    normed = []
    for r in rows:
        league = _norm_league(r.get("league") or r.get("League"))
        game_time = r.get("game_time") or r.get("kickoff_time") or r.get("datetime")
        game_time = _coerce_iso(game_time)
        home = _norm_team(r.get("home_team") or r.get("Home") or r.get("home"))
        away = _norm_team(r.get("away_team") or r.get("Away") or r.get("away"))
        if not (league and game_time and home and away):
            continue
        # Coerce types in-place so we don't merge strings later
        for f in NUM_FIELDS:
            if f in r and r[f] is not None:
                try: r[f] = float(r[f])
                except: pass
        for f in BOOL_FIELDS:
            if f in r:
                r[f] = _to_boolish(r[f])
        normed.append({**r, "league": league, "game_time": game_time, "home_team": home, "away_team": away})

    cache_set("model_rows", normed)
    return normed

# -------------------- Script Model loader (predict.py) --------------------
_predict_module = None

def get_predict_module():
    """Dynamically import the user's predict.py when MODEL_SOURCE=script."""
    global _predict_module
    if _predict_module is not None:
        return _predict_module
    if MODEL_SOURCE != "script" or not MODEL_PATH:
        _predict_module = None
        return None
    spec = importlib.util.spec_from_file_location("user_predict", MODEL_PATH)
    if spec is None or spec.loader is None:
        _predict_module = None
        return None
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        log.error("Failed to import predict script at %s: %s", MODEL_PATH, e)
        _predict_module = None
        return None
    _predict_module = mod
    return mod

# -------------------- Odds / Slates / Weather adapters --------------------
PREFERRED_BOOKS = ["pinnacle","circa","bookmaker","draftkings","fanduel","betmgm"]

def pick_market(bookmakers: List[Dict[str, Any]], key: str):
    for bk in PREFERRED_BOOKS:
        b = next((x for x in bookmakers or [] if x.get("key")==bk), None)
        m = next((m for m in (b.get("markets") or []) if m.get("key")==key), None) if b else None
        if m: return m, bk
    if bookmakers:
        m0 = next((m for m in (bookmakers[0].get("markets") or []) if m.get("key")==key), None)
        if m0: return m0, bookmakers[0].get("key")
    return None, None

async def fetch_odds_for(sport_key: str, start_iso: str, end_iso: str) -> List[Dict[str, Any]]:
    if not ODDS_API_KEY:
        raise HTTPException(status_code=500, detail="ODDS_API_KEY not set")
    params = {
        "regions": "us",
        "markets": "spreads,totals,h2h",
        "oddsFormat": "american",
        "dateFormat": "iso",
        "commenceTimeFrom": start_iso,
        "commenceTimeTo":   end_iso,
        "apiKey": ODDS_API_KEY,
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(20, connect=10)) as c:
        r = await c.get(f"{ODDS_BASE}/sports/{sport_key}/odds", params=params)
        if r.status_code == 429:
            raise HTTPException(status_code=429, detail="The Odds API rate limit reached")
        r.raise_for_status()
        data = r.json()
        return data if isinstance(data, list) else []

async def fetch_all_odds() -> List[Dict[str, Any]]:
    start_iso, end_iso = current_window()
    tasks = [fetch_odds_for(sk, start_iso, end_iso) for sk in SPORT_KEYS]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    out: List[Dict[str, Any]] = []
    had_error = False
    for sk, res in zip(SPORT_KEYS, results):
        if isinstance(res, Exception):
            had_error = True
            log.error("Odds fetch failed for %s: %s", sk, res)
            continue
        if not res:
            log.warning("Odds fetch returned 0 events for %s in window %s → %s", sk, start_iso, end_iso)
        out.extend(res)

    if not out and had_error:
        raise HTTPException(status_code=502, detail="All odds fetches failed; check /debug/odds")

    return out

async def fetch_slates_api_sports_by_date(date_ymd: str) -> List[Dict[str, Any]]:
    """
    API-SPORTS (American Football) via RapidAPI.
    GET https://api-american-football.p.rapidapi.com/games?date=YYYY-MM-DD
    Normalizes date to ISO string so our join_key matches Odds API commence_time.
    """
    def _flatten_dt(raw):
        if not raw:
            return None
        if isinstance(raw, str):
            try:
                return parse_iso(raw).isoformat().replace("+00:00","Z")
            except Exception:
                if len(raw) == 10:
                    return f"{raw}T00:00:00Z"
                return raw
        if isinstance(raw, dict):
            ts = raw.get("timestamp")
            if ts is not None:
                try:
                    return datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat().replace("+00:00","Z")
                except Exception:
                    pass
            d, t = raw.get("date"), raw.get("time")
            if d and t:
                return f"{d}T{t}:00Z" if len(t) == 5 else f"{d}T{t}Z"
            if d:
                return f"{d}T00:00:00Z"
        return None

    key = APISPORTS_KEY or os.getenv("RAPIDAPI_KEY")
    if not key:
        return []
    headers = {"X-RapidAPI-Key": key, "X-RapidAPI-Host": APISPORTS_HOST}
    params = {"date": date_ymd}

    async with httpx.AsyncClient(timeout=20) as c:
        r = await c.get(f"{APISPORTS_BASE}/games", headers=headers, params=params)
        if r.status_code >= 400:
            return []
        data = r.json().get("response", [])
        rows = []
        for g in data:
            league = (g.get("league") or {}).get("name") or g.get("league")
            season = (g.get("league") or {}).get("season") or g.get("season")
            week   = (g.get("week") or {}).get("number") or g.get("week")
            raw_dt = g.get("date") or (g.get("game") or {}).get("date") or (g.get("fixture") or {}).get("date")
            dt_iso = _flatten_dt(raw_dt)

            teams = g.get("teams") or {}
            home = (teams.get("home") or {}).get("name") or g.get("home")
            away = (teams.get("away") or {}).get("name") or g.get("away")

            venue = (g.get("game") or {}).get("venue") or g.get("venue") or {}
            lat = (venue.get("coordinates") or {}).get("latitude") or venue.get("lat")
            lon = (venue.get("coordinates") or {}).get("longitude") or venue.get("lon")

            game_id = g.get("id") or (g.get("game") or {}).get("id") or (g.get("fixture") or {}).get("id")

            rows.append({
                "source": "api-sports",
                "league": league, "season": season, "week": week,
                "game_id": game_id, "game_time": dt_iso,
                "home_team": home, "away_team": away,
                "venue_lat": lat, "venue_lon": lon,
            })
        return rows

async def fetch_cfbd_venues_map() -> Dict[str, Dict[str, Any]]:
    cached = cache_get("cfbd_venues")
    if cached is not None:
        return cached
    if not CFBD_KEY:
        cache_set("cfbd_venues", {})
        return {}
    headers = {"Authorization": f"Bearer {CFBD_KEY}"}
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.get(f"{CFBD_BASE}/venues", headers=headers)
        r.raise_for_status()
        arr = r.json() if isinstance(r.json(), list) else []
        m = {}
        for v in arr:
            lat, lon = v.get("latitude"), v.get("longitude")
            tz = v.get("timezone")
            name = (v.get("name") or "").strip()
            city = (v.get("city") or "").strip()
            school = (v.get("school") or "").strip()
            if lat is None or lon is None: continue
            entry = {"lat": lat, "lon": lon, "tz": tz, "venue": name, "city": city}
            if school: m[f"school::{school.lower()}"] = entry
            if name:   m[f"venue::{name.lower()}"]   = entry
        cache_set("cfbd_venues", m)
        return m

async def get_weather_points(lat: float, lon: float, start_iso: str, end_iso: str) -> List[Dict[str, Any]]:
    if not TIO_API_KEY:
        return []
    headers = {"apikey": TIO_API_KEY}
    body = {
        "location": f"{lat},{lon}",
        "fields": ["temperature","windSpeed","windGust","precipitationType",
                   "precipitationIntensity","humidity","visibility","cloudCover"],
        "timesteps": ["1h"],
        "startTime": start_iso, "endTime": end_iso,
        "units": "imperial"
    }
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(TIO_URL, headers=headers, json=body)
        if r.status_code >= 400:
            return []
        data = r.json().get("data", {}).get("timelines", [])
        hourly = next((x.get("intervals", []) for x in data if x.get("timestep")=="1h"), [])
        return hourly

def pick_outcome(market: Optional[Dict[str, Any]], name: str) -> Dict[str, Any]:
    return next((o for o in (market or {}).get("outcomes", []) if o.get("name")==name), {})

# -------------------- Core: build unified rows --------------------
async def build_rows() -> List[Dict[str, Any]]:
    cached = cache_get("model_data_rows")
    if cached is not None:
        return cached

    # 1) Odds (NFL + NCAAF)
    odds = await fetch_all_odds()

    # 2) Slates for next 7 days (for season/week/venues; optional)
    today = datetime.utcnow().date()
    slate_rows: List[Dict[str, Any]] = []
    try:
        tasks = [fetch_slates_api_sports_by_date((today + timedelta(days=i)).isoformat()) for i in range(7)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, list):
                slate_rows.extend(res)
    except Exception:
        pass
    slate_idx: Dict[str, Dict[str, Any]] = {
        join_key(g.get("league"), g.get("game_time"), g.get("away_team"), g.get("home_team")): g
        for g in slate_rows
    }

    # 3) Model rows (precomputed CSV/JSON) if configured
    model_rows = await load_model_rows() if MODEL_SOURCE in ("file","url") else []
    model_idx = {
        join_key(m.get("league"), m.get("game_time"), m.get("away_team"), m.get("home_team")): m
        for m in model_rows
    }

    # 4) CFBD venues (NCAA fallback)
    cfbd_map = await fetch_cfbd_venues_map()

    rows: List[Dict[str, Any]] = []
    for ev in odds:
        lg = league_label_from_odds_key(ev.get("sport_key") or ev.get("sport_title", ""))
        if not lg:
            continue
        commence_time = ev.get("commence_time")
        home = ev.get("home_team")
        away = ev.get("away_team")
        k = join_key(lg, commence_time, away, home)

        # markets
        m_spreads, bk_spread = pick_market(ev.get("bookmakers") or [], "spreads")
        m_totals,  bk_total  = pick_market(ev.get("bookmakers") or [], "totals")
        m_h2h,     bk_h2h    = pick_market(ev.get("bookmakers") or [], "h2h")

        # outcomes
        ml_home = pick_outcome(m_h2h, home)
        ml_away = pick_outcome(m_h2h, away)
        over    = next((o for o in (m_totals or {}).get("outcomes", []) if (o.get("name") or "").lower()=="over"), {})
        under   = next((o for o in (m_totals or {}).get("outcomes", []) if (o.get("name") or "").lower()=="under"), {})
        sp_home = pick_outcome(m_spreads, home)

        row: Dict[str, Any] = {
            "date": commence_time,
            "league": lg, "season": None, "week": None,
            "game_id": ev.get("id"),
            "game_time": commence_time,
            "away_team": away, "home_team": home,

            # sportsbook values
            "sportsbook_spread": sp_home.get("point"),
            "sportsbook_total": over.get("point") or under.get("point"),
            "sportsbook_ml_odds_home": ml_home.get("price"),
            "sportsbook_ml_odds_away": ml_away.get("price"),
            "book_spread": bk_spread, "book_total": bk_total, "book_h2h": bk_h2h,

            # model fields (populated from model file or script)
            "model_spread": None, "model_total": None, "model_pick": None,
            "confidence_pct": None, "edge_vs_line": None,
            "ou_pick": None, "ou_confidence_pct": None, "total_edge": None,
            "model_ml_prob_home": None, "model_ml_prob_away": None, "ml_value_flag": None,

            # signals
            "trap_alert": None, "sharp_flag": None,
            "weather_alert": None, "volatility_score": None,
        }

        # season/week if slate had it
        s = slate_idx.get(k)
        if s:
            row["season"] = s.get("season")
            row["week"]   = s.get("week")

        # merge precomputed model fields (if any)
        m = model_idx.get(k)
        if m:
            for fld in [
                "model_spread","model_total","model_pick","confidence_pct","edge_vs_line",
                "ou_pick","ou_confidence_pct","total_edge",
                "model_ml_prob_home","model_ml_prob_away","ml_value_flag",
                "trap_alert","sharp_flag","volatility_score"
            ]:
                if m.get(fld) is not None and m.get(fld) != "":
                    row[fld] = m[fld]

        # compute edges if inputs exist and model didn't provide them
        try:
            if row.get("model_spread") is not None and row.get("sportsbook_spread") is not None and row.get("edge_vs_line") is None:
                row["edge_vs_line"] = float(row["model_spread"]) - float(row["sportsbook_spread"])
        except Exception:
            pass
        try:
            if row.get("model_total") is not None and row.get("sportsbook_total") is not None and row.get("total_edge") is None:
                row["total_edge"] = float(row["model_total"]) - float(row["sportsbook_total"])
        except Exception:
            pass

        # ---------- Weather (venue resolution) ----------
        venue_lat = (s or {}).get("venue_lat")
        venue_lon = (s or {}).get("venue_lon")

        # NCAA fallback via CFBD school mapping
        if (venue_lat is None or venue_lon is None) and lg == "NCAA" and CFBD_KEY:
            m_school = cfbd_map.get(f"school::{(home or '').lower()}")
            if m_school:
                venue_lat, venue_lon = m_school["lat"], m_school["lon"]

        # NFL fallback via static stadium map
        if (venue_lat is None or venue_lon is None) and lg == "NFL":
            st = NFL_STADIUMS.get(home or "")
            if st:
                venue_lat, venue_lon = st["lat"], st["lon"]

        # if we have coords + kickoff, pull a timeline slice and compute alert
        if venue_lat is not None and venue_lon is not None and commence_time:
            k_dt = parse_iso(commence_time)
            win_start = iso((k_dt - timedelta(hours=2)).replace(minute=0, second=0, microsecond=0))
            win_end   = iso((k_dt + timedelta(hours=2)).replace(minute=0, second=0, microsecond=0))
            try:
                pts = await get_weather_points(float(venue_lat), float(venue_lon), win_start, win_end)
                nearest = nearest_hour_point(pts, commence_time)
                values = (nearest or {}).get("values", {})
                row["weather_alert"] = compute_weather_alert(values)
                row["weather_snapshot"] = values
            except Exception as e:
                log.warning("Weather fetch failed for %s @ %s: %s", lg, commence_time, e)

        # stable primary key for Retool tables
        row["row_key"] = row.get("game_id") or synth_row_key(lg, commence_time, away, home)

        # Finally, coerce types in case upstream gave strings
        coerce_types(row)

        rows.append(row)

    # --- Run script model if available (MODEL_SOURCE=script) ---
    pm = get_predict_module()
    if pm and hasattr(pm, "predict_games"):
        try:
            inputs = [{
                "league": r.get("league"),
                "game_time": r.get("game_time"),
                "home_team": r.get("home_team"),
                "away_team": r.get("away_team"),
                "sportsbook_spread": r.get("sportsbook_spread"),
                "sportsbook_total": r.get("sportsbook_total"),
                "sportsbook_ml_odds_home": r.get("sportsbook_ml_odds_home"),
                "sportsbook_ml_odds_away": r.get("sportsbook_ml_odds_away"),
            } for r in rows]

            preds = pm.predict_games(inputs) or []
            pred_idx = {
                join_key(p.get("league"), p.get("game_time"), p.get("away_team"), p.get("home_team")): p
                for p in preds if p.get("league") and p.get("game_time")
            }

            for r in rows:
                pk = join_key(r.get("league"), r.get("game_time"), r.get("away_team"), r.get("home_team"))
                p = pred_idx.get(pk)
                if not p:
                    continue
                for fld in [
                    "model_ml_prob_home","model_ml_prob_away","model_pick","confidence_pct","ml_value_flag",
                    "model_spread","model_total","edge_vs_line","ou_pick","ou_confidence_pct","total_edge",
                    "trap_alert","sharp_flag","volatility_score"
                ]:
                    if fld in p and p[fld] is not None and p[fld] != "":
                        r[fld] = p[fld]
                # Re-coerce after merge
                coerce_types(r)

                # fill edges if still missing
                try:
                    if r.get("model_spread") is not None and r.get("sportsbook_spread") is not None and r.get("edge_vs_line") is None:
                        r["edge_vs_line"] = float(r["model_spread"]) - float(r["sportsbook_spread"])
                except Exception:
                    pass
                try:
                    if r.get("model_total") is not None and r.get("sportsbook_total") is not None and r.get("total_edge") is None:
                        r["total_edge"] = float(r["model_total"]) - float(r["sportsbook_total"])
                except Exception:
                    pass

        except Exception as e:
            log.error("predict_games failed: %s", e)

    cache_set("model_data_rows", rows)
    return rows

# -------------------- Routes --------------------
@app.get("/")
async def root():
    start_iso, end_iso = current_window()
    return {
        "ok": True,
        "endpoints": ["/health", "/version", "/api/model-data", "/api/history",
                      "/api/model/debug", "/debug/env", "/debug/odds", "/debug/slates",
                      "/debug/coverage", "/debug/model-script"],
        "window": {"from": start_iso, "to": end_iso}
    }

@app.get("/api/model-data")
async def api_model_data(refresh: bool = False):
    if refresh:
        cache_clear("model_data_rows", "model_rows", "cfbd_venues")
    return await build_rows()

@app.get("/api/history")
async def api_history(refresh: bool = False):
    if refresh:
        cache_clear("history_rows")
    return await load_history_rows()

@app.get("/api/model/debug")
async def api_model_debug():
    rows = await load_model_rows() if MODEL_SOURCE in ("file","url") else []
    return {
        "MODEL_SOURCE": MODEL_SOURCE,
        "MODEL_PATH": MODEL_PATH,
        "MODEL_URL": MODEL_URL,
        "rows_detected": len(rows),
        "first_row": rows[0] if rows else None
    }

@app.get("/debug/model-script")
def debug_model_script():
    pm = get_predict_module()
    return {
        "enabled": MODEL_SOURCE == "script",
        "path": MODEL_PATH,
        "has_predict_games": bool(pm and hasattr(pm, "predict_games"))
    }

@app.get("/debug/env")
def debug_env():
    start_iso, end_iso = current_window()
    return {
        "has_ODDS_API_KEY": bool(ODDS_API_KEY),
        "has_TIO_API_KEY": bool(TIO_API_KEY),
        "has_APISPORTS_KEY": bool(APISPORTS_KEY or os.getenv("RAPIDAPI_KEY")),
        "has_CFBD_KEY": bool(CFBD_KEY),
        "cors_origin": CORS_ORIGIN,
        "window": {"from": start_iso, "to": end_iso},
        "sport_keys": SPORT_KEYS,
        "cache_ttl": CACHE_TTL,
        "log_level": LOG_LEVEL,
        "days_back": DAYS_BACK,
        "days_ahead": DAYS_AHEAD,
        "model_source": MODEL_SOURCE,
        "model_path": MODEL_PATH,
        "model_url": MODEL_URL,
        "model_weights_path": MODEL_WEIGHTS_PATH,
    }

@app.get("/debug/odds")
async def debug_odds():
    if not ODDS_API_KEY:
        raise HTTPException(status_code=500, detail="ODDS_API_KEY not set")
    start_iso, end_iso = current_window()
    results = {}
    async with httpx.AsyncClient(timeout=httpx.Timeout(20, connect=10)) as c:
        for sk in SPORT_KEYS:
            url = f"{ODDS_BASE}/sports/{sk}/odds"
            params = {
                "regions": "us",
                "markets": "spreads,totals,h2h",
                "oddsFormat": "american",
                "dateFormat": "iso",
                "commenceTimeFrom": start_iso,
                "commenceTimeTo":   end_iso,
                "apiKey": ODDS_API_KEY,
            }
            try:
                r = await c.get(url, params=params)
                ct = r.headers.get("content-type","")
                try:
                    body = r.json()
                except Exception:
                    body = r.text[:500]
                results[sk] = {
                    "status": r.status_code,
                    "url": _redact_api_key(str(r.url)),
                    "count": len(body) if isinstance(body, list) else None,
                    "sample": body[:1] if isinstance(body, list) else body,
                    "content_type": ct
                }
            except Exception as e:
                results[sk] = {"error": str(e)}
    return {"window": {"from": start_iso, "to": end_iso}, "results": results}

# Show next 14 days of slates from API-SPORTS and how many have venue coords
@app.get("/debug/slates")
async def debug_slates():
    today = datetime.utcnow().date()
    tasks = [fetch_slates_api_sports_by_date((today + timedelta(days=i)).isoformat()) for i in range(14)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    rows = []
    for res in results:
        if isinstance(res, list):
            rows.extend(res)
    with_coords = [r for r in rows if r.get("venue_lat") is not None and r.get("venue_lon") is not None]
    return {
        "total_slates": len(rows),
        "with_coords": len(with_coords),
        "sample": rows[:3],
        "sample_with_coords": with_coords[:3]
    }

# Model coverage: how many odds rows matched a model row OR script prediction
@app.get("/debug/coverage")
async def debug_coverage():
    rows = await build_rows()
    total = len(rows)
    with_model = sum(
        1 for r in rows
        if any(r.get(f) not in (None, "") for f in ["model_spread","model_total","model_pick","model_ml_prob_home","model_ml_prob_away"])
    )
    return {"total_rows": total, "rows_with_any_model_field": with_model}

@app.post("/admin/refresh")
def admin_refresh():
    _cache.clear()
    return {"ok": True, "cleared_keys": True}

@app.get("/health")
async def health():
    return {"ok": True, "last_sync": cache_get("model_data_rows") is not None}

@app.get("/version")
async def version():
    return {"model": "rf_script_or_file", "features": 18, "source": "odds+slate+weather+model", "ttl_seconds": CACHE_TTL}

# ----- Script-model debug (raw output from predict.py) -----
import importlib

try:
    _predict_mod = importlib.import_module("predict")
    _HAS_SCRIPT_MODEL = hasattr(_predict_mod, "predict_games")
except Exception:
    _predict_mod = None
    _HAS_SCRIPT_MODEL = False

@app.get("/debug/model-script")
def debug_model_script(limit: int = 10):
    if not _HAS_SCRIPT_MODEL:
        return {"enabled": False, "reason": "predict.predict_games() not found"}
    try:
        out = _predict_mod.predict_games()
        if not isinstance(out, list):
            return {"enabled": True, "note": "predict_games did not return a list", "type": str(type(out))}
        return {"enabled": True, "count": len(out), "sample": out[:max(0, limit)]}
    except Exception as e:
        return {"enabled": True, "error": str(e)}
