# main.py  (repo root)

import os, asyncio, time, logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta, timezone

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import re  # at the top with imports if not already

def _redact_api_key(url_str: str) -> str:
    return re.sub(r'(apiKey=)[^&]+', r'\1REDACTED', url_str)

# ========= Env / Config =========
ODDS_API_KEY   = os.getenv("ODDS_API_KEY")          # The Odds API (official)
APISPORTS_KEY  = os.getenv("APISPORTS_KEY")         # API-SPORTS (via RapidAPI) for slates/injuries
CFBD_KEY       = os.getenv("CFBD_KEY")              # CollegeFootballData (CFB venues fallback)
TIO_API_KEY    = os.getenv("TIO_API_KEY")           # Tomorrow.io (Timeline API)
CACHE_TTL      = int(os.getenv("CACHE_TTL", "600")) # seconds
CORS_ORIGIN    = [o.strip() for o in os.getenv("CORS_ORIGIN", "*").split(",")]
LOG_LEVEL      = os.getenv("LOG_LEVEL", "INFO")

# Odds window (so we don't get an empty [] on quiet days)
DAYS_BACK  = int(os.getenv("DAYS_BACK",  "1"))
DAYS_AHEAD = int(os.getenv("DAYS_AHEAD", "14"))

# API bases
ODDS_BASE       = "https://api.the-odds-api.com/v4"
APISPORTS_HOST  = "api-american-football.p.rapidapi.com"
APISPORTS_BASE  = f"https://{APISPORTS_HOST}"
CFBD_BASE       = "https://api.collegefootballdata.com"
TIO_URL         = "https://api.tomorrow.io/v4/timelines"

SPORT_KEYS = ["americanfootball_nfl", "americanfootball_ncaaf"]  # Odds API sport keys

# ========= App =========
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGIN if CORS_ORIGIN != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL.upper(), logging.INFO))
log = logging.getLogger("betting-machine")

# ========= In-memory cache =========
_cache: Dict[str, Dict[str, Any]] = {}  # key -> {"ts": epoch, "data": object}

def cache_get(key: str):
    item = _cache.get(key)
    if not item: return None
    if time.time() - item["ts"] > CACHE_TTL: return None
    return item["data"]

def cache_set(key: str, data: Any):
    _cache[key] = {"ts": time.time(), "data": data}
    return data

# ========= Helpers =========
def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

def parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)

def current_window():
    now = datetime.utcnow().replace(tzinfo=timezone.utc)
    return iso(now - timedelta(days=DAYS_BACK)), iso(now + timedelta(days=DAYS_AHEAD))

def nearest_hour_point(points: List[Dict[str, Any]], kickoff_iso: str) -> Optional[Dict[str, Any]]:
    if not points: return None
    k = parse_iso(kickoff_iso)
    best = None
    best_delta = None
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

# ========= Adapters =========
async def fetch_odds_for(sport_key: str, start_iso: str, end_iso: str) -> List[Dict[str, Any]]:
    """
    The Odds API v4 supports commenceTimeFrom/commenceTimeTo to widen the window.
    """
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
        # Surface that something failed so you don't just see [] forever
        raise HTTPException(status_code=502, detail="All odds fetches failed; check /debug/odds")

    return out

async def fetch_slates_api_sports_by_date(date_ymd: str) -> List[Dict[str, Any]]:
    """
    API-SPORTS (American Football) via RapidAPI.
    GET https://api-american-football.p.rapidapi.com/games?date=YYYY-MM-DD
    """
    key = APISPORTS_KEY or os.getenv("RAPIDAPI_KEY")
    if not key:
        return []
    headers = {"X-RapidAPI-Key": key, "X-RapidAPI-Host": APISPORTS_HOST}
    params = {"date": date_ymd}
    async with httpx.AsyncClient(timeout=20) as c:
        r = await c.get(f"{APISPORTS_BASE}/games", headers=headers, params=params)
        if r.status_code >= 400:
            return []  # be forgiving
        data = r.json().get("response", [])
        rows = []
        for g in data:
            league = (g.get("league") or {}).get("name") or g.get("league")
            season = (g.get("league") or {}).get("season") or g.get("season")
            week   = (g.get("week") or {}).get("number") or g.get("week")
            game_id = g.get("id") or (g.get("game") or {}).get("id") or (g.get("fixture") or {}).get("id")
            dt = g.get("date") or (g.get("game") or {}).get("date") or (g.get("fixture") or {}).get("date")
            teams = g.get("teams") or {}
            home = (teams.get("home") or {}).get("name") or g.get("home")
            away = (teams.get("away") or {}).get("name") or g.get("away")
            venue = (g.get("game") or {}).get("venue") or g.get("venue") or {}
            lat = (venue.get("coordinates") or {}).get("latitude") or venue.get("lat")
            lon = (venue.get("coordinates") or {}).get("longitude") or venue.get("lon")
            rows.append({
                "source": "api-sports",
                "league": league, "season": season, "week": week,
                "game_id": game_id, "game_time": dt,
                "home_team": home, "away_team": away,
                "venue_lat": lat, "venue_lon": lon,
            })
        return rows

async def fetch_cfbd_venues_map() -> Dict[str, Dict[str, Any]]:
    """
    CFBD venues for college: map by school/venue -> {lat, lon, tz}
    """
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

# ========= Core: build unified rows =========
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

async def build_rows() -> List[Dict[str, Any]]:
    cached = cache_get("model_data_rows")
    if cached is not None:
        return cached

    # 1) Odds (NFL + NCAAF) across a wider window so we don't get []
    odds = await fetch_all_odds()

    # 2) Optional: slates for venue/season/week (next 7 days)
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

    # 3) CFBD venues (fallback for NCAA)
    cfbd_map = await fetch_cfbd_venues_map()

    rows: List[Dict[str, Any]] = []
    for ev in odds:
        lg = league_label_from_odds_key(ev.get("sport_key") or ev.get("sport_title", ""))
        if not lg: continue
        commence_time = ev.get("commence_time")
        home = ev.get("home_team")
        away = ev.get("away_team")
        k = join_key(lg, commence_time, away, home)

        # markets
        m_spreads, bk_spread = pick_market(ev.get("bookmakers") or [], "spreads")
        m_totals,  bk_total  = pick_market(ev.get("bookmakers") or [], "totals")
        m_h2h,     bk_h2h    = pick_market(ev.get("bookmakers") or [], "h2h")

        # outcomes
        ml_home = next((o for o in (m_h2h or {}).get("outcomes", []) if o.get("name")==home), {})
        ml_away = next((o for o in (m_h2h or {}).get("outcomes", []) if o.get("name")==away), {})
        over    = next((o for o in (m_totals or {}).get("outcomes", []) if (o.get("name") or "").lower()=="over"), {})
        under   = next((o for o in (m_totals or {}).get("outcomes", []) if (o.get("name") or "").lower()=="under"), {})
        sp_home = next((o for o in (m_spreads or {}).get("outcomes", []) if o.get("name")==home), {})
        sp_away = next((o for o in (m_spreads or {}).get("outcomes", []) if o.get("name")==away), {})

        row = {
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

            # model fields (to be filled once your model is merged)
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

        # ---------- Weather (auto venue resolution) ----------
        venue_lat = (s or {}).get("venue_lat")
        venue_lon = (s or {}).get("venue_lon")

        # NCAA fallback via CFBD school mapping
        if (venue_lat is None or venue_lon is None) and lg == "NCAA" and CFBD_KEY:
            m = cfbd_map.get(f"school::{(home or '').lower()}")
            if m:
                venue_lat, venue_lon = m["lat"], m["lon"]

        if venue_lat is not None and venue_lon is not None and commence_time:
            k_dt = parse_iso(commence_time)
            win_start = iso((k_dt - timedelta(hours=2)).replace(minute=0, second=0, microsecond=0))
            win_end   = iso((k_dt + timedelta(hours=2)).replace(minute=0, second=0, microsecond=0))
            try:
                pts = await get_weather_points(float(venue_lat), float(venue_lon), win_start, win_end)
                nearest = nearest_hour_point(pts, commence_time)
                values = (nearest or {}).get("values", {})
                row["weather_alert"] = compute_weather_alert(values)
                row["weather_snapshot"] = values  # optional for detail UI
            except Exception as e:
                log.warning("Weather fetch failed for %s vs %s: %s", away, home, e)

        rows.append(row)

    cache_set("model_data_rows", rows)
    return rows

# ========= Routes =========
@app.get("/")
async def root():
    start_iso, end_iso = current_window()
    return {
        "ok": True,
        "endpoints": ["/health", "/version", "/api/model-data", "/debug/env", "/debug/odds"],
        "window": {"from": start_iso, "to": end_iso}
    }

@app.get("/api/model-data")
async def model_data():
    return await build_rows()

@app.get("/health")
async def health():
    return {"ok": True, "last_sync": cache_get("model_data_rows") is not None}

@app.get("/version")
async def version():
    return {"model": "lr_v1", "features": 18, "source": "odds+slate+weather", "ttl_seconds": CACHE_TTL}

# ---- Debug endpoints ----
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
