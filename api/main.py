# ── api/main.py ────────────────────────────────────────────────
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import json, os

# ── FastAPI APP (only one!) ───────────────────────────────────
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later restrict to Retool domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Health-check route ────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "The ATS Model API is live!"}

# ── Load model metadata once at startup ───────────────────────
META_PATH = os.path.join(os.path.dirname(__file__), "model_metadata.txt")
with open(META_PATH, "r") as f:
    meta = json.load(f)

FEATURES = meta["features_used"]
WEIGHTS   = np.array(meta["weights"])
INTERCEPT = meta.get("intercept", 0.0)

import httpx, os, asyncio

# --- Rapid-API credentials pulled from env --------------------
RAPID_KEY  = os.getenv("RAPIDAPI_KEY")
ODDS_HOST  = os.getenv("RAPIDAPI_HOST_ODDS")  # e.g. americanodds.p.rapidapi.com
SPORT      = "ncaaf"                         # change to "nfl" when needed

async def fetch_odds():
    url = f"https://{ODDS_HOST}/odds/{SPORT}"
    headers = {
        "X-RapidAPI-Key": RAPID_KEY,
        "X-RapidAPI-Host": ODDS_HOST
    }
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url, headers=headers)
        r.raise_for_status()
        return r.json()

def parse_games(raw):
    """Turn Rapid-API response into the minimal structure our
       score() function expects (id, home_team, away_team, spread)."""
    games = []
    for g in raw:
        home  = g["homeTeam"]
        away  = g["awayTeam"]
        # sample path – adjust to the exact JSON fields you receive
        spread = float(g["markets"][0]["outcomes"][0]["point"])
        games.append({
            "id": str(g["id"]),
            "home_team": home,
            "away_team": away,
            "commence_time": g["commenceTime"],
            "spread": spread
        })
    return games

@app.get("/api/model-data")
async def model_data():
    try:
        raw_odds = await fetch_odds()
        games    = parse_games(raw_odds)
    except Exception as e:
        # fallback: return empty list so the frontend doesn’t crash
        print("Rapid-API fetch failed:", e)
        return []

    return [score(g) for g in games]
