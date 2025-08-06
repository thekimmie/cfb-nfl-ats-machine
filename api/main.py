# api/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import pickle
import numpy as np
import httpx

# ── App setup ───────────────────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # later restrict to your Retool domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "The ATS Model API is live!"}


# ── Load model metadata & weights ───────────────────────────────────────────
META_PATH = os.path.join(os.path.dirname(__file__), "model_metadata.txt")
with open(META_PATH, "r") as f:
    meta = json.load(f)

FEATURES  = meta["features_used"]
WEIGHTS   = np.array(meta["weights"])
INTERCEPT = meta.get("intercept", 0.0)


# ── Load your trained model (optional if you want to use sklearn predict_proba) ──
MODEL_PATH = os.path.join(os.path.dirname(__file__), "final_ats_model.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


# ── RapidAPI config ─────────────────────────────────────────────────────────
RAPID_KEY     = os.getenv("RAPIDAPI_KEY")
ODDS_HOST     = os.getenv("RAPIDAPI_HOST_ODDS")  # e.g. americanodds.p.rapidapi.com
SPORT         = "ncaaf"                         # or "nfl"


async def fetch_odds():
    url = f"https://{ODDS_HOST}/odds/{SPORT}"
    headers = {
        "X-RapidAPI-Key": RAPID_KEY,
        "X-RapidAPI-Host": ODDS_HOST
    }
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()
        return resp.json()


def parse_games(raw):
    """Normalize RapidAPI response into our minimal game dicts."""
    games = []
    for g in raw:
        home   = g["homeTeam"]
        away   = g["awayTeam"]
        # adjust this path if your JSON differs
        spread = float(g["markets"][0]["outcomes"][0]["point"])
        games.append({
            "id":            str(g["id"]),
            "home_team":     home,
            "away_team":     away,
            "commence_time": g["commenceTime"],
            "spread":        spread
        })
    return games


def score(game: dict) -> dict:
    """Attach model_probability and model_pick to each game."""
    # build feature vector in the correct order
    x = np.array([game.get(f, 0) for f in FEATURES]).reshape(1, -1)

    # try sklearn-style predict_proba first
    try:
        prob = float(model.predict_proba(x)[0, 1])
    except Exception:
        # fallback to manual linear calculation
        logit = float(INTERCEPT + np.dot(WEIGHTS, x.flatten()))
        prob  = 1 / (1 + np.exp(-logit))

    return {
        **game,
        "model_probability": prob,
        "model_pick":        prob > 0.5
    }


# ── API endpoint ─────────────────────────────────────────────────────────────
@app.get("/api/model-data")
async def model_data():
    try:
        raw_odds = await fetch_odds()
        games    = parse_games(raw_odds)
    except Exception as e:
        # if the odds fetch fails, return an empty list (so Retool doesn’t break)
        print("RapidAPI fetch failed:", e)
        return []

    # score + return
    return [score(g) for g in games]
