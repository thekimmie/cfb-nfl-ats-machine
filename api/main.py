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
BASE_DIR    = os.path.dirname(__file__)
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
MODEL_PATH  = os.path.join(PROJECT_DIR, "final_ats_model.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


# ── RapidAPI config ─────────────────────────────────────────────────────────
RAPID_KEY = os.getenv("RAPIDAPI_KEY")
ODDS_HOST = os.getenv("RAPIDAPI_HOST")  # e.g. "americanodds.p.rapidapi.com"
SPORT     = "ncaaf"                    # or "nfl"


async def fetch_odds():
    url = f"https://{ODDS_HOST}/odds/{SPORT}"
    headers = {
        "X-RapidAPI-Key": RAPID_KEY,
        "X-RapidAPI-Host": ODDS_HOST
    }
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url, headers=headers)

        # DEBUG: log status code and first 200 characters of the body
        print("⚙️  RapidAPI status:", resp.status_code)
        print("⚙️  RapidAPI body:", resp.text[:200])

        resp.raise_for_status()
        return resp.json()


def parse_games(raw):
    """Normalize RapidAPI response into our minimal game dicts."""
    games = []
    for g in raw:
        home   = g["homeTeam"]
        away   = g["awayTeam"]
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
    x = np.array([game.get(f, 0) for f in FEATURES]).reshape(1, -1)

    try:
        prob = float(model.predict_proba(x)[0, 1])
    except Exception:
        logit = float(INTERCEPT + np.dot(WEIGHTS, x.flatten()))
        prob  = 1 / (1 + np.exp(-logit))

    return {
        **game,
        "model_probability": prob,
        "model_pick":        prob > 0.5
    }


@app.get("/api/model-data")
async def model_data():
    # Let any errors bubble up so we can see them in the logs
    raw_odds = await fetch_odds()
    games    = parse_games(raw_odds)
    return [score(g) for g in games]
