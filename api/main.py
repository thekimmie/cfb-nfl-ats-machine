from fastapi import FastAPI

app = FastAPI()  # ✅ Define app first

@app.get("/")    # ✅ Then use it
def root():
    return {"message": "The ATS Model API is live!"}

# api/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import requests
import os
import json
from datetime import datetime

app = FastAPI()

# Enable CORS for Retool
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict to Retool's domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuration ---
API_KEY = "your_odds_api_key_here"  # Replace with your actual Odds API key
ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/americanfootball_ncaaf/odds"

# --- Load model metadata ---
METADATA_PATH = os.path.join(os.path.dirname(__file__), "model_metadata.txt")
with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

features_used = metadata["features_used"]
weights = np.array(metadata["weights"])
intercept = metadata.get("intercept", 0.0)

# --- Example Dummy Feature Constructor ---
def construct_features_for_game(game):
    # 🔮 NOTE: These values are placeholders. Replace with real data pulls as available.
    return {
        "custom_qbr": 0.5,
        "explosive_plays": 0.3,
        "turnovers": -0.2,
        "epa_over_opponent_avg": 0.1,
        "opponent_def_efficiency": 0.6,
        "rush_yards": 120,
        "pass_yards": 250,
        "home_away": 1 if game.get("home_team") else 0,
        "weather_impact": 0.1,
        "special_teams_efficiency": 0.5,
        "pass_vs_weak_secondary": 0.3,
        "run_vs_weak_rundef": 0.2,
        "o_line_rating": 0.4,
        "d_line_disruption": 0.3,
        "climate_mismatch": 0,
        "travel_distance": 150,
        "letdown_spot": 0,
        "short_rest": 0
    }

# --- Model Scoring Function ---
def score_game(game, features_dict, sportsbook_spread):
    x = np.array([features_dict[feat] for feat in features_used])
    logits = np.dot(x, weights) + intercept
    prob = 1 / (1 + np.exp(-logits))
    confidence_pct = round(prob * 100, 1) if prob >= 0.5 else round((1 - prob) * 100, 1)
    model_spread = round(logits, 2)
    edge_vs_line = round(model_spread - sportsbook_spread, 2)
    model_pick_team = game["away_team"] if prob >= 0.5 else game["home_team"]
    model_pick = f"{model_pick_team} +{abs(sportsbook_spread)}"

    return {
        "game_id": game["id"],
        "home_team": game["home_team"],
        "away_team": game["away_team"],
        "game_time": game.get("commence_time"),
        "sportsbook_spread": sportsbook_spread,
        "model_spread": model_spread,
        "model_pick": model_pick,
        "confidence_pct": confidence_pct,
        "edge_vs_line": edge_vs_line,
        "trap_alert": edge_vs_line < -3,
        "sharp_flag": edge_vs_line > 3,
        "volatility_score": round(np.std(x), 2)
    }

# --- API Endpoint ---
@app.get("/api/model-data")
def get_model_predictions():
    # 🧪 Dummy Data Instead of Live API
    games = [
        {
            "id": "1",
            "home_team": "Georgia",
            "away_team": "Florida",
            "commence_time": "2025-08-30T19:00:00Z",
            "bookmakers": [
                {
                    "markets": [
                        {
                            "key": "spreads",
                            "outcomes": [
                                { "name": "Florida", "point": -10.0 },
                                { "name": "Georgia", "point": 10.0 }
                            ]
                        }
                    ]
                }
            ]
        },
        {
            "id": "2",
            "home_team": "Alabama",
            "away_team": "LSU",
            "commence_time": "2025-08-30T22:00:00Z",
            "bookmakers": [
                {
                    "markets": [
                        {
                            "key": "spreads",
                            "outcomes": [
                                { "name": "LSU", "point": -2.5 },
                                { "name": "Alabama", "point": 2.5 }
                            ]
                        }
                    ]
                }
            ]
        },
        # ⬇️ Add remaining games here...
    ]

    predictions = []

    for game in games:
        try:
            home_team = game["home_team"]
            away_team = game["away_team"]
            spread = None

            # Same spread logic
            for bookmaker in game.get("bookmakers", []):
                for market in bookmaker.get("markets", []):
                    if market["key"] == "spreads":
                        for outcome in market.get("outcomes", []):
                            if outcome["name"] == away_team:
                                spread = float(outcome.get("point", 0.0))
                                break
                if spread is not None:
                    break

            if spread is None:
                continue

            game_data = {
                "id": game["id"],
                "home_team": home_team,
                "away_team": away_team,
                "commence_time": game.get("commence_time")
            }

            features = construct_features_for_game(game_data)
            prediction = score_game(game_data, features, spread)
            predictions.append(prediction)
        except Exception as e:
            print(f"Error processing game: {e}")
            continue

    return predictions
