# api/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests
import json
import numpy as np
import os

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model metadata
with open("model_metadata.txt", "r") as f:
    metadata = json.load(f)

features_used = metadata["features_used"]
weights = np.array(metadata["weights"])
intercept = metadata.get("intercept", 0.0)

ODDS_API_KEY = os.getenv("ODDS_API_KEY")  # You can also hardcode it for now if needed
ODDS_URL = f"https://api.the-odds-api.com/v4/sports/americanfootball_ncaaf/odds?regions=us&markets=spreads&apiKey={ODDS_API_KEY}"

@app.get("/api/model-data")
async def get_model_predictions():
    # Step 1: Pull live odds
    odds_response = requests.get(ODDS_URL)
    games = odds_response.json()

    model_results = []
    for game in games:
        try:
            home_team = game["home_team"]
            away_team = game["away_team"]
            sportsbook_spread = float(game["bookmakers"][0]["markets"][0]["outcomes"][0]["point"])

            # === FAKE FEATURES (replace with real ones later) ===
            features_vector = np.random.rand(len(features_used))  # Placeholder
            logits = np.dot(features_vector, weights) + intercept
            prob = 1 / (1 + np.exp(-logits))
            confidence_pct = round(prob * 100, 1) if prob >= 0.5 else round((1 - prob) * 100, 1)
            model_pick = away_team if prob >= 0.5 else home_team
            model_spread = round(logits, 2)
            edge_vs_line = round(model_spread - sportsbook_spread, 2)
            volatility_score = round(np.std(features_vector), 2)

            result = {
                "game_id": game["id"],
                "home_team": home_team,
                "away_team": away_team,
                "game_time": game["commence_time"],
                "sportsbook_spread": sportsbook_spread,
                "model_spread": model_spread,
                "model_pick": f"{model_pick} +{abs(sportsbook_spread)}",
                "confidence_pct": confidence_pct,
                "edge_vs_line": edge_vs_line,
                "trap_alert": edge_vs_line < -3.0,
                "sharp_flag": edge_vs_line > 3.0,
                "volatility_score": volatility_score,
            }

            model_results.append(result)
        except Exception as e:
            print(f"Error processing game: {e}")

    return model_results
