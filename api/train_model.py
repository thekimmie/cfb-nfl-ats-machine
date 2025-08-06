import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

# 1. Load the data
df = pd.read_csv("qb_team_ats_data.csv")

# 2. Choose your features & target
features = [
  "spread",
  "qb_passing_yards",
  "qb_rushing_yards",
  "qb_total_epa",
  "qb_turnovers",
  "opponent_def_epa_allowed",
  "opponent_def_pass_yards_allowed"
]
X = df[features]
y = df["covered_spread"]

# 3. Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 4. Save the trained model to disk
with open("final_ats_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as final_ats_model.pkl")
