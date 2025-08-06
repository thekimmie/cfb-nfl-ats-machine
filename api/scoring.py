# api/scoring.py
import numpy as np
from .main import FEATURES, WEIGHTS, INTERCEPT, model  # or just import model directly

def score(game):
    # 1) Extract your feature vector in the same order as your metadata:
    x = np.array([game.get(f, 0) for f in FEATURES])
    # 2) Compute a linear score or let your pickle-model do it:
    #    If you're using weights manually:
    logit = INTERCEPT + np.dot(WEIGHTS, x)
    prob  = 1 / (1 + np.exp(-logit))
    #    Or, if you loaded a full sklearn model:
    # prob = model.predict_proba(x.reshape(1, -1))[0,1]

    # 3) Build your output shape—add any fields you want in Retool:
    return {
      **game,
      "model_probability": float(prob),
      "model_pick": prob > 0.5,
      # …and any other fields you calculate
    }
