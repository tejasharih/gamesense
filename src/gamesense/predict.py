from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from gamesense.model import load_model


def confidence_tier(prob: float) -> str:
    edge = abs(prob - 0.5)
    if edge >= 0.25:
        return "Very High"
    if edge >= 0.15:
        return "High"
    if edge >= 0.08:
        return "Medium"
    return "Low"


def _contributions(artifact: Dict[str, object], features: Dict[str, float]) -> List[Tuple[str, float]]:
    pipe = artifact["pipeline"]
    feature_columns = artifact["feature_columns"]
    scaler = pipe.named_steps["scaler"]
    clf = pipe.named_steps["clf"]

    x = np.array([features[c] for c in feature_columns], dtype=float)
    x_scaled = (x - scaler.mean_) / scaler.scale_
    contrib = x_scaled * clf.coef_[0]

    pairs = list(zip(feature_columns, contrib.tolist()))
    pairs.sort(key=lambda t: abs(t[1]), reverse=True)
    return pairs


def predict_matchup(model_path: Path, features: Dict[str, float]) -> Dict[str, object]:
    artifact = load_model(model_path)
    feature_columns = artifact["feature_columns"]
    row = pd.DataFrame([{c: features[c] for c in feature_columns}])

    pipe = artifact["pipeline"]
    home_prob = float(pipe.predict_proba(row)[:, 1][0])

    top_factors = _contributions(artifact, features)[:3]
    formatted_factors = [
        {
            "feature": feat,
            "impact": round(score, 3),
            "direction": "home_edge" if score >= 0 else "away_edge",
        }
        for feat, score in top_factors
    ]

    return {
        "home_win_probability": round(home_prob, 4),
        "away_win_probability": round(1.0 - home_prob, 4),
        "confidence": confidence_tier(home_prob),
        "top_factors": formatted_factors,
    }


def sample_input(league: str = "NBA") -> Dict[str, float]:
    league_upper = league.upper()
    is_nba = 1.0 if league_upper == "NBA" else 0.0
    return {
        "league_is_nba": is_nba,
        "elo_diff": 0.85,
        "offensive_diff": 0.60,
        "defensive_diff": 0.45,
        "injury_diff": 0.90,
        "rest_diff": 0.30,
        "form_diff": 0.70,
        "qb_status_diff": 0.0 if is_nba == 1.0 else 0.8,
        "travel_diff": 0.40,
        "market_spread": -3.5,
    }
