from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import math
import random
from typing import Dict, Iterable, List

import pandas as pd


NBA_TEAMS = [
    "BOS", "NYK", "MIL", "MIA", "PHI", "CLE", "IND", "ORL", "LAL", "GSW",
    "DEN", "MIN", "DAL", "PHX", "OKC", "SAC", "MEM", "NOP", "LAC", "HOU",
]

NFL_TEAMS = [
    "KC", "BUF", "BAL", "SF", "DAL", "PHI", "MIA", "DET", "CIN", "LAR",
    "GB", "HOU", "JAX", "PIT", "SEA", "NYJ", "MIN", "TB", "NO", "CLE",
]

FEATURE_COLUMNS = [
    "league_is_nba",
    "elo_diff",
    "offensive_diff",
    "defensive_diff",
    "injury_diff",
    "rest_diff",
    "form_diff",
    "qb_status_diff",
    "travel_diff",
    "market_spread",
]


@dataclass
class TeamLatent:
    elo: float
    offense: float
    defense: float


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _init_team_latents(teams: Iterable[str], rng: random.Random) -> Dict[str, TeamLatent]:
    latents: Dict[str, TeamLatent] = {}
    for t in teams:
        latents[t] = TeamLatent(
            elo=1500 + rng.gauss(0, 80),
            offense=110 + rng.gauss(0, 8),
            defense=110 + rng.gauss(0, 8),
        )
    return latents


def _sample_game_features(
    league: str,
    home: TeamLatent,
    away: TeamLatent,
    rng: random.Random,
) -> Dict[str, float]:
    league_is_nba = 1.0 if league == "NBA" else 0.0
    elo_diff = (home.elo - away.elo) / 100.0
    offensive_diff = (home.offense - away.offense) / 10.0
    defensive_diff = (away.defense - home.defense) / 10.0
    injury_diff = max(-4.0, min(4.0, rng.gauss(0.0, 1.2)))
    rest_diff = max(-3.0, min(3.0, rng.gauss(0.0, 1.1)))
    form_diff = max(-2.5, min(2.5, rng.gauss(0.0, 1.0)))
    qb_status_diff = 0.0 if league == "NBA" else max(-2.0, min(2.0, rng.gauss(0.0, 0.7)))
    travel_diff = max(-3.0, min(3.0, rng.gauss(0.0, 1.0)))

    # Simulate market spread (negative = home favored).
    market_spread = (
        -2.0 * elo_diff
        - 1.4 * offensive_diff
        - 1.0 * defensive_diff
        - 0.7 * rest_diff
        - 0.5 * form_diff
        + rng.gauss(0.0, 1.3)
    )
    if league == "NFL":
        market_spread -= 0.8 * qb_status_diff

    return {
        "league_is_nba": league_is_nba,
        "elo_diff": elo_diff,
        "offensive_diff": offensive_diff,
        "defensive_diff": defensive_diff,
        "injury_diff": injury_diff,
        "rest_diff": rest_diff,
        "form_diff": form_diff,
        "qb_status_diff": qb_status_diff,
        "travel_diff": travel_diff,
        "market_spread": market_spread,
    }


def _home_win_probability(league: str, feats: Dict[str, float], rng: random.Random) -> float:
    # Explicitly explainable scoring rule; later replace with learned model from real APIs.
    score = (
        0.34 * feats["elo_diff"]
        + 0.23 * feats["offensive_diff"]
        + 0.18 * feats["defensive_diff"]
        + 0.12 * feats["injury_diff"]
        + 0.10 * feats["rest_diff"]
        + 0.11 * feats["form_diff"]
        + 0.08 * feats["travel_diff"]
        - 0.20 * feats["market_spread"]
    )
    if league == "NFL":
        score += 0.20 * feats["qb_status_diff"]
        score += 0.22  # Home advantage a bit stronger in NFL sample generation.
    else:
        score += 0.16

    score += rng.gauss(0.0, 0.20)
    return max(0.03, min(0.97, _sigmoid(score)))


def generate_synthetic_dataset(n_games_per_league: int = 1500, seed: int = 7) -> pd.DataFrame:
    rng = random.Random(seed)
    nba_latents = _init_team_latents(NBA_TEAMS, rng)
    nfl_latents = _init_team_latents(NFL_TEAMS, rng)

    start = date(2023, 9, 1)
    rows: List[Dict[str, float]] = []

    for league, teams, latents in [
        ("NBA", NBA_TEAMS, nba_latents),
        ("NFL", NFL_TEAMS, nfl_latents),
    ]:
        for i in range(n_games_per_league):
            home_name, away_name = rng.sample(teams, 2)
            feats = _sample_game_features(league, latents[home_name], latents[away_name], rng)
            p = _home_win_probability(league, feats, rng)
            home_win = 1 if rng.random() < p else 0

            rows.append(
                {
                    "game_id": f"{league}-{i+1:05d}",
                    "game_date": start + timedelta(days=i),
                    "league": league,
                    "home_team": home_name,
                    "away_team": away_name,
                    **feats,
                    "home_win": home_win,
                }
            )

    df = pd.DataFrame(rows)
    return df.sort_values("game_date").reset_index(drop=True)
