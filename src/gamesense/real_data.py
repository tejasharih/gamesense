from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

from gamesense.balldontlie import BallDontLieClient


@dataclass
class TeamState:
    elo: float = 1500.0
    wins: int = 0
    losses: int = 0
    games: int = 0
    points_for: float = 0.0
    points_against: float = 0.0
    recent_results: tuple[int, ...] = ()
    last_game_day: pd.Timestamp | None = None


def _to_abbr(team: dict) -> str:
    return team.get("abbreviation", "UNK")


def fetch_nba_games(seasons: Iterable[int], out_path: Path) -> pd.DataFrame:
    client = BallDontLieClient.from_env()
    seasons = list(seasons)
    partial_rows: list[dict] = []

    def _on_page(page_number: int, total_rows: int) -> None:
        print(f"Fetched page {page_number} for seasons {seasons}; raw rows so far: {total_rows}", flush=True)

    games = client.get_nba_games(seasons=seasons, on_page=_on_page)
    rows = []
    for game in games:
        if game.get("postseason"):
            continue
        home_score = game.get("home_team_score")
        away_score = game.get("visitor_team_score")
        if home_score is None or away_score is None:
            continue
        rows.append(
            {
                "game_id": game["id"],
                "game_date": game["date"],
                "season": game["season"],
                "league": "NBA",
                "home_team": _to_abbr(game["home_team"]),
                "away_team": _to_abbr(game["visitor_team"]),
                "home_score": int(home_score),
                "away_score": int(away_score),
            }
            )
        partial_rows.append(rows[-1])
        if len(partial_rows) % 250 == 0:
            (
                pd.DataFrame(partial_rows)
                .drop_duplicates(subset=["game_id"])
                .sort_values(["game_date", "game_id"])
                .to_csv(out_path, index=False)
            )
            print(f"Saved partial sync to {out_path} with {len(partial_rows)} rows", flush=True)
    df = pd.DataFrame(rows).drop_duplicates(subset=["game_id"]).sort_values(["game_date", "game_id"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Completed NBA sync for seasons {seasons}. Final rows: {len(df)}", flush=True)
    return df


def build_nba_feature_table(games_df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    df = games_df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["game_date", "game_id"]).reset_index(drop=True)

    team_state: Dict[str, TeamState] = defaultdict(TeamState)
    rows: list[dict] = []

    for row in df.itertuples(index=False):
        home = team_state[row.home_team]
        away = team_state[row.away_team]

        home_avg_margin = _avg_margin(home)
        away_avg_margin = _avg_margin(away)
        home_form = _recent_form(home)
        away_form = _recent_form(away)

        home_rest = _rest_days(home, row.game_date)
        away_rest = _rest_days(away, row.game_date)

        elo_diff = (home.elo - away.elo) / 100.0
        offensive_diff = ((home.points_for / max(home.games, 1)) - (away.points_for / max(away.games, 1))) / 10.0
        defensive_diff = ((away.points_against / max(away.games, 1)) - (home.points_against / max(home.games, 1))) / 10.0
        rest_diff = (home_rest - away_rest) / 3.0
        form_diff = home_form - away_form
        travel_diff = 0.15
        injury_diff = 0.0
        qb_status_diff = 0.0
        market_spread = -(0.9 * elo_diff + 0.7 * (home_avg_margin - away_avg_margin) + 0.45 * rest_diff)
        home_win = 1 if row.home_score > row.away_score else 0

        rows.append(
            {
                "game_id": row.game_id,
                "game_date": row.game_date.date().isoformat(),
                "season": row.season,
                "league": "NBA",
                "home_team": row.home_team,
                "away_team": row.away_team,
                "league_is_nba": 1.0,
                "elo_diff": round(elo_diff, 4),
                "offensive_diff": round(offensive_diff, 4),
                "defensive_diff": round(defensive_diff, 4),
                "injury_diff": injury_diff,
                "rest_diff": round(rest_diff, 4),
                "form_diff": round(form_diff, 4),
                "qb_status_diff": qb_status_diff,
                "travel_diff": travel_diff,
                "market_spread": round(market_spread, 4),
                "home_win": home_win,
            }
        )

        _update_team_state(home, row.home_score, row.away_score, row.game_date, did_win=home_win == 1)
        _update_team_state(away, row.away_score, row.home_score, row.game_date, did_win=home_win == 0)
        _apply_elo(home, away, home_win)

    features_df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(out_path, index=False)
    return features_df


def _avg_margin(team: TeamState) -> float:
    if team.games == 0:
        return 0.0
    return (team.points_for - team.points_against) / team.games


def _recent_form(team: TeamState) -> float:
    if not team.recent_results:
        return 0.0
    return sum(team.recent_results[-5:]) / min(5, len(team.recent_results))


def _rest_days(team: TeamState, game_date: pd.Timestamp) -> int:
    if team.last_game_day is None:
        return 3
    return max(0, int((game_date - team.last_game_day).days) - 1)


def _update_team_state(team: TeamState, points_for: int, points_against: int, game_date: pd.Timestamp, *, did_win: bool) -> None:
    team.points_for += points_for
    team.points_against += points_against
    team.games += 1
    team.wins += 1 if did_win else 0
    team.losses += 0 if did_win else 1
    team.recent_results = (*team.recent_results[-4:], 1 if did_win else 0)
    team.last_game_day = game_date


def _apply_elo(home: TeamState, away: TeamState, home_win: int, k: float = 20.0) -> None:
    expected_home = 1.0 / (1.0 + 10 ** ((away.elo - home.elo) / 400.0))
    result = float(home_win)
    home.elo += k * (result - expected_home)
    away.elo += k * ((1.0 - result) - (1.0 - expected_home))
