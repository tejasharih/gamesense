from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from gamesense.model import load_model, train_and_save
from gamesense.predict import predict_matchup, sample_input


ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT_DIR / "models" / "gamesense_model.pkl"
SYNTHETIC_DATA_PATH = ROOT_DIR / "data" / "synthetic_games.csv"
REAL_GAMES_PATH = ROOT_DIR / "data" / "nba_games_real.csv"
REAL_FEATURES_PATH = ROOT_DIR / "data" / "nba_features_real.csv"
WEB_DIR = ROOT_DIR / "web"


def active_data_path() -> Path:
    return REAL_FEATURES_PATH if REAL_FEATURES_PATH.exists() else SYNTHETIC_DATA_PATH


def ensure_artifacts() -> dict:
    if not MODEL_PATH.exists() or not active_data_path().exists():
        train_and_save(MODEL_PATH, SYNTHETIC_DATA_PATH)
    return load_model(MODEL_PATH)


def latest_matchups() -> list[dict]:
    source_path = REAL_GAMES_PATH if REAL_GAMES_PATH.exists() else active_data_path()
    df = pd.read_csv(source_path)
    date_column = "game_date" if "game_date" in df.columns else "date"
    rows = []
    for league in sorted(df["league"].unique()):
        league_df = df[df["league"] == league].tail(3)
        for _, row in league_df.iterrows():
            home_win = int(row["home_win"]) if "home_win" in row else int(row["home_score"] > row["away_score"])
            rows.append(
                {
                    "league": league,
                    "home_team": row["home_team"],
                    "away_team": row["away_team"],
                    "game_date": row[date_column],
                    "market_spread": round(float(row["market_spread"]), 1) if "market_spread" in row else None,
                    "home_win": home_win,
                }
            )
    return rows


def team_profiles() -> dict:
    if not REAL_GAMES_PATH.exists():
        return {}

    df = pd.read_csv(REAL_GAMES_PATH)
    df["game_date"] = pd.to_datetime(df["game_date"])
    profiles: dict = {}

    teams = sorted(set(df["home_team"]).union(set(df["away_team"])))
    for team in teams:
        home_games = df[df["home_team"] == team].copy()
        away_games = df[df["away_team"] == team].copy()

        wins = int((home_games["home_score"] > home_games["away_score"]).sum() + (away_games["away_score"] > away_games["home_score"]).sum())
        losses = int(len(home_games) + len(away_games) - wins)
        points_for = float(home_games["home_score"].sum() + away_games["away_score"].sum())
        points_against = float(home_games["away_score"].sum() + away_games["home_score"].sum())

        combined = pd.concat(
            [
                home_games.assign(result=(home_games["home_score"] > home_games["away_score"]).astype(int)),
                away_games.assign(result=(away_games["away_score"] > away_games["home_score"]).astype(int)),
            ],
            ignore_index=True,
        ).sort_values("game_date")

        last_five = combined["result"].tail(5).tolist()
        profiles[team] = {
            "wins": wins,
            "losses": losses,
            "win_pct": round(wins / max(wins + losses, 1), 3),
            "points_for": round(points_for / max(wins + losses, 1), 1),
            "points_against": round(points_against / max(wins + losses, 1), 1),
            "last_five": last_five,
        }

    return profiles


def team_snapshots() -> dict:
    if not REAL_GAMES_PATH.exists():
        return {}

    df = pd.read_csv(REAL_GAMES_PATH)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["game_date", "game_id"]).reset_index(drop=True)

    snapshots: dict = {}
    teams = sorted(set(df["home_team"]).union(set(df["away_team"])))
    for team in teams:
        team_games = []
        for _, row in df.iterrows():
            if row["home_team"] == team:
                team_games.append(
                    {
                        "date": row["game_date"],
                        "points_for": int(row["home_score"]),
                        "points_against": int(row["away_score"]),
                        "result": 1 if row["home_score"] > row["away_score"] else 0,
                    }
                )
            elif row["away_team"] == team:
                team_games.append(
                    {
                        "date": row["game_date"],
                        "points_for": int(row["away_score"]),
                        "points_against": int(row["home_score"]),
                        "result": 1 if row["away_score"] > row["home_score"] else 0,
                    }
                )

        if not team_games:
            continue

        wins = sum(game["result"] for game in team_games)
        losses = len(team_games) - wins
        points_for = sum(game["points_for"] for game in team_games)
        points_against = sum(game["points_against"] for game in team_games)
        avg_margin = (points_for - points_against) / max(len(team_games), 1)
        last_five = [game["result"] for game in team_games[-5:]]
        elo = 1500 + avg_margin * 12 + (wins - losses) * 2.5

        snapshots[team] = {
            "games": len(team_games),
            "wins": wins,
            "losses": losses,
            "win_pct": round(wins / max(len(team_games), 1), 3),
            "points_for": round(points_for / max(len(team_games), 1), 1),
            "points_against": round(points_against / max(len(team_games), 1), 1),
            "avg_margin": round(avg_margin, 2),
            "form": round(sum(last_five) / max(len(last_five), 1), 3),
            "last_five": last_five,
            "elo": round(elo, 2),
            "last_game_date": team_games[-1]["date"].date().isoformat(),
        }

    return snapshots


def teams_for_league() -> dict:
    if REAL_GAMES_PATH.exists():
        games_df = pd.read_csv(REAL_GAMES_PATH)
        nba_teams = sorted(set(games_df["home_team"]).union(set(games_df["away_team"])))
    else:
        nba_teams = sorted({preset["home_team"] for preset in preset_matchups()} | {preset["away_team"] for preset in preset_matchups()})

    nfl_teams = [
        "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET", "GB",
        "HOU", "IND", "JAX", "KC", "LAC", "LAR", "LV", "MIA", "MIN", "NE", "NO", "NYG",
        "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
    ]
    return {"NBA": nba_teams, "NFL": nfl_teams}


def team_labels() -> dict:
    return {
        "ATL": "Atlanta Hawks",
        "BKN": "Brooklyn Nets",
        "BOS": "Boston Celtics",
        "CHA": "Charlotte Hornets",
        "CHI": "Chicago Bulls",
        "CLE": "Cleveland Cavaliers",
        "DAL": "Dallas Mavericks",
        "DEN": "Denver Nuggets",
        "DET": "Detroit Pistons",
        "GSW": "Golden State Warriors",
        "HOU": "Houston Rockets",
        "IND": "Indiana Pacers",
        "LAC": "LA Clippers",
        "LAL": "Los Angeles Lakers",
        "MEM": "Memphis Grizzlies",
        "MIA": "Miami Heat",
        "MIL": "Milwaukee Bucks",
        "MIN": "Minnesota Timberwolves",
        "NOP": "New Orleans Pelicans",
        "NYK": "New York Knicks",
        "OKC": "Oklahoma City Thunder",
        "ORL": "Orlando Magic",
        "PHI": "Philadelphia 76ers",
        "PHX": "Phoenix Suns",
        "POR": "Portland Trail Blazers",
        "SAC": "Sacramento Kings",
        "SAS": "San Antonio Spurs",
        "TOR": "Toronto Raptors",
        "UTA": "Utah Jazz",
        "WAS": "Washington Wizards",
    }


def preset_matchups() -> list[dict]:
    if REAL_FEATURES_PATH.exists():
        df = pd.read_csv(REAL_FEATURES_PATH).tail(4)
        presets = []
        for _, row in df.iterrows():
            presets.append(
                {
                    "label": f"{row['away_team']} at {row['home_team']}",
                    "league": row["league"],
                    "home_team": row["home_team"],
                    "away_team": row["away_team"],
                    "features": {
                        "league_is_nba": float(row["league_is_nba"]),
                        "elo_diff": float(row["elo_diff"]),
                        "offensive_diff": float(row["offensive_diff"]),
                        "defensive_diff": float(row["defensive_diff"]),
                        "injury_diff": float(row["injury_diff"]),
                        "rest_diff": float(row["rest_diff"]),
                        "form_diff": float(row["form_diff"]),
                        "qb_status_diff": float(row["qb_status_diff"]),
                        "travel_diff": float(row["travel_diff"]),
                        "market_spread": float(row["market_spread"]),
                    },
                }
            )
        return presets

    return [
        {
            "label": "Celtics vs Knicks",
            "league": "NBA",
            "home_team": "BOS",
            "away_team": "NYK",
            "features": {
                "league_is_nba": 1.0,
                "elo_diff": 1.1,
                "offensive_diff": 0.7,
                "defensive_diff": 0.5,
                "injury_diff": 0.4,
                "rest_diff": 0.3,
                "form_diff": 0.8,
                "qb_status_diff": 0.0,
                "travel_diff": 0.2,
                "market_spread": -5.5,
            },
        },
        {
            "label": "Lakers vs Nuggets",
            "league": "NBA",
            "home_team": "LAL",
            "away_team": "DEN",
            "features": {
                "league_is_nba": 1.0,
                "elo_diff": -0.2,
                "offensive_diff": 0.3,
                "defensive_diff": -0.4,
                "injury_diff": -0.2,
                "rest_diff": 0.6,
                "form_diff": 0.1,
                "qb_status_diff": 0.0,
                "travel_diff": 0.2,
                "market_spread": 1.5,
            },
        },
        {
            "label": "Chiefs vs Bills",
            "league": "NFL",
            "home_team": "KC",
            "away_team": "BUF",
            "features": {
                "league_is_nba": 0.0,
                "elo_diff": 0.5,
                "offensive_diff": 0.4,
                "defensive_diff": 0.3,
                "injury_diff": 0.1,
                "rest_diff": 0.2,
                "form_diff": 0.5,
                "qb_status_diff": 0.4,
                "travel_diff": 0.3,
                "market_spread": -2.5,
            },
        },
        {
            "label": "49ers vs Eagles",
            "league": "NFL",
            "home_team": "SF",
            "away_team": "PHI",
            "features": {
                "league_is_nba": 0.0,
                "elo_diff": 0.2,
                "offensive_diff": 0.1,
                "defensive_diff": 0.4,
                "injury_diff": -0.3,
                "rest_diff": 0.0,
                "form_diff": 0.2,
                "qb_status_diff": -0.2,
                "travel_diff": 0.4,
                "market_spread": -1.0,
            },
        },
    ]


def bootstrap_payload() -> dict:
    artifact = ensure_artifacts()
    metadata = artifact.get("metadata", {})
    data_source = metadata.get("data_source", active_data_path().name)
    profiles = team_profiles()
    snapshots = team_snapshots()
    return {
        "metrics": artifact["metrics"],
        "data_source": data_source,
        "metadata": metadata,
        "team_profiles": profiles,
        "team_snapshots": snapshots,
        "teams": teams_for_league(),
        "team_labels": team_labels(),
        "sample_input": {
            "NBA": sample_input("NBA"),
            "NFL": sample_input("NFL"),
        },
        "presets": preset_matchups(),
        "latest_matchups": latest_matchups(),
    }


class GameSenseHandler(BaseHTTPRequestHandler):
    def _set_no_cache_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")

    def _send_json(self, payload: dict, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._set_no_cache_headers()
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, file_path: Path) -> None:
        if not file_path.exists() or not file_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return

        content_type = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".json": "application/json; charset=utf-8",
        }.get(file_path.suffix, "text/plain; charset=utf-8")

        body = file_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self._set_no_cache_headers()
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/bootstrap":
            self._send_json(bootstrap_payload())
            return

        if parsed.path in ("/", "/index.html"):
            self._serve_file(WEB_DIR / "index.html")
            return

        asset_path = (WEB_DIR / parsed.path.lstrip("/")).resolve()
        if WEB_DIR.resolve() not in asset_path.parents and asset_path != WEB_DIR.resolve():
            self.send_error(HTTPStatus.FORBIDDEN, "Forbidden")
            return
        self._serve_file(asset_path)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/predict":
            self.send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length)
        payload = json.loads(body.decode("utf-8"))
        features = payload.get("features", {})

        result = predict_matchup(MODEL_PATH, features)
        self._send_json(result)


def main() -> None:
    ensure_artifacts()
    port = int(os.getenv("PORT", "8000"))
    server = ThreadingHTTPServer(("127.0.0.1", port), GameSenseHandler)
    print(f"GameSense dashboard running at http://127.0.0.1:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
