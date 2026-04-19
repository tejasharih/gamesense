from __future__ import annotations

import argparse
from pathlib import Path
from pprint import pprint

import pandas as pd

from gamesense.model import train_and_save, train_from_dataframe
from gamesense.predict import predict_matchup, sample_input
from gamesense.real_data import build_nba_feature_table, fetch_nba_games


DEFAULT_MODEL = Path("models/gamesense_model.pkl")
DEFAULT_DATA = Path("data/synthetic_games.csv")
DEFAULT_REAL_GAMES = Path("data/nba_games_real.csv")
DEFAULT_REAL_FEATURES = Path("data/nba_features_real.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GameSense CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    train_cmd = sub.add_parser("train", help="Generate data, train model, save artifact")
    train_cmd.add_argument("--seed", type=int, default=7)

    sync_cmd = sub.add_parser("sync-nba", help="Fetch real NBA historical games from BALLDONTLIE")
    sync_cmd.add_argument("--seasons", type=int, nargs="+", required=True)
    sync_cmd.add_argument("--append", action="store_true", help="Append newly fetched seasons to the existing local games file")

    real_cmd = sub.add_parser("train-real", help="Train on real NBA feature data")
    real_cmd.add_argument("--seasons", type=int, nargs="+", required=True)
    real_cmd.add_argument("--refresh", action="store_true", help="Refetch seasons from BALLDONTLIE before training")

    pred_cmd = sub.add_parser("predict", help="Run prediction with sample features")
    pred_cmd.add_argument("--league", choices=["NBA", "NFL"], default="NBA")
    pred_cmd.add_argument("--elo-diff", type=float)
    pred_cmd.add_argument("--offensive-diff", type=float)
    pred_cmd.add_argument("--defensive-diff", type=float)
    pred_cmd.add_argument("--injury-diff", type=float)
    pred_cmd.add_argument("--rest-diff", type=float)
    pred_cmd.add_argument("--form-diff", type=float)
    pred_cmd.add_argument("--qb-status-diff", type=float)
    pred_cmd.add_argument("--travel-diff", type=float)
    pred_cmd.add_argument("--market-spread", type=float)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "train":
        metrics = train_and_save(DEFAULT_MODEL, DEFAULT_DATA, seed=args.seed)
        print("Model trained. Backtest metrics:")
        pprint(metrics)
        return

    if args.command == "sync-nba":
        existing_df = None
        if args.append and DEFAULT_REAL_GAMES.exists():
            existing_df = pd.read_csv(DEFAULT_REAL_GAMES)

        games_df = fetch_nba_games(args.seasons, DEFAULT_REAL_GAMES)
        if existing_df is not None:
            games_df = (
                pd.concat([existing_df, games_df], ignore_index=True)
                .drop_duplicates(subset=["game_id"])
                .sort_values(["game_date", "game_id"])
            )
            games_df.to_csv(DEFAULT_REAL_GAMES, index=False)

        feature_df = build_nba_feature_table(games_df, DEFAULT_REAL_FEATURES)
        print("Synced real NBA data:")
        pprint(
            {
                "games_rows": len(games_df),
                "feature_rows": len(feature_df),
                "games_path": str(DEFAULT_REAL_GAMES),
                "features_path": str(DEFAULT_REAL_FEATURES),
            }
        )
        return

    if args.command == "train-real":
        if args.refresh or not DEFAULT_REAL_GAMES.exists() or not DEFAULT_REAL_FEATURES.exists():
            games_df = fetch_nba_games(args.seasons, DEFAULT_REAL_GAMES)
            feature_df = build_nba_feature_table(games_df, DEFAULT_REAL_FEATURES)
        else:
            feature_df = pd.read_csv(DEFAULT_REAL_FEATURES)
            games_df = pd.read_csv(DEFAULT_REAL_GAMES)
            feature_df["game_date"] = pd.to_datetime(feature_df["game_date"])
            if "season" not in feature_df.columns:
                season_lookup = games_df[["game_id", "season"]].drop_duplicates()
                feature_df = feature_df.merge(season_lookup, on="game_id", how="left")

        requested_seasons = set(args.seasons)
        available_seasons = set(pd.Series(games_df["season"]).astype(int).unique())
        if not requested_seasons.issubset(available_seasons):
            missing = sorted(requested_seasons - available_seasons)
            raise RuntimeError(
                f"Missing locally synced seasons: {missing}. Run sync-nba for those seasons first or pass --refresh."
            )

        filtered_df = feature_df[pd.Series(feature_df["season"]).astype(int).isin(args.seasons)].copy()
        filtered_df["game_date"] = filtered_df["game_date"].dt.date.astype(str)
        metrics = train_from_dataframe(
            filtered_df,
            DEFAULT_MODEL,
            DEFAULT_REAL_FEATURES,
            metadata={
                "data_source": "real_nba_history",
                "seasons": sorted(args.seasons),
                "games_path": str(DEFAULT_REAL_GAMES),
                "features_path": str(DEFAULT_REAL_FEATURES),
            },
        )
        print("Real-data model trained. Backtest metrics:")
        pprint(metrics)
        return

    if args.command == "predict":
        feats = sample_input(args.league)
        overrides = {
            "elo_diff": args.elo_diff,
            "offensive_diff": args.offensive_diff,
            "defensive_diff": args.defensive_diff,
            "injury_diff": args.injury_diff,
            "rest_diff": args.rest_diff,
            "form_diff": args.form_diff,
            "qb_status_diff": args.qb_status_diff,
            "travel_diff": args.travel_diff,
            "market_spread": args.market_spread,
        }
        for k, v in overrides.items():
            if v is not None:
                feats[k] = v

        result = predict_matchup(DEFAULT_MODEL, feats)
        print("Prediction:")
        pprint(result)
        print("\nFeatures:")
        pprint(feats)


if __name__ == "__main__":
    main()
