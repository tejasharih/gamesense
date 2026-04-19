# GameSense

GameSense is a sports matchup intelligence platform focused on explainable NBA and NFL forecasting. It combines real historical game data, engineered matchup signals, and a visual dashboard so users can see not only a win probability, but also what drove the edge.

## What it does
- Trains a matchup model on historical results
- Generates home/away win probabilities
- Surfaces top drivers behind each prediction
- Exposes a local web dashboard for exploring signals and recent matchups

## Why this project exists
Sports data is scattered across schedules, standings, injury notes, and box scores. GameSense turns that noise into a single decision layer that answers two questions clearly: who has the edge, and why?

## Why it stands out
- Uses real NBA historical data from BALLDONTLIE, not only toy examples
- Benchmarks against a naive home-team baseline
- Includes an end-to-end product surface, not just a notebook or script
- Keeps the forecasting logic explainable instead of hiding behind a black box

## Feature Inputs
- `elo_diff`
- `offensive_diff`
- `defensive_diff`
- `injury_diff`
- `rest_diff`
- `form_diff`
- `qb_status_diff` (NFL-centric)
- `travel_diff`
- `market_spread`
- `league_is_nba`

## Quick Start
```bash
cd /Users/tejashariharan/Downloads/gamesense
PYTHONPATH=src python3 -m gamesense.cli train
PYTHONPATH=src python3 -m gamesense.cli predict --league NBA
PYTHONPATH=src python3 -m gamesense.cli predict --league NFL --elo-diff 0.3 --market-spread 1.5
```

## Dashboard
```bash
cd /Users/tejashariharan/Downloads/gamesense
PYTHONPATH=src python3 -m gamesense.dashboard
```

Then open `http://127.0.0.1:8000`.

If `8000` is already in use:

```bash
PORT=8001 PYTHONPATH=src python3 -m gamesense.dashboard
```

## BALLDONTLIE Real Data
Create an API key at [BALLDONTLIE](https://www.balldontlie.io/) and export it:

```bash
export BALLDONTLIE_API_KEY="your_key_here"
```

Pull real NBA history and train on it:

```bash
cd /Users/tejashariharan/Downloads/gamesense
PYTHONPATH=src python3 -m gamesense.cli sync-nba --seasons 2022 2023 2024
PYTHONPATH=src python3 -m gamesense.cli train-real --seasons 2022 2023 2024
```

The free BALLDONTLIE tier includes `games` for NBA and NFL, which is enough for real historical-result modeling. NBA docs: [docs.balldontlie.io](https://docs.balldontlie.io/). NFL docs: [nfl.balldontlie.io](https://nfl.balldontlie.io/).

If you already synced the CSVs once, `train-real` will train from the local cached files. Use `--refresh` only when you explicitly want to refetch from the API.

## Repo Structure
- `src/gamesense/` core model, API client, CLI, and dashboard server
- `web/` local dashboard UI
- `tests/` pipeline smoke tests
- `data/` generated local datasets
- `models/` trained local artifacts

## Current Status
- Real NBA seasons synced locally: `2022`, `2023`, `2024`
- Real-data backtest accuracy observed locally: about `67-68%`
- Baseline accuracy observed locally: about `53-54%`
- Dashboard available locally at `http://127.0.0.1:8000`

## How To Demo It
1. Start the dashboard.
2. Pick an away team and home team from the matchup builder.
3. Click `Build Matchup`.
4. Use the hero forecast, matchup profile, comparison bars, and top drivers to explain the edge.

## Publish To GitHub
```bash
cd /Users/tejashariharan/Downloads/gamesense
git init
git add .
git commit -m "Build GameSense matchup intelligence platform"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```

The repo is set up to keep generated `data/` and `models/` artifacts out of version control through `.gitignore`.

## Next upgrades (real-data version)
1. Add NFL historical sync and features
2. Add richer NBA features such as injuries, standings, and season averages if the API tier supports them
3. Add calibration plots and per-team bias checks
4. Ship a hosted frontend/backend version after local iteration stabilizes
