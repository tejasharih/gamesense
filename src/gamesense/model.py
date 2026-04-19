from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from gamesense.data import FEATURE_COLUMNS, generate_synthetic_dataset


@dataclass
class EvalMetrics:
    accuracy: float
    log_loss: float
    brier: float
    baseline_accuracy: float
    baseline_log_loss: float


def per_league_accuracy(
    pipe: Pipeline, test_df: pd.DataFrame, feature_cols: list
) -> Dict[str, float]:
    results: Dict[str, float] = {}
    for league in sorted(test_df["league"].unique()):
        df = test_df[test_df["league"] == league]
        if df.empty:
            continue
        probs = pipe.predict_proba(df[feature_cols])[:, 1]
        preds = (probs >= 0.5).astype(int)
        results[f"{league.lower()}_accuracy"] = float(accuracy_score(df["home_win"], preds))
    return results


def time_split(df: pd.DataFrame, train_frac: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(df) * train_frac)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, solver="lbfgs")),
        ]
    )


def evaluate_model(pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> EvalMetrics:
    probs = pipe.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    baseline_probs = np.full(shape=len(y_test), fill_value=0.57)  # home-team baseline prior
    baseline_preds = np.ones_like(y_test)

    return EvalMetrics(
        accuracy=float(accuracy_score(y_test, preds)),
        log_loss=float(log_loss(y_test, probs)),
        brier=float(brier_score_loss(y_test, probs)),
        baseline_accuracy=float(accuracy_score(y_test, baseline_preds)),
        baseline_log_loss=float(log_loss(y_test, baseline_probs)),
    )


def train_and_save(model_path: Path, data_path: Path, seed: int = 7) -> Dict[str, float]:
    df = generate_synthetic_dataset(seed=seed)
    return train_from_dataframe(df, model_path, data_path, metadata={"data_source": data_path.name})


def train_from_dataframe(
    df: pd.DataFrame,
    model_path: Path,
    data_path: Path,
    metadata: Dict[str, object] | None = None,
) -> Dict[str, float]:
    data_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_path, index=False)

    train_df, test_df = time_split(df)
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["home_win"]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["home_win"]

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    metrics = evaluate_model(pipe, X_test, y_test)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    league_breakdown = per_league_accuracy(pipe, test_df, FEATURE_COLUMNS)

    artifact = {
        "pipeline": pipe,
        "feature_columns": FEATURE_COLUMNS,
        "metrics": {**metrics.__dict__, **league_breakdown},
        "metadata": {
            "data_source": data_path.name,
            "row_count": int(len(df)),
            **(metadata or {}),
        },
    }
    with model_path.open("wb") as f:
        pickle.dump(artifact, f)

    return {**metrics.__dict__, **league_breakdown}


def load_model(model_path: Path) -> Dict[str, object]:
    with model_path.open("rb") as f:
        return pickle.load(f)
