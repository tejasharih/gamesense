from pathlib import Path
import tempfile
import unittest

from gamesense.dashboard import bootstrap_payload
from gamesense.model import load_model, train_and_save
from gamesense.predict import predict_matchup, sample_input


class TestPipeline(unittest.TestCase):
    def test_train_and_predict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / "model.pkl"
            data_path = root / "games.csv"

            metrics = train_and_save(model_path, data_path, seed=11)
            self.assertGreater(metrics["accuracy"], 0.58)
            self.assertGreater(metrics["accuracy"], metrics["baseline_accuracy"])
            artifact = load_model(model_path)
            self.assertEqual(artifact["metadata"]["data_source"], "games.csv")

            result = predict_matchup(model_path, sample_input("NFL"))
            self.assertGreaterEqual(result["home_win_probability"], 0.0)
            self.assertLessEqual(result["home_win_probability"], 1.0)
            self.assertEqual(len(result["top_factors"]), 3)

    def test_bootstrap_payload_contains_builder_context(self) -> None:
        payload = bootstrap_payload()
        self.assertIn("teams", payload)
        self.assertIn("NBA", payload["teams"])
        self.assertGreater(len(payload["teams"]["NBA"]), 0)
        self.assertIn("team_labels", payload)
        self.assertIn("sample_input", payload)


if __name__ == "__main__":
    unittest.main()
