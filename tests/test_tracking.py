from types import SimpleNamespace
import pandas as pd
import pytest
from backend.src import tracking


def test_best_run_searches_named_experiment_in_descending_f1_order(monkeypatch):
    monkeypatch.setattr(tracking.mlflow, "get_experiment_by_name",
                        lambda name: SimpleNamespace(experiment_id="123"))
    def search(**kwargs):
        assert kwargs["experiment_ids"] == ["123"]
        assert kwargs["order_by"][0] == "metrics.F1_score_test DESC"
        assert "FINISHED" in kwargs["filter_string"]
        return pd.DataFrame([{"run_id": "best", "metrics.F1_score_test": 1.0}])
    monkeypatch.setattr(tracking.mlflow, "search_runs", search)
    assert tracking.best_run()["run_id"] == "best"


def test_missing_experiment_fails_explicitly(monkeypatch):
    monkeypatch.setattr(tracking.mlflow, "get_experiment_by_name", lambda name: None)
    with pytest.raises(RuntimeError, match="experiment not found"):
        tracking.best_run()
