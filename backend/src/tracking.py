"""Tracking configuration and deterministic best-run selection."""
import os
import mlflow


def configure_tracking():
    mlflow.set_tracking_uri(os.getenv(
        "MLFLOW_TRACKING_URI",
        "https://dagshub.com/rami4real/mymlproject.mlflow",
    ))


def best_run(pipeline_only=False):
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "churn-experiment")
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment not found: {experiment_name}")
    filters = "attributes.status = 'FINISHED' AND metrics.F1_score_test >= 0"
    if pipeline_only:
        filters += " AND tags.input_format = 'raw-v1'"
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=filters,
        order_by=["metrics.F1_score_test DESC", "start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        raise RuntimeError("No completed runs with an F1 score are available")
    return runs.iloc[0]
