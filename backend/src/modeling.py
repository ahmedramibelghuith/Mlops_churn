"""Run from the repository root: python -m backend.src.modeling."""
import argparse
from pathlib import Path
import pandas as pd
import mlflow
from .data_preprocessing_training import transform_data
from .tracking import configure_tracking
from .treatement import xgboost_model, random_forest_model
import os


def main():
    parser = argparse.ArgumentParser(description="Train churn pipelines and log to MLflow")
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[2] / "data/customer_churn_dataset.csv")
    parser.add_argument("--version", default="v2.0")
    args = parser.parse_args()
    configure_tracking()
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "churn-experiment"))
    df = pd.read_csv(args.data)
    X_train, X_test, y_train, y_test = transform_data(df)
    for train in (xgboost_model, random_forest_model):
        train(args.data, args.version, df, X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
