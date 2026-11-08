"""Register an evaluated run and record its immutable serving URI."""
import argparse
import json
from pathlib import Path
import mlflow
from backend.src.tracking import best_run, configure_tracking


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-f1", type=float, required=True,
                        help="Minimum acceptable macro F1, between 0 and 1")
    parser.add_argument("--name", default="CustomerChurn")
    args = parser.parse_args()
    if not 0 <= args.min_f1 <= 1:
        parser.error("--min-f1 must be between 0 and 1")
    configure_tracking()
    run = best_run(pipeline_only=True)
    if float(run["metrics.F1_score_test"]) < args.min_f1:
        raise RuntimeError("Best candidate does not meet the F1 threshold")
    if run.get("tags.input_format") != "raw-v1":
        raise RuntimeError("Retrain using the current pipeline before registration")
    uri = f"runs:/{run['run_id']}/ML_models"
    version = mlflow.register_model(uri, args.name)
    path = Path("artifacts/production_model.json")
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({"model_uri": uri, "registered_name": args.name,
                               "version": str(version.version),
                               "macro_f1": float(run["metrics.F1_score_test"])}, indent=2) + "\n")
    print(f"Registered model. Set MODEL_URI={uri} when serving.")


if __name__ == "__main__":
    main()
