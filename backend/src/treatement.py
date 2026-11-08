"""Train and log complete pipelines; processors travel with each model."""
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from xgboost import XGBClassifier
from .features import NUMERICAL, CATEGORICAL


def train_model(name, classifier, data_url, version, df, x_train, y_train, x_test, y_test):
    preprocessing = ColumnTransformer([
        ("numerical", StandardScaler(), NUMERICAL),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])
    pipeline = Pipeline([("preprocessing", preprocessing), ("classifier", classifier)])
    with mlflow.start_run(run_name=name):
        mlflow.log_params({"data_url": str(data_url), "data_version": version,
                           "input_rows": len(df), **classifier.get_params()})
        mlflow.set_tags({"model": name, "input_format": "raw-v1"})
        pipeline.fit(x_train, y_train)
        predicted = pipeline.predict(x_test)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, predicted, average="macro", zero_division=0)
        mlflow.log_metrics({"Precision_test": precision, "Recall_test": recall,
                            "F1_score_test": f1})
        mlflow.sklearn.log_model(
            pipeline, artifact_path="ML_models",
            signature=infer_signature(x_test, predicted),
            input_example=x_train.head(2), metadata={"input_format": "raw-v1"},
        )


def random_forest_model(*args):
    return train_model("Random Forest", RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42), *args)


def xgboost_model(*args):
    return train_model("XGBoost", XGBClassifier(
        n_estimators=100, max_depth=10, learning_rate=0.1, random_state=42), *args)
