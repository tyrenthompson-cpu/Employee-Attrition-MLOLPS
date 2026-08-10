import abc

import yaml
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from src.preprocessing import build_preprocessing_pipeline
from src.utils import get_file_hash


def main():
    # Load config
    with open("configs/train_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Load data
    df = pd.read_csv(config["data"]["path"])
    target = config["data"]["target"]

    X = df.drop(columns=[target])
    y = df[target]

    # Identify feature types
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    # Build preprocessing pipeline
    preprocessor = build_preprocessing_pipeline(categorical_cols, numeric_cols)

    # Build model
    model = RandomForestClassifier(
        n_estimators=config["model"]["n_estimators"],
        max_depth=config["model"]["max_depth"],
        random_state=config["training"]["random_state"]
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["training"]["test_size"],
        random_state=config["training"]["random_state"]
    )

    # MLflow experiment
    mlflow.set_experiment("employee_attrition")

    with mlflow.start_run():
        # Train model
        pipeline.fit(X_train, y_train)

        # Predictions
        preds = pipeline.predict(X_test)
        probs = pipeline.predict_proba(X_test)[:, 1]

        # Metrics
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)

        # Log ALL hyperparameters
        mlflow.log_params(config["model"])
        mlflow.log_params(config["training"])
        mlflow.log_param("categorical_features", categorical_cols)
        mlflow.log_param("numeric_features", numeric_cols)
        mlflow.log_param("data_hash", get_file_hash(config["data"]["path"]))

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", auc)

        # Log model
        mlflow.sklearn.log_model(pipeline, "model")

        # Print metrics
        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC-AUC: {auc:.4f}")

        # Exit non-zero if below threshold
        if acc < config["training"]["min_accuracy"]:
            print("Model performance below threshold.")
            exit(1)


if __name__ == "__main__":
    main()
