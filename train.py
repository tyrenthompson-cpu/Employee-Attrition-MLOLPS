import mlflow
import mlflow.sklearn
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from preprocessing import build_preprocessing_pipeline

def main():
    with open("configs/train_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    df = pd.read_csv(config["data"]["path"])
    target = config["data"]["target"]

    X = df.drop(columns=[target])
    y = df[target]

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    preprocessor = build_preprocessing_pipeline(categorical_cols, numeric_cols)

    model = RandomForestClassifier(
        n_estimators=config["model"]["n_estimators"],
        max_depth=config["model"]["max_depth"],
        random_state=42
    )

    from sklearn.pipeline import Pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["training"]["test_size"], random_state=42
    )

    mlflow.set_experiment("employee_attrition")

    with mlflow.start_run():
        pipeline.fit(X_train, y_train)

        preds = pipeline.predict(X_test)

        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, preds)

        mlflow.log_params(config["model"])
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("auc", auc)

        mlflow.sklearn.log_model(pipeline, "model")

        print(f"Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
