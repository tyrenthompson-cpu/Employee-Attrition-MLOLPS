import sys
import mlflow
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import yaml

def load_config(path="config/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess(df):
    df = df.copy()

    # Encode target
    le = LabelEncoder()
    df["Attrition"] = le.fit_transform(df["Attrition"])

    # Simple numeric-only model for demonstration
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    X = df[numeric_cols].drop("Attrition", axis=1)
    y = df["Attrition"]

    return X, y

def evaluate_model(model, X_test, y_test):
    # Probabilities required for ROC-AUC
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)

    return acc, auc, precision, recall

def main():
    config = load_config()

    df = load_data(config["data"]["path"])
    X, y = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"]
    )

    model = LogisticRegression(max_iter=config["model"]["max_iter"])
    model.fit(X_train, y_train)

    acc, auc, precision, recall = evaluate_model(model, X_test, y_test)

    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC: {auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    with mlflow.start_run():

        # Log ALL hyperparameters from ALL config sections
        mlflow.log_params({
            f"{section}.{key}": value
            for section, params in config.items()
            for key, value in params.items()
        })

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        # Log model
        mlflow.sklearn.log_model(model, "model")

    # Non-zero exit if threshold not met
    threshold = config["evaluation"]["min_accuracy"]
    if acc < threshold:
        print(f"Evaluation failed: accuracy {acc:.4f} < required {threshold}")
        sys.exit(1)

    print("Evaluation passed.")
    sys.exit(0)

if __name__ == "__main__":
    main()

