import pandas as pd
from sklearn.model_selection import train_test_split
from src.train import build_pipeline
from src.utils import split_features_target

def test_model_meets_min_accuracy():
    df = pd.read_csv("data/employee_attrition.csv")
    X, y = split_features_target(df, "Attrition")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_pipeline(X)
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    acc = (preds == y_test).mean()

    assert acc > 0.60
