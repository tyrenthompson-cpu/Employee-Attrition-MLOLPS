import pandas as pd
from src.preprocessing import build_preprocessing_pipeline

def test_missing_values():
    df = pd.DataFrame({"A": [1, None, 3], "B": ["x", "y", None]})
    pipeline = build_preprocessing_pipeline(["B"], ["A"])
    transformed = pipeline.fit_transform(df)
    assert transformed.shape[0] == 3

def test_no_mutation():
    df = pd.DataFrame({"A": [1, 2], "B": ["x", "y"]})
    df_copy = df.copy()
    pipeline = build_preprocessing_pipeline(["B"], ["A"])
    pipeline.fit_transform(df)
    assert df.equals(df_copy)
