import pandas as pd
import pytest
from src.preprocessing import build_preprocessing_pipeline

def test_missing_values_imputed():
    df = pd.DataFrame({"Age": [25, None, 40], "Department": ["Sales", "HR", None]})
    pipeline = build_preprocessing_pipeline(["Department"], ["Age"])
    transformed = pipeline.fit_transform(df)
    assert transformed.shape[0] == 3

def test_categorical_encoding():
    df = pd.DataFrame({"Age": [30, 40], "Department": ["Sales", "HR"]})
    pipeline = build_preprocessing_pipeline(["Department"], ["Age"])
    transformed = pipeline.fit_transform(df)
    assert transformed.shape[1] == 3

def test_no_dataframe_mutation():
    df = pd.DataFrame({"Age": [30, 40], "Department": ["Sales", "HR"]})
    df_copy = df.copy()
    pipeline = build_preprocessing_pipeline(["Department"], ["Age"])
    pipeline.fit_transform(df)
    assert df.equals(df_copy)

def test_invalid_input_raises_error():
    pipeline = build_preprocessing_pipeline([], [])
    with pytest.raises(Exception):
        pipeline.fit_transform("not a dataframe")

def test_numeric_imputation():
    df = pd.DataFrame({"Age": [30, None, 50], "Department": ["Sales", "HR", "Sales"]})
    pipeline = build_preprocessing_pipeline(["Department"], ["Age"])
    transformed = pipeline.fit_transform(df)
    assert transformed.shape[0] == 3

def test_output_shape_consistent():
    df = pd.DataFrame({"Age": [30, 40, 50], "Department": ["Sales", "HR", "Sales"]})
    pipeline = build_preprocessing_pipeline(["Department"], ["Age"])
    transformed = pipeline.fit_transform(df)
    assert transformed.shape[1] == 3
