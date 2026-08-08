import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def build_preprocessing_pipeline(categorical_cols, numeric_cols):
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    preprocessor = ColumnTransformer([
        ("categorical", cat_pipeline, categorical_cols),
        ("numerical", num_pipeline, numeric_cols)
    ])

    return preprocessor

def load_data(path):
    df = pd.read_csv(path)
    return df
