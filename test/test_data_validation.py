import pandas as pd
import numpy as np
import subprocess
from sklearn.metrics import roc_auc_score

###############################################
# DATA VALIDATION TESTS
###############################################

def test_columns_present():
    df = pd.read_csv("data/employee_attrition.csv")
    expected = {"Age", "Attrition", "JobRole"}
    assert expected.issubset(df.columns)

def test_column_types():
    df = pd.read_csv("data/employee_attrition.csv")
    assert df["Age"].dtype in ["int64", "float64"]
    assert df["Attrition"].dtype == "object"

def test_no_missing_values():
    df = pd.read_csv("data/employee_attrition.csv")
    assert df.isnull().sum().sum() == 0


###############################################
# MODEL VALIDATION TESTS
###############################################

def test_model_output_shape(model, sample_input):
    y_pred = model.predict(sample_input)
    assert y_pred.shape == (len(sample_input), 1)
    assert np.isfinite(y_pred).all()

def test_model_probability_range(model, sample_input):
    probs = model.predict_proba(sample_input)[:, 1]
    assert (probs >= 0).all() and (probs <= 1).all()


###############################################
# METRIC VALIDATION TESTS
###############################################

def test_roc_auc_uses_probabilities(model, X_test, y_test):
    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    assert auc > 0  # basic sanity check


###############################################
# MLFLOW VALIDATION TESTS
###############################################

def test_mlflow_logs_all_params(mlflow_client, run_id, config):
    params = mlflow_client.get_run(run_id).data.params
    for section, values in config.items():
        for key in values:
            assert f"{section}.{key}" in params


###############################################
# TRAINING EXIT VALIDATION TEST
###############################################

def test_training_exits_nonzero_on_failure():
    result = subprocess.run(["python", "train.py"], capture_output=True)
    assert result.returncode != 0

