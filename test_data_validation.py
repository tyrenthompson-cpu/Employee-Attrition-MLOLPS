import pandas as pd

def test_columns_present():
    df = pd.read_csv("data/employee_attrition.csv")
    expected = {"Age", "Attrition", "JobRole"}
    assert expected.issubset(df.columns)
