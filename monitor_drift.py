import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset


def main():
    reference = pd.read_csv("data/employee_attrition.csv")
    production = reference.sample(frac=0.3).copy()

    # simulate drift
    production["Age"] = production["Age"] + 5

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=production)

    report.save_html("reports/drift_report.html")

    drift_share = report.as_dict()["metrics"][0]["result"]["dataset_drift"]["drift_share"]
    print("Drift share:", drift_share)

    if drift_share > 0.3:
        exit(1)

if __name__ == "__main__":
    main()
