"""Validate transaction features and score a locally trained pipeline."""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

FEATURES = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]
DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "models" / "fraud_pipeline.joblib"


def validate_features(frame):
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Provide at least one transaction as a pandas DataFrame.")
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate column names are not allowed.")
    missing = sorted(set(FEATURES) - set(frame.columns))
    extra = sorted(set(frame.columns) - set(FEATURES))
    if missing or extra:
        raise ValueError(f"Missing columns: {missing}; unexpected columns: {extra}")
    result = frame.loc[:, FEATURES].apply(pd.to_numeric, errors="raise")
    if np.isinf(result.to_numpy(dtype=float)).any():
        raise ValueError("Infinite feature values are not allowed.")
    if (result[["Time", "Amount"]] < 0).any().any():
        raise ValueError("Time and Amount must be non-negative.")
    # Missing feature values are handled by the training-fitted median imputer.
    return result


def load_model(path=DEFAULT_MODEL):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Model not found: {path}. Run the notebook first.")
    # Joblib can execute code: load only an artifact you trained or trust.
    return joblib.load(path)


def predict_transactions(frame, model=None):
    features = validate_features(frame)
    if model is None:
        model = load_model()
    probabilities = model.predict_proba(features)[:, 1]
    return pd.DataFrame({
        "fraud_score": probabilities,
        "prediction": (probabilities >= 0.5).astype(int),
    }, index=frame.index)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="CSV with exactly the 30 feature columns, without Class")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    print(predict_transactions(pd.read_csv(args.csv), load_model(args.model)).to_string(index=False))


if __name__ == "__main__":
    main()
