"""Shared feature engineering for the Telco Customer Churn project.

This module is the single source of truth for the two engineered features used
by the Decision Tree model:

- ``TotalServices``: total number of services subscribed (0-9)
- ``tenure_group``: tenure (months) bucketed into 6 ranges

It is imported by both:
- ``notebook/churn_analysis.ipynb`` (Day 1) — applied to the training/test data
- ``app.py`` (Day 2) — applied to a single incoming customer record

so that the exact same logic produces these features whether it's run on the
training data or on a brand-new customer record at prediction time.
"""

import pandas as pd

# Columns used to compute TotalServices: Yes/No service flags.
# (InternetService is handled separately below since it has 3 categories, not Yes/No.)
service_cols = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]

# tenure (months) bucket edges/labels used for tenure_group
tenure_bins = [-1, 12, 24, 36, 48, 60, 72]
tenure_labels = ["0-12", "13-24", "25-36", "37-48", "49-60", "61-72"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with TotalServices and tenure_group added.

    Works on:
    - the full training/test dataset (many rows), and
    - a single new customer record (a 1-row DataFrame), e.g. built from the
      raw JSON body of the future FastAPI /predict endpoint.

    Required raw columns: everything listed in `service_cols`, plus
    "InternetService" and "tenure".
    """
    df = df.copy()
    df["TotalServices"] = (df[service_cols] == "Yes").sum(axis=1) + (df["InternetService"] != "No").astype(int)
    df["tenure_group"] = pd.cut(df["tenure"], bins=tenure_bins, labels=tenure_labels)
    return df
