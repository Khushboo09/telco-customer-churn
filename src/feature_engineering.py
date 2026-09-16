
import pandas as pd

service_cols = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]

tenure_bins = [-1, 12, 24, 36, 48, 60, 72]
tenure_labels = ["0-12", "13-24", "25-36", "37-48", "49-60", "61-72"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TotalServices"] = (df[service_cols] == "Yes").sum(axis=1) + (df["InternetService"] != "No").astype(int)
    df["tenure_group"] = pd.cut(df["tenure"], bins=tenure_bins, labels=tenure_labels)
    return df
