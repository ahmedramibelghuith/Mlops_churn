"""One input contract shared by training and both prediction endpoints."""
FEATURES = [
    "Age", "Gender", "Tenure", "Usage Frequency", "Support Calls",
    "Payment Delay", "Subscription Type", "Contract Length", "Total Spend",
    "Last Interaction",
]
NUMERICAL = [c for c in FEATURES if c not in
             ("Gender", "Subscription Type", "Contract Length")]
CATEGORICAL = ["Gender", "Subscription Type", "Contract Length"]
ALIASES = {name.replace(" ", "_"): name for name in FEATURES}


def select_features(df):
    frame = df.rename(columns=ALIASES).copy()
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate feature columns are not allowed")
    missing = [name for name in FEATURES if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    frame = frame.loc[:, FEATURES]
    if frame.empty or frame.isna().any().any():
        raise ValueError("Provide at least one row with no missing feature values")
    return frame
