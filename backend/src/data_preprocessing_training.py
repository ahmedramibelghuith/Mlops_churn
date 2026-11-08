"""Split raw features before fitting any preprocessing."""
from sklearn.model_selection import train_test_split
from .features import select_features


def transform_data(df):
    frame = df.dropna().copy()
    X = select_features(frame)
    y = frame["Churn"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
