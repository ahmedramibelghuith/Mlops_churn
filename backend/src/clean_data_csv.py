"""Compatibility preprocessing for previously trained standalone classifiers."""
import pickle
from functools import lru_cache
from pathlib import Path
if __package__:
    from .features import select_features, NUMERICAL, CATEGORICAL
else:
    from features import select_features, NUMERICAL, CATEGORICAL


@lru_cache(maxsize=1)
def load_processors():
    directory = Path(__file__).resolve().parents[1]
    with (directory / "label_encodersFinal.pkl").open("rb") as stream:
        encoders = pickle.load(stream)
    with (directory / "scalerFinal.pkl").open("rb") as stream:
        scaler = pickle.load(stream)
    return encoders, scaler


def clean_data(df):
    frame = select_features(df)
    encoders, scaler = load_processors()
    frame[NUMERICAL] = scaler.transform(frame[NUMERICAL])
    for column in CATEGORICAL:
        frame[column] = encoders[column].transform(frame[column])
    return frame
