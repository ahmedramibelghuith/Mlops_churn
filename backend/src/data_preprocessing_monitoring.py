"""Prepare legacy monitoring inputs without leaking the target into features."""
if __package__:
    from .clean_data_csv import clean_data
else:
    from clean_data_csv import clean_data


def transform_data(df):
    frame = df.dropna().copy()
    return clean_data(frame), frame["Churn"].copy()
