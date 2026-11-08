import pandas as pd
import pytest
from backend.src.features import FEATURES, select_features
from backend.src.data_preprocessing_training import transform_data


def customers(count=20):
    return pd.DataFrame([{
        "CustomerID": i, "Age": 22, "Gender": "Female", "Tenure": 25,
        "Usage Frequency": 14, "Support Calls": 4, "Payment Delay": 27,
        "Subscription Type": "Basic", "Contract Length": "Monthly",
        "Total Spend": 598.5, "Last Interaction": 9, "Churn": i % 2,
    } for i in range(count)])


def test_features_exclude_label_and_identifier_and_restore_order():
    frame = customers().iloc[:, ::-1]
    assert select_features(frame).columns.tolist() == FEATURES


def test_training_split_contains_no_target_or_customer_id():
    frame = customers()
    train, test, y_train, y_test = transform_data(frame)
    assert train.columns.tolist() == test.columns.tolist() == FEATURES
    assert len(train) + len(test) == len(frame)
    assert set(train.index).isdisjoint(test.index)
    assert set(y_train.unique()) == set(y_test.unique()) == {0, 1}
    assert "Churn" in frame  # Caller input is preserved.


def test_aliases_are_accepted():
    frame = customers().rename(columns=lambda name: name.replace(" ", "_"))
    assert select_features(frame).columns.tolist() == FEATURES


def test_missing_features_are_rejected():
    with pytest.raises(ValueError, match="Missing columns"):
        select_features(customers().drop(columns="Age"))


def test_duplicate_aliases_are_rejected():
    frame = customers()
    frame["Support_Calls"] = 1
    with pytest.raises(ValueError, match="Duplicate"):
        select_features(frame)
