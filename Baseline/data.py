"""Shared data loading for EEE 405 Team 17. Everyone imports load_split(); don't rewrite it."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_split(path="creditcard.csv"):
    """Return X_train, X_test, y_train, y_test per the team spec.

    - Stratified 80/20 split, random_state=42
    - StandardScaler on Time and Amount, fit on TRAIN only
    - No resampling here: SMOTE / undersampling go on the train set in your own script
    - Outputs are pandas DataFrames/Series (V1..V28, Time, Amount; label = Class)
    """
    df = pd.read_csv(path)
    X = df.drop(columns="Class")
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    X_train, X_test = X_train.copy(), X_test.copy()
    scaler = StandardScaler().fit(X_train[["Time", "Amount"]])  # fit on train only
    X_train[["Time", "Amount"]] = scaler.transform(X_train[["Time", "Amount"]])
    X_test[["Time", "Amount"]] = scaler.transform(X_test[["Time", "Amount"]])

    return X_train, X_test, y_train, y_test
