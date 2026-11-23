"""Short EDA: class counts + class-balance plot."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data import load_split

os.makedirs("results", exist_ok=True)

df = pd.read_csv("creditcard.csv")
counts = df["Class"].value_counts().sort_index()
n, n_fraud = len(df), int(counts[1])
print(f"Rows: {n:,}  |  Features: {df.shape[1] - 1}  |  Missing values: {int(df.isna().sum().sum())}")
print(f"Legit (0): {int(counts[0]):,}  |  Fraud (1): {n_fraud:,}  |  Fraud rate: {n_fraud / n:.4%}")
print(f"Imbalance ratio: ~{int(counts[0]) // n_fraud}:1")

_, _, y_train, y_test = load_split()
print(f"Train: {len(y_train):,} rows, {int(y_train.sum())} frauds ({y_train.mean():.4%})")
print(f"Test:  {len(y_test):,} rows, {int(y_test.sum())} frauds ({y_test.mean():.4%})")

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, log in zip(axes, [False, True]):
    bars = ax.bar(["Legit (0)", "Fraud (1)"], counts.values, color=["#4C72B0", "#C44E52"])
    ax.bar_label(bars, labels=[f"{v:,}" for v in counts.values])
    ax.set_ylabel("Transactions")
    ax.set_title("Class balance" + (" (log scale)" if log else ""))
    if log:
        ax.set_yscale("log")
fig.suptitle(f"Credit Card Fraud: {n_fraud / n:.3%} positive")
fig.tight_layout()
fig.savefig("results/class_balance.png", dpi=150)
print("Saved results/class_balance.png")
