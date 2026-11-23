"""Person 1: logistic regression baseline, NO imbalance handling."""
from sklearn.linear_model import LogisticRegression

from data import load_split
from metrics import evaluate, save_results

X_train, X_test, y_train, y_test = load_split()

clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train, y_train)

y_prob = clf.predict_proba(X_test)[:, 1]
metrics = evaluate(y_test, y_prob)
save_results("baseline_logreg", metrics, y_prob=y_prob)

print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()})
