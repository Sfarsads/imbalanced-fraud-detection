import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("creditcard.csv")
X = df.drop(columns="Class")
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
cols = ["Time", "Amount"]
X_train[cols] = scaler.fit_transform(X_train[cols])
X_test[cols] = scaler.transform(X_test[cols])

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
print(y_train.sum(), y_test.sum())

import torch

from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(42)

X_tr = torch.tensor(X_train.values, dtype=torch.float32)
y_tr = torch.tensor(y_train.values, dtype=torch.float32)
X_te = torch.tensor(X_test.values, dtype=torch.float32)
y_te = torch.tensor(y_test.values, dtype=torch.float32)

loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=1024, shuffle=True)

print(X_tr.shape, y_tr.dtype)
xb, yb = next(iter(loader))
print(xb.shape, yb.shape, yb.sum())

import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, d_in, hidden=64, p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)

model = MLP(X_tr.shape[1])
print(model)
out = model(xb)
print(out.shape)

pos_weight = (y_tr ==0).sum() / (y_tr == 1).sum()
print("pos_weight:", pos_weight.item())

loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(20):
    model.train()
    total = 0.0
    for xb, yb in loader:
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
        total += loss.item()
    print(epoch + 1, total / len(loader))

    