import torch
import numpy as np

# ---- Data ----
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
m = len(X)
X_biased = np.c_[np.ones((m, 1)), X]

X_biased = torch.from_numpy(X_biased).float()
y = torch.from_numpy(y).float()

W = torch.randn(2, 1, requires_grad=True)

epochs = 1000
lr = 1e-2
epsilon = 1e-6
old_loss = 0

optimizer = torch.optim.SGD([W], lr)

for _ in range(epochs):
    indices = torch.randperm(m)
    X_shuffled = X_biased[indices]
    y_shuffled = y[indices]

    batch_size = 32
    for i in range(0, m, batch_size):
        x_batch = X_shuffled[i: i + batch_size]
        y_batch = y_shuffled[i: i + batch_size]
        pred = x_batch @ W

        loss = ((pred - y_batch) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        full_batch_pred = X_shuffled @ W
        full_batch_loss = ((full_batch_pred - y_shuffled) ** 2).mean()
    if torch.abs(old_loss - full_batch_loss) < epsilon:
        print(_)
        break
    old_loss = full_batch_loss


print(W)