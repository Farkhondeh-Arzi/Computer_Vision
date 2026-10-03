import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def loss_function(p, y):
    return np.mean((p - y) ** 2)


class Optimizer:
    def __init__(self, learning_rate, state=None):
        self.W = np.random.randn(2, 1)
        self.learning_rate = learning_rate
        self.state = state

    def step(self, x_batch, y_batch):
        pass


class SGDOptimizer(Optimizer):
    def step(self, x_batch, y_batch):
        batch_size = len(y_batch)
        grad = (2 / batch_size) * x_batch.T.dot(x_batch.dot(self.W) - y_batch)
        self.W -= self.learning_rate * grad
        return self.W


class SGDMomentumOptimizer(Optimizer):
    def __init__(self, learning_rate):
        self.state = {'v_t': 0, 'beta': 0.5}
        super().__init__(learning_rate, self.state)

    def step(self, x_batch, y_batch):
        batch_size = len(y_batch)
        grad = (2 / batch_size) * x_batch.T.dot(x_batch.dot(self.W) - y_batch)

        beta = self.state.get('beta')
        velocity = beta * self.state.get('v_t') + (1 - beta) * grad
        self.state['v_t'] = velocity
        self.W -= self.learning_rate * velocity
        return self.W


class RMSPropOptimizer(Optimizer):
    def __init__(self, learning_rate):
        self.state = {'s_t': 0, 'rho': 0.9}
        super().__init__(learning_rate, self.state)

    def step(self, x_batch, y_batch):
        batch_size = len(y_batch)
        grad = (2 / batch_size) * x_batch.T.dot(x_batch.dot(self.W) - y_batch)

        rho = self.state.get('rho')
        cache = rho * self.state.get('s_t') + (1 - rho) * grad ** 2
        self.state['s_t'] = cache
        self.W -= self.learning_rate * (1 / (np.sqrt(cache) + 1e-9)) * grad
        return self.W


class ADAMOptimizer(Optimizer):
    def __init__(self, learning_rate):
        self.state = {'v_t': 0, 's_t': 0, 'beta1': 0.9, 'beta2': 0.999, 't': 0}
        super().__init__(learning_rate, self.state)

    def step(self, x_batch, y_batch):
        self.state['t'] += 1
        t = self.state['t']

        batch_size = len(y_batch)
        grad = (2 / batch_size) * x_batch.T.dot(x_batch.dot(self.W) - y_batch)

        beta1 = self.state.get('beta1')
        beta2 = self.state.get('beta2')

        velocity = beta1 * self.state.get('v_t') + (1 - beta1) * grad
        velocity_corrected = velocity / (1 - beta1 ** t)

        cache = beta2 * self.state.get('s_t') + (1 - beta2) * grad ** 2
        cache_corrected = cache / (1 - beta2 ** t)

        self.state['v_t'] = velocity
        self.state['s_t'] = cache

        self.W -= self.learning_rate * (1 / (np.sqrt(cache_corrected) + 1e-9)) * velocity_corrected
        return self.W


def training_loop_with_path(X, y, optimizer, epochs=100, batch_size=1):
    m = len(X)
    X_biased = np.c_[np.ones((m, 1)), X]

    loss_history = []
    W_history = [optimizer.W.copy().ravel()]

    for epoch in range(epochs):
        indices = np.random.permutation(m)
        X_shuffled = X_biased[indices]
        y_shuffled = y[indices]

        for i in range(0, m, batch_size):
            x_batch = X_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i + batch_size]
            optimizer.step(x_batch, y_batch)

        predictions = X_biased.dot(optimizer.W)
        loss = loss_function(predictions, y)
        loss_history.append(loss)

        W_history.append(optimizer.W.copy().ravel())

    return optimizer.W, loss_history, np.array(W_history)


def compute_loss_grid(X_biased, y, w0_range, w1_range):
    W0, W1 = np.meshgrid(w0_range, w1_range)
    Z = np.zeros_like(W0)

    for i in range(W0.shape[0]):
        for j in range(W0.shape[1]):
            W_candidate = np.array([[W0[i, j]], [W1[i, j]]])
            predictions = X_biased.dot(W_candidate)
            Z[i, j] = loss_function(predictions, y)

    return W0, W1, Z


# ---- Data ----
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
m = len(X)
X_biased = np.c_[np.ones((m, 1)), X]

optimizers_to_compare = {
    'SGD': (SGDOptimizer, dict(learning_rate=0.1), 'tab:blue'),
    'SGD + Momentum': (SGDMomentumOptimizer, dict(learning_rate=0.1), 'tab:orange'),
    'RMSProp': (RMSPropOptimizer, dict(learning_rate=0.1), 'tab:green'),
    'Adam': (ADAMOptimizer, dict(learning_rate=0.1), 'tab:red'),
}

paths = {}

for name, (OptimizerClass, kwargs, color) in optimizers_to_compare.items():
    np.random.seed(0)  # نقطه‌ی شروع یکسان برای مقایسه‌ی منصفانه
    optimizer = OptimizerClass(**kwargs)
    W_final, loss_history, W_history = training_loop_with_path(
        X, y, optimizer, epochs=100, batch_size=1
    )
    paths[name] = (W_history, color)
    print(f"{name}: final W = {W_final.ravel()}, final loss = {loss_history[-1]:.4f}")

w0_range = np.linspace(0, 6, 100)
w1_range = np.linspace(0, 6, 100)
W0_grid, W1_grid, Z = compute_loss_grid(X_biased, y, w0_range, w1_range)

# ---- Plot ----
plt.figure(figsize=(8, 7))
contour_levels = np.logspace(0, 3, 20)
plt.contour(W0_grid, W1_grid, Z, levels=contour_levels, cmap='Blues', alpha=0.6)

for name, (W_history, color) in paths.items():
    plt.plot(W_history[:, 0], W_history[:, 1], marker='o', markersize=2,
              linewidth=1.2, color=color, label=name)
    plt.plot(W_history[0, 0], W_history[0, 1], marker='s', color=color, markersize=8)
    plt.plot(W_history[-1, 0], W_history[-1, 1], marker='*', color=color, markersize=14)

plt.xlabel('W0 (intercept)')
plt.ylabel('W1 (slope)')
plt.title('Convergence Path of Optimizers on Loss Surface')
plt.legend()
plt.tight_layout()
plt.show()