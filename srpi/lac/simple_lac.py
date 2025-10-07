import numpy as np

class SimpleLAC:
    """
    A tiny stub mapping (reflection text length + local features) to an advantage estimate.
    Replace with a real encoder for serious experiments.
    """
    def __init__(self, input_dim: int, hidden: int = 64, lr: float = 1e-3, sigma_max: float = 1.0):
        rng = np.random.default_rng(0)
        self.W1 = rng.normal(scale=0.1, size=(input_dim, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(scale=0.1, size=(hidden, 2))  # mean, logvar
        self.b2 = np.zeros(2)
        self.lr = lr
        self.sigma_max = sigma_max

    def forward(self, x):
        h = np.tanh(x @ self.W1 + self.b1)
        out = h @ self.W2 + self.b2
        mean = out[0]
        logvar = out[1]
        sigma2 = np.exp(logvar)
        return mean, sigma2, h

    def predict(self, x):
        m, s2, _ = self.forward(x)
        return m, s2

    def update(self, xs, targets):
        # simple heteroscedastic regression loss
        dW1 = np.zeros_like(self.W1)
        db1 = np.zeros_like(self.b1)
        dW2 = np.zeros_like(self.W2)
        db2 = np.zeros_like(self.b2)
        n = len(xs)
        for x, y in zip(xs, targets):
            m, s2, h = self.forward(x)
            # NLL of y ~ N(m, s2)
            # d/dm: (m - y)/s2
            dm = (m - y) / (s2 + 1e-6)
            # d/dlogvar: 0.5*( (m-y)^2 / s2 - 1 )
            dlogv = 0.5 * (((m - y)**2) / (s2 + 1e-6) - 1.0)

            dout = np.array([dm, dlogv])
            dW2 += np.outer(h, dout)
            db2 += dout
            dh = (1 - h**2) * (dout @ self.W2.T)
            dW1 += np.outer(x, dh)
            db1 += dh

        self.W1 -= self.lr * dW1 / n
        self.b1 -= self.lr * db1 / n
        self.W2 -= self.lr * dW2 / n
        self.b2 -= self.lr * db2 / n
