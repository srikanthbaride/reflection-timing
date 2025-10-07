import numpy as np

class MLPPolicy:
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64, lr: float = 5e-3, entropy_coef: float = 0.01, kl_coef: float = 0.0):
        rng = np.random.default_rng(0)
        # simple 2-layer MLP
        self.W1 = rng.normal(scale=0.1, size=(obs_dim, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(scale=0.1, size=(hidden, act_dim))
        self.b2 = np.zeros(act_dim)
        self.lr = lr
        self.entropy_coef = entropy_coef
        self.kl_coef = kl_coef
        self.prev_logits = None  # for KL penalty

    def forward(self, obs):
        h = np.tanh(obs @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        return logits, h

    def policy(self, obs):
        logits, _ = self.forward(obs)
        # softmax
        z = logits - logits.max()
        probs = np.exp(z) / np.exp(z).sum()
        return probs, logits

    def sample(self, obs):
        probs, logits = self.policy(obs)
        a = np.random.choice(len(probs), p=probs)
        logp = np.log(probs[a] + 1e-9)
        return a, logp, probs, logits

    def update(self, batch_obs, batch_acts, batch_advs):
        # simple REINFORCE with baseline omitted for brevity; adds entropy bonus
        grads_W1 = np.zeros_like(self.W1)
        grads_b1 = np.zeros_like(self.b1)
        grads_W2 = np.zeros_like(self.W2)
        grads_b2 = np.zeros_like(self.b2)

        for obs, a, adv in zip(batch_obs, batch_acts, batch_advs):
            logits, h = self.forward(obs)
            z = logits - logits.max()
            probs = np.exp(z) / np.exp(z).sum()
            # grad log pi(a|s) for softmax linear head
            grad_logits = -probs
            grad_logits[a] += 1.0
            grad_logits *= adv

            grads_W2 += np.outer(h, grad_logits)
            grads_b2 += grad_logits

            dh = (1 - h**2) * (grad_logits @ self.W2.T)
            grads_W1 += np.outer(obs, dh)
            grads_b1 += dh

            # entropy bonus
            ent = -(probs * np.log(probs + 1e-9)).sum()
            ent_grad_logits = -np.log(probs + 1e-9) - 1.0
            grads_W2 += self.entropy_coef * np.outer(h, ent_grad_logits)
            grads_b2 += self.entropy_coef * ent_grad_logits

        n = len(batch_obs)
        self.W1 += self.lr * grads_W1 / n
        self.b1 += self.lr * grads_b1 / n
        self.W2 += self.lr * grads_W2 / n
        self.b2 += self.lr * grads_b2 / n
