"""Variational quantum classifier trained with exact parameter-shift gradients on the NumPy simulator.

Circuit: per layer [optional data encoding RY(x) on each qubit] + trainable RY, RZ on each qubit + CNOT ring.
Output f = <Z_0> + b; class 1 when f > 0; loss = mean squared error to targets +-1.
Every trainable angle feeds exactly one gate, so the two-term parameter-shift rule gives exact gradients.

Author: Rayyan Anwar
"""
from __future__ import annotations

import numpy as np

from . import quantum as q


class VQC:
    def __init__(self, n_qubits: int = 2, layers: int = 3, reupload: bool = True, seed: int = 0):
        if not 1 <= n_qubits <= 6 or not 1 <= layers <= 8:
            raise ValueError("n_qubits must be in [1, 6] and layers in [1, 8]")
        self.n, self.layers, self.reupload = n_qubits, layers, reupload
        rng = np.random.default_rng(seed)
        self.theta = rng.normal(0, 1.0, size=(layers, n_qubits, 2))
        self.bias = 0.0
        self.rng = rng

    @property
    def n_params(self) -> int:
        return self.theta.size + 1

    def forward_many(self, X: np.ndarray, thetas: np.ndarray) -> np.ndarray:
        """f for R parameter sets on B samples at once: thetas (R, layers, n, 2) -> (R, B)."""
        X = np.atleast_2d(np.asarray(X, float))
        R, B, d = thetas.shape[0], X.shape[0], X.shape[1]
        state = q.zero_state(self.n, R * B)
        Xt = np.tile(X, (R, 1))
        for l in range(self.layers):
            if self.reupload or l == 0:
                for k in range(self.n):
                    state = q.apply_1q(state, q.ry(Xt[:, k % d]), k)
            for k in range(self.n):
                state = q.apply_1q(state, q.ry(np.repeat(thetas[:, l, k, 0], B)), k)
                state = q.apply_1q(state, q.rz(np.repeat(thetas[:, l, k, 1], B)), k)
            if self.n > 1:
                for k in range(self.n):
                    state = q.apply_cnot(state, k, (k + 1) % self.n) if self.n > 2 or k == 0 else state
        return q.expect_z(state, 0).reshape(R, B)

    def decision(self, X: np.ndarray) -> np.ndarray:
        return self.forward_many(X, self.theta[None])[0] + self.bias

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.decision(X) > 0).astype(int)

    def loss_and_grad(self, X: np.ndarray, y: np.ndarray):
        t = 2.0 * np.asarray(y) - 1.0
        P = self.theta.size
        shifts = np.zeros((2 * P,) + self.theta.shape)
        flat = shifts.reshape(2 * P, P)
        for j in range(P):
            flat[2 * j, j], flat[2 * j + 1, j] = np.pi / 2, -np.pi / 2
        f_all = self.forward_many(X, np.concatenate([self.theta[None], self.theta[None] + shifts]))
        f0, fs = f_all[0], f_all[1:].reshape(P, 2, -1)
        resid = f0 + self.bias - t
        df = (fs[:, 0] - fs[:, 1]) / 2.0  # (P, B)
        g_theta = (2.0 * resid[None, :] * df).mean(axis=1).reshape(self.theta.shape)
        return float((resid ** 2).mean()), g_theta, float((2.0 * resid).mean())

    def fit(self, X, y, epochs: int = 60, lr: float = 0.1, batch_size: int | None = None, seed: int = 0) -> dict:
        X, y = np.asarray(X, float), np.asarray(y)
        if epochs < 1 or lr <= 0:
            raise ValueError("epochs must be >= 1 and lr > 0")
        rng = np.random.default_rng(seed)
        params = np.concatenate([self.theta.ravel(), [self.bias]])
        m, v = np.zeros_like(params), np.zeros_like(params)
        hist = {"loss": [], "train_acc": []}
        step = 0
        for _ in range(epochs):
            idx = rng.permutation(len(X))
            for chunk in np.array_split(idx, max(1, len(X) // (batch_size or len(X)))):
                self.theta, self.bias = params[:-1].reshape(self.theta.shape), float(params[-1])
                loss, gt, gb = self.loss_and_grad(X[chunk], y[chunk])
                g = np.concatenate([gt.ravel(), [gb]])
                step += 1
                m = 0.9 * m + 0.1 * g
                v = 0.999 * v + 0.001 * g * g
                params = params - lr * (m / (1 - 0.9 ** step)) / (np.sqrt(v / (1 - 0.999 ** step)) + 1e-8)
            self.theta, self.bias = params[:-1].reshape(self.theta.shape), float(params[-1])
            hist["loss"].append(float(((self.decision(X) - (2.0 * y - 1.0)) ** 2).mean()))
            hist["train_acc"].append(float((self.predict(X) == y).mean()))
        return hist
