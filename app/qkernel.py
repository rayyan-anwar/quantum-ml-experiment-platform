"""Quantum kernel: fidelity between states prepared by a ZZ-style feature map, used with a classical SVM.

K(x, x') = |<psi(x)|psi(x')>|^2. Computed exactly on the simulator (no shot noise, no hardware noise).
Author: Rayyan Anwar
"""
from __future__ import annotations

import numpy as np

from . import quantum as q


def feature_map_states(X: np.ndarray, n_qubits: int | None = None, reps: int = 2) -> np.ndarray:
    X = np.atleast_2d(np.asarray(X, float))
    B, d = X.shape
    n = n_qubits or d
    s = q.zero_state(n, B)
    for _ in range(reps):
        for k in range(n):
            s = q.apply_1q(s, q.H, k)
        for k in range(n):
            s = q.apply_1q(s, q.rz(2 * X[:, k % d]), k)
        for k in range(n - 1):
            s = q.apply_cnot(s, k, k + 1)
            s = q.apply_1q(s, q.rz(2 * (np.pi - X[:, k % d]) * (np.pi - X[:, (k + 1) % d])), k + 1)
            s = q.apply_cnot(s, k, k + 1)
    return s.reshape(B, -1)


def kernel_matrix(A: np.ndarray, B: np.ndarray | None = None, n_qubits: int | None = None, reps: int = 2) -> np.ndarray:
    SA = feature_map_states(A, n_qubits, reps)
    SB = SA if B is None else feature_map_states(B, n_qubits, reps)
    return np.abs(SA.conj() @ SB.T) ** 2
