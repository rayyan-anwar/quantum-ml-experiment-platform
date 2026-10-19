"""Independent dense-matrix reference used ONLY to validate the fast simulator (tests and evaluation).

Builds full 2^n x 2^n operators with Kronecker products / explicit basis permutations, sharing no code with quantum.py.
Author: Rayyan Anwar
"""
from __future__ import annotations

import numpy as np


def op_1q(n: int, U: np.ndarray, k: int) -> np.ndarray:
    m = np.array([[1.0 + 0j]])
    for q in range(n):
        m = np.kron(m, U if q == k else np.eye(2))
    return m


def _perm(n: int, fn) -> np.ndarray:
    d = 2 ** n
    m = np.zeros((d, d), dtype=complex)
    for b in range(d):
        bits = [(b >> (n - 1 - q)) & 1 for q in range(n)]
        out, phase = fn(bits)
        j = sum(bit << (n - 1 - q) for q, bit in enumerate(out))
        m[j, b] = phase
    return m


def op_cnot(n: int, c: int, t: int) -> np.ndarray:
    def fn(bits):
        out = list(bits)
        if bits[c]:
            out[t] ^= 1
        return out, 1.0
    return _perm(n, fn)


def op_cz(n: int, a: int, b: int) -> np.ndarray:
    return _perm(n, lambda bits: (bits, -1.0 if bits[a] and bits[b] else 1.0))


def op_swap(n: int, a: int, b: int) -> np.ndarray:
    def fn(bits):
        out = list(bits)
        out[a], out[b] = bits[b], bits[a]
        return out, 1.0
    return _perm(n, fn)


def run(n: int, ops: list[np.ndarray]) -> np.ndarray:
    v = np.zeros(2 ** n, dtype=complex)
    v[0] = 1
    for m in ops:
        v = m @ v
    return v
