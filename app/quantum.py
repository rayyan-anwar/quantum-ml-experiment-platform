"""Batched state-vector quantum simulator in NumPy.

Convention: qubit 0 is the LEFTMOST bit of a basis label (most significant), so |q0 q1 ... q(n-1)>. State arrays have
shape (batch, 2, ..., 2) with axis k+1 belonging to qubit k. Exact (noise-free) simulation of ideal qubits; practical
up to roughly 20 qubits in memory, capped much lower by the applications.

Author: Rayyan Anwar
"""
from __future__ import annotations

import numpy as np

SQ2 = 1 / np.sqrt(2)
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) * SQ2
S = np.array([[1, 0], [0, 1j]], dtype=complex)
SDG = S.conj().T
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
TDG = T.conj().T
FIXED = {"i": I2, "x": X, "y": Y, "z": Z, "h": H, "s": S, "sdg": SDG, "t": T, "tdg": TDG}


def rx(theta):
    t = np.asarray(theta, dtype=float)
    c, s = np.cos(t / 2), np.sin(t / 2)
    return _stack(c, -1j * s, -1j * s, c)


def ry(theta):
    t = np.asarray(theta, dtype=float)
    c, s = np.cos(t / 2), np.sin(t / 2)
    return _stack(c, -s, s, c)


def rz(theta):
    t = np.asarray(theta, dtype=float)
    return _stack(np.exp(-1j * t / 2), 0 * t, 0 * t, np.exp(1j * t / 2))


def _stack(a, b, c, d):
    m = np.stack([a, b, c, d], axis=-1).astype(complex)
    return m.reshape(m.shape[:-1] + (2, 2))


def zero_state(n: int, batch: int = 1) -> np.ndarray:
    if n < 1:
        raise ValueError("need at least one qubit")
    s = np.zeros((batch,) + (2,) * n, dtype=complex)
    s[(slice(None),) + (0,) * n] = 1.0
    return s


def _check(state, *qubits):
    n = state.ndim - 1
    for q in qubits:
        if not 0 <= q < n:
            raise ValueError(f"qubit {q} out of range for {n} qubits")
    if len(set(qubits)) != len(qubits):
        raise ValueError("qubits must be distinct")


def apply_1q(state: np.ndarray, U: np.ndarray, k: int) -> np.ndarray:
    """U is (2,2) shared across the batch or (batch,2,2) per-sample."""
    _check(state, k)
    s = np.moveaxis(state, k + 1, -1)
    s = s @ U.T if U.ndim == 2 else np.einsum("b...j,bij->b...i", s, U)
    return np.moveaxis(s, -1, k + 1)


def apply_cnot(state: np.ndarray, c: int, t: int) -> np.ndarray:
    _check(state, c, t)
    out = state.copy()
    sl = [slice(None)] * state.ndim
    sl[c + 1] = 1
    ax = t + 1 if t < c else t  # target axis inside the sliced (control=1) view
    out[tuple(sl)] = np.flip(state[tuple(sl)], axis=ax)
    return out


def apply_cz(state: np.ndarray, a: int, b: int) -> np.ndarray:
    _check(state, a, b)
    out = state.copy()
    sl = [slice(None)] * state.ndim
    sl[a + 1], sl[b + 1] = 1, 1
    out[tuple(sl)] *= -1
    return out


def apply_swap(state: np.ndarray, a: int, b: int) -> np.ndarray:
    _check(state, a, b)
    return np.swapaxes(state, a + 1, b + 1).copy()


def probabilities(state: np.ndarray) -> np.ndarray:
    """(batch, 2**n) probabilities in big-endian basis order."""
    p = (state.real ** 2 + state.imag ** 2).reshape(state.shape[0], -1)
    return p


def expect_z(state: np.ndarray, k: int) -> np.ndarray:
    _check(state, k)
    p = (state.real ** 2 + state.imag ** 2)
    p = np.moveaxis(p, k + 1, 1).reshape(state.shape[0], 2, -1).sum(-1)
    return p[:, 0] - p[:, 1]


_BASIS_CHANGE = {"X": H, "Y": H @ SDG}


def expect_pauli(state: np.ndarray, pauli: str) -> np.ndarray:
    """Expectation of a Pauli string such as 'XZI' (one letter per qubit, I/X/Y/Z)."""
    n = state.ndim - 1
    if len(pauli) != n or set(pauli) - set("IXYZ"):
        raise ValueError(f"pauli string must have {n} letters from I, X, Y, Z")
    s = state
    for k, ch in enumerate(pauli):
        if ch in _BASIS_CHANGE:
            s = apply_1q(s, _BASIS_CHANGE[ch], k)
    p = (s.real ** 2 + s.imag ** 2)
    sign = np.ones(p.shape[1:])
    for k, ch in enumerate(pauli):
        if ch != "I":
            shape = [1] * n
            shape[k] = 2
            sign = sign * np.array([1.0, -1.0]).reshape(shape)
    return (p * sign).reshape(p.shape[0], -1).sum(-1)


def reduced_density(vec: np.ndarray, k: int) -> np.ndarray:
    """2x2 reduced density matrix of qubit k from a single state (shape (2,)*n)."""
    n = vec.ndim
    m = np.moveaxis(vec, k, 0).reshape(2, -1)
    return m @ m.conj().T


def bloch_vector(vec: np.ndarray, k: int) -> tuple[float, float, float]:
    r = reduced_density(vec, k)
    return float(2 * r[0, 1].real), float(-2 * r[0, 1].imag), float(r[0, 0].real - r[1, 1].real)


def qubit_entropy(vec: np.ndarray, k: int) -> float:
    """Von Neumann entropy (bits) of qubit k: 0 = unentangled with the rest, 1 = maximally entangled."""
    ev = np.clip(np.linalg.eigvalsh(reduced_density(vec, k)), 0, 1)
    ev = ev[ev > 1e-12]
    return float(-(ev * np.log2(ev)).sum()) if len(ev) else 0.0


def basis_labels(n: int) -> list[str]:
    return [format(i, f"0{n}b") for i in range(2 ** n)]


def sample_counts(probs: np.ndarray, shots: int, seed: int | None = None) -> dict[str, int]:
    p = np.asarray(probs, dtype=float)
    p = p / p.sum()
    n = int(np.log2(len(p)))
    draw = np.random.default_rng(seed).multinomial(shots, p)
    return {format(i, f"0{n}b"): int(c) for i, c in enumerate(draw) if c}
