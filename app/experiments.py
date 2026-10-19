"""Run quantum and classical models on identical splits; aggregate over seeds; ablations and training curves.

Baselines are tuned with 3-fold CV on the training split only (small grids), as is the quantum-kernel SVM.
Author: Rayyan Anwar
"""
from __future__ import annotations

import time
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from .data import DATASETS, load_dataset, split_scale
from .qkernel import kernel_matrix
from .vqc import VQC

MODELS = ("logreg", "svm_rbf", "mlp", "quantum_kernel_svm", "vqc")
DEFAULT_VQC = {"n_qubits": 2, "layers": 3, "reupload": True, "epochs": 60, "lr": 0.1}


def run_model(name: str, Xtr, ytr, Xte, yte, seed: int = 0, vqc_cfg: dict | None = None) -> dict:
    cfg = {**DEFAULT_VQC, **(vqc_cfg or {})}
    t0 = time.perf_counter()
    extra: dict = {}
    if name == "logreg":
        m = GridSearchCV(LogisticRegression(max_iter=2000), {"C": [0.1, 1, 10]}, cv=3).fit(Xtr, ytr)
        ptr, pte, extra = m.predict(Xtr), m.predict(Xte), {"params": 3, "best": m.best_params_}
    elif name == "svm_rbf":
        m = GridSearchCV(SVC(), {"C": [0.1, 1, 10], "gamma": ["scale", 0.5, 2.0]}, cv=3).fit(Xtr, ytr)
        ptr, pte = m.predict(Xtr), m.predict(Xte)
        extra = {"params": int(m.best_estimator_.n_support_.sum()), "best": {k: str(v) for k, v in m.best_params_.items()}}
    elif name == "mlp":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)  # reported via the 'converged' field instead
            m = MLPClassifier((8,), max_iter=2000, random_state=seed).fit(Xtr, ytr)
        ptr, pte = m.predict(Xtr), m.predict(Xte)
        extra = {"params": sum(w.size for w in m.coefs_) + sum(b.size for b in m.intercepts_), "converged": bool(m.n_iter_ < 2000)}
    elif name == "quantum_kernel_svm":
        Ktr = kernel_matrix(Xtr)
        m = GridSearchCV(SVC(kernel="precomputed"), {"C": [0.1, 1, 10]}, cv=3).fit(Ktr, ytr)
        ptr, pte = m.predict(Ktr), m.predict(kernel_matrix(Xte, Xtr))
        extra = {"params": int(m.best_estimator_.n_support_.sum()), "best": m.best_params_}
    elif name == "vqc":
        m = VQC(cfg["n_qubits"], cfg["layers"], cfg["reupload"], seed)
        hist = m.fit(Xtr, ytr, epochs=cfg["epochs"], lr=cfg["lr"], seed=seed)
        ptr, pte = m.predict(Xtr), m.predict(Xte)
        extra = {"params": m.n_params, "loss_curve": hist["loss"], "train_acc_curve": hist["train_acc"], "config": cfg}
    else:
        raise ValueError(f"unknown model '{name}'; choose from {MODELS}")
    return {"model": name, "train_acc": float((ptr == ytr).mean()), "test_acc": float((pte == yte).mean()),
            "test_f1": float(f1_score(yte, pte, average="macro")), "seconds": time.perf_counter() - t0, **extra}


def run_suite(datasets=DATASETS, models=MODELS, seeds=(0, 1, 2, 3, 4), vqc_cfg: dict | None = None) -> dict:
    rows = []
    for ds in datasets:
        for s in seeds:
            X, y = load_dataset(ds, s)
            Xtr, Xte, ytr, yte = split_scale(X, y, s)
            for m in models:
                r = run_model(m, Xtr, ytr, Xte, yte, s, vqc_cfg)
                r.pop("loss_curve", None), r.pop("train_acc_curve", None)
                rows.append({"dataset": ds, "seed": s, **r})
    agg = {}
    for ds in datasets:
        agg[ds] = {}
        for m in models:
            a = np.array([r["test_acc"] for r in rows if r["dataset"] == ds and r["model"] == m])
            f = np.array([r["test_f1"] for r in rows if r["dataset"] == ds and r["model"] == m])
            tr = np.array([r["train_acc"] for r in rows if r["dataset"] == ds and r["model"] == m])
            agg[ds][m] = {"test_acc_mean": float(a.mean()), "test_acc_std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
                          "test_f1_mean": float(f.mean()), "train_acc_mean": float(tr.mean()), "n_seeds": int(len(a)),
                          "params": next(r.get("params") for r in rows if r["dataset"] == ds and r["model"] == m)}
    return {"seeds": list(seeds), "vqc_config": {**DEFAULT_VQC, **(vqc_cfg or {})}, "rows": rows, "aggregate": agg}


def ablation(dataset: str = "moons", layers=(1, 2, 3, 4), seeds=(0, 1, 2, 3, 4), epochs: int = 60) -> dict:
    out = {}
    for reup in (False, True):
        for L in layers:
            accs = []
            for s in seeds:
                X, y = load_dataset(dataset, s)
                Xtr, Xte, ytr, yte = split_scale(X, y, s)
                accs.append(run_model("vqc", Xtr, ytr, Xte, yte, s, {"layers": L, "reupload": reup, "epochs": epochs})["test_acc"])
            out[f"reupload={reup},layers={L}"] = {"test_acc_mean": float(np.mean(accs)), "test_acc_std": float(np.std(accs, ddof=1)),
                                                  "params": L * 2 * 2 + 1}
    return {"dataset": dataset, "seeds": list(seeds), "results": out}


class InputError(ValueError):
    """Invalid user input."""


def run_single(dataset: str, model: str, seed: int = 0, n_qubits: int = 2, layers: int = 3, reupload: bool = True,
               epochs: int = 40, lr: float = 0.1) -> dict:
    if dataset not in DATASETS:
        raise InputError(f"dataset must be one of {list(DATASETS)}")
    if model not in MODELS:
        raise InputError(f"model must be one of {list(MODELS)}")
    for name, v, lo, hi in (("seed", seed, 0, 10_000), ("n_qubits", n_qubits, 1, 4), ("layers", layers, 1, 5), ("epochs", epochs, 1, 100)):
        if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
            raise InputError(f"{name} must be an integer in [{lo}, {hi}]")
    if not isinstance(lr, (int, float)) or isinstance(lr, bool) or not 0 < lr <= 1:
        raise InputError("lr must be a number in (0, 1]")
    if not isinstance(reupload, bool):
        raise InputError("reupload must be a boolean")
    X, y = load_dataset(dataset, seed)
    Xtr, Xte, ytr, yte = split_scale(X, y, seed)
    r = run_model(model, Xtr, ytr, Xte, yte, seed, {"n_qubits": n_qubits, "layers": layers, "reupload": reupload, "epochs": epochs, "lr": lr})
    return {"dataset": dataset, "seed": seed, "train_n": int(len(ytr)), "test_n": int(len(yte)), **r}
