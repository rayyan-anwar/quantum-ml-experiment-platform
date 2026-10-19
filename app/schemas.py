"""Pydantic request models. Author: Rayyan Anwar"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ExperimentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset: Literal["moons", "circles", "xor_blobs", "iris_vv"] = "moons"
    model: Literal["logreg", "svm_rbf", "mlp", "quantum_kernel_svm", "vqc"] = "vqc"
    seed: int = Field(0, ge=0, le=10_000)
    n_qubits: int = Field(2, ge=1, le=4)
    layers: int = Field(3, ge=1, le=5)
    reupload: bool = True
    epochs: int = Field(40, ge=1, le=100)
    lr: float = Field(0.1, gt=0, le=1)
