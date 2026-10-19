# Quantum ML Experiment Platform

**Author: Rayyan Anwar** — AI/ML Engineer | Researcher | Technology Manager | Software Developer

A reproducible harness for comparing a **simulated variational quantum classifier (VQC)** and a **quantum-kernel SVM** with tuned classical baselines. It includes its own batched NumPy state-vector simulator (validated against an independent dense-matrix implementation), exact parameter-shift gradients, ablations, an API and a small UI.

> **Simulator only, no advantage claimed.** Ideal noise-free qubits on tiny toy datasets. In the committed run a tuned classical RBF SVM was best or tied on 3 of 4 datasets. See [docs/EVALUATION.md](docs/EVALUATION.md).

## Results (committed run, 5 seeds, `results/RESULTS.md`)
| Dataset | RBF SVM | VQC | Quantum-kernel SVM | MLP | Logistic regression |
|---|---:|---:|---:|---:|---:|
| moons | 0.954 | 0.908 | 0.746 | 0.842 | 0.850 |
| circles | 0.996 | 0.921 | 0.852 | 0.731 | 0.479 |
| xor_blobs | 0.933 | 0.940 | 0.762 | 0.762 | 0.485 |
| iris (versicolor vs virginica) | 0.955 | 0.950 | 0.715 | 0.945 | 0.940 |

Mean test accuracy; standard deviations (typically 0.02 to 0.07) are in the results file. Re-uploading data helps the VQC only from 3 layers up.

## Implemented vs optional
| Feature | Status |
|---|---|
| Batched state-vector simulator (1-qubit gates, CNOT, CZ, SWAP, Pauli expectations, Bloch vectors, entropy, sampling) | Implemented; checked against a dense reference on random circuits |
| VQC with data re-uploading, exact parameter-shift gradients, Adam | Implemented; gradients verified against finite differences |
| Quantum fidelity kernel + SVM | Implemented, tested (symmetric, PSD, unit diagonal) |
| Tuned classical baselines, multi-seed protocol, ablation, training curve | Implemented |
| FastAPI service + browser UI | Implemented; API tests written but **not executed in the authoring sandbox** |
| Hardware execution, noise models, shot noise, barren-plateau studies, larger qubit counts | Not implemented |

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest -q
python scripts/evaluate.py             # about a minute
uvicorn app.main:app --port 8017       # http://127.0.0.1:8017
```
`curl -s localhost:8017/api/experiment -H 'content-type: application/json' -d '{"dataset":"moons","model":"vqc","epochs":30}'`

Endpoints: `GET /health`, `/api/info`, `/api/results`; `POST /api/experiment`; docs at `/docs`.

## Documentation
[Architecture](docs/ARCHITECTURE.md) · [Evaluation](docs/EVALUATION.md) · [System card](docs/MODEL_CARD.md) · [Deployment](docs/DEPLOYMENT.md)

## Testing status
18 tests executed and passing (`python -m unittest`); 2 API tests skipped in the authoring sandbox (FastAPI/httpx unavailable). Docker not built; Ruff not run; UI JavaScript syntax-checked only.

## Known limitations
Noise-free simulation; tiny datasets and qubit counts; baselines lightly tuned (MLP untuned); few seeds; results depend on the chosen ansatz and feature map; no speed or hardware claims; no authentication.

## GitHub repository metadata
- **Description:** Compare a simulated variational quantum classifier and quantum kernel with tuned classical baselines; NumPy simulator validated against a dense reference.
- **Topics:** `quantum-machine-learning`, `variational-quantum-circuit`, `quantum-kernel`, `state-vector-simulator`, `scikit-learn`, `reproducible-research`, `fastapi`, `python`

## License
MIT © Rayyan Anwar. See [LICENSE](LICENSE).
