# Deployment

Author: Rayyan Anwar

## Local
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest -q
python scripts/evaluate.py          # about a minute; regenerates results/
uvicorn app.main:app --port 8017    # http://127.0.0.1:8017
```
## Docker
`docker compose up --build` (port 8017, health `/health`). Written but **not built or run** in the authoring sandbox; CI has a `docker build` job.
## Cloud
Any container host: build and push (your account), expose 8017 behind HTTPS, probe `/health`. Experiments are CPU-bound and run one at a time (HTTP 409 if busy); give the service a dedicated vCPU. Inputs are capped (4 qubits, 5 layers, 100 epochs). No secrets, database or storage. No authentication is built in; add platform access control if public. Not deployed by the author.
## Screenshots
Run a VQC experiment and view the committed comparison tables; capture into `docs/screenshots/` (none bundled).
