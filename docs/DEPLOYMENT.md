# Deployment

Author: Rayyan Anwar

## Local development

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
python -m pytest -q
python scripts/evaluate.py
uvicorn app.main:app --host 127.0.0.1 --port 8017
```

Open http://127.0.0.1:8017. The API health endpoint is /health.

## Docker

Build and start with:

```bash
docker compose up --build
```

The service listens on port 8017. Check http://127.0.0.1:8017/health.
The image includes the application, frontend and committed results. Dataset generation
uses scikit-learn and does not require a separate data directory.

## Continuous integration

GitHub Actions runs the complete pytest suite and builds the Docker image on pushes
to main or fix branches, pull requests targeting main, and manual dispatches.
The first run must complete successfully before treating CI as verified.

## Production notes

This is a CPU-bound research demo. Experiments run one at a time; concurrent requests
receive HTTP 409. Use HTTPS and platform-level access control before exposing it
publicly. The application does not include user authentication, persistent user
accounts, or managed experiment storage. Do not describe it as a production-secure
multi-user service without adding those controls.

The quantum implementation is an ideal, noise-free NumPy simulator. No quantum
hardware execution or quantum advantage is claimed.
