"""FastAPI application for the Quantum ML Experiment Platform. Author: Rayyan Anwar"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from . import __author__, __version__
from .data import DATASETS
from .experiments import MODELS, InputError, run_single
from .schemas import ExperimentRequest

ROOT = Path(__file__).resolve().parent.parent
_busy = threading.Lock()
app = FastAPI(title="Quantum ML Experiment Platform", version=__version__,
              description="Compare a simulated variational quantum classifier and a quantum kernel with classical baselines. "
                          "Noise-free simulation only. Author: Rayyan Anwar.",
              contact={"name": __author__}, license_info={"name": "MIT"})


@app.exception_handler(InputError)
async def on_input_error(_: Request, exc: InputError):
    return JSONResponse(status_code=422, content={"error": {"code": "invalid_input", "message": str(exc)}})


@app.get("/health")
def health():
    return {"status": "ok", "version": __version__, "author": __author__}


@app.get("/api/info")
def info():
    return {"name": "Quantum ML Experiment Platform", "author": __author__, "version": __version__,
            "simulator": "exact noise-free state vector (NumPy)", "datasets": list(DATASETS), "models": list(MODELS)}


@app.post("/api/experiment")
async def experiment(req: ExperimentRequest):
    if not _busy.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="another experiment is running; retry shortly")

    def work():
        try:
            return run_single(**req.model_dump())
        finally:
            _busy.release()
    return await run_in_threadpool(work)


@app.get("/api/results")
def results():
    p = ROOT / "results" / "results.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="no results yet; run: python scripts/evaluate.py")
    return json.loads(p.read_text())


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(ROOT / "frontend" / "index.html")
