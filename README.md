# ⚛️ Quantum ML Experiment Platform

<div align="center">

  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:080B1E,35:312E81,70:6D28D9,100:06B6D4&height=240&section=header&text=Quantum%20ML%20Lab&fontSize=44&fontColor=FFFFFF&animation=fadeIn&fontAlignY=35&desc=Quantum%20Circuits%20%7C%20Machine%20Learning%20%7C%20Reproducible%20Research&descSize=13&descAlignY=58" width="100%" alt="Animated Quantum ML research banner" />

  <h3>Where Quantum Algorithms Meet Classical Machine Learning</h3>

  **Created by Rayyan Anwar**

  *AI/ML Engineer · Researcher · Technology Manager · Software Developer*

  <p>
    <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
    <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy" />
    <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="scikit-learn" />
    <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
  </p>

  <p>
    <img src="https://img.shields.io/badge/Tests-18%20Passed-success?style=flat-square" alt="18 tests reported passing" />
    <img src="https://img.shields.io/badge/Research-Reproducible-blueviolet?style=flat-square" alt="Reproducible research" />
    <img src="https://img.shields.io/badge/Execution-Simulator%20Only-orange?style=flat-square" alt="Simulator only" />
    <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="MIT License" />
  </p>

  <a href="https://github.com/rayyan-anwar">GitHub Profile</a> ·
  <a href="https://github.com/rayyan-anwar?tab=repositories">Explore Projects</a>

</div>

---

## 🧬 About the Project

The **Quantum ML Experiment Platform** is a reproducible research environment for comparing simulated quantum machine-learning models against classical baselines on small classification datasets.

It combines a custom batched NumPy state-vector simulator, a Variational Quantum Classifier (VQC), a quantum fidelity kernel with an SVM, and classical machine-learning models in one experimental workflow.

The goal is straightforward: **measure performance, examine the evidence, and understand when a quantum-inspired approach does or does not outperform a classical method.**

> [!IMPORTANT]
> **Simulator only. No quantum advantage claimed.** Experiments use ideal, noise-free quantum simulation and small datasets. The reported results do not establish an advantage on real quantum hardware.

## ✨ Key Features

<table>
  <tr>
    <td width="50%">
      <h3>⚛️ Custom Quantum Simulator</h3>
      Batched NumPy state-vector simulation with single-qubit gates, CNOT, CZ, SWAP, and quantum-state analysis.
    </td>
    <td width="50%">
      <h3>🧠 Variational Quantum Classifier</h3>
      Train parameterized quantum circuits using data re-uploading, exact parameter-shift gradients, and Adam optimization.
    </td>
  </tr>
  <tr>
    <td width="50%">
      <h3>🔗 Quantum Fidelity Kernel</h3>
      Construct a quantum fidelity kernel and use it with an SVM for classification experiments.
    </td>
    <td width="50%">
      <h3>📈 Classical Baselines</h3>
      Compare quantum approaches against RBF SVM, multilayer perceptron, and logistic regression.
    </td>
  </tr>
  <tr>
    <td width="50%">
      <h3>🔬 Reproducible Experiments</h3>
      Evaluate multiple random seeds, compare model accuracy, and investigate experimental ablations.
    </td>
    <td width="50%">
      <h3>🌐 API & Browser UI</h3>
      Run experiments through a FastAPI service and an accompanying browser interface.
    </td>
  </tr>
</table>

## 📊 Benchmark Results

The following table reproduces the reported mean test accuracies from the committed evaluation, using five seeds.

| Dataset | RBF SVM | VQC | Quantum-Kernel SVM | MLP | Logistic Regression |
|---|---:|---:|---:|---:|---:|
| Moons | **0.954** | 0.908 | 0.746 | 0.842 | 0.850 |
| Circles | **0.996** | 0.921 | 0.852 | 0.731 | 0.479 |
| XOR Blobs | 0.933 | **0.940** | 0.762 | 0.762 | 0.485 |
| Iris: Versicolor vs. Virginica | **0.955** | 0.950 | 0.715 | 0.945 | 0.940 |

<sub>Metric: mean test accuracy. Results are from the committed experiment report and are not independently re-run by this README. Standard deviations, typically 0.02 to 0.07, are reported in <code>results/RESULTS.md</code>.</sub>

### What the results suggest

- **RBF SVM is the strongest overall baseline** in these experiments, achieving the best or tied performance on three of four datasets.
- **VQC performs competitively on XOR Blobs**, where its reported accuracy is 0.940 compared with 0.933 for RBF SVM.
- **Quantum-kernel SVM does not dominate** the classical baselines on these particular datasets.
- **Data re-uploading matters:** the reported experiments show a VQC benefit from re-uploading at three or more layers, according to the committed evaluation.

These observations apply to the selected datasets, circuit architectures, optimization settings, and seeds. They should not be generalized to all quantum machine-learning tasks.

## 🔬 Research Methodology

The platform supports a workflow for making comparisons more systematic.

1. **Dataset selection:** Choose a supported toy classification dataset.
2. **Model selection:** Select VQC, quantum-kernel SVM, or a classical baseline.
3. **Training:** Run the model using the configured training procedure.
4. **Evaluation:** Measure test accuracy across random seeds.
5. **Ablation:** Investigate changes to circuit depth and data re-uploading.
6. **Comparison:** Review results and training curves to identify differences between methods.

For a fair comparison, preprocessing, data splits, tuning budgets, and evaluation procedures should be documented and controlled. The existing baselines are only lightly tuned, and the MLP is untuned, so the reported ranking is not a definitive comparison of each model's best possible performance.

## ⚙️ Quantum Simulation Capabilities

The custom simulator implements:

| Capability | Description |
|---|---|
| State-vector evolution | Track quantum amplitudes through circuit operations |
| Single-qubit gates | Apply supported one-qubit operations |
| Multi-qubit gates | CNOT, CZ, and SWAP |
| Pauli expectations | Evaluate supported quantum observables |
| Bloch vectors | Analyze single-qubit state properties |
| Entropy | Calculate supported quantum entropy measures |
| Measurement sampling | Generate simulated measurement outcomes |
| Parameter-shift gradients | Calculate gradients for supported variational parameters |

### Verification

The simulator was compared against an independent dense-matrix implementation on random circuits. The VQC's parameter-shift gradients were also checked against finite differences.

The committed test and evaluation reports should be consulted for exact scope, tolerances, and limitations.

## 🏗️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core implementation and experiment orchestration |
| NumPy | Batched numerical computation and state-vector simulation |
| scikit-learn | Classical models, SVMs, and evaluation utilities |
| FastAPI | Experiment API |
| HTML, CSS, JavaScript | Browser interface |
| pytest / unittest | Automated tests |
| Matplotlib or project-supported plotting tools | Experiment visualization, where implemented |

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or a compatible version supported by the project's dependencies.
- Git.
- A terminal or command prompt.

### 1. Clone the repository

```bash
git clone https://github.com/rayyan-anwar/quantum-ml-experiment-platform.git
cd quantum-ml-experiment-platform
```

If your repository has a different name, replace the example URL and directory with the actual values.

### 2. Create a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

### 4. Run the tests

```bash
python -m pytest -q
```

The committed report records 18 passing tests. Two API tests were not executed in the original authoring sandbox, so verify the current test output in your environment.

### 5. Run the evaluation

```bash
python scripts/evaluate.py
```

The original report indicates a runtime of approximately one minute. Actual runtime depends on your hardware and environment.

### 6. Launch the application

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8017
```

Open the following URLs:

| Resource | URL |
|---|---|
| Browser application | http://127.0.0.1:8017 |
| Interactive API documentation | http://127.0.0.1:8017/docs |
| Health check | http://127.0.0.1:8017/health |
| Application information | http://127.0.0.1:8017/api/info |
| Experiment results | http://127.0.0.1:8017/api/results |

## 🔌 API Example

Submit an experiment using the API:

```bash
curl -s http://127.0.0.1:8017/api/experiment \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": "moons",
    "model": "vqc",
    "epochs": 30
  }'
```

The request demonstrates the experiment endpoint with the `moons` dataset, VQC model, and 30 training epochs. Consult the running `/docs` page for the exact request schema and supported parameters.

## 🧪 Testing & Verification Status

| Item | Reported status |
|---|---|
| Automated tests | 18 executed and passing |
| Quantum simulator | Compared with an independent dense-matrix reference |
| Parameter-shift gradients | Verified against finite differences |
| Quantum fidelity kernel | Symmetry, positive-semidefinite, and unit-diagonal checks implemented |
| API tests | Written but not executed in the original authoring sandbox |
| Docker build | Not verified |
| Ruff linting | Not run |
| Browser UI | JavaScript syntax-checked; full browser behavior not verified |

The committed report documents the tested implementation, not a guarantee that every environment or future revision will behave identically.

## 📚 Documentation

| Document | Description |
|---|---|
| [Architecture](docs/ARCHITECTURE.md) | Components and system design |
| [Evaluation](docs/EVALUATION.md) | Experimental methodology, metrics, and caveats |
| [Model Card](docs/MODEL_CARD.md) | Intended use and limitations |
| [Deployment](docs/DEPLOYMENT.md) | Deployment instructions |
| [Results](results/RESULTS.md) | Committed benchmark results |

## ⚠️ Known Limitations

- Ideal, noise-free quantum simulation only.
- Small datasets and limited qubit counts.
- Classical baselines are not all equally tuned.
- Results use a limited number of random seeds.
- Results depend on the chosen feature map, ansatz, optimizer, and hyperparameters.
- No claims of quantum speedup or real-hardware advantage.
- No hardware execution, noise models, or shot-noise studies.
- No barren-plateau investigation.
- No authentication in the current application.

## 🛣️ Future Research Directions

The following are potential extensions, not implemented features:

- [ ] Quantum circuit noise and finite-shot simulation.
- [ ] Barren-plateau and gradient-variance studies.
- [ ] Expanded hyperparameter optimization for all baselines.
- [ ] Confidence intervals and statistical significance analysis.
- [ ] Additional datasets and larger multi-seed experiments.
- [ ] Integration with established quantum SDKs.
- [ ] Optional real-hardware execution backends.
- [ ] Docker build verification and expanded CI coverage.
- [ ] Browser end-to-end tests and improved experiment visualizations.

## 👨‍💻 About the Author

<div align="center">

  <h3>Rayyan Anwar</h3>

  **AI/ML Engineer · Researcher · Technology Manager · Software Developer**

  Exploring artificial intelligence, quantum computing, scientific experimentation, and responsible software engineering.

  <a href="https://github.com/rayyan-anwar">
    <img src="https://img.shields.io/badge/GitHub-rayyan--anwar-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub Profile" />
  </a>

  <p>
    <a href="https://github.com/rayyan-anwar?tab=repositories">Explore My Repositories</a>
  </p>

</div>

## 📄 License

Released under the **MIT License**.

Copyright © Rayyan Anwar. See [`LICENSE`](LICENSE) for details.

---

<div align="center">

  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:06B6D4,45:6D28D9,100:080B1E&height=120&section=footer" width="100%" alt="Quantum ML decorative footer" />

  **Experiment rigorously. Compare honestly. Advance quantum machine learning.**

  <sub>Research first. Evidence always. No advantage claimed without proof.</sub>

</div>
