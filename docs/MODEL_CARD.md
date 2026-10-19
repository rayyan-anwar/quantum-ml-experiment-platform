# System Card

Author: Rayyan Anwar

**System:** a research harness comparing a simulated variational quantum classifier (VQC), a simulated quantum-kernel SVM, and classical baselines (logistic regression, RBF SVM, small MLP) on four small 2-D toy datasets.

**Intended use:** learning and reproducible experimentation with hybrid quantum-classical ML on a simulator. **Out of scope:** claims of quantum advantage, statements about real hardware (noise, connectivity, shot cost, queue time), or production ML.

**Data:** generated (moons, circles, XOR blobs) or bundled with scikit-learn (Iris versicolor vs virginica, petal length/width). No personal data.

**Limitations and risks:** ideal qubits only (no noise, no finite shots); 2-4 qubits on 2-D data cannot say anything about scaling; baselines are small and lightly tuned (the MLP is untuned and sometimes does not converge); 5 seeds give wide intervals; results are sensitive to the feature map and ansatz chosen; classical simulation cost grows as 2^n, so there is no speed claim either.
