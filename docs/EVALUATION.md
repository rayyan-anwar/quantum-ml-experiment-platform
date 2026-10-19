# Evaluation

Author: Rayyan Anwar

## Research questions
1. On small 2-D toy datasets, how does a simulated variational quantum classifier (VQC) compare with tuned classical baselines under identical splits?
2. Does a fidelity quantum kernel with a ZZ-style feature map help an SVM?
3. Which VQC design choices (depth, data re-uploading) matter?

## Method
`python scripts/evaluate.py` runs 4 datasets x 5 models x 5 seeds (seed changes the data draw, the 60/40 stratified split and the initialisation), plus a VQC ablation on moons. Logistic regression, RBF SVM and the quantum-kernel SVM are tuned by 3-fold CV on training data only; the MLP (8 hidden units) is untuned (17 of 20 runs converged within 2000 iterations); the VQC uses 2 qubits, 3 layers, re-uploading, 60 epochs of Adam (lr 0.1).

## Committed results (test accuracy, mean ± std over 5 seeds; see `results/RESULTS.md`)
| Dataset | logreg | RBF SVM | MLP | quantum-kernel SVM | VQC (13 params) |
|---|---:|---:|---:|---:|---:|
| moons | 0.850 ± 0.037 | **0.954 ± 0.017** | 0.842 ± 0.033 | 0.746 ± 0.068 | 0.908 ± 0.050 |
| circles | 0.479 ± 0.035 | **0.996 ± 0.006** | 0.731 ± 0.206 | 0.852 ± 0.045 | 0.921 ± 0.071 |
| xor_blobs | 0.485 ± 0.108 | 0.933 ± 0.019 | 0.762 ± 0.174 | 0.762 ± 0.046 | **0.940 ± 0.020** |
| iris_vv | 0.940 ± 0.038 | **0.955 ± 0.027** | 0.945 ± 0.037 | 0.715 ± 0.034 | 0.950 ± 0.035 |

## Findings
- **No quantum advantage is shown.** The tuned classical RBF SVM is best or tied on three of four datasets, and clearly better on moons and circles. The VQC wins nominally only on XOR blobs, within noise of the SVM (0.940 vs 0.933).
- The VQC beats the small, untuned MLP on three datasets, but the MLP baseline is weak (high variance, not tuned), so this is **not** evidence for the VQC being a better learner.
- Linear logistic regression fails on circles and XOR as expected (about chance), which confirms the datasets need non-linear models.
- The quantum-kernel SVM with this feature map is weaker than the RBF SVM on every dataset; the feature map was fixed, not optimised, and results depend heavily on that choice.
- Ablation (moons): re-uploading only helps once there are at least 3 layers (0.908 and 0.929 vs 0.840 and 0.850 without); with 1 layer both settings are identical by construction (0.669).
- The simulator is verified, not just used: agreement with an independent dense-matrix implementation to about 1e-10 on random circuits, and parameter-shift gradients match finite differences to about 1e-10.

## Caveats
Ideal noise-free qubits; 2-4 qubits on 2-D data; 5 seeds, so differences below about one standard deviation are not meaningful; no significance tests; classical simulation cost doubles per qubit, so nothing here speaks to speed or to hardware. Run timings (about 0.8 s per VQC fit) are from a 1-core sandbox.
