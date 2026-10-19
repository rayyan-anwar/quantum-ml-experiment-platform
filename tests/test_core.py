"""Tests: simulator vs dense reference, known quantum states, VQC gradients/training, kernel properties, experiments, API.

Author: Rayyan Anwar
"""
import unittest

import numpy as np

from app import dense_reference as D
from app import quantum as q
from app.data import DATASETS, load_dataset, split_scale
from app.experiments import MODELS, InputError, run_model, run_single, run_suite
from app.qkernel import feature_map_states, kernel_matrix
from app.vqc import VQC


def random_circuit(n, depth, rng):
    state, ops = q.zero_state(n), []
    for _ in range(depth):
        kind = rng.integers(0, 6)
        a = int(rng.integers(0, n))
        if kind == 0:
            th = float(rng.uniform(0, 6.3)); g = [q.rx, q.ry, q.rz][int(rng.integers(0, 3))](th)
            state = q.apply_1q(state, g, a); ops.append(D.op_1q(n, g, a))
        elif kind == 1:
            name = ["h", "x", "y", "z", "s", "t", "sdg", "tdg"][int(rng.integers(0, 8))]
            state = q.apply_1q(state, q.FIXED[name], a); ops.append(D.op_1q(n, q.FIXED[name], a))
        elif n > 1:
            b = int((a + 1 + rng.integers(0, n - 1)) % n)
            fn, ref = [(q.apply_cnot, D.op_cnot), (q.apply_cz, D.op_cz), (q.apply_swap, D.op_swap), (q.apply_cnot, D.op_cnot)][kind - 2]
            state = fn(state, a, b); ops.append(ref(n, a, b))
    return state, ops


class TestSimulator(unittest.TestCase):
    def test_random_circuits_match_dense_reference(self):
        rng = np.random.default_rng(0)
        for n in range(1, 6):
            for _ in range(20):
                state, ops = random_circuit(n, 25, rng)
                np.testing.assert_allclose(state.reshape(-1), D.run(n, ops), atol=1e-10)
                self.assertAlmostEqual(float(np.linalg.norm(state)), 1.0, places=10)

    def test_all_gates_unitary(self):
        for name, g in q.FIXED.items():
            np.testing.assert_allclose(g.conj().T @ g, np.eye(2), atol=1e-12, err_msg=name)
        for f in (q.rx, q.ry, q.rz):
            for th in (0.0, 0.3, 2.0, np.pi):
                g = f(th)
                np.testing.assert_allclose(g.conj().T @ g, np.eye(2), atol=1e-12)

    def test_batched_per_sample_gates_match_loop(self):
        angles = np.array([0.1, 1.0, 2.5, 3.0])
        batch = q.apply_1q(q.zero_state(2, 4), q.ry(angles), 1)
        for i, a in enumerate(angles):
            single = q.apply_1q(q.zero_state(2, 1), q.ry(a), 1)
            np.testing.assert_allclose(batch[i], single[0], atol=1e-12)

    def test_bell_state_properties(self):
        s = q.apply_cnot(q.apply_1q(q.zero_state(2), q.H, 0), 0, 1)
        np.testing.assert_allclose(q.probabilities(s)[0], [0.5, 0, 0, 0.5], atol=1e-12)
        for pauli, expected in (("ZZ", 1), ("XX", 1), ("YY", -1), ("ZI", 0), ("IZ", 0)):
            self.assertAlmostEqual(float(q.expect_pauli(s, pauli)[0]), expected, places=10, msg=pauli)
        self.assertAlmostEqual(q.qubit_entropy(s[0], 0), 1.0, places=10)
        self.assertEqual(q.basis_labels(2), ["00", "01", "10", "11"])

    def test_bloch_vectors_and_entropy_of_product_states(self):
        plus = q.apply_1q(q.zero_state(1), q.H, 0)[0]
        self.assertTrue(np.allclose(q.bloch_vector(plus, 0), (1, 0, 0), atol=1e-12))
        zero = q.zero_state(1)[0]
        self.assertTrue(np.allclose(q.bloch_vector(zero, 0), (0, 0, 1), atol=1e-12))
        iplus = q.apply_1q(plus[None], q.S, 0)[0]
        self.assertTrue(np.allclose(q.bloch_vector(iplus, 0), (0, 1, 0), atol=1e-12))
        self.assertAlmostEqual(q.qubit_entropy(plus, 0), 0.0, places=10)

    def test_expect_z_and_qubit_ordering(self):
        s = q.apply_1q(q.zero_state(3), q.X, 0)  # |100>
        self.assertEqual(int(np.argmax(q.probabilities(s)[0])), 0b100)
        self.assertAlmostEqual(float(q.expect_z(s, 0)[0]), -1.0)
        self.assertAlmostEqual(float(q.expect_z(s, 2)[0]), 1.0)

    def test_invalid_qubits_rejected(self):
        s = q.zero_state(2)
        for call in (lambda: q.apply_1q(s, q.X, 2), lambda: q.apply_cnot(s, 0, 0), lambda: q.apply_cz(s, 0, 5),
                     lambda: q.zero_state(0), lambda: q.expect_pauli(s, "XYZ"), lambda: q.expect_pauli(s, "AB")):
            with self.assertRaises(ValueError):
                call()

    def test_sampling(self):
        c = q.sample_counts(np.array([0.5, 0, 0, 0.5]), 1000, seed=1)
        self.assertEqual(sum(c.values()), 1000)
        self.assertEqual(set(c), {"00", "11"})
        self.assertEqual(c, q.sample_counts(np.array([0.5, 0, 0, 0.5]), 1000, seed=1))
        self.assertAlmostEqual(c["00"] / 1000, 0.5, delta=0.07)


class TestVQC(unittest.TestCase):
    def test_parameter_shift_matches_finite_difference(self):
        X, y = load_dataset("moons", 0)
        Xtr, _, ytr, _ = split_scale(X, y, 0)
        for n, layers, reup in ((2, 2, True), (3, 2, False), (1, 2, True)):
            m = VQC(n, layers, reup, seed=1)
            _, g, gb = m.loss_and_grad(Xtr[:16], ytr[:16])
            num = np.zeros_like(m.theta)
            for idx in np.ndindex(m.theta.shape):
                m.theta[idx] += 1e-6; lp = m.loss_and_grad(Xtr[:16], ytr[:16])[0]
                m.theta[idx] -= 2e-6; lm = m.loss_and_grad(Xtr[:16], ytr[:16])[0]
                m.theta[idx] += 1e-6; num[idx] = (lp - lm) / 2e-6
            np.testing.assert_allclose(g, num, atol=1e-6, err_msg=f"n={n}")
            m.bias += 1e-6; lp = m.loss_and_grad(Xtr[:16], ytr[:16])[0]; m.bias -= 2e-6; lm = m.loss_and_grad(Xtr[:16], ytr[:16])[0]
            self.assertAlmostEqual(gb, (lp - lm) / 2e-6, places=5)

    def test_forward_many_matches_single_evaluations(self):
        X = np.random.default_rng(0).uniform(0, np.pi, (5, 2))
        m = VQC(2, 2, True, 0)
        thetas = np.stack([m.theta, m.theta + 0.3])
        both = m.forward_many(X, thetas)
        self.assertEqual(both.shape, (2, 5))
        np.testing.assert_allclose(both[0], m.forward_many(X, thetas[:1])[0], atol=1e-12)
        self.assertTrue((np.abs(both) <= 1 + 1e-9).all())

    def test_training_reduces_loss_and_is_deterministic(self):
        X, y = load_dataset("moons", 0)
        Xtr, Xte, ytr, yte = split_scale(X, y, 0)
        a, b = VQC(2, 3, True, 0), VQC(2, 3, True, 0)
        ha, hb = a.fit(Xtr, ytr, epochs=25), b.fit(Xtr, ytr, epochs=25)
        self.assertLess(ha["loss"][-1], ha["loss"][0] * 0.8)
        self.assertEqual(ha["loss"], hb["loss"])
        self.assertGreater((a.predict(Xte) == yte).mean(), 0.6)

    def test_invalid_configuration(self):
        for kw in ({"n_qubits": 0}, {"n_qubits": 7}, {"layers": 0}, {"layers": 9}):
            with self.assertRaises(ValueError):
                VQC(**kw)
        with self.assertRaises(ValueError):
            VQC().fit(np.zeros((4, 2)), np.zeros(4), epochs=0)


class TestKernel(unittest.TestCase):
    def test_kernel_is_valid_fidelity_kernel(self):
        X = np.random.default_rng(1).uniform(0, np.pi, (25, 2))
        K = kernel_matrix(X)
        np.testing.assert_allclose(K, K.T, atol=1e-12)
        np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-10)
        self.assertGreater(np.linalg.eigvalsh(K).min(), -1e-10)
        self.assertTrue(((K > -1e-12) & (K < 1 + 1e-10)).all())

    def test_feature_map_states_are_normalised_and_cross_kernel_shape(self):
        S = feature_map_states(np.random.default_rng(2).uniform(0, np.pi, (6, 3)))
        np.testing.assert_allclose(np.linalg.norm(S, axis=1), 1.0, atol=1e-10)
        self.assertEqual(kernel_matrix(np.zeros((4, 2)), np.zeros((7, 2))).shape, (4, 7))


class TestExperiments(unittest.TestCase):
    def test_datasets_and_split_scaling(self):
        for ds in DATASETS:
            X, y = load_dataset(ds, 0)
            Xtr, Xte, ytr, yte = split_scale(X, y, 0)
            self.assertEqual(set(np.unique(y)), {0, 1})
            self.assertGreaterEqual(Xtr.min(), 0)
            self.assertLessEqual(max(Xtr.max(), Xte.max()), np.pi + 1e-12)
            self.assertAlmostEqual(abs(ytr.mean() - y.mean()), 0, delta=0.06)
        with self.assertRaises(ValueError):
            load_dataset("mnist")
        with self.assertRaises(ValueError):
            load_dataset("moons", n=5)

    def test_every_model_runs_and_beats_chance_on_easy_data(self):
        X, y = load_dataset("iris_vv", 0)
        Xtr, Xte, ytr, yte = split_scale(X, y, 0)
        for m in MODELS:
            r = run_model(m, Xtr, ytr, Xte, yte, 0, {"epochs": 15})
            self.assertGreater(r["test_acc"], 0.6, msg=m)
            self.assertTrue(0 <= r["train_acc"] <= 1)
        with self.assertRaises(ValueError):
            run_model("transformer", Xtr, ytr, Xte, yte)

    def test_suite_small_structure(self):
        r = run_suite(datasets=("iris_vv",), models=("logreg", "vqc"), seeds=(0, 1), vqc_cfg={"epochs": 8})
        self.assertEqual(len(r["rows"]), 4)
        self.assertEqual(r["aggregate"]["iris_vv"]["vqc"]["n_seeds"], 2)

    def test_run_single_validation(self):
        ok = run_single("iris_vv", "vqc", epochs=5)
        self.assertEqual(len(ok["loss_curve"]), 5)
        for kw in ({"dataset": "x"}, {"model": "x"}, {"layers": 9}, {"n_qubits": 5}, {"epochs": 0}, {"lr": 0}, {"seed": -1},
                   {"reupload": "yes"}, {"epochs": True}):
            with self.assertRaises(InputError, msg=str(kw)):
                run_single(**{"dataset": "moons", "model": "vqc", **kw})


try:
    import fastapi  # noqa: F401
    import httpx  # noqa: F401
    HAVE_API = True
except ImportError:
    HAVE_API = False


@unittest.skipUnless(HAVE_API, "fastapi/httpx not installed - API tests are skipped, not passed")
class TestApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        from app.main import app
        cls.c = TestClient(app)

    def test_basics(self):
        self.assertEqual(self.c.get("/health").json()["author"], "Rayyan Anwar")
        self.assertIn("Rayyan Anwar", self.c.get("/").text)

    def test_experiment_and_validation(self):
        r = self.c.post("/api/experiment", json={"dataset": "iris_vv", "model": "vqc", "epochs": 5})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(len(r.json()["loss_curve"]), 5)
        for bad in ({"dataset": "x"}, {"layers": 9}, {"epochs": 1000}, {"extra": 1}, {"lr": 0}):
            self.assertEqual(self.c.post("/api/experiment", json=bad).status_code, 422, msg=str(bad))


if __name__ == "__main__":
    unittest.main()
