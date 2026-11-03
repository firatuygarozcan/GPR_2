"""GPyTorch backend: interface and numerical equivalence with the sklearn model.

Small synthetic data only; no SimBench data, no experiment outputs.
"""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'seasonal'))
import dsse
from config import NETWORK
from gpytorch_models import GPyTorchDSSEGaussianProcess, sklearn_hyperparameter_spec


def synthetic_data(n_train=120, n_test=40, n_inputs=5, n_outputs=4, seed=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_train + n_test, n_inputs)) * rng.uniform(0.1, 10.0, size=n_inputs)
    weights = rng.normal(size=(n_inputs, n_outputs))
    Y = 1.0 + 0.01 * np.tanh(X @ weights / 10.0) + 1e-4 * rng.normal(size=(n_train + n_test, n_outputs))
    Y[:, -1] = 1.02  # constant target: exercises the zero-std normalization guard
    columns = [f'bus_{bus}_vm_pu' for bus in range(n_outputs)]
    X = pd.DataFrame(X, columns=[f'x{j}' for j in range(n_inputs)])
    Y = pd.DataFrame(Y, columns=columns)
    return X.iloc[:n_train], Y.iloc[:n_train], X.iloc[n_train:]


def sklearn_hyperparameters(model):
    kernel = model.model.kernel_
    return {
        'outputscale': float(kernel.k1.k1.constant_value),
        'lengthscale': float(kernel.k1.k2.length_scale),
        'noise': float(kernel.k2.noise_level),
    }


class DefaultBackendTests(unittest.TestCase):
    def test_default_backend_is_sklearn(self):
        self.assertIsInstance(dsse.make_gp_model(), dsse.DSSEGaussianProcess)

    def test_default_run_directory_is_unchanged(self):
        self.assertEqual(dsse.model_run_suffix(), '')
        self.assertEqual(
            dsse.seasonal_run_directory(NETWORK).name,
            '4gpr_constant_rbf_white_alpha1e-10_train_30_all_remaining_test_noisy',
        )
        self.assertEqual(dsse.model_run_suffix('gpytorch', 'shared'), '_gpytorch_shared')

    def test_bounds_and_initial_values_come_from_the_sklearn_kernel(self):
        spec = sklearn_hyperparameter_spec()
        self.assertEqual(spec['outputscale'], (1.0, (1e-3, 1e3)))
        self.assertEqual(spec['lengthscale'], (1.0, (1e-2, 1e3)))
        self.assertEqual(spec['noise'], (1e-3, (1e-8, 1.0)))


class SharedVariantEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X_train, cls.Y_train, cls.X_test = synthetic_data()
        cls.sk = dsse.DSSEGaussianProcess().fit(cls.X_train, cls.Y_train)
        cls.sk_mean, cls.sk_std = cls.sk.predict(cls.X_test)

    def test_fixed_sklearn_hyperparameters_reproduce_sklearn(self):
        gp = GPyTorchDSSEGaussianProcess().fit(
            self.X_train, self.Y_train, hyperparameters=sklearn_hyperparameters(self.sk)
        )
        mean, std = gp.predict(self.X_test)
        self.assertEqual(list(mean.columns), list(self.Y_train.columns))
        self.assertTrue(mean.index.equals(self.X_test.index))
        np.testing.assert_allclose(mean.to_numpy(), self.sk_mean.to_numpy(), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(std.to_numpy(), self.sk_std.to_numpy(), rtol=1e-9, atol=0)
        summary = gp.kernel_summary().iloc[0]
        sk_summary = self.sk.kernel_summary().iloc[0]
        self.assertAlmostEqual(
            summary['log_marginal_likelihood'] / sk_summary['log_marginal_likelihood'], 1.0, places=10
        )

    def test_own_optimization_reaches_the_sklearn_optimum(self):
        gp = GPyTorchDSSEGaussianProcess().fit(self.X_train, self.Y_train)
        summary = gp.kernel_summary().iloc[0]
        sk_summary = self.sk.kernel_summary().iloc[0]
        # GPyTorch must find at least the sklearn optimum (up to optimizer tolerance).
        self.assertGreaterEqual(
            summary['log_marginal_likelihood'], sk_summary['log_marginal_likelihood'] - 1e-3
        )
        mean, std = gp.predict(self.X_test)
        np.testing.assert_allclose(mean.to_numpy(), self.sk_mean.to_numpy(), rtol=0, atol=1e-6)
        np.testing.assert_allclose(std.to_numpy(), self.sk_std.to_numpy(), rtol=1e-2)
        self.assertEqual(len(gp.optimization_log()), 5)


if __name__ == '__main__':
    unittest.main()
