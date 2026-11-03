"""GPyTorch backend: interface and numerical equivalence with the sklearn model.

Small synthetic data only; no SimBench data, no experiment outputs.
"""
from pathlib import Path
import sys
import unittest

import gpytorch
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'seasonal'))
import dsse
from config import NETWORK
from gpytorch_models import (
    GPyTorchDSSEGaussianProcess,
    ICMKernelGP,
    SharedKernelGP,
    sklearn_hyperparameter_spec,
)


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


def small_icm(rank=2, n=25, n_tasks=4, seed=5):
    """A small ICM model with random (non-trivial) hyperparameters."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    y = rng.normal(size=(n, n_tasks))
    model = ICMKernelGP(x, y, sklearn_hyperparameter_spec(), rank=rank)
    model.set_hyperparameters(
        outputscale=1.7,
        lengthscale=1.3,
        task_noises=rng.uniform(0.05, 0.3, size=n_tasks),
        covar_factor=rng.normal(size=(n_tasks, rank)),
        task_var=0.4,
    )
    return model, rng


def dense_icm_covariance(model):
    """Dense Sigma = B kron K + D kron I (task-major ordering) built by autograd-friendly ops."""
    K = model.data_kernel(model.train_x).to_dense()
    B = model.task_covariance()
    d = model.likelihood.task_noises + model.jitter
    n = K.shape[0]
    return torch.kron(B, K) + torch.kron(torch.diag(d), torch.eye(n, dtype=K.dtype))


class ICMVariantTests(unittest.TestCase):
    def test_log_marginal_likelihood_and_gradient_match_dense_computation(self):
        model, _ = small_icm()
        lml = model.log_marginal_likelihood()
        lml.backward()
        ours = {name: p.grad.clone() for name, p in model.named_parameters()}
        model.zero_grad()

        sigma = dense_icm_covariance(model)
        y = model.train_y.T.reshape(-1)  # vec(Y), task-major
        dense = torch.distributions.MultivariateNormal(torch.zeros_like(y), covariance_matrix=sigma).log_prob(y)
        dense.backward()
        self.assertAlmostEqual(float(lml), float(dense), places=9)
        for name, p in model.named_parameters():
            np.testing.assert_allclose(ours[name].numpy(), p.grad.numpy(), rtol=1e-7, atol=1e-9, err_msg=name)

    def test_predictions_match_gpytorch_exact_multitask_gp(self):
        model, rng = small_icm(rank=2)
        x_test = torch.as_tensor(rng.normal(size=(7, 3)))
        mean, variance = model.predict(x_test)

        class ExactICM(gpytorch.models.ExactGP):
            def __init__(self, x, y, likelihood, covar_module):
                super().__init__(x, y, likelihood)
                self.mean_module = gpytorch.means.MultitaskMean(gpytorch.means.ZeroMean(), num_tasks=y.shape[1])
                self.covar_module = covar_module

            def forward(self, x):
                return gpytorch.distributions.MultitaskMultivariateNormal(
                    self.mean_module(x), self.covar_module(x)
                )

        likelihood = model.likelihood
        reference = ExactICM(model.train_x, model.train_y, likelihood, model.covar_module).double()
        reference.eval(), likelihood.eval()
        with torch.no_grad(), gpytorch.settings.max_cholesky_size(10**6), gpytorch.settings.fast_computations(
            False, False, False
        ):
            predictive = likelihood(reference(x_test))
            np.testing.assert_allclose(mean, predictive.mean.numpy(), rtol=1e-8, atol=1e-10)
            # The reference adds the jitter-free task noise as well (GPR_ALPHA is only in training).
            np.testing.assert_allclose(variance, predictive.variance.numpy(), rtol=1e-6, atol=1e-9)

    def test_identity_task_covariance_reduces_to_the_shared_model(self):
        rng = np.random.default_rng(11)
        x, y = rng.normal(size=(30, 3)), rng.normal(size=(30, 5))
        spec = sklearn_hyperparameter_spec()
        shared = SharedKernelGP(x, y, spec)
        shared.set_hyperparameters(outputscale=2.0, lengthscale=0.8, noise=0.05)
        icm = ICMKernelGP(x, y, spec, rank=1)
        icm.set_hyperparameters(
            outputscale=2.0, lengthscale=0.8, task_noises=0.05, covar_factor=np.zeros((5, 1)), task_var=1.0
        )
        self.assertAlmostEqual(float(icm.log_marginal_likelihood()), float(shared.log_marginal_likelihood()), places=8)
        shared_mean, shared_var = shared.predict(x[:4])
        icm_mean, icm_var = icm.predict(x[:4])
        np.testing.assert_allclose(icm_mean, shared_mean, rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(icm_var, np.repeat(shared_var[:, None], 5, axis=1), rtol=1e-9)

    def test_icm_wrapper_interface_and_warm_start(self):
        X_train, Y_train, X_test = synthetic_data()
        shared = GPyTorchDSSEGaussianProcess().fit(X_train, Y_train)
        icm = GPyTorchDSSEGaussianProcess(variant='icm', rank=2).fit(X_train, Y_train)
        mean, std = icm.predict(X_test)
        self.assertEqual(mean.shape, (len(X_test), Y_train.shape[1]))
        self.assertTrue((std.to_numpy() > 0).all())
        # Starting at the shared optimum, ICM can only improve the likelihood.
        self.assertGreaterEqual(
            icm.kernel_summary().iloc[0]['log_marginal_likelihood'],
            shared.kernel_summary().iloc[0]['log_marginal_likelihood'] - 1e-6,
        )
        outputs = icm.extra_outputs()
        self.assertEqual(set(outputs), {'optimization_log', 'task_parameters', 'task_covariance'})
        self.assertEqual(len(outputs['task_covariance']), Y_train.shape[1] ** 2)


if __name__ == '__main__':
    unittest.main()
