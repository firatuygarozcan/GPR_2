"""GPyTorch backend for the season-specific DSSE models (MODEL_BACKEND = "gpytorch").

``GPyTorchDSSEGaussianProcess`` has the same interface as
``dsse.DSSEGaussianProcess``: ``fit(X, Y)``, ``predict(X) -> (mean, std)`` per
bus, and ``kernel_summary()`` with the same columns.

Variant "shared" is the GPyTorch counterpart of the sklearn model:

* inputs standardized with a StandardScaler fitted on the training inputs,
* every bus output normalized by its own training mean/std (normalize_y=True),
* one ScaleKernel(RBFKernel) and one Gaussian noise variance shared by all
  bus outputs; the outputs are independent given these hyperparameters,
* the training covariance additionally gets the numerical jitter GPR_ALPHA,
* the predictive variance includes the Gaussian noise (likelihood variance),
  but not the jitter, exactly like sklearn's WhiteKernel inside the kernel.

The log marginal likelihood is the sum of the per-bus likelihoods, as in
sklearn's multi-output GaussianProcessRegressor. All bus outputs share one
covariance matrix, so it is factorized once and solved for all outputs at
once. (GPyTorch's ExactGP would repeat the covariance for every output.)
"""

import math
import time
from contextlib import contextmanager

import gpytorch
import numpy as np
import pandas as pd
import torch
from gpytorch.constraints import Interval
from gpytorch.utils.transforms import inv_sigmoid
from sklearn.preprocessing import StandardScaler

from config import (
    GPR_ALPHA,
    GPYTORCH_LBFGS_MAX_ITER,
    GPYTORCH_N_RESTARTS,
    GPYTORCH_SEED,
)
from dsse import build_kernel

DTYPE = torch.float64
HYPERPARAMETERS = ["outputscale", "lengthscale", "noise"]


@contextmanager
def float64_default():
    """GPyTorch creates constraint buffers in the default dtype; keep them float64."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(DTYPE)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def sklearn_hyperparameter_spec():
    """Initial values and bounds of the sklearn kernel, keyed by GPyTorch name.

    ConstantKernel.constant_value -> outputscale, RBF.length_scale ->
    lengthscale, WhiteKernel.noise_level -> Gaussian noise variance.
    """
    kernel = build_kernel()
    constant, rbf, white = kernel.k1.k1, kernel.k1.k2, kernel.k2
    return {
        "outputscale": (float(constant.constant_value), tuple(constant.constant_value_bounds)),
        "lengthscale": (float(rbf.length_scale), tuple(rbf.length_scale_bounds)),
        "noise": (float(white.noise_level), tuple(white.noise_level_bounds)),
    }


def normalization_scale(std):
    """sklearn's _handle_zeros_in_scale: constant targets keep a unit scale."""
    std = np.asarray(std, dtype=float).copy()
    std[std < 10 * np.finfo(std.dtype).eps] = 1.0
    return std


class LogInterval(Interval):
    """Bounded constraint that is uniform in log space.

    sklearn optimizes log-hyperparameters inside box bounds; this keeps the
    same geometry for GPyTorch: value = exp(log lo + (log hi - log lo) * sigmoid(raw)).
    """

    def transform(self, tensor):
        log_lower, log_upper = self.lower_bound.log(), self.upper_bound.log()
        return torch.exp(log_lower + (log_upper - log_lower) * torch.sigmoid(tensor))

    def inverse_transform(self, transformed_tensor):
        log_lower, log_upper = self.lower_bound.log(), self.upper_bound.log()
        return inv_sigmoid((transformed_tensor.log() - log_lower) / (log_upper - log_lower))


class SharedKernelGP(gpytorch.Module):
    """Exact GP: one shared ScaleKernel(RBFKernel) + Gaussian noise, T normalized outputs."""

    def __init__(self, train_x, train_y, spec, jitter=GPR_ALPHA):
        with float64_default():
            super().__init__()
            self.register_buffer("train_x", torch.as_tensor(train_x, dtype=DTYPE))
            self.register_buffer("train_y", torch.as_tensor(train_y, dtype=DTYPE))
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(lengthscale_constraint=LogInterval(*spec["lengthscale"][1])),
                outputscale_constraint=LogInterval(*spec["outputscale"][1]),
            )
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
                noise_constraint=LogInterval(*spec["noise"][1])
            )
        self.jitter = float(jitter)
        self.spec = spec
        self.set_hyperparameters(**{name: value for name, (value, _) in spec.items()})
        self.chol_ = None
        self.alpha_ = None

    def set_hyperparameters(self, outputscale, lengthscale, noise):
        self.initialize(
            **{
                "covar_module.outputscale": torch.tensor(outputscale, dtype=DTYPE),
                "covar_module.base_kernel.lengthscale": torch.tensor(lengthscale, dtype=DTYPE),
                "likelihood.noise_covar.noise": torch.tensor(noise, dtype=DTYPE),
            }
        )

    def hyperparameters(self):
        return {
            "outputscale": float(self.covar_module.outputscale.detach()),
            "lengthscale": float(self.covar_module.base_kernel.lengthscale.detach()),
            "noise": float(self.likelihood.noise.detach()),
        }

    @property
    def kernel_(self):
        """sklearn-style description, so existing log lines keep working."""
        h = self.hyperparameters()
        return (
            f"{math.sqrt(h['outputscale']):.3g}**2 * RBF(length_scale={h['lengthscale']:.3g}) "
            f"+ WhiteKernel(noise_level={h['noise']:.3g})  "
            "[GPyTorch ScaleKernel(RBFKernel) + GaussianLikelihood, shared]"
        )

    def _train_cholesky(self):
        n = self.train_x.shape[0]
        covariance = self.covar_module(self.train_x).to_dense()
        diagonal = self.likelihood.noise.reshape(()) + self.jitter
        covariance = covariance + diagonal * torch.eye(n, dtype=DTYPE)
        return torch.linalg.cholesky(covariance)

    def log_marginal_likelihood(self):
        """Sum over outputs of log N(y_j | 0, K + (noise + jitter) I)."""
        n, n_outputs = self.train_y.shape
        chol = self._train_cholesky()
        alpha = torch.cholesky_solve(self.train_y, chol)
        return (
            -0.5 * (self.train_y * alpha).sum()
            - n_outputs * torch.log(torch.diagonal(chol)).sum()
            - 0.5 * n * n_outputs * math.log(2 * math.pi)
        )

    def condition(self):
        """Cache the Cholesky factor and weights at the current hyperparameters."""
        with torch.no_grad():
            self.chol_ = self._train_cholesky()
            self.alpha_ = torch.cholesky_solve(self.train_y, self.chol_)

    def predict(self, x):
        """Normalized predictive mean (m, T) and variance (m,) including noise."""
        if self.chol_ is None:
            self.condition()
        x = torch.as_tensor(x, dtype=DTYPE)
        with torch.no_grad():
            cross = self.covar_module(x, self.train_x).to_dense()
            mean = cross @ self.alpha_
            v = torch.linalg.solve_triangular(self.chol_, cross.T, upper=False)
            prior = self.covar_module(x, diag=True)
            variance = prior - (v**2).sum(dim=0) + self.likelihood.noise.reshape(())
            # sklearn sets slightly negative variances to zero as well.
            variance = torch.clamp(variance, min=0.0)
        return mean.numpy(), variance.numpy()

    def optimize(self, starts, max_iter=GPYTORCH_LBFGS_MAX_ITER):
        """Maximize the log marginal likelihood with L-BFGS from several starts."""
        n, n_outputs = self.train_y.shape
        records = []
        best_state, best_lml = None, -math.inf
        for index, start in enumerate(starts):
            self.set_hyperparameters(**start)
            optimizer = torch.optim.LBFGS(
                self.parameters(),
                lr=1.0,
                max_iter=max_iter,
                tolerance_grad=1e-10,
                tolerance_change=1e-14,
                history_size=50,
                line_search_fn="strong_wolfe",
            )
            n_evaluations = 0

            def closure():
                nonlocal n_evaluations
                n_evaluations += 1
                optimizer.zero_grad()
                loss = -self.log_marginal_likelihood() / (n * n_outputs)
                loss.backward()
                return loss

            started = time.perf_counter()
            status = "ok"
            try:
                optimizer.step(closure)
                with torch.no_grad():
                    lml = float(self.log_marginal_likelihood())
                if not math.isfinite(lml):
                    status = "non_finite"
            except (RuntimeError, torch.linalg.LinAlgError) as exc:
                lml, status = -math.inf, f"failed: {type(exc).__name__}"
            records.append(
                {
                    "start_index": index,
                    **{f"start_{name}": value for name, value in start.items()},
                    **{f"final_{name}": value for name, value in self.hyperparameters().items()},
                    "log_marginal_likelihood": lml,
                    "n_evaluations": n_evaluations,
                    "seconds": time.perf_counter() - started,
                    "status": status,
                }
            )
            if status == "ok" and lml > best_lml:
                best_lml = lml
                best_state = {key: value.clone() for key, value in self.state_dict().items()}
        if best_state is None:
            raise RuntimeError("GPyTorch hyperparameter optimization failed from every start.")
        self.load_state_dict(best_state)
        return records


def optimization_starts(spec, n_restarts=GPYTORCH_N_RESTARTS, seed=GPYTORCH_SEED):
    """sklearn's initial values first, then log-uniform random draws inside the bounds."""
    rng = np.random.default_rng(seed)
    starts = [{name: value for name, (value, _) in spec.items()}]
    for _ in range(int(n_restarts)):
        starts.append(
            {
                name: float(np.exp(rng.uniform(np.log(bounds[0]), np.log(bounds[1]))))
                for name, (_, bounds) in spec.items()
            }
        )
    return starts


class GPyTorchDSSEGaussianProcess:
    """GPyTorch season model with the interface of dsse.DSSEGaussianProcess."""

    def __init__(self, variant="shared"):
        if variant != "shared":
            raise NotImplementedError(
                f"GP_VARIANT {variant!r} is not implemented yet; only 'shared' is available."
            )
        self.variant = variant
        self.scaler = StandardScaler()
        self.model = None
        self.target_columns = None
        self.y_mean_ = None
        self.y_std_ = None
        self.optimization_ = []
        self.fixed_hyperparameters_ = False

    @staticmethod
    def _seed():
        torch.manual_seed(GPYTORCH_SEED)
        np.random.seed(GPYTORCH_SEED)

    def fit(self, X, Y, hyperparameters=None):
        """Fit like DSSEGaussianProcess; ``hyperparameters`` skips optimization.

        ``hyperparameters`` is a dict with outputscale, lengthscale and noise
        (sklearn's constant_value, length_scale and noise_level).
        """
        if not isinstance(Y, pd.DataFrame):
            Y = pd.DataFrame(Y, index=X.index)
        if len(X) != len(Y):
            raise ValueError("X and Y must contain the same number of samples.")

        self._seed()
        self.target_columns = list(Y.columns)
        print("\nTraining one season-specific multi-output GPyTorch GP (shared hyperparameters)")
        X_scaled = self.scaler.fit_transform(X)
        targets = Y.to_numpy(float)
        self.y_mean_ = targets.mean(axis=0)
        self.y_std_ = normalization_scale(targets.std(axis=0))
        normalized = (targets - self.y_mean_) / self.y_std_

        spec = sklearn_hyperparameter_spec()
        self.model = SharedKernelGP(X_scaled, normalized, spec)
        if hyperparameters is None:
            self.optimization_ = self.model.optimize(optimization_starts(spec))
            self.fixed_hyperparameters_ = False
        else:
            self.model.set_hyperparameters(**hyperparameters)
            self.optimization_ = []
            self.fixed_hyperparameters_ = True
        self.model.condition()

        print(
            f"  fitted one GPyTorch GP: {X.shape[1]} inputs → "
            f"{Y.shape[1]} voltage outputs"
        )
        return self

    def predict(self, X):
        if self.model is None or self.target_columns is None:
            raise RuntimeError("Call fit() before predict().")

        mean, variance = self.model.predict(self.scaler.transform(X))
        mean = mean * self.y_std_ + self.y_mean_
        std = np.sqrt(variance)[:, None] * self.y_std_
        return (
            pd.DataFrame(mean, index=X.index, columns=self.target_columns),
            pd.DataFrame(std, index=X.index, columns=self.target_columns),
        )

    def kernel_summary(self):
        if self.model is None:
            return pd.DataFrame()

        values = self.model.hyperparameters()
        with torch.no_grad():
            lml = float(self.model.log_marginal_likelihood())
        best = max(
            (row for row in self.optimization_ if row["status"] == "ok"),
            key=lambda row: row["log_marginal_likelihood"],
            default=None,
        )
        return pd.DataFrame(
            [
                {
                    "model_type": "single_multioutput_gpr",
                    "n_outputs": len(self.target_columns),
                    "optimized_kernel": str(self.model.kernel_),
                    "constant_value": values["outputscale"],
                    "rbf_length_scale": values["lengthscale"],
                    "white_noise_level": values["noise"],
                    "alpha": self.model.jitter,
                    "log_marginal_likelihood": lml,
                    "backend": "gpytorch",
                    "gp_variant": self.variant,
                    "fixed_hyperparameters": self.fixed_hyperparameters_,
                    "n_starts": len(self.optimization_),
                    "best_start_index": None if best is None else best["start_index"],
                    "n_failed_starts": sum(row["status"] != "ok" for row in self.optimization_),
                }
            ]
        )

    def optimization_log(self):
        return pd.DataFrame(self.optimization_)
