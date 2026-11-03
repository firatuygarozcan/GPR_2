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

Variant "icm" (intrinsic coregionalization model) uses the same inputs and
output normalization with MultitaskKernel(ScaleKernel(RBFKernel), rank r) and
a MultitaskGaussianLikelihood with one noise variance per bus; see ICMKernelGP.
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
    GP_ICM_RANK,
    GPR_ALPHA,
    GPYTORCH_ICM_N_STARTS,
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


class _ICMLogMarginalLikelihood(torch.autograd.Function):
    """log N(vec(Y) | 0, B ⊗ K + D ⊗ I) with an analytic backward pass.

    Y is (n, T), K the (n, n) data covariance, B the (T, T) task covariance and
    d the T noise variances. With B~ = D^-1/2 B D^-1/2 = U_B L_B U_B^T and
    K = U_K L_K U_K^T, the covariance is (D^1/2 ⊗ I)(U_B ⊗ U_K)(L_B ⊗ L_K + I)
    (U_B ⊗ U_K)^T (D^1/2 ⊗ I), so the solve and the log-determinant only need
    the two small eigendecompositions. The gradient is the usual
    1/2 tr((a a^T - Sigma^-1) dSigma) written in the same eigenbasis; it never
    differentiates eigenvectors, which is unstable for the many nearly equal
    small eigenvalues of K.
    """

    @staticmethod
    def forward(ctx, K, B, d, Y):
        n, n_tasks = Y.shape
        lam_k, u_k = torch.linalg.eigh(K)
        lam_k = lam_k.clamp(min=0.0)
        s = d.rsqrt()
        lam_b, u_b = torch.linalg.eigh(s[:, None] * B * s[None, :])
        lam_b = lam_b.clamp(min=0.0)
        denominator = lam_k[:, None] * lam_b[None, :] + 1.0
        rotated = u_k.T @ (Y * s) @ u_b
        weighted = rotated / denominator
        quadratic = (rotated * weighted).sum()
        logdet = n * torch.log(d).sum() + torch.log(denominator).sum()
        A = (u_k @ weighted @ u_b.T) * s  # Sigma^-1 vec(Y), reshaped to (n, T)
        ctx.save_for_backward(K, B, d, A, lam_k, u_k, lam_b, u_b, denominator)
        return -0.5 * quadratic - 0.5 * logdet - 0.5 * n * n_tasks * math.log(2 * math.pi)

    @staticmethod
    def backward(ctx, grad):
        K, B, d, A, lam_k, u_k, lam_b, u_b, denominator = ctx.saved_tensors
        s = d.rsqrt()
        c = (lam_b[None, :] / denominator).sum(dim=1)
        grad_k = 0.5 * (A @ B @ A.T - (u_k * c) @ u_k.T)
        e = (lam_k[:, None] / denominator).sum(dim=0)
        grad_b = 0.5 * (A.T @ K @ A - s[:, None] * ((u_b * e) @ u_b.T) * s[None, :])
        h = (1.0 / denominator).sum(dim=0)
        grad_d = 0.5 * ((A**2).sum(dim=0) - ((u_b**2) @ h) / d)
        return grad * grad_k, grad * grad_b, grad * grad_d, None


class ICMKernelGP(gpytorch.Module):
    """Exact ICM GP: MultitaskKernel(ScaleKernel(RBFKernel), rank r) + per-bus noise.

    Covariance of the normalized outputs: B ⊗ K + D ⊗ I with B = W W^T + diag(v)
    (IndexKernel) and D the per-bus noise of a rank-0 MultitaskGaussianLikelihood.
    Inference uses the Kronecker eigendecomposition (see
    _ICMLogMarginalLikelihood) instead of GPyTorch's generic multitask path,
    which would differentiate eigenvectors during training and build an
    (n T) x (m T) solve for the predictive variance.
    """

    def __init__(self, train_x, train_y, spec, rank, jitter=GPR_ALPHA):
        n_tasks = int(np.asarray(train_y).shape[1])
        with float64_default():
            super().__init__()
            self.register_buffer("train_x", torch.as_tensor(train_x, dtype=DTYPE))
            self.register_buffer("train_y", torch.as_tensor(train_y, dtype=DTYPE))
            self.covar_module = gpytorch.kernels.MultitaskKernel(
                gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel(lengthscale_constraint=LogInterval(*spec["lengthscale"][1])),
                    outputscale_constraint=LogInterval(*spec["outputscale"][1]),
                ),
                num_tasks=n_tasks,
                rank=int(rank),
            )
            self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
                num_tasks=n_tasks,
                rank=0,
                has_global_noise=False,
                noise_constraint=LogInterval(*spec["noise"][1]),
            )
        self.rank = int(rank)
        self.jitter = float(jitter)
        self.spec = spec
        self.cache_ = None

    @property
    def data_kernel(self):
        return self.covar_module.data_covar_module

    def task_covariance(self):
        return self.covar_module.task_covar_module._eval_covar_matrix()

    def set_hyperparameters(self, outputscale, lengthscale, task_noises, covar_factor, task_var):
        n_tasks = self.train_y.shape[1]
        self.initialize(
            **{
                "covar_module.data_covar_module.outputscale": torch.tensor(outputscale, dtype=DTYPE),
                "covar_module.data_covar_module.base_kernel.lengthscale": torch.tensor(lengthscale, dtype=DTYPE),
                "covar_module.task_covar_module.covar_factor": torch.as_tensor(covar_factor, dtype=DTYPE),
                "covar_module.task_covar_module.var": torch.as_tensor(task_var, dtype=DTYPE).expand(n_tasks),
                "likelihood.task_noises": torch.as_tensor(task_noises, dtype=DTYPE).expand(n_tasks),
            }
        )

    def hyperparameters(self):
        with torch.no_grad():
            return {
                "outputscale": float(self.data_kernel.outputscale),
                "lengthscale": float(self.data_kernel.base_kernel.lengthscale),
                "task_noises": self.likelihood.task_noises.detach().numpy().copy(),
                "task_covariance": self.task_covariance().detach().numpy().copy(),
            }

    @property
    def kernel_(self):
        h = self.hyperparameters()
        signal = h["outputscale"] * np.diag(h["task_covariance"])
        return (
            f"ICM rank {self.rank}: ScaleKernel(RBF(length_scale={h['lengthscale']:.3g})) ⊗ B, "
            f"signal variance median={np.median(signal):.3g}, "
            f"task noise median={np.median(h['task_noises']):.3g} "
            f"[{h['task_noises'].min():.2g}, {h['task_noises'].max():.2g}]"
        )

    def _noise(self):
        return self.likelihood.task_noises

    def log_marginal_likelihood(self):
        K = self.data_kernel(self.train_x).to_dense()
        return _ICMLogMarginalLikelihood.apply(K, self.task_covariance(), self._noise() + self.jitter, self.train_y)

    def condition(self):
        """Cache the eigendecompositions and weights at the current hyperparameters."""
        with torch.no_grad():
            K = self.data_kernel(self.train_x).to_dense()
            B = self.task_covariance()
            d = self._noise() + self.jitter
            s = d.rsqrt()
            lam_k, u_k = torch.linalg.eigh(K)
            lam_k = lam_k.clamp(min=0.0)
            lam_b, u_b = torch.linalg.eigh(s[:, None] * B * s[None, :])
            lam_b = lam_b.clamp(min=0.0)
            denominator = lam_k[:, None] * lam_b[None, :] + 1.0
            weighted = (u_k.T @ (self.train_y * s) @ u_b) / denominator
            A = (u_k @ weighted @ u_b.T) * s
            self.cache_ = {
                "u_k": u_k,
                "mean_weights": A @ B,  # predictive mean = k(x, X) A B
                "inverse_denominator": 1.0 / denominator,
                "task_projection_sq": (u_b.T @ (s[:, None] * B)) ** 2,
                "prior_task_variance": torch.diagonal(B).clone(),
            }

    def predict(self, x):
        """Normalized predictive mean (m, T) and variance (m, T) including bus noise.

        variance[m, t] = k(x, x) B_tt - sum_ij P_im^2 Q_jt^2 / (l_K,i l_B,j + 1) + d_t
        with P = U_K^T k(X, x) and Q = U_B^T D^-1/2 B; the jitter is excluded.
        """
        if self.cache_ is None:
            self.condition()
        cache = self.cache_
        x = torch.as_tensor(x, dtype=DTYPE)
        with torch.no_grad():
            cross = self.data_kernel(x, self.train_x).to_dense()
            mean = cross @ cache["mean_weights"]
            projected_sq = (cross @ cache["u_k"]) ** 2
            reduction = projected_sq @ cache["inverse_denominator"] @ cache["task_projection_sq"]
            prior = self.data_kernel(x, diag=True)[:, None] * cache["prior_task_variance"][None, :]
            variance = torch.clamp(prior - reduction, min=0.0) + self._noise()[None, :]
        return mean.numpy(), variance.numpy()

    def optimize(self, starts, max_iter=GPYTORCH_LBFGS_MAX_ITER):
        """Maximize the log marginal likelihood with L-BFGS from the given starts."""
        n, n_tasks = self.train_y.shape
        records = []
        best_state, best_lml = None, -math.inf
        for index, start in enumerate(starts):
            label = start.get("label", str(index))
            self.set_hyperparameters(**{key: value for key, value in start.items() if key != "label"})
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
                loss = -self.log_marginal_likelihood() / (n * n_tasks)
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
            h = self.hyperparameters()
            records.append(
                {
                    "stage": "icm",
                    "start_index": index,
                    "start_label": label,
                    "start_outputscale": start["outputscale"],
                    "start_lengthscale": start["lengthscale"],
                    "final_outputscale": h["outputscale"],
                    "final_lengthscale": h["lengthscale"],
                    "final_task_noise_median": float(np.median(h["task_noises"])),
                    "final_signal_variance_median": float(
                        np.median(h["outputscale"] * np.diag(h["task_covariance"]))
                    ),
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
            raise RuntimeError("ICM hyperparameter optimization failed from every start.")
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


def icm_starts(shared_hyperparameters, n_tasks, rank, n_starts=GPYTORCH_ICM_N_STARTS, seed=GPYTORCH_SEED):
    """ICM starts: the fitted shared model (B close to I), then sklearn's initial values.

    With B = I and equal bus noise the ICM likelihood equals the shared one, so
    the first start begins at the shared optimum. covar_factor gets a small
    seeded perturbation because W = 0 is a stationary point of B = W W^T + diag(v).
    """
    rng = np.random.default_rng(seed)
    initial = sklearn_hyperparameter_spec()
    candidates = [
        {
            "label": "shared_optimum",
            "outputscale": shared_hyperparameters["outputscale"],
            "lengthscale": shared_hyperparameters["lengthscale"],
            "task_noises": shared_hyperparameters["noise"],
        },
        {
            "label": "sklearn_initial",
            "outputscale": initial["outputscale"][0],
            "lengthscale": initial["lengthscale"][0],
            "task_noises": initial["noise"][0],
        },
    ]
    starts = []
    for candidate in candidates[: int(n_starts)]:
        starts.append({**candidate, "covar_factor": 0.1 * rng.standard_normal((n_tasks, rank)), "task_var": 1.0})
    return starts


class GPyTorchDSSEGaussianProcess:
    """GPyTorch season model with the interface of dsse.DSSEGaussianProcess."""

    VARIANTS = ("shared", "icm")

    def __init__(self, variant="shared", rank=GP_ICM_RANK):
        if variant not in self.VARIANTS:
            raise NotImplementedError(
                f"GP_VARIANT {variant!r} is not implemented; available: {', '.join(self.VARIANTS)}."
            )
        self.variant = variant
        self.rank = int(rank) if variant == "icm" else None
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

        For the shared variant ``hyperparameters`` is a dict with outputscale,
        lengthscale and noise (sklearn's constant_value, length_scale and
        noise_level). The ICM variant first fits the shared model and starts
        its own optimization from there (see icm_starts).
        """
        if not isinstance(Y, pd.DataFrame):
            Y = pd.DataFrame(Y, index=X.index)
        if len(X) != len(Y):
            raise ValueError("X and Y must contain the same number of samples.")

        self._seed()
        self.target_columns = list(Y.columns)
        label = "shared hyperparameters" if self.variant == "shared" else f"ICM rank {self.rank}"
        print(f"\nTraining one season-specific multi-output GPyTorch GP ({label})")
        X_scaled = self.scaler.fit_transform(X)
        targets = Y.to_numpy(float)
        self.y_mean_ = targets.mean(axis=0)
        self.y_std_ = normalization_scale(targets.std(axis=0))
        normalized = (targets - self.y_mean_) / self.y_std_

        spec = sklearn_hyperparameter_spec()
        shared = SharedKernelGP(X_scaled, normalized, spec)
        if hyperparameters is None:
            shared_log = shared.optimize(optimization_starts(spec))
            self.fixed_hyperparameters_ = False
        else:
            if self.variant != "shared":
                raise ValueError("Fixed hyperparameters are only supported for the shared variant.")
            shared.set_hyperparameters(**hyperparameters)
            shared_log = []
            self.fixed_hyperparameters_ = True

        if self.variant == "shared":
            self.model = shared
            self.optimization_ = shared_log
        else:
            self.model = ICMKernelGP(X_scaled, normalized, spec, rank=self.rank)
            starts = icm_starts(shared.hyperparameters(), normalized.shape[1], self.rank)
            icm_log = self.model.optimize(starts)
            self.optimization_ = [{"stage": "shared_warm_start", **row} for row in shared_log] + icm_log
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
        if variance.ndim == 1:
            # shared variant: one normalized variance per test point for all buses
            variance = variance[:, None]
        std = np.sqrt(variance) * self.y_std_
        return (
            pd.DataFrame(mean, index=X.index, columns=self.target_columns),
            pd.DataFrame(std, index=X.index, columns=self.target_columns),
        )

    def _best_start(self, stage):
        return max(
            (
                row
                for row in self.optimization_
                if row["status"] == "ok" and row.get("stage", stage) == stage
            ),
            key=lambda row: row["log_marginal_likelihood"],
            default=None,
        )

    def kernel_summary(self):
        if self.model is None:
            return pd.DataFrame()

        values = self.model.hyperparameters()
        with torch.no_grad():
            lml = float(self.model.log_marginal_likelihood())
        stage = "shared" if self.variant == "shared" else "icm"
        best = self._best_start(stage)
        row = {
            "model_type": "single_multioutput_gpr" if self.variant == "shared" else "icm_multitask_gpr",
            "n_outputs": len(self.target_columns),
            "optimized_kernel": str(self.model.kernel_),
            "constant_value": values["outputscale"],
            "rbf_length_scale": values["lengthscale"],
            # ICM has one noise variance per bus; the summary reports the median.
            "white_noise_level": (
                values["noise"] if self.variant == "shared" else float(np.median(values["task_noises"]))
            ),
            "alpha": self.model.jitter,
            "log_marginal_likelihood": lml,
            "backend": "gpytorch",
            "gp_variant": self.variant,
            "icm_rank": self.rank,
            "fixed_hyperparameters": self.fixed_hyperparameters_,
            "n_starts": sum(row.get("stage", stage) == stage for row in self.optimization_),
            "best_start_index": None if best is None else best["start_index"],
            "n_failed_starts": sum(
                row["status"] != "ok" and row.get("stage", stage) == stage for row in self.optimization_
            ),
        }
        if self.variant == "icm":
            signal = values["outputscale"] * np.diag(values["task_covariance"])
            row.update(
                {
                    "task_noise_min": float(values["task_noises"].min()),
                    "task_noise_max": float(values["task_noises"].max()),
                    "signal_variance_median": float(np.median(signal)),
                }
            )
        return pd.DataFrame([row])

    def optimization_log(self):
        return pd.DataFrame(self.optimization_)

    def task_parameters(self):
        """Per-bus ICM parameters (normalized and in pu^2); empty for the shared variant."""
        if self.variant != "icm" or self.model is None:
            return pd.DataFrame()
        values = self.model.hyperparameters()
        B = values["task_covariance"]
        return pd.DataFrame(
            {
                "target_column": self.target_columns,
                "y_train_std_pu": self.y_std_,
                "task_noise": values["task_noises"],
                "task_noise_pu2": values["task_noises"] * self.y_std_**2,
                "task_covariance_diag": np.diag(B),
                "signal_variance": values["outputscale"] * np.diag(B),
            }
        )

    def task_covariance(self):
        """Long table of the learned (normalized) task covariance B; empty for shared."""
        if self.variant != "icm" or self.model is None:
            return pd.DataFrame()
        B = self.model.hyperparameters()["task_covariance"]
        columns = np.asarray(self.target_columns)
        rows, cols = np.meshgrid(np.arange(len(columns)), np.arange(len(columns)), indexing="ij")
        return pd.DataFrame(
            {
                "target_column_i": columns[rows.ravel()],
                "target_column_j": columns[cols.ravel()],
                "task_covariance": B.ravel(),
                "task_correlation": (B / np.sqrt(np.outer(np.diag(B), np.diag(B)))).ravel(),
            }
        )

    def extra_outputs(self):
        """Additional per-season tables saved next to the run outputs."""
        outputs = {"optimization_log": self.optimization_log()}
        if self.variant == "icm":
            outputs["task_parameters"] = self.task_parameters()
            outputs["task_covariance"] = self.task_covariance()
        return outputs
