from __future__ import annotations
import math
from typing import Optional, Tuple

import torch


class SVGPSurrogate:
    """Sparse Variational Gaussian Process surrogate over grid mutation parameters.

    Implements the standard SVGP formulation (Titsias, 2009; Hensman et al.,
    2013) with a variational distribution over the inducing variables, giving

        mu(u)      = k(u,Z)^T K_ZZ^-1 m
        sigma^2(u) = k(u,u) + k(u,Z)^T K_ZZ^-1 (S - K_ZZ) K_ZZ^-1 k(Z,u)

    The variational parameters and kernel hyperparameters are fitted by
    maximizing the ELBO with Adam, one step per search step.

    Internally the model is whitened: with K_ZZ = L L^T we parameterize
    v_w = L^-1 v, so that p(v_w) = N(0, I) and q(v_w) = N(m_w, S_w). This is a
    reparameterization of the same model -- it corresponds exactly to
    m = L m_w and S = L S_w L^T in the equations above, and leaves the
    predictive distribution unchanged -- but it avoids forming K_ZZ^-1
    explicitly, which is badly conditioned when inducing points are close
    relative to the kernel lengthscale.

    The number of active inducing points is m_act = min(m, |D|), so it never
    exceeds the observations available in the current target-specific search.
    """

    def __init__(
        self,
        dim: int,
        m: int,
        device: torch.device,
        bound: float = 0.1,
        lengthscale: Optional[float] = None,
        outputscale: float = 1.0,
        noise: float = 0.01,
        lr: float = 0.05,
        jitter: float = 1e-6,
    ):
        self.dim = dim
        self.m = m
        self.device = device
        self.bound = bound
        self.jitter = jitter
        self.lr = lr
        self.dtype = torch.float64

        if lengthscale is None:
            lengthscale = max(bound, 1e-3)

        self.Z = ((torch.rand(m, dim, device=device, dtype=self.dtype) * 2.0) - 1.0) * bound

        def _param(value: float) -> torch.Tensor:
            return torch.tensor(math.log(value), device=device, dtype=self.dtype, requires_grad=True)

        self.raw_lengthscale = _param(lengthscale)
        self.raw_outputscale = _param(outputscale)
        self.raw_noise = _param(noise)

        self.q_mean = torch.zeros(m, device=device, dtype=self.dtype, requires_grad=True)
        self.q_sqrt = torch.eye(m, device=device, dtype=self.dtype).clone().requires_grad_(True)

        self.optimizer = torch.optim.Adam(
            [self.q_mean, self.q_sqrt, self.raw_lengthscale, self.raw_outputscale, self.raw_noise],
            lr=lr,
        )

        self.X: Optional[torch.Tensor] = None
        self.y: Optional[torch.Tensor] = None

    @property
    def m_act(self) -> int:
        if self.X is None:
            return 0
        return min(self.m, int(self.X.size(0)))

    def _kernel(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        lengthscale = self.raw_lengthscale.exp()
        outputscale = self.raw_outputscale.exp()
        d2 = torch.cdist(A, B) ** 2
        return outputscale * torch.exp(-0.5 * d2 / (lengthscale ** 2 + 1e-12))

    def _active(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        M = self.m_act
        return self.Z[:M], self.q_mean[:M], self.q_sqrt[:M, :M].tril()

    def _posterior(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Variational predictive mean and variance at X."""
        Z, q_mean, q_sqrt = self._active()
        M = Z.size(0)

        Kzz = self._kernel(Z, Z) + self.jitter * torch.eye(M, device=self.device, dtype=self.dtype)
        Lz = torch.linalg.cholesky(Kzz)

        Kxz = self._kernel(X, Z)
        B = torch.linalg.solve_triangular(Lz, Kxz.transpose(-1, -2), upper=False).transpose(-1, -2)

        mean = B @ q_mean

        Kxx_diag = self.raw_outputscale.exp().expand(X.size(0))

        k_tilde = Kxx_diag - (B ** 2).sum(dim=1)

        BS = B @ q_sqrt
        var = k_tilde + (BS ** 2).sum(dim=1)
        return mean, var.clamp_min(1e-10)

    def _kl_divergence(self) -> torch.Tensor:
        """KL(q(v_w) || N(0, I)) under the whitened parameterization."""
        _, q_mean, q_sqrt = self._active()
        M = q_mean.size(0)
        diag = torch.diagonal(q_sqrt).abs() + 1e-12
        trace_term = (q_sqrt ** 2).sum()
        mahalanobis = (q_mean ** 2).sum()
        logdet_s = 2.0 * torch.log(diag).sum()
        return 0.5 * (trace_term + mahalanobis - M - logdet_s)

    def add_observation(self, u: torch.Tensor, y: float) -> None:
        u = u.detach().reshape(1, -1).to(device=self.device, dtype=self.dtype)
        y_t = torch.tensor([float(y)], device=self.device, dtype=self.dtype)
        self.X = u if self.X is None else torch.cat([self.X, u], dim=0)
        self.y = y_t if self.y is None else torch.cat([self.y, y_t], dim=0)

    def fit_step(self) -> Optional[float]:
        """Take one Adam step on the ELBO. Called once per search step."""
        if self.X is None or self.X.size(0) == 0:
            return None

        self.optimizer.zero_grad(set_to_none=True)
        mean, var = self._posterior(self.X)
        noise = self.raw_noise.exp()

        residual = self.y - mean
        expected_ll = -0.5 * (
            math.log(2.0 * math.pi)
            + torch.log(noise)
            + (residual ** 2) / noise
            + var / noise
        ).sum()

        elbo = expected_ll - self._kl_divergence()
        (-elbo).backward()
        self.optimizer.step()
        return float(elbo.detach().item())

    @torch.no_grad()
    def predict(self, U: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predictive mean and standard deviation for candidate parameters."""
        if self.X is None or self.X.size(0) == 0:
            n = U.size(0)
            return torch.zeros(n, device=self.device), torch.ones(n, device=self.device)
        mean, var = self._posterior(U.to(device=self.device, dtype=self.dtype))
        return mean.float(), var.sqrt().float()
