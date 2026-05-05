"""GPyTorch adapter — independent SVGP per continuous node.

Cannot represent joint factorization with discrete evidence; such queries
return ``not_supported`` per ``BaselineAdapter`` contract.
"""
from __future__ import annotations

import torch

from benchmarking.baselines.base import BaselineAdapter
from benchmarking.domains.base import BenchmarkProblem, Query


class GPyTorchAdapter(BaselineAdapter):
    name = "gpytorch"
    supports = {"continuous"}

    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.gps: dict = {}
        self.problem: BenchmarkProblem | None = None

    def fit(self, problem: BenchmarkProblem) -> None:
        try:
            import gpytorch
        except ImportError as e:
            raise ImportError("GPyTorchAdapter needs gpytorch: pip install gpytorch") from e

        self.problem = problem
        self.gps = {}
        # One independent SVGP per continuous leaf node, with parents as features.
        for node, (kind, _) in problem.variables.items():
            if kind != "continuous":
                continue
            x = problem.train_data[node].to(self.device).float()
            parents = [p for p, c in problem.dag if c == node]
            if not parents:
                # Root: just store mean/std
                self.gps[node] = {"mean": x.mean(0), "std": x.std(0).clamp_min(1e-3)}
                continue
            pa = torch.cat([problem.train_data[p].to(self.device).float()
                            for p in parents], dim=-1)
            # Lightweight SVGP — tiny inducing set for benchmark speed
            n_ind = min(64, pa.shape[0])
            ind = pa[:n_ind].clone()

            class _GP(gpytorch.models.ApproximateGP):
                def __init__(self, ind):
                    var_dist = gpytorch.variational.CholeskyVariationalDistribution(ind.size(0))
                    var_strat = gpytorch.variational.VariationalStrategy(
                        self, ind, var_dist, learn_inducing_locations=True,
                    )
                    super().__init__(var_strat)
                    self.mean_module = gpytorch.means.ConstantMean()
                    self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
                def forward(self, x):
                    return gpytorch.distributions.MultivariateNormal(
                        self.mean_module(x), self.covar_module(x),
                    )

            gp = _GP(ind).to(self.device)
            lik = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
            opt = torch.optim.Adam(list(gp.parameters()) + list(lik.parameters()), lr=5e-2)
            mll = gpytorch.mlls.VariationalELBO(lik, gp, num_data=x.shape[0])
            gp.train(); lik.train()
            for _ in range(40):
                opt.zero_grad()
                output = gp(pa)
                loss = -mll(output, x.squeeze(-1))
                loss.backward(); opt.step()
            gp.eval(); lik.eval()
            self.gps[node] = {"gp": gp, "lik": lik, "parents": parents}

    def query(self, q: Query) -> torch.Tensor:
        if any(self.problem.variables[k][0] == "discrete" for k in q.evidence):
            raise NotImplementedError("gpytorch adapter cannot condition on discrete evidence")
        target = q.targets[0]
        spec = self.gps.get(target)
        if spec is None:
            raise NotImplementedError(f"target '{target}' not handled by GPyTorch adapter")
        if "mean" in spec:
            return spec["mean"].cpu()
        # Build parent tensor.  PR-B §A.8: use torch.as_tensor instead of
        # torch.tensor(tensor) so we don't trip the "copy construct from
        # a tensor" UserWarning when q.evidence values are already tensors.
        pa = torch.cat([
            torch.as_tensor(
                q.evidence.get(p, 0.0), device=self.device, dtype=torch.float32,
            ).reshape(1, -1)
            for p in spec["parents"]
        ], dim=-1)
        with torch.no_grad():
            pred = spec["lik"](spec["gp"](pa))
            mean = pred.mean
        return mean.cpu()

    # ------------------------------------------------------------------
    # Batched samples (v0.4)
    # ------------------------------------------------------------------

    def query_batch_samples(
        self, q: Query, n_samples: int = 2000,
    ) -> torch.Tensor:
        """Return ``[B, n_samples, |targets|]`` GP-posterior samples.

        The SVGP per node yields a ``MultivariateNormal(mean, covariance)``
        at any query point.  Drawing ``n_samples`` from that and shaping
        gives ``[B, n_samples, 1]`` for single-D continuous targets.

        GPyTorch handles only continuous evidence.  If any discrete
        evidence is in the batch, this returns a ``[0, n_samples, |T|]``
        NaN sentinel so the runner can skip the row gracefully.
        """
        assert self.problem is not None
        if any(self.problem.variables[k][0] == "discrete" for k in q.evidence):
            return torch.full(
                (0, n_samples, len(q.targets)), float("nan"),
            )
        target = q.targets[0]
        spec = self.gps.get(target)
        b = self._batch_size(q)
        if spec is None:
            return torch.full(
                (b, n_samples, len(q.targets)), float("nan"),
            )
        if "mean" in spec:
            # Root continuous node: marginal Gaussian
            mean = spec["mean"].cpu().reshape(-1)
            std = spec["std"].cpu().reshape(-1)
            noise = torch.randn(b, n_samples, len(q.targets))
            return mean.reshape(1, 1, -1) + std.reshape(1, 1, -1) * noise

        parents = spec["parents"]
        pa_rows = []
        for i in range(b):
            cols = []
            for p in parents:
                ev = q.evidence.get(p, 0.0)
                if isinstance(ev, torch.Tensor):
                    val = ev[i] if ev.dim() >= 1 else ev
                    val = val.float().reshape(1)
                else:
                    val = torch.tensor([float(ev)])
                cols.append(val)
            pa_rows.append(torch.cat(cols, dim=-1))
        pa = torch.stack(pa_rows, dim=0).to(self.device).float()  # [B, P]

        with torch.no_grad():
            pred = spec["lik"](spec["gp"](pa))
            samples = pred.sample(torch.Size([n_samples]))  # [n_samples, B]
            samples = samples.movedim(0, 1).unsqueeze(-1)    # [B, n_samples, 1]
        return samples.cpu()

    def _batch_size(self, q: Query) -> int:
        for v in q.evidence.values():
            if isinstance(v, torch.Tensor) and v.dim() >= 1:
                return int(v.shape[0])
        return 1

    def teardown(self) -> None:
        self.gps = {}
        self.problem = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
