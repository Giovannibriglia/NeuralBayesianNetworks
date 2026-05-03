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
        # Build parent tensor
        pa = torch.cat([
            torch.tensor(q.evidence.get(p, 0.0)).reshape(1, -1).to(self.device).float()
            for p in spec["parents"]
        ], dim=-1)
        with torch.no_grad():
            pred = spec["lik"](spec["gp"](pa))
            mean = pred.mean
        return mean.cpu()

    def teardown(self) -> None:
        self.gps = {}
        self.problem = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
