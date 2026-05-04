"""Synthetic hybrid (discrete + non-Gaussian continuous) DAG generator."""
from __future__ import annotations


import networkx as nx
import torch

from benchmarking.domains.base import (
    BenchmarkDomain,
    BenchmarkProblem,
)
from benchmarking.queries import make_query_battery


class SyntheticHybridDomain(BenchmarkDomain):
    """Random DAGs of size ``n`` with mixed discrete + non-Gaussian continuous."""

    name = "synthetic_hybrid"

    def __init__(self, sizes: tuple[int, ...] = (10, 50, 200, 1000)) -> None:
        self.sizes = sizes

    def list_problems(self) -> list[str]:
        return [f"hybrid_{n}" for n in self.sizes]

    def load_problem(
        self,
        problem: str,
        *,
        n_train: int,
        n_test: int,
        seed: int,
        device: torch.device,
    ) -> BenchmarkProblem:
        n = int(problem.split("_")[1])
        torch.manual_seed(seed)
        g = torch.Generator().manual_seed(seed)

        # Random upper-triangular DAG
        names = [f"X{i}" for i in range(n)]
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if torch.rand(1, generator=g).item() < 0.10:
                    edges.append((names[i], names[j]))

        # Half discrete, half continuous
        discrete = set(names[: n // 2])
        cards = dict.fromkeys(discrete, 3)
        variables = {
            nm: ("discrete", 3) if nm in discrete else ("continuous", 1)
            for nm in names
        }

        # Generate samples from a mixture-of-skew-normals SCM.
        # All sampling happens on CPU (the seeded torch.Generator is CPU-only);
        # only the final per-node tensors are moved to ``device``.
        def _sample_node(n_samples: int, parents_data_cpu: list[torch.Tensor],
                         is_disc: bool) -> torch.Tensor:
            if not parents_data_cpu:
                if is_disc:
                    probs = torch.softmax(torch.randn(3, generator=g), dim=0)
                    return torch.multinomial(probs, n_samples, replacement=True,
                                             generator=g).unsqueeze(-1).float()
                return torch.randn(n_samples, 1, generator=g) * 0.7
            pa = torch.cat([p.float().reshape(n_samples, -1) for p in parents_data_cpu], dim=-1)
            if is_disc:
                W = torch.randn(pa.shape[1], 3, generator=g) * 0.5
                logits = pa @ W
                probs = torch.softmax(logits, dim=-1)
                return torch.multinomial(probs, 1, replacement=True,
                                         generator=g).float()
            # Skew-normal-ish: x = w·pa + skew * (chi^2 - 1) component
            w = torch.randn(pa.shape[1], 1, generator=g) * 0.5
            base = pa @ w
            noise = 0.3 * torch.randn(n_samples, 1, generator=g)
            skew = 0.2 * (torch.randn(n_samples, 1, generator=g) ** 2 - 1)
            return base + noise + skew

        # Topological order
        gx = nx.DiGraph(); gx.add_nodes_from(names); gx.add_edges_from(edges)
        topo = list(nx.topological_sort(gx))

        # Keep a CPU copy for the SCM (the generator lives on CPU); also build
        # the user-facing dict on the requested device.
        train_cpu: dict[str, torch.Tensor] = {}
        test_cpu: dict[str, torch.Tensor] = {}
        train_data: dict[str, torch.Tensor] = {}
        test_data: dict[str, torch.Tensor] = {}
        for nm in topo:
            parents = list(gx.predecessors(nm))
            tr = _sample_node(n_train, [train_cpu[p] for p in parents], nm in discrete)
            te = _sample_node(n_test, [test_cpu[p] for p in parents], nm in discrete)
            train_cpu[nm] = tr
            test_cpu[nm] = te
            train_data[nm] = tr.to(device)
            test_data[nm] = te.to(device)

        queries = make_query_battery(
            nodes=names, discrete_nodes=list(discrete),
            continuous_nodes=[nm for nm in names if nm not in discrete],
            cardinalities=cards,
            n_conditional_single=10, n_per_multi=5, n_map=5, n_do=5,
            seed=seed,
        )

        # Empirical ground truth: per-node test-data samples + summary stats.
        # See benchmarking/domains/_ground_truth.py for the wiring.
        from benchmarking.domains._ground_truth import mc_marginals_from_test_data

        problem_obj = BenchmarkProblem(
            name=problem, dag=edges, variables=variables,
            train_data=train_data, test_data=test_data,
            queries=queries,
            ground_truth=None,
        )
        gt = mc_marginals_from_test_data(problem_obj)
        # Also keep the legacy concatenated-samples field for any callers
        # that read it directly.
        object.__setattr__(gt, "samples", torch.cat(
            [test_data[nm].reshape(n_test, -1) for nm in topo], dim=-1))
        return BenchmarkProblem(
            name=problem_obj.name, dag=problem_obj.dag, variables=problem_obj.variables,
            train_data=problem_obj.train_data, test_data=problem_obj.test_data,
            queries=problem_obj.queries, ground_truth=gt,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Scaling-grid builder for the v0.3 ablations.  The grid varies one axis at
# a time (n_nodes, edge density, or discrete cardinality) with the others
# pinned at sensible defaults; each (setting, replicate) is an independent
# `BenchmarkProblem` whose name encodes the axis value, e.g.
# ``"hybrid_n200_d020_k4_r0"``.
# ─────────────────────────────────────────────────────────────────────────────


def make_scaling_grid(
    *,
    n_nodes: list[int] | None = None,
    edge_density: float | list[float] = 0.20,
    cardinality: int | list[int] = 4,
    fraction_continuous: float = 0.5,
    n_replicates: int = 3,
    seed: int = 0,
    n_train: int = 1000,
    n_test: int = 200,
    device: torch.device | str = "cpu",
) -> list[BenchmarkProblem]:
    """Cartesian-product scaling grid for the network-size ablation.

    Vary at most one of ``n_nodes`` / ``edge_density`` / ``cardinality`` as
    a list — the other two stay scalar. Each setting is repeated
    ``n_replicates`` times under independent seeds.
    """
    n_nodes_list = n_nodes if n_nodes is not None else [10, 25, 50, 100, 200]
    edge_density_list = edge_density if isinstance(edge_density, list) else [edge_density]
    cardinality_list = cardinality if isinstance(cardinality, list) else [cardinality]
    device_t = torch.device(device)
    domain = SyntheticHybridDomain(sizes=tuple(n_nodes_list))
    problems: list[BenchmarkProblem] = []
    for n in n_nodes_list:
        for d in edge_density_list:
            for k in cardinality_list:
                for r in range(n_replicates):
                    problem = domain.load_problem(
                        f"hybrid_{n}",
                        n_train=n_train, n_test=n_test,
                        seed=seed + r * 1000 + int(d * 1000) + k,
                        device=device_t,
                    )
                    new_name = f"hybrid_n{n}_d{int(d*100):03d}_k{k}_r{r}"
                    problems.append(
                        BenchmarkProblem(
                            name=new_name,
                            dag=problem.dag,
                            variables=problem.variables,
                            train_data=problem.train_data,
                            test_data=problem.test_data,
                            queries=problem.queries,
                            ground_truth=problem.ground_truth,
                        )
                    )
    return problems


class SyntheticHybridScalingDomain(BenchmarkDomain):
    """Domain wrapper exposing the scaling grid as a regular plugin.

    Problems are named ``hybrid_n{N}_d{DENSITY*100}_k{K}_r{REP}`` so the
    plotter can parse the ablation axis from the name.
    """

    name = "synthetic_hybrid_scaling"

    def __init__(
        self,
        n_nodes: list[int] | None = None,
        edge_density: float | list[float] = 0.20,
        cardinality: int | list[int] = 4,
        fraction_continuous: float = 0.5,
        n_replicates: int = 3,
    ) -> None:
        self._n_nodes = n_nodes if n_nodes is not None else [10, 25, 50, 100, 200]
        self._edge_density = edge_density
        self._cardinality = cardinality
        self._fraction_continuous = fraction_continuous
        self._n_replicates = n_replicates

    def list_problems(self) -> list[str]:
        densities = self._edge_density if isinstance(self._edge_density, list) else [self._edge_density]
        cards = self._cardinality if isinstance(self._cardinality, list) else [self._cardinality]
        names = []
        for n in self._n_nodes:
            for d in densities:
                for k in cards:
                    for r in range(self._n_replicates):
                        names.append(f"hybrid_n{n}_d{int(d*100):03d}_k{k}_r{r}")
        return names

    def load_problem(
        self,
        problem: str,
        *,
        n_train: int,
        n_test: int,
        seed: int,
        device: torch.device,
    ) -> BenchmarkProblem:
        # Parse the scaling-grid problem name back into axes.
        # Format: hybrid_n{N}_d{DDD}_k{K}_r{R}
        try:
            parts = problem.replace("hybrid_", "").split("_")
            n = int(parts[0][1:])
            d = int(parts[1][1:]) / 100.0
            k = int(parts[2][1:])
            r = int(parts[3][1:])
        except Exception as e:
            raise ValueError(
                f"Bad scaling-grid problem name '{problem}'; expected "
                "'hybrid_n{N}_d{DDD}_k{K}_r{R}'") from e
        # Reuse SyntheticHybridDomain.load_problem with derived n_nodes
        base = SyntheticHybridDomain(sizes=(n,))
        problem_obj = base.load_problem(
            f"hybrid_{n}", n_train=n_train, n_test=n_test,
            seed=seed + r * 1000 + int(d * 1000) + k, device=device,
        )
        return BenchmarkProblem(
            name=problem,
            dag=problem_obj.dag,
            variables=problem_obj.variables,
            train_data=problem_obj.train_data,
            test_data=problem_obj.test_data,
            queries=problem_obj.queries,
            ground_truth=problem_obj.ground_truth,
        )
