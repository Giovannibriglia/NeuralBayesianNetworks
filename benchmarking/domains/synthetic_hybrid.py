"""Synthetic hybrid (discrete + non-Gaussian continuous) DAG generator."""
from __future__ import annotations


import networkx as nx
import torch

from benchmarking.domains.base import (
    BenchmarkDomain,
    BenchmarkProblem,
    GroundTruth,
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

        # Generate samples from a mixture-of-skew-normals SCM
        def _sample_node(n_samples: int, parents_data: list[torch.Tensor],
                         is_disc: bool) -> torch.Tensor:
            if not parents_data:
                if is_disc:
                    probs = torch.softmax(torch.randn(3, generator=g), dim=0)
                    return torch.multinomial(probs, n_samples, replacement=True,
                                             generator=g).unsqueeze(-1).float()
                return torch.randn(n_samples, 1, generator=g) * 0.7
            pa = torch.cat([p.float().reshape(n_samples, -1) for p in parents_data], dim=-1)
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

        train_data: dict[str, torch.Tensor] = {}
        test_data: dict[str, torch.Tensor] = {}
        for nm in topo:
            parents = list(gx.predecessors(nm))
            tr = _sample_node(n_train, [train_data[p] for p in parents], nm in discrete)
            te = _sample_node(n_test, [test_data[p] for p in parents], nm in discrete)
            train_data[nm] = tr.to(device)
            test_data[nm] = te.to(device)

        # Empirical ground truth: just the marginal samples themselves.
        gt_samples = torch.cat(
            [test_data[nm].reshape(n_test, -1) for nm in topo], dim=-1
        )

        queries = make_query_battery(
            nodes=names, discrete_nodes=list(discrete),
            continuous_nodes=[nm for nm in names if nm not in discrete],
            cardinalities=cards,
            n_conditional_single=10, n_per_multi=5, n_map=5, n_do=5,
            seed=seed,
        )

        return BenchmarkProblem(
            name=problem, dag=edges, variables=variables,
            train_data=train_data, test_data=test_data,
            queries=queries,
            ground_truth=GroundTruth(samples=gt_samples),
        )
