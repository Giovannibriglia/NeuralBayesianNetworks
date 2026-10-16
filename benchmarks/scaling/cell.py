"""Run ONE (task, method) on ONE problem under a compute budget; print JSON.

Usage:  python cell.py <problem.pt> <task> <method> [--device cpu|mps]
        [--budget-s 30] [--threads 1] [--opt key=value ...]

The budget is enforced in two layers:
* here: torch / BLAS threads pinned, and SIGALRM at ``budget_s`` seconds after
  the timed region starts (pure-Python loops are interrupted promptly; a long
  C call is interrupted when it returns);
* in ``run.py``: a hard wall-clock kill and an RSS watchdog (kill above the
  memory cap), so a cell can never exceed the budget silently.

Only the timed region is charged: imports and problem loading are not.

Tasks / methods
---------------
discrete:  fit    -> nbn | pgmpy
           infer  -> nbn_ve | pgmpy_ve | nbn_lw
           sets   -> nbn_interval | nbn_l1
nongauss:  fit    -> nbn_local | nbn_stacked | pyro_svi
           infer  -> nbn_lw | pyro_is | pyro_vec
"""
from __future__ import annotations

import os
import sys

_THREADS = "1"
for _i, _a in enumerate(sys.argv):
    if _a == "--threads" and _i + 1 < len(sys.argv):
        _THREADS = sys.argv[_i + 1]
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = _THREADS

import argparse  # noqa: E402
import json  # noqa: E402
import resource  # noqa: E402
import signal  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(int(_THREADS))
warnings.filterwarnings("ignore")


class Budget(Exception):
    pass


def _alarm(signum, frame):
    raise Budget("time budget exceeded")


class Timer:
    def __init__(self, budget_s):
        self.budget_s = budget_s

    def __enter__(self):
        signal.signal(signal.SIGALRM, _alarm)
        signal.setitimer(signal.ITIMER_REAL, self.budget_s)
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        signal.setitimer(signal.ITIMER_REAL, 0)
        self.elapsed = time.perf_counter() - self.t0
        return False


def peak_rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 1024


# --------------------------------------------------------------------- #
# Query sets (deterministic per problem)
# --------------------------------------------------------------------- #

def discrete_queries(pr, n_q=10, n_ev=5, seed=123):
    rng = np.random.default_rng(seed)
    nodes = pr["nodes"]
    test = pr["test"]
    qs = []
    for q in range(n_q):
        pick = rng.choice(len(nodes), size=min(len(nodes), n_ev + 1), replace=False)
        tgt = nodes[pick[0]]
        row = int(rng.integers(test[tgt].shape[0]))
        ev = {nodes[i]: int(test[nodes[i]][row]) for i in pick[1:]}
        qs.append((tgt, ev))
    return qs


def continuous_queries(pr, n_q=10, n_ev=3, seed=123):
    """Queries whose evidence is relevant: P(Y > median(Y) | e).

    The target has at least one parent; evidence is drawn from its Markov
    blanket and grandparents (random nodes in a sparse graph are nearly
    independent of the target, which would make every answer ~0.5), with
    values taken from one test row.
    """
    rng = np.random.default_rng(seed)
    nodes = pr["nodes"]
    test, train = pr["test"], pr["train"]
    pa = {v: [a for a, b in pr["edges"] if b == v] for v in nodes}
    ch = {v: [] for v in nodes}
    for a, b in pr["edges"]:
        ch[a].append(b)
    with_pa = [v for v in nodes if pa[v]] or nodes
    qs = []
    for q in range(n_q):
        tgt = with_pa[int(rng.integers(len(with_pa)))]
        pool = set(pa[tgt]) | set(ch[tgt])
        for c in ch[tgt]:
            pool |= set(pa[c])
        for a in pa[tgt]:
            pool |= set(pa[a])
        pool.discard(tgt)
        pool = sorted(pool)
        k = min(n_ev, len(pool))
        ev_nodes = list(rng.choice(pool, size=k, replace=False)) if k else []
        while len(ev_nodes) < min(n_ev, len(nodes) - 1):
            v = nodes[int(rng.integers(len(nodes)))]
            if v != tgt and v not in ev_nodes:
                ev_nodes.append(v)
        row = int(rng.integers(test[tgt].shape[0]))
        ev = {v: float(test[v][row]) for v in ev_nodes}
        thr = float(train[tgt].median())
        qs.append((tgt, ev, thr))
    return qs


# --------------------------------------------------------------------- #
# Discrete
# --------------------------------------------------------------------- #

def true_cpts(model):
    """node -> (parents, probs [*parent_cards, K]) from the true model."""
    out = {}
    for v in model.dag.topological_order():
        m = model.mechanisms[v]
        pa = list(model.dag.parents(v))
        logits = m.tabulate([3] * len(pa)).detach()
        out[v] = (pa, torch.softmax(logits, -1))
    return out


def to_pgmpy(model, k):
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.models import DiscreteBayesianNetwork
    bn = DiscreteBayesianNetwork()
    bn.add_nodes_from(list(model.dag.topological_order()))
    bn.add_edges_from(list(model.dag.networkx_graph.edges()))
    cpds = []
    for v, (pa, p) in true_cpts(model).items():
        vals = p.reshape(-1, k).T.double().numpy()  # [K, prod cards], C-order cols
        vals = vals / vals.sum(0, keepdims=True)
        cpds.append(TabularCPD(v, k, vals, evidence=pa or None,
                               evidence_card=[k] * len(pa) or None))
    bn.add_cpds(*cpds)
    return bn


def d_fit(pr, method, dev, budget, opt):
    from nbn.core.network import NeuralBayesianNetwork
    from nbn.mechanisms.parametric.categorical_table import CategoricalTableMechanism
    k = pr["k"]
    nodes = pr["nodes"]
    res = {}
    if method in ("nbn", "nbn_sparse"):
        from nbn.mechanisms.parametric.sparse_categorical_table import (
            SparseCategoricalTableMechanism,
        )
        cls = CategoricalTableMechanism if method == "nbn" else SparseCategoricalTableMechanism
        data = {v: pr["train"][v].to(dev) for v in nodes}
        with Timer(budget) as t:
            m = NeuralBayesianNetwork(pr["edges"], variables=dict.fromkeys(nodes, ("discrete", k)),
                                      device=str(dev))
            for v in nodes:
                m.set_mechanism(v, cls(alpha=0.0))
            m.fit(data, consolidate=False)
            if dev.type == "mps":
                torch.mps.synchronize()
        res["time_s"] = t.elapsed
        if pr["family"] == "discadd":
            return _additive_accuracy(pr, res, lambda v, rows: _nbn_row_probs(m, v, rows))
        learned = {v: torch.softmax(m.mechanisms[v].tabulate().detach().cpu().double(), -1)
                   .reshape(-1, k) for v in nodes}
    elif method == "pgmpy":
        import pandas as pd
        from pgmpy.estimators import MaximumLikelihoodEstimator
        from pgmpy.models import DiscreteBayesianNetwork
        df = pd.DataFrame({v: pr["train"][v].numpy() for v in nodes})
        with Timer(budget) as t:
            bn = DiscreteBayesianNetwork()
            bn.add_nodes_from(nodes)
            bn.add_edges_from(pr["edges"])
            cpds = MaximumLikelihoodEstimator(
                model=bn, data=df, state_names={v: list(range(k)) for v in nodes}
            ).get_parameters(n_jobs=1)
        res["time_s"] = t.elapsed
        learned = {}
        for c in cpds:
            v = c.variable
            pa_true = [a for a, b in pr["edges"] if b == v]
            # reorder pgmpy evidence axes to the canonical (edge-list) parent order
            vals = torch.as_tensor(c.get_values())  # [K, prod]
            ev = list(c.variables[1:])
            if ev:
                arr = vals.reshape(k, *[k] * len(ev))
                perm = [0] + [1 + ev.index(p) for p in pa_true]
                arr = arr.permute(*perm)
                vals = arr.reshape(k, -1)
            learned[v] = vals.T.double()
        if pr["family"] == "discadd":
            def _pg_rows(v, rows):
                pa = [a for a, b in pr["edges"] if b == v]
                idx = torch.zeros(rows[v].shape[0], dtype=torch.long)
                for a in pa:
                    idx = idx * k + rows[a]
                return learned[v].reshape(-1, k)[idx]
            return _additive_accuracy(pr, res, _pg_rows)
    else:
        raise ValueError(method)
    # accuracy: TV distance to the true CPT, averaged over parent rows weighted
    # by their training counts (rows never seen carry no information)
    truth = true_cpts(pr["true_model"])
    tv_num = tv_den = 0.0
    for v in nodes:
        pa = truth[v][0]
        p_true = truth[v][1].reshape(-1, k).double()
        if pa:
            idx = torch.zeros(pr["train"][v].shape[0], dtype=torch.long)
            for a in pa:
                idx = idx * k + pr["train"][a]
            cnt = torch.bincount(idx, minlength=p_true.shape[0]).double()
        else:
            cnt = torch.tensor([float(pr["train"][v].shape[0])])
        tv = 0.5 * (learned[v].reshape(-1, k) - p_true).abs().sum(-1)
        tv = torch.nan_to_num(tv, nan=1.0)
        tv_num += float((tv * cnt).sum())
        tv_den += float(cnt.sum())
    res["acc_tv"] = tv_num / tv_den
    res["cpt_hash"] = float(sum(float(torch.nan_to_num(learned[v]).sum()) for v in nodes))
    return res


def _nbn_row_probs(m, v, rows):
    pa = list(m.dag.parents(v))
    mech = m.mechanisms[v]
    n = rows[v].shape[0]
    P = torch.stack([rows[a] for a in pa], -1) if pa else None
    with torch.no_grad():
        out = [mech.log_prob(torch.full((n,), c), P).exp() for c in range(mech.n_classes)]
    return torch.stack(out, -1).double().cpu()


def _additive_accuracy(pr, res, row_probs):
    """Mean TV between learned and true rows over the training rows."""
    k, nodes = pr["k"], pr["nodes"]
    rows = pr["train"]
    tv = []
    for j, v in enumerate(nodes):
        lg = torch.tensor(pr["add_B"][j]).expand(rows[v].shape[0], k).clone()
        for t, i in enumerate(pr["add_parents"][j]):
            lg += torch.tensor(pr["add_W"][j][t])[rows[nodes[i]]]
        p_true = torch.softmax(lg, -1)
        tv.append(float(0.5 * (row_probs(v, rows) - p_true).abs().sum(-1).mean()))
    res["acc_tv"] = float(np.mean(tv))
    return res


def d_infer(pr, method, dev, budget, opt):
    qs = discrete_queries(pr, n_q=int(opt.get("n_q", 10)), n_ev=int(opt.get("n_ev", 5)))
    model = pr["true_model"]
    answers = []
    res = {"n_q": len(qs)}
    if method in ("nbn_ve", "nbn_lw"):
        model = model.to(dev)
        if method == "nbn_ve":
            from nbn.inference.tensor_ve import TensorVariableElimination
            # Tell NBN the budget so its guard refuses a query it cannot fit
            # (instead of the watchdog killing the process): 75% of the cap,
            # the rest is the interpreter and the model.
            eng = TensorVariableElimination(
                memory_budget=float(opt.get("ve_mem_bytes", 0.75 * float(opt.get("mem_gb", 4)) * 2**30)))
        else:
            from nbn.inference.likelihood_weighting import LikelihoodWeightingEngine
            eng = LikelihoodWeightingEngine(n_samples=int(opt.get("n_samples", 100_000)))
        per_q = []
        with Timer(budget) as t:
            for tgt, ev in qs:
                t1 = time.perf_counter()
                if method == "nbn_lw":
                    p, ess = model.query([tgt], evidence=ev, engine=eng, return_ess=True)
                    res.setdefault("ess_frac", []).append(float(ess.reshape(-1)[0]))
                else:
                    try:
                        p = model.query([tgt], evidence=ev, engine=eng)
                    except Budget:
                        raise
                    except Exception as e:  # noqa: BLE001  guard refused the query
                        res.setdefault("q_errors", []).append(f"{type(e).__name__}: {str(e)[:160]}")
                        answers.append(None)
                        per_q.append(time.perf_counter() - t1)
                        continue
                if dev.type == "mps":
                    torch.mps.synchronize()
                answers.append([float(x) for x in p.reshape(-1).cpu()])
                per_q.append(time.perf_counter() - t1)
        res["time_s"] = t.elapsed
        res["q_time_s"] = per_q
        try:  # width bounds need this branch's NBN (absent on master)
            from nbn.inference._elimination_order import query_width_bounds
        except ImportError:
            query_width_bounds = None
        if method == "nbn_ve" and query_width_bounds is not None:
            # width bounds per query (not timed): lower = minor-min-width on the
            # pruned, evidence-reduced moral graph; upper = NBN's min-fill order
            wb = []
            for tgt, ev in qs:
                _, rel = eng._plan_with_relevant(model, tgt, tuple(sorted(ev)))
                g = model.dag.networkx_graph
                scope = set(rel)
                for v in rel:
                    scope |= set(g.predecessors(v))
                wb.append(list(query_width_bounds(g.subgraph(scope), targets=[tgt],
                                                  evidence=list(ev))))
            res["width_bounds"] = wb
        res["n_answered"] = sum(a is not None for a in answers)
        if res["n_answered"] == 0:
            raise RuntimeError("out of memory: every query refused by the VE guard; "
                               + (res.get("q_errors") or [""])[0])
    elif method == "nbn_cert":
        from nbn.inference.certified import certified_query
        model = model.to(dev)
        per_q, lows, ups, widths = [], [], [], []
        S = int(opt.get("n_samples", 200_000))
        with Timer(budget) as t:
            for tgt, ev in qs:
                t1 = time.perf_counter()
                a = certified_query(model, tgt, ev, n_samples=S,
                                    delta=float(opt.get("delta", 0.05)))
                answers.append([float(x) for x in a.estimate])
                lows.append([float(x) for x in a.lower])
                ups.append([float(x) for x in a.upper])
                widths.append(a.max_width)
                per_q.append(time.perf_counter() - t1)
        res.update(time_s=t.elapsed, q_time_s=per_q, lower=lows, upper=ups,
                   max_width=widths, n_answered=len(answers), n_samples=S)
    elif method == "pgmpy_ve":
        from pgmpy.inference import VariableElimination
        per_q = []
        with Timer(budget) as t:
            bn = to_pgmpy(model, pr["k"])
            inf = VariableElimination(bn)
            for tgt, ev in qs:
                t1 = time.perf_counter()
                f = inf.query([tgt], evidence=ev, show_progress=False,
                              elimination_order=opt.get("order", "greedy"))
                answers.append([float(x) for x in f.values.reshape(-1)])
                per_q.append(time.perf_counter() - t1)
        res["time_s"] = t.elapsed
        res["q_time_s"] = per_q
        res["n_answered"] = len(answers)
    else:
        raise ValueError(method)
    res["answers"] = answers
    return res


def d_sets(pr, method, dev, budget, opt):
    from nbn.core.network import NeuralBayesianNetwork
    from nbn.uncertainty import interval_set, l1_ball_set
    k = pr["k"]
    nodes = pr["nodes"]
    m = NeuralBayesianNetwork(pr["edges"], variables=dict.fromkeys(nodes, ("discrete", k)),
                              device=str(dev))
    data = {v: pr["train"][v].to(dev) for v in nodes}
    delta = 0.05 / len(nodes)
    width = []
    if method == "nbn_intervals_batched":
        from nbn.uncertainty import interval_sets
        with Timer(budget) as t:
            sets = interval_sets(m, data, nodes, delta=delta, method=opt.get("imethod", "kl"))
        width = [float((s.upper - s.lower).mean()) for s in sets.values()]
        return {"time_s": t.elapsed, "mean_width": float(np.mean(width))}
    with Timer(budget) as t:
        for v in nodes:
            if method == "nbn_interval":
                s = interval_set(m, data, v, delta=delta, method=opt.get("imethod", "kl"))
                width.append(float((s.upper - s.lower).mean()))
            elif method == "nbn_l1":
                s = l1_ball_set(m, data, v, delta=delta)
                width.append(float(s.radius.clamp(max=2).mean()))
            else:
                raise ValueError(method)
    return {"time_s": t.elapsed, "mean_width": float(np.mean(width))}


# --------------------------------------------------------------------- #
# Continuous non-Gaussian
# --------------------------------------------------------------------- #

FIT_DEFAULTS = dict(epochs=20, batch_size=512, lr=1e-3, hidden=(32,), K=3)


def _test_ll_rows(model, test):
    with torch.no_grad():
        return model.log_prob({k: v.reshape(-1, 1) for k, v in test.items()})


def _test_ll_nbn(model, test):
    return float(_test_ll_rows(model, test).mean())


def c_fit(pr, method, dev, budget, opt):
    from nbn.core.network import NeuralBayesianNetwork
    from nbn.mechanisms.parametric.mdn import MDNMechanism
    cfg = dict(FIT_DEFAULTS)
    cfg.update({k: (int(v) if k in ("epochs", "batch_size", "K") else float(v))
                for k, v in opt.items() if k in ("epochs", "batch_size", "lr", "K")})
    nodes = pr["nodes"]
    res = dict(cfg=dict(cfg, hidden=list(cfg["hidden"])))
    torch.manual_seed(0)
    if method in ("nbn_local", "nbn_stacked"):
        data = {v: pr["train"][v].reshape(-1, 1).to(dev) for v in nodes}
        m = NeuralBayesianNetwork(pr["edges"], variables=dict.fromkeys(nodes, ("continuous", 1)),
                                  device=str(dev))
        for v in nodes:
            m.set_mechanism(v, MDNMechanism(num_components=cfg["K"], hidden=cfg["hidden"]))
        with Timer(budget) as t:
            m.fit(data, method="local" if method == "nbn_local" else "stacked",
                  epochs=cfg["epochs"], batch_size=cfg["batch_size"], lr=cfg["lr"],
                  consolidate=False)
            if dev.type == "mps":
                torch.mps.synchronize()
        res["time_s"] = t.elapsed
        m = m.to("cpu")
        rows = _test_ll_rows(m, pr["test"])
        res["test_ll"] = float(rows.mean()) / len(nodes)
        res["test_ll_median"] = float(rows.median()) / len(nodes)
    elif method == "pyro_svi":
        import pyro
        import pyro.distributions as dist
        from pyro.infer import SVI, Trace_ELBO
        from pyro.optim import Adam
        pyro.set_rng_seed(0)
        pyro.clear_param_store()
        X = torch.stack([pr["train"][v] for v in nodes], 1).to(dev)
        Xt = torch.stack([pr["test"][v] for v in nodes], 1).to(dev)
        col = {v: i for i, v in enumerate(nodes)}
        pa_idx = {v: [col[a] for a, b in pr["edges"] if b == v] for v in nodes}
        mu, sd = X.mean(0), X.std(0).clamp_min(1e-3)
        K = cfg["K"]
        nets = {}
        for v in nodes:
            p = len(pa_idx[v])
            if p:
                layers, prev = [], p
                for h in cfg["hidden"]:
                    layers += [torch.nn.Linear(prev, h), torch.nn.ReLU()]
                    prev = h
                layers.append(torch.nn.Linear(prev, 3 * K))
                nets[v] = torch.nn.Sequential(*layers).to(dev)

        def node_dist(v, Z):
            # standardised units, same parameterisation as NBN's MDN
            if v in nets:
                out = pyro.module(f"net_{v}", nets[v])(Z[:, pa_idx[v]])
                logits, loc, ls = out[:, :K], out[:, K:2 * K], out[:, 2 * K:]
            else:
                logits = pyro.param(f"lg_{v}", torch.zeros(K, device=dev))
                loc = pyro.param(f"loc_{v}", torch.linspace(-1, 1, K, device=dev))
                ls = pyro.param(f"ls_{v}", torch.zeros(K, device=dev))
            scale = torch.exp(ls.clamp(max=7.0)).clamp_min(1e-3)
            return dist.MixtureSameFamily(dist.Categorical(logits=logits),
                                          dist.Normal(loc, scale))

        def model(Xb, n_total):
            Z = (Xb - mu) / sd
            with pyro.plate("data", n_total, subsample=torch.arange(Xb.shape[0], device=dev)):
                for v in nodes:
                    pyro.sample(v, node_dist(v, Z), obs=Z[:, col[v]])

        def guide(Xb, n_total):
            pass

        svi = SVI(model, guide, Adam({"lr": cfg["lr"]}), loss=Trace_ELBO())
        N = X.shape[0]
        bs = cfg["batch_size"]
        with Timer(budget) as t:
            for ep in range(cfg["epochs"]):
                perm = torch.randperm(N, device=dev)
                for i in range(0, N, bs):
                    Xb = X[perm[i:i + bs]]
                    svi.step(Xb, N)
            if dev.type == "mps":
                torch.mps.synchronize()
        res["time_s"] = t.elapsed
        with torch.no_grad():
            Z = (Xt - mu) / sd
            rows = torch.zeros(Z.shape[0], device=dev)
            for v in nodes:
                rows += node_dist(v, Z).log_prob(Z[:, col[v]]) - torch.log(sd[col[v]])
        res["test_ll"] = float(rows.mean()) / len(nodes)
        res["test_ll_median"] = float(rows.median()) / len(nodes)
    else:
        raise ValueError(method)
    rows = _test_ll_rows(pr["true_model"], pr["test"])
    res["true_test_ll"] = float(rows.mean()) / len(nodes)
    res["true_test_ll_median"] = float(rows.median()) / len(nodes)
    return res


def c_infer(pr, method, dev, budget, opt):
    qs = continuous_queries(pr, n_q=int(opt.get("n_q", 10)))
    S = int(opt.get("n_samples", 10_000))
    model = pr["true_model"].to(dev)
    nodes = pr["nodes"]
    est, ess = [], []
    per_q = []
    if method == "nbn_lw":
        from nbn.inference.likelihood_weighting import LikelihoodWeightingEngine
        eng = LikelihoodWeightingEngine(n_samples=S)
        with Timer(budget) as t:
            for tgt, ev, thr in qs:
                t1 = time.perf_counter()
                (w, x), e = model.query([tgt], evidence=ev, engine=eng, return_ess=True)
                w = w.reshape(-1)
                x = x.reshape(-1)
                est.append(float((w * (x > thr).float()).sum() / w.sum()))
                ess.append(float(e.reshape(-1)[0]))
                per_q.append(time.perf_counter() - t1)
    elif method in ("pyro_is", "pyro_vec"):
        import pyro
        from pyro import poutine
        pyro.set_rng_seed(0)
        mechs = {v: model.mechanisms[v] for v in nodes}
        parents = {v: list(model.dag.parents(v)) for v in nodes}

        import pyro.distributions as pdist

        def node_dist(v, pa, n):
            logits, loc, scale = mechs[v]._params_from_parents(pa, (n,))
            return pdist.MixtureSameFamily(pdist.Categorical(logits=logits),
                                           pdist.Normal(loc[..., 0], scale[..., 0]))

        def pmodel(ev, n):
            vals = {}
            with pyro.plate("particles", n, dim=-1):
                for v in nodes:
                    pa = (torch.stack([vals[a] for a in parents[v]], -1)
                          if parents[v] else None)
                    obs = torch.full((n,), ev[v], device=dev) if v in ev else None
                    vals[v] = pyro.sample(v, node_dist(v, pa, n), obs=obs)
            return vals

        with Timer(budget) as t:
            for tgt, ev, thr in qs:
                t1 = time.perf_counter()
                if method == "pyro_vec":
                    with torch.no_grad():
                        tr = poutine.trace(pmodel).get_trace(ev, S)
                    lw = torch.zeros(S, device=dev)
                    for name, site in tr.nodes.items():
                        if site["type"] == "sample" and site["is_observed"]:
                            lw = lw + site["fn"].log_prob(site["value"]).reshape(S)
                    xs = tr.nodes[tgt]["value"].reshape(S)
                    w = torch.softmax(lw, 0)
                else:
                    from pyro.infer import Importance
                    imp = Importance(lambda ev=ev, tgt=tgt: pmodel(ev, 1)[tgt], guide=None,
                                     num_samples=S)
                    with torch.no_grad():
                        imp.run()
                    lw = torch.stack(imp.log_weights).reshape(-1)
                    xs = torch.stack([tr.nodes["_RETURN"]["value"] for tr in imp.exec_traces]).reshape(-1)
                    w = torch.softmax(lw, 0)
                est.append(float((w * (xs > thr).float()).sum()))
                ess.append(float(1.0 / (w.pow(2).sum() * S)))
                per_q.append(time.perf_counter() - t1)
    else:
        raise ValueError(method)
    return {"time_s": t.elapsed, "q_time_s": per_q, "answers": est, "ess_frac": ess,
            "n_q": len(qs), "n_samples": S}


TASKS = {
    ("discrete", "fit"): d_fit,
    ("discadd", "fit"): d_fit,
    ("discrete", "infer"): d_infer,
    ("discrete", "sets"): d_sets,
    ("nongauss", "fit"): c_fit,
    ("nongauss", "infer"): c_infer,
    ("ngs", "fit"): c_fit,
    ("ngs", "infer"): c_infer,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("problem")
    ap.add_argument("task")
    ap.add_argument("method")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--budget-s", type=float, default=30.0)
    ap.add_argument("--threads", default="1")
    ap.add_argument("--opt", nargs="*", default=[])
    a = ap.parse_args()
    opt = dict(kv.split("=", 1) for kv in a.opt)
    base_rss = peak_rss_mb()
    pr = torch.load(a.problem, weights_only=False)
    out = dict(problem=pr["name"], family=pr["family"], topology=pr["topology"],
               n=pr["n"], deg=pr["deg"], n_edges=pr["n_edges"], max_in=pr["max_in"],
               window=pr["window"], seed=pr["seed"], task=a.task, method=a.method,
               device=a.device, threads=int(a.threads), budget_s=a.budget_s, opt=opt)
    dev = torch.device(a.device)
    try:
        r = TASKS[(pr["family"], a.task)](pr, a.method, dev, a.budget_s, opt)
        out.update(r)
        out["status"] = "ok"
    except Budget:
        out["status"] = "timeout"
    except MemoryError:
        out["status"] = "oom"
    except Exception as e:  # noqa: BLE001
        msg = f"{type(e).__name__}: {e}"
        out["status"] = "oom" if ("memory" in msg.lower() or "MPSTensorLimit" in msg) else "error"
        out["error"] = msg[:500]
        out["trace"] = traceback.format_exc()[-1500:]
    out["peak_rss_mb"] = peak_rss_mb()
    out["load_rss_mb"] = base_rss
    print("RESULT " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
