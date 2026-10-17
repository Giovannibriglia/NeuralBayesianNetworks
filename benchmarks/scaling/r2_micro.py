"""Round-2 micro-benchmarks (1 thread): kNN search backends and FlexCode grid density.

Writes results/r2_micro.jsonl.  Run after the sweeps (it is short).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import torch  # noqa: E402

torch.set_num_threads(1)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

from nbn.mechanisms.non_parametric.flexcode import FlexCodeMechanism  # noqa: E402
from nbn.mechanisms.non_parametric.knn_conditional import KNNConditionalMechanism  # noqa: E402


def timeit(f, reps=3):
    f()
    t = time.perf_counter()
    for _ in range(reps):
        f()
    return (time.perf_counter() - t) / reps


def main():
    out = []
    g = torch.Generator().manual_seed(0)
    for n in (2_000, 10_000, 50_000):
        for d in (1, 2, 4, 8):
            pa = torch.randn(n, d, generator=g)
            y = torch.randn(n, 1, generator=g)
            q = torch.randn(10_000, d, generator=g)
            row = dict(kind="knn_search", n_train=n, d_pa=d, m_query=10_000)
            for label, kw in (("old_any", dict(tie_break="any")),
                              ("exact_brute", dict(search="brute")),
                              ("exact_kdtree", dict(search="kdtree"))):
                m = KNNConditionalMechanism(**kw)
                m.fit_local(y, pa)
                xs = m._std_parents(q)
                row[label] = timeit(lambda m=m, xs=xs: m._knn_search(xs), reps=2)
            # discrete-like parents: few distinct rows -> dedup
            pad = (pa * 2).round().clamp(-2, 2)
            qd = (q * 2).round().clamp(-2, 2)
            # old: every query row searched (no dedup, float32 cdist + topk)
            m = KNNConditionalMechanism(tie_break="any")
            m.fit_local(y, pad)
            xs = m._std_parents(qd)
            row["disc_old_any"] = timeit(lambda m=m, xs=xs: m._brute_any(xs), reps=2)
            m = KNNConditionalMechanism()
            m.fit_local(y, pad)
            xs = m._std_parents(qd)
            row["disc_new"] = timeit(lambda m=m, xs=xs: m._knn_search(xs), reps=2)
            row["disc_unique_rows"] = int(torch.unique(xs, dim=0).shape[0])
            out.append(row)
            print(row, flush=True)
    # FlexCode grid density
    m = FlexCodeMechanism(epochs=2, hidden=(32,))
    pa = torch.randn(3000, 2, generator=g)
    m.fit_local(pa.sum(1, keepdim=True) + 0.3 * torch.randn(3000, 1, generator=g), pa)
    for rows in (10_000, 100_000):
        coef = m._coef(torch.randn(rows, 2, generator=g), rows).detach()
        old = timeit(lambda c=coef: (c.unsqueeze(1) * m._basis_grid.unsqueeze(0)).sum(-1).clamp_min(0.0))
        new = timeit(lambda c=coef: m._grid_density(c))
        out.append(dict(kind="flex_grid", rows=rows, old=old, new=new))
        print(out[-1], flush=True)
    (HERE / "results" / "r2_micro.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))


if __name__ == "__main__":
    main()
