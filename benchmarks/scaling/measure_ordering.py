"""Split exact-query time into "choose the elimination order" and "the rest".

Run once per NBN checkout (``PYTHONPATH`` selects it); prints one JSON line.
Used by ``explain.py`` (plot 03).  One thread, the 10 queries of the sweep.
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
import torch  # noqa: E402

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cell  # noqa: E402
import nbn.inference.tensor_ve as tve  # noqa: E402
from nbn.inference.tensor_ve import TensorVariableElimination  # noqa: E402

label = sys.argv[1]
out = []
for name in ("discrete_local_w5_n1000_deg2_k3_s0", "discrete_local_w5_n2000_deg2_k3_s0"):
    pr = torch.load(os.path.join(HERE, "problems", f"{name}.pt"), weights_only=False)
    m = pr["true_model"]
    spent = [0.0]
    orig = tve.get_order

    def timed(*a, _o=orig, spent=spent, **k):
        t = time.perf_counter()
        r = _o(*a, **k)
        spent[0] += time.perf_counter() - t
        return r

    tve.get_order = timed
    eng = TensorVariableElimination()
    t0 = time.perf_counter()
    for tgt, ev in cell.discrete_queries(pr):
        m.query([tgt], evidence=ev, engine=eng)
    total = time.perf_counter() - t0
    tve.get_order = orig
    out.append(dict(code=label, problem=name, n=pr["n"], total_s=total, ordering_s=spent[0]))
print(json.dumps(out))
