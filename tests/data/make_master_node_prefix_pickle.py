"""Regenerate ``master_node_prefix_pickle.pt`` with *origin/master* code.

A whole-model ``torch.save(model)`` pickle whose node names start with
``node__`` — the escape prefix the branch's ``_AliasModuleDict`` introduced
(review finding N4).  It must be produced by the old code::

    git archive origin/master | tar -x -C <scratch>/master_src
    PYTHONPATH=<scratch>/master_src .venv/bin/python \\
        tests/data/make_master_node_prefix_pickle.py \\
        tests/data/master_node_prefix_pickle.pt

(origin/master at 64c5f0c when the committed file was made.)
"""
from __future__ import annotations

import sys

import torch

import nbn
from nbn import NeuralBayesianNetwork as NBN
from nbn.mechanisms import CategoricalTableMechanism, LinearGaussianMechanism


def main(out: str) -> None:
    assert "master_src" in nbn.__file__, f"not running master code: {nbn.__file__}"
    torch.manual_seed(0)
    n = 200
    a = torch.randint(0, 3, (n,))
    b = a.float().unsqueeze(1) + 0.1 * torch.randn(n, 1)
    data = {"node__a": a, "b.c": b}
    model = NBN(
        [("node__a", "b.c")],
        {"node__a": ("discrete", 3), "b.c": ("continuous", 1)}, device="cpu",
    )
    model.set_mechanism("node__a", CategoricalTableMechanism())
    model.set_mechanism("b.c", LinearGaussianMechanism())
    model.fit(data)
    torch.save({"model": model, "data": data, "ref": model.log_prob(data).detach()}, out)


if __name__ == "__main__":
    main(sys.argv[1])
