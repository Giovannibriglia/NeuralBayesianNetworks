"""Regenerate ``master_mdn_flow_{model,ref}.pt`` with *origin/master* code.

These fixtures pin backward compatibility with checkpoints written before the
MDN / normalizing-flow standardisation buffers existed (review finding R1).
They must be produced by the old code, not by the branch under test::

    git archive origin/master nbn | tar -x -C <scratch>/master_src
    PYTHONPATH=<scratch>/master_src .venv/bin/python \\
        tests/data/make_master_mdn_flow_checkpoint.py \\
        tests/data/master_mdn_flow_model.pt tests/data/master_mdn_flow_ref.pt

(origin/master at 64c5f0c when the committed files were made.)
"""
from __future__ import annotations

import sys

import torch

import nbn
from nbn import NeuralBayesianNetwork as NBN
from nbn.mechanisms import MDNMechanism
from nbn.mechanisms.parametric.normalizing_flow import NormalizingFlowMechanism


def main(out_model: str, out_ref: str) -> None:
    assert "master_src" in nbn.__file__, f"not running master code: {nbn.__file__}"
    torch.manual_seed(0)
    n = 200
    a = 50.0 + 5.0 * torch.randn(n, 1)
    b = 0.5 * a + torch.randn(n, 1)
    c = torch.sin(b) + 0.1 * torch.randn(n, 1)
    d = 3.0 + torch.randn(n, 1)
    data = {"A": a, "B": b, "C": c, "D": d}
    model = NBN(
        [("A", "B"), ("B", "C")],
        dict.fromkeys(data, ("continuous", 1)), device="cpu",
    )
    model.set_mechanism("A", MDNMechanism(num_components=2, hidden=(8,)))
    model.set_mechanism("B", MDNMechanism(num_components=2, hidden=(8,)))
    model.set_mechanism("C", NormalizingFlowMechanism(num_transforms=1, hidden=(8,), bins=4))
    model.set_mechanism("D", NormalizingFlowMechanism(num_transforms=1, hidden=(8,), bins=4))
    model.fit(data, epochs=3)
    model.eval()
    with torch.no_grad():
        per_node = model.log_prob(data, per_node=True)
    model.save(out_model)
    torch.save({
        "data": data,
        "per_node": {k: v.detach().clone() for k, v in per_node.items()},
        "state_dicts": {
            node: {key: t.detach().clone() for key, t in mech.state_dict().items()}
            for node, mech in model.mechanisms.items()
        },
    }, out_ref)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
