"""Regression: NBN must tolerate node names containing '.' (e.g. bnlearn's
``pm2.5`` in mehra, ``YR.GLASS`` in magic-niab).

``nn.ModuleDict`` rejects dotted keys, so every NBN mechanism crashed on these
networks. ``_AliasModuleDict`` stores a dot-free alias and reverses it on read,
so all call sites keep using the original node name.
"""
import torch

from nbn.core.network import NeuralBayesianNetwork, _AliasModuleDict
from nbn.mechanisms.categorical_table import CategoricalTableMechanism


class TestAliasModuleDict:
    """The storage wrapper round-trips dotted keys via the mapping protocol."""

    def test_round_trip(self):
        d = _AliasModuleDict()
        m = CategoricalTableMechanism()
        d["pm2.5"] = m
        # getitem / contains / keys / items / iter all speak original names
        assert d["pm2.5"] is m
        assert "pm2.5" in d
        assert list(d.keys()) == ["pm2.5"]
        assert [k for k, _ in d.items()] == ["pm2.5"]
        assert list(d) == ["pm2.5"]
        assert dict(d) == {"pm2.5": m}
        # the underlying submodule key is dot-free (what nn.ModuleDict requires)
        assert "pm2.5" not in d._modules
        assert "pm2__DOT__5" in d._modules

    def test_delete(self):
        d = _AliasModuleDict()
        d["a.b"] = CategoricalTableMechanism()
        del d["a.b"]
        assert "a.b" not in d
        assert len(d) == 0


class TestDottedNodeNamesEndToEnd:
    """A 2-node net with dotted names fits + queries under both a plain-table
    and a neural mechanism (both register through network.py's ModuleDict)."""

    def _build(self):
        return NeuralBayesianNetwork(
            [("x.y", "a.b")],
            variables={"x.y": ("discrete", 2), "a.b": ("discrete", 3)},
            device="cpu",
        )

    def _data(self):
        return {
            "x.y": torch.randint(0, 2, (200,)),
            "a.b": torch.randint(0, 3, (200,)),
        }

    def _run(self, default_discrete):
        model = self._build()
        model.auto_mechanisms(default_discrete=default_discrete)
        # registration + round-trip via the original dotted names
        assert "x.y" in model.mechanisms and "a.b" in model.mechanisms
        assert sorted(dict(model.mechanisms)) == ["a.b", "x.y"]
        model.fit(self._data(), epochs=2)
        q = model.query(["a.b"], evidence={"x.y": 1}).detach()
        assert q.shape == (3,)
        assert torch.isfinite(q).all()
        assert abs(float(q.sum()) - 1.0) < 1e-4

    def test_categorical_table(self):
        self._run("categorical_table")

    def test_neural_categorical(self):
        self._run("neural_categorical")
