# Persistence and `from_bif`

## `save` and `load`

`model.save(path)` writes one `.pt` checkpoint with the graph, the variable
specs and the fitted mechanism modules; `NeuralBayesianNetwork.load(path)`
returns a model that answers queries right away.

```python
import os
import tempfile

import torch
from nbn import NeuralBayesianNetwork, seed_all

seed_all(0)
n = 2000
rain = torch.bernoulli(torch.full((n,), 0.3)).long()
wet = torch.bernoulli(torch.where(rain == 1, 0.9, 0.2)).long()
model = NeuralBayesianNetwork([("Rain", "WetGrass")],
                              {"Rain": ("discrete", 2), "WetGrass": ("discrete", 2)}, device="cpu")
model.auto_mechanisms()
model.fit({"Rain": rain, "WetGrass": wet})

path = os.path.join(tempfile.mkdtemp(), "model.pt")
model.save(path)
again = NeuralBayesianNetwork.load(path, map_location="cpu")
print(torch.equal(model.query(["Rain"], {"WetGrass": 1}), again.query(["Rain"], {"WetGrass": 1})))
```

`map_location` is the device the loaded model lives on. The mechanisms are
pickled with `torch.save`, because most of them create their parameters
during `fit`; so **load only checkpoints you trust**, as with any pickled
PyTorch module. A `StructuralCausalModel` ({doc}`scm`) saves its noise tying
and declared confounders too.

## `state_dict`

A model is an `nn.Module`, so `model.state_dict()` and
`model.load_state_dict(...)` work, for example to checkpoint during training.
Loading a state dict needs a model whose mechanisms are already fitted (the
parameters must exist). To snapshot and restore parameters around an
optimiser step, **copy** the state dict: `state_dict()` shares storage with
the live parameters, so an uncopied snapshot changes with them.

```python
import copy

snap = copy.deepcopy(model.state_dict())
# ... an optimiser step or a warm-started fit ...
model.load_state_dict(snap)
```

## Reading a `.bif` file

`NeuralBayesianNetwork.from_bif(path)` builds a discrete network from a BIF
file (the format of the [bnlearn repository](https://www.bnlearn.com/bnrepository/)).
Every node gets a `CategoricalTableMechanism` holding the file's table, so the
model is ready to query; its marginals match pgmpy's own inference on the same
file. It needs `pgmpy` (`pip install pgmpy`, or the `bench` extra).

```python
BIF = """network toy { }
variable rain { type discrete [ 2 ] { no, yes }; }
variable wet { type discrete [ 2 ] { no, yes }; }
probability ( rain ) { table 0.8, 0.2; }
probability ( wet | rain ) {
  (no) 0.9, 0.1;
  (yes) 0.2, 0.8;
}
"""
bif_path = os.path.join(tempfile.mkdtemp(), "toy.bif")
with open(bif_path, "w") as f:
    f.write(BIF)

toy = NeuralBayesianNetwork.from_bif(bif_path, device="cpu")
print(toy.state_names)                                   # label <-> index per node
yes = toy.state_names["wet"].index("yes")
print(toy.query(["rain"], {"wet": yes}))                 # P(rain | wet = yes) = [1/3, 2/3]
```

States are integer indices; `model.state_names[node][i]` is the label of
state `i`, and survives `save` / `load`. The repository ships three bnlearn
networks in `tests/data/bnlearn/` (asia, alarm, sachs).

The notebook {doc}`../examples/07_batching_devices_persistence` round-trips a
model through `save` / `load` and loads the Asia network with `from_bif`.
