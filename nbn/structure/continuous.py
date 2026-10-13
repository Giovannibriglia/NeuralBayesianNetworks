"""Continuous and mixed factors: fit every candidate once, calibrate one set per factor (#318).

:meth:`HypothesisSet.fit` fits one mechanism per factor ``(node, parents)``:

* **continuous target, linear location** — ``AdditiveNoiseMechanism(location="linear")``.
  The weighted Gram matrix of the union design ``[U_j, 1]`` and its cross
  terms with ``x`` are computed **once** per node; every candidate's normal
  equations are sub-blocks of them (the linear-Gaussian sufficient statistics
  are the moments of ``{X} ∪ U``: Koller & Friedman 2009, §17.2.4, p. 730).
* **continuous target, MLP location** — all candidates trained together
  (:mod:`nbn.structure.batched`), then turned into standard
  ``AdditiveNoiseMechanism(location="mlp")`` objects with their own
  standardisation buffers and a noise law fitted on their own residuals.
* **discrete target with a continuous parent** — the same batched trainer
  with a categorical head, turned into ``StructuralNeuralCategorical``.
* **discrete target, discrete parents** — a ``CategoricalTableMechanism`` from
  the factor cache (#312).

Because every result is a standard NBN mechanism, ``model_for``, the
calibration tools and counterfactuals work unchanged.

Calibrated sets (:meth:`HypothesisSet.calibrated_sets`)
-------------------------------------------------------
One calibrated set per factor on a **calibration split not used for
fitting**: split conformal intervals (``"cqr"`` / ``"residual"``) or conformal
sets for discrete children, or a DKW band on the PIT (``"pit"``).  The level
is split over the distinct factors by a union bound (valid whatever the
dependence created by sharing the calibration rows).  What the level means
is declared per factor:

The budget is split over **all** factors of the hypothesis set (also those
handled by ``factor_sets``), so calibrated sets at ``level`` and PAC table
sets of ``factor_sets(delta = 1 - level)`` hold together at ``level``.

* conformal results are ``"marginal"``: coverage on average over the parents,
  **not** for each parent value (Kochenderfer et al., *Algorithms for
  Validation*, §12.2.3, p. 340; Ex. 12.7, p. 345), so they do not support a
  per-row worst case in robust planning;
* PIT bands are ``"pac_if_pit_independent"``: they bound every conditional at
  once only if the PIT is independent of the parents (declared, testable only
  for violations).

**No selection on the calibration split:** keep every candidate.  Choosing a
candidate by its calibration score and quoting that candidate's guarantee is
not supported; use a fresh split for that.
"""
from __future__ import annotations

import copy
import zlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Mapping, Tuple

import torch
import torch.nn as nn

from nbn.learning.weighting import validate_weights, weighted_moments
from nbn.structure.batched import StackedMLP, train_stacked
from nbn.utils.device import to_device

if TYPE_CHECKING:  # pragma: no cover
    from nbn.structure.hypotheses import HypothesisSet

Factor = Tuple[str, Tuple[str, ...]]
Location = Literal["linear", "mlp"]
CalibrationMethod = Literal["cqr", "residual", "pit"]


@dataclass(eq=False)
class FactorMechanisms:
    """Fitted mechanisms per factor, from :meth:`HypothesisSet.fit`."""

    hypothesis_set: HypothesisSet = field(repr=False)
    mechanisms: Dict[Factor, nn.Module] = field(repr=False)
    # Fitted root mechanisms for the inputs that some candidate uses.
    roots: Dict[str, nn.Module] = field(repr=False)
    kinds: Dict[Factor, str] = field(default_factory=dict)
    # Final training loss per factor (MLP kinds), for diagnostics.
    train_loss: Dict[Factor, float] = field(default_factory=dict)
    config: dict = field(default_factory=dict)

    def __getitem__(self, factor: Factor) -> nn.Module:
        node, parents = factor
        return self.mechanisms[(node, tuple(parents))]


@dataclass(eq=False)
class CalibratedFactorSets:
    """One calibrated set per factor, all holding together at :attr:`level`."""

    hypothesis_set: HypothesisSet = field(repr=False)
    sets: Dict[Factor, Any] = field(repr=False)
    guarantees: Dict[Factor, str] = field(default_factory=dict)
    method: str = "cqr"
    level: float = 0.9
    # The union-bound split: every factor's set is built at 1 - per_factor.
    per_factor: float = 0.1
    skipped: Dict[Factor, str] = field(default_factory=dict)
    models: Dict[Factor, Any] = field(default_factory=dict, repr=False)

    def __getitem__(self, factor: Factor) -> Any:
        node, parents = factor
        return self.sets[(node, tuple(parents))]


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def fit_factors(
    hs: HypothesisSet,
    data: Mapping[str, torch.Tensor],
    location: Location = "mlp",
    noise: Literal["gaussian", "mixture"] = "gaussian",
    hidden: int = 64,
    categorical_hidden: Tuple[int, ...] = (64, 64),
    epochs: int | None = None,
    lr: float | None = None,
    batch_size: int | None = None,
    weights: torch.Tensor | None = None,
    seed: int = 0,
    coupling: str = "gumbel_max",
    alpha: float = 0.0,
    consolidate: bool = True,
    device: str | torch.device | None = None,
) -> FactorMechanisms:
    """Fit every factor of ``hs`` once; see the module docstring.

    ``alpha`` is the Dirichlet smoothing of the table factors and discrete
    roots; ``consolidate`` stores the online-update state of the neural
    categorical factors (so ``model.update`` works), as ``fit`` does.  Each
    candidate's initial weights come from a generator seeded by ``seed`` and
    the factor itself, so adding or reordering candidates does not change
    another candidate's fit.
    """
    from nbn.scm.categorical import StructuralCategoricalTable, StructuralNeuralCategorical
    from nbn.scm.mechanisms import AdditiveNoiseMechanism
    from nbn.update import online_laplace

    if location not in ("linear", "mlp"):
        raise ValueError(f"location must be 'linear' or 'mlp'; got {location!r}.")
    first = data[hs.targets[0]]
    dev = torch.device(device) if device is not None else first.device
    data = to_device(dict(data), dev)
    n = data[hs.targets[0]].shape[0]
    w = validate_weights(weights, n, where="HypothesisSet.fit", device=dev)
    mechs: Dict[Factor, nn.Module] = {}
    kinds: Dict[Factor, str] = {}
    losses: Dict[Factor, float] = {}
    counts = None

    for node in hs.targets:
        cands = hs.candidates[node]
        var = hs.variables[node]
        union = tuple(dict.fromkeys(p for c in cands for p in c))
        cont_parent = {c: any(not hs.variables[p].is_discrete for p in c) for c in cands}
        if var.is_discrete:
            table_cands = [c for c in cands if not cont_parent[c]]
            neural_cands = [c for c in cands if cont_parent[c]]
            if table_cands:
                if counts is None:
                    counts = hs.counts(data, weights=w, input_marginals=False, discrete_only=True)
                for c in table_cands:
                    mech = StructuralCategoricalTable(alpha=alpha, coupling=coupling)
                    mech.fit_from_counts(counts.tables[(node, c)], counts.parent_cards[(node, c)])
                    mechs[(node, c)], kinds[(node, c)] = mech, "table"
            if neural_cands:
                k = int(var.cardinality)
                y = data[node].reshape(n, -1)[:, 0].long()
                sub_union = tuple(p for p in union if any(p in c for c in neural_cands))
                z, col_of = _columns(data, sub_union, n)
                masks = _masks(col_of, z.shape[1], neural_cands)
                gens = [torch.Generator().manual_seed(_seed(seed, node, c)) for c in neural_cands]
                orders = [_cols_in_candidate_order(col_of, c) for c in neural_cands]
                net = StackedMLP(masks, categorical_hidden, k, activation="relu", generators=gens,
                                 input_orders=orders).to(dev)
                final = train_stacked(
                    net, z, y, "cross_entropy", weights=w,
                    epochs=100 if epochs is None else epochs, lr=1e-3 if lr is None else lr,
                    batch_size=512 if batch_size is None else batch_size,
                    order_generator=torch.Generator().manual_seed(_seed(seed, node, ())),
                )
                for i, c in enumerate(neural_cands):
                    mech = StructuralNeuralCategorical(n_classes=k, hidden=categorical_hidden,
                                                       activation="relu", coupling=coupling)
                    mech.set_network(net.export(i, order=_cols_in_candidate_order(col_of, c)),
                                     d_pa=sum(len(col_of[p]) for p in c))
                    mech.noise.n_classes = k
                    mech.to(dev)
                    if consolidate:
                        pa_c = z[:, _cols_in_candidate_order(col_of, c)]
                        online_laplace.consolidate(mech, y, pa_c, weights=w)
                    mechs[(node, c)], kinds[(node, c)] = mech, "categorical_mlp"
                    losses[(node, c)] = final[i]
            continue

        # Continuous target: x = g(pa) + U.
        x = data[node].reshape(n, -1).to(torch.float32)
        roots = [c for c in cands if len(c) == 0]
        with_parents = [c for c in cands if len(c) > 0]
        for c in roots:
            mech = AdditiveNoiseMechanism(location="linear", noise=noise)
            mech.fit_local(x, None, weights=w)
            mechs[(node, c)], kinds[(node, c)] = mech, "root"
        if not with_parents:
            continue
        pa_union, col_of = _columns(data, union, n)
        if location == "linear":
            for c, mech in _fit_linear_subblocks(x, pa_union, col_of, with_parents, w, noise).items():
                mechs[(node, c)], kinds[(node, c)] = mech, "linear"
        else:
            mean, std = weighted_moments(pa_union, w, unbiased=True)
            mean = mean.to(torch.float32)
            std = std.to(torch.float32).nan_to_num(1.0).clamp_min(1e-6)
            z = (pa_union - mean) / std
            masks = _masks(col_of, z.shape[1], with_parents)
            gens = [torch.Generator().manual_seed(_seed(seed, node, c)) for c in with_parents]
            orders = [_cols_in_candidate_order(col_of, c) for c in with_parents]
            net = StackedMLP(masks, (hidden, hidden), x.shape[1], activation="silu", generators=gens,
                             input_orders=orders).to(dev)
            final = train_stacked(
                net, z, x, "mse", weights=w,
                epochs=300 if epochs is None else epochs, lr=1e-2 if lr is None else lr,
                batch_size=batch_size, order_generator=torch.Generator().manual_seed(_seed(seed, node, ())),
            )
            for i, c in enumerate(with_parents):
                cols = _cols_in_candidate_order(col_of, c)
                mech = AdditiveNoiseMechanism(location="mlp", noise=noise, hidden=hidden)
                mech.set_mlp_location(net.export(i, order=cols),
                                      mean[cols], std[cols], output_dim=x.shape[1])
                mech.fit_noise(x, pa_union[:, cols], w)
                mechs[(node, c)], kinds[(node, c)] = mech, "mlp"
                losses[(node, c)] = final[i]

    root_mechs = _fit_roots(hs, data, w, noise, alpha, coupling)
    return FactorMechanisms(
        hypothesis_set=hs, mechanisms=mechs, roots=root_mechs, kinds=kinds, train_loss=losses,
        config={"location": location, "noise": noise, "hidden": hidden, "seed": seed,
                "categorical_hidden": list(categorical_hidden), "coupling": coupling, "alpha": alpha},
    )


def _columns(data, variables: Tuple[str, ...], n: int) -> Tuple[torch.Tensor, Dict[str, List[int]]]:
    """``[N, D]`` float design over ``variables`` (multi-dimensional ones span several columns)."""
    parts, col_of, start = [], {}, 0
    for v in variables:
        part = data[v].reshape(n, -1).to(torch.float32)
        parts.append(part)
        col_of[v] = list(range(start, start + part.shape[1]))
        start += part.shape[1]
    return torch.cat(parts, -1), col_of


def _masks(col_of: Dict[str, List[int]], d: int, cands) -> torch.Tensor:
    masks = torch.zeros(len(cands), d)
    for i, c in enumerate(cands):
        for p in c:
            masks[i, col_of[p]] = 1.0
    return masks


def _cols_in_candidate_order(col_of: Dict[str, List[int]], c: Tuple[str, ...]) -> List[int]:
    return [j for p in c for j in col_of[p]]


def _fit_linear_subblocks(x, pa_union, col_of, cands, w, noise) -> Dict[Tuple[str, ...], nn.Module]:
    """Weighted ridge per candidate from one union Gram matrix (sub-blocks)."""
    from nbn.scm.mechanisms import AdditiveNoiseMechanism, linear_moments

    out: Dict[Tuple[str, ...], nn.Module] = {}
    gram, cross = linear_moments(x, pa_union, w)
    bias_col = gram.shape[0] - 1
    for c in cands:
        mech = AdditiveNoiseMechanism(location="linear", noise=noise)
        cols = _cols_in_candidate_order(col_of, c)
        idx = cols + [bias_col]
        mech.set_linear_from_moments(gram[idx][:, idx], cross[idx], device=x.device)
        mech.fit_noise(x, pa_union[:, cols], w)
        out[c] = mech
    return out


def _fit_roots(hs, data, w, noise, alpha: float = 0.0, coupling: str = "gumbel_max") -> Dict[str, nn.Module]:
    """A fitted root mechanism for every input that some candidate uses."""
    from nbn.scm.categorical import StructuralCategoricalTable
    from nbn.scm.mechanisms import AdditiveNoiseMechanism
    from nbn.uncertainty.counts import count_table_for

    used = {p for cs in hs.candidates.values() for c in cs for p in c}
    roots: Dict[str, nn.Module] = {}
    for v in hs.inputs:
        if v not in used or v not in data:
            continue
        var = hs.variables[v]
        if var.is_discrete:
            mech = StructuralCategoricalTable(alpha=alpha, coupling=coupling)
            table, _ = count_table_for(hs.variables, data, v, (), w)
            mech.fit_from_counts(table, [])
        else:
            mech = AdditiveNoiseMechanism(location="linear", noise=noise)
            mech.fit_local(data[v].reshape(data[v].shape[0], -1).to(torch.float32), None, weights=w)
        roots[v] = mech
    return roots


def _seed(seed: int, node: str, parents: Tuple[str, ...]) -> int:
    """A generator seed from ``seed`` and the factor itself (stable under adding / reordering)."""
    key = f"{node}|{','.join(parents)}".encode()
    return (int(seed) * 1_000_003 + zlib.crc32(key)) % (2**63 - 1)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def calibrate_factors(
    hs: HypothesisSet,
    fitted: FactorMechanisms,
    calibration_data: Mapping[str, torch.Tensor],
    method: CalibrationMethod = "cqr",
    level: float = 0.9,
    groups: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> CalibratedFactorSets:
    """One calibrated set per neural / linear factor, the level split over factors."""
    from nbn.scm import conformal_intervals, conformal_sets, pit_band_set

    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1); got {level}.")
    if method not in ("cqr", "residual", "pit"):
        raise ValueError(f"method must be 'cqr', 'residual' or 'pit'; got {method!r}.")
    if fitted.hypothesis_set is not hs:
        raise ValueError("These mechanisms were fitted by another HypothesisSet.")
    eligible = [f for f in hs.factors if fitted.kinds[f] != "table"]
    skipped = {f: "count-based table: use factor_sets (PAC)" for f in hs.factors if fitted.kinds[f] == "table"}
    if not eligible:
        raise ValueError("No continuous or neural factor to calibrate; use factor_sets.")
    # The budget is split over ALL factors of the set, as factor_sets does, so the
    # calibrated sets here and the PAC table sets of factor_sets(delta = 1 - level)
    # hold together at `level` (union bound over every factor).
    per_factor = (1.0 - level) / hs.n_factors
    calibration_data = to_device(dict(calibration_data), _device_of(fitted))
    sets: Dict[Factor, Any] = {}
    guarantees: Dict[Factor, str] = {}
    models: Dict[Factor, Any] = {}
    for f in eligible:
        node, parents = f
        model = _factor_model(hs, fitted, node, parents)
        models[f] = model
        discrete = hs.variables[node].is_discrete
        if method == "pit":
            res = pit_band_set(model, calibration_data, node, delta=per_factor, groups=groups,
                               generator=generator)
        elif discrete:
            res = conformal_sets(model, calibration_data, node, alpha=per_factor, groups=groups,
                                 generator=generator)
        else:
            res = conformal_intervals(model, calibration_data, node, alpha=per_factor,
                                      method=method, groups=groups, generator=generator)
        sets[f] = res
        guarantees[f] = res.guarantee
    return CalibratedFactorSets(
        hypothesis_set=hs, sets=sets, guarantees=guarantees, method=method, level=level,
        per_factor=per_factor, skipped=skipped, models=models,
    )


def _factor_model(hs, fitted: FactorMechanisms, node: str, parents: Tuple[str, ...]):
    """A small SCM ``parents -> node`` carrying the fitted factor (for the calibration tools)."""
    from nbn.scm import StructuralCausalModel

    variables = {v: hs.variables[v] for v in (*parents, node)}
    model = StructuralCausalModel([(p, node) for p in parents], variables, device=str(_device_of(fitted)))
    model.set_mechanism(node, fitted.mechanisms[(node, parents)])
    for p in parents:
        if p in fitted.roots:
            model.set_mechanism(p, fitted.roots[p])
    return model


def _device_of(fitted: FactorMechanisms) -> torch.device:
    for m in fitted.mechanisms.values():
        for t in list(m.parameters()) + list(m.buffers()):
            return t.device
    return torch.device("cpu")


def model_from_fitted(hs: HypothesisSet, fitted: FactorMechanisms, hypothesis: Mapping[str, Any],
                      device: str | torch.device | None = None):
    """A ``StructuralCausalModel`` for one hypothesis with copies of the fitted mechanisms (on ``device``)."""
    from nbn.scm import StructuralCausalModel

    hs._check_hypothesis(hypothesis)
    used = set(hs.targets) | {p for parents in hypothesis.values() for p in parents}
    variables = {v: spec for v, spec in hs.variables.items() if v in used}
    model = StructuralCausalModel(hs.edges(hypothesis), variables, device=str(_device_of(fitted)))
    for node in model.dag.nodes():
        if node in hs.candidates:
            parents = tuple(hypothesis[node])
            if tuple(model.dag.parents(node)) != parents:
                raise RuntimeError(f"Parent order of '{node}' changed inside the DAG.")
            model.set_mechanism(node, copy.deepcopy(fitted.mechanisms[(node, parents)]))
        elif node in fitted.roots:
            model.set_mechanism(node, copy.deepcopy(fitted.roots[node]))
        else:
            raise ValueError(f"No fitted mechanism for the input '{node}'.")
    if device is not None:
        model.to(torch.device(device))
    return model
