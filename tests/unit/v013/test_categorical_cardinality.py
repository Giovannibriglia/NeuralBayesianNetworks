"""Bug 1a (#127): CategoricalTableMechanism.fit_local must honor the
variable's declared cardinality rather than inferring it from observed
distinct training values.

Pre-fix, training data that never observed a high class index of a node
fitted a truncated CPT, causing factor-axis mismatches ("tensor a (10)
must match b (9)") and ``select() index out of range`` errors at query
time on bnlearn networks with rare states (barley: 7/48 nodes).

Scope: cardinality fix + Laplace smoothing of unobserved classes. A full
configurable Dirichlet prior is deferred to Bug 1b.

See docs/v0.13-nbn-cat-ve-investigation.md.
"""
from __future__ import annotations

import torch

from nbn.mechanisms.parametric.categorical_table import CategoricalTableMechanism


class TestDeclaredCardinality:
    def test_declared_cardinality_honored(self):
        """fit_local spans the declared cardinality even when training data
        observes only a subset of classes."""
        # Observes only classes 0-7 despite declared cardinality 10.
        x = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7] * 4, dtype=torch.long)
        mech = CategoricalTableMechanism()
        mech.fit_local(x, None, n_classes=10)

        assert mech.n_classes == 10
        assert mech.cpt.shape[-1] == 10

    def test_laplace_smoothing_unobserved_classes(self):
        """Declared classes never seen in training get nonzero probability."""
        x = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
        mech = CategoricalTableMechanism()
        mech.fit_local(x, None, n_classes=4)

        probs = mech.cpt[0]  # root node → single CPT row
        assert probs.shape[-1] == 4
        assert (probs > 0).all(), f"unobserved classes have zero prob: {probs}"
        # Observed classes still dominate the unobserved (smoothed) ones.
        assert probs[0] > probs[2]
        assert probs[1] > probs[3]

    def test_backward_compat_when_n_classes_none(self):
        """Without n_classes, falls back to observed-max + 1 (legacy)."""
        x = torch.tensor([0, 1, 2, 1, 0], dtype=torch.long)
        mech = CategoricalTableMechanism()
        mech.fit_local(x, None)

        assert mech.n_classes == 3

    def test_observed_exceeding_declared_is_defended(self):
        """If an observed value exceeds the declared cardinality, the CPT
        still spans the observed range (a too-narrow CPT would truncate)."""
        x = torch.tensor([0, 1, 2, 5], dtype=torch.long)  # max index 5
        mech = CategoricalTableMechanism()
        mech.fit_local(x, None, n_classes=4)  # declared smaller than observed

        assert mech.n_classes == 6  # max(4, observed_max + 1)

    def test_fully_observed_is_noop_vs_legacy(self):
        """When every declared class is observed, the fitted CPT is identical
        whether or not n_classes is supplied (no spurious smoothing)."""
        x = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=torch.long)

        legacy = CategoricalTableMechanism()
        legacy.fit_local(x, None)

        declared = CategoricalTableMechanism()
        declared.fit_local(x, None, n_classes=3)

        assert legacy.n_classes == declared.n_classes == 3
        assert torch.allclose(legacy.cpt, declared.cpt)


class TestDeclaredParentCardinality:
    def test_parent_cardinality_honored(self):
        """Parent axes use declared cardinality, so the CPT row count matches
        the declared parent state space even when a parent value is unobserved.

        VE reads parent cardinality from the *parent's* mechanism while
        ``tabulate`` reshapes via the child's stored ``_parent_cards`` — both
        must agree on the declared value or factor multiplication mismatches.
        """
        # Parent declared cardinality 3, but training only ever sees 0 and 1.
        parents = torch.tensor([[0], [1], [0], [1]], dtype=torch.long)
        x = torch.tensor([0, 1, 1, 0], dtype=torch.long)

        mech = CategoricalTableMechanism()
        mech.fit_local(x, parents, parent_cards=[3], n_classes=2)

        # CPT has one row per declared parent state (3), not observed (2).
        assert mech.cpt.shape[0] == 3
        # tabulate() reshapes to [*parent_cards, K] = [3, 2].
        assert tuple(mech.tabulate().shape) == (3, 2)

    def test_parent_cards_never_below_observed(self):
        """A declared parent cardinality smaller than observed is widened to
        fit observed parent values (avoids index-out-of-range on the row map)."""
        parents = torch.tensor([[0], [1], [2], [3]], dtype=torch.long)  # max 3
        x = torch.tensor([0, 1, 0, 1], dtype=torch.long)

        mech = CategoricalTableMechanism()
        mech.fit_local(x, parents, parent_cards=[2], n_classes=2)  # declared 2

        assert mech.cpt.shape[0] == 4  # max(2, observed_max + 1)
