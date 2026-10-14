"""Confounded action rows from several regimes (#315).

Ported from the scm_to_mdp ``confounding/`` reference package (its test oracle,
including Pearl's printed IV bounds), plus the NBN-specific parts (table layout,
devices, the torch box, the gate).  The port was also checked once against the
reference package directly: identical rows (max |diff| 1.1e-16) on 4 random
confounded MDPs x 4 methods x 2 sampling modes.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from nbn.uncertainty import confounded_interval_set, confounding_gate
from nbn.uncertainty.confounded import (
    IVProgram,
    ProgramTooLarge,
    RefutedAssumption,
    RegimeBox,
    natural_rows,
    rows_from_box,
    sensitivity_row,
    sensitivity_row_box,
)
from tests.conftest import available_devices
from tests.unit.scm.confounded_sim import (
    ConfoundedMDP,
    as_data,
    contains,
    model_for,
    pooled,
    random_confounded_mdp,
    rows_as_sas,
)

try:
    import scipy  # noqa: F401

    HAS_SCIPY = True
except ImportError:  # pragma: no cover - depends on the environment
    HAS_SCIPY = False
needs_scipy = pytest.mark.skipif(not HAS_SCIPY, reason="the LP and sensitivity paths need SciPy")


def pearl_cells(z0, z1):
    """Book order (y0x0, y0x1, y1x0, y1x1) for each z -> P[k=z, a=x, y]."""
    P = np.zeros((2, 2, 2))
    for k, (y0x0, y0x1, y1x0, y1x1) in enumerate((z0, z1)):
        P[k, 0, 0], P[k, 1, 0], P[k, 0, 1], P[k, 1, 1] = y0x0, y0x1, y1x0, y1x1
    return P


def ace_bounds(P):
    prog = IVProgram(RegimeBox.point(P))
    return prog.bounds(prog.indicator(1, 1) - prog.indicator(0, 1))


@needs_scipy
# --- external oracles: Pearl, Causality 2e, Ch. 8 (data and bounds as printed) --------------
def test_lipid_ace_matches_printed_bounds():
    # data p. 270; 0.392 <= ACE <= 0.780 p. 271
    lo, hi = ace_bounds(pearl_cells((0.919, 0.000, 0.081, 0.000), (0.315, 0.139, 0.073, 0.473)))
    assert lo == pytest.approx(0.392, abs=1e-5) and hi == pytest.approx(0.780, abs=1e-5)


@needs_scipy
def test_peptaid_ace_matches_printed_bounds_and_beats_natural():
    # data and -0.23 <= ACE <= -0.15 on p. 272
    P = pearl_cells((0.32, 0.32, 0.04, 0.32), (0.02, 0.17, 0.67, 0.14))
    lo, hi = ace_bounds(P)
    assert lo == pytest.approx(-0.23, abs=0.005) and hi == pytest.approx(-0.15, abs=0.005)
    s_lo, s_hi = IVProgram(RegimeBox.point(P)).rows()
    n_lo, n_hi = natural_rows(RegimeBox.point(P))
    assert np.all((n_hi[1] - n_lo[1]) - (s_hi[1] - s_lo[1]) > 0.1)    # the LP is much tighter


@needs_scipy
def test_pearl_table_8_1_collapses_to_a_point():
    # Table 8.1, p. 278; ACE = 0.55 on p. 277
    lo, hi = ace_bounds(pearl_cells((0.55, 0.45, 0.0, 0.0), (0.45, 0.0, 0.0, 0.55)))
    assert lo == pytest.approx(0.55, abs=1e-5) and hi == pytest.approx(0.55, abs=1e-5)


@needs_scipy
def test_instrumental_inequality_violation_is_refuted():
    # max_x sum_y max_z P(y, x | z) = 1.8 > 1 (Pearl, Causality 2e, (8.21), p. 274)
    P = np.zeros((2, 2, 2))
    P[0, 0, 0], P[0, 1, 0] = 0.9, 0.1
    P[1, 0, 1], P[1, 1, 1] = 0.9, 0.1
    with pytest.raises(RefutedAssumption):
        IVProgram(RegimeBox.point(P)).rows()


# --- structure of the LP --------------------------------------------------------------------
def random_iv_instance(rng, nA, nY, K):
    T = np.array(list(np.ndindex(*(nY,) * nA)))
    mu = rng.dirichlet(np.ones(len(T)) * 0.5)
    split = rng.dirichlet(np.ones(nA) * 0.3, size=(K, len(T)))
    P = np.zeros((K, nA, nY))
    for k in range(K):
        for i, t in enumerate(T):
            for a in range(nA):
                P[k, a, t[a]] += mu[i] * split[k, i, a]
    truth = np.array([[mu[T[:, a] == y].sum() for y in range(nY)] for a in range(nA)])
    return P, truth


@needs_scipy
def test_hand_example_lp_point_vs_intersection_half():
    P = np.zeros((2, 2, 2))
    P[0, 0, 1], P[0, 1, 0] = 0.5, 0.5
    P[1, 0, 1], P[1, 1, 1] = 0.5, 0.5
    lo, hi = IVProgram(RegimeBox.point(P)).rows()
    assert hi[0, 0] == pytest.approx(0.0, abs=1e-5)
    assert natural_rows(RegimeBox.point(P))[1][0, 0] == pytest.approx(0.5)


@needs_scipy
@pytest.mark.parametrize("nA,nY,K", [(2, 2, 2), (2, 3, 3), (3, 2, 2)])
def test_lp_sound_and_inside_intersection(nA, nY, K):
    rng = np.random.default_rng(0)
    for _ in range(10):
        P, truth = random_iv_instance(rng, nA, nY, K)
        lo, hi = IVProgram(RegimeBox.point(P)).rows()
        n_lo, n_hi = natural_rows(RegimeBox.point(P))
        assert np.all(lo - 1e-7 <= truth) and np.all(truth <= hi + 1e-7)
        assert np.all(n_lo <= lo + 1e-5) and np.all(hi <= n_hi + 1e-5)


@needs_scipy
def test_declared_randomised_regime_point_identifies():
    rng = np.random.default_rng(1)
    P, truth = random_iv_instance(rng, 2, 2, 1)
    P2 = np.concatenate([P, np.stack([0.7 * truth[0], 0.3 * truth[1]])[None]])
    lo, hi = IVProgram(RegimeBox.point(P2), declared_unconfounded=[1]).rows()
    u_lo, u_hi = IVProgram(RegimeBox.point(P2)).rows()
    assert np.all(hi[1] - lo[1] < 1e-4) and np.all(lo[1] - 1e-6 <= truth[1]) and np.all(truth[1] <= hi[1] + 1e-6)
    assert (u_hi[1] - u_lo[1]).max() > 0.05


@needs_scipy
def test_box_union_contains_point_answer_and_soundness_on_random_boxes():
    rng = np.random.default_rng(7)
    for _ in range(25):
        nA, nY, K = rng.integers(2, 4), rng.integers(2, 4), rng.integers(1, 4)
        P, truth = random_iv_instance(rng, nA, nY, K)
        comp = P.sum(-1, keepdims=True) - P

        def w(shape=P.shape):
            return rng.uniform(0, 0.05, size=shape)

        box = RegimeBox(np.clip(P - w(), 0, 1), np.clip(P + w(), 0, 1),
                        np.clip(comp - w(), 0, 1), np.clip(comp + w(), 0, 1), np.full(K, 100.0))
        assert box.contains(P)
        lo, hi, _ = rows_from_box(box, "lp")
        n_lo, n_hi, _ = rows_from_box(box, "intersection")
        assert np.all(lo - 1e-7 <= truth) and np.all(truth <= hi + 1e-7)
        assert np.all(n_lo <= lo + 1e-9) and np.all(hi <= n_hi + 1e-9)


@needs_scipy
def test_too_large_is_refused():
    with pytest.raises(ProgramTooLarge):
        IVProgram(RegimeBox.point(np.full((2, 3, 10), 1 / 30)), max_types=100)


# --- closed-form rows and sensitivity -------------------------------------------------------
def test_manski_formula_and_simplex_tightening():
    p_ay = np.array([0.1, 0.3])                       # P(a, y), P(a) = 0.4
    lo, hi = natural_rows(RegimeBox.point(np.stack([p_ay, np.array([0.3, 0.3])])[None]))
    assert np.allclose(lo[0], [0.1, 0.3]) and np.allclose(hi[0], [0.7, 0.9])
    # P(a=0) = 0.5, all of it on y = 0: y = 0 in [0.5, 1], the others in [0, 0.5]
    t_lo, t_hi, _ = rows_from_box(RegimeBox.point(np.array([[[0.5, 0.0, 0.0], [0.2, 0.2, 0.1]]])), "manski")
    assert np.allclose(t_lo[0], [0.5, 0.0, 0.0]) and np.allclose(t_hi[0], [1.0, 0.5, 0.5])


def test_sensitivity_limits():
    p_ay = np.array([0.1, 0.25, 0.05])
    p_a = p_ay.sum()
    lo, hi = sensitivity_row(p_ay, p_a, 1.0)
    assert np.allclose(lo, p_ay / p_a) and np.allclose(hi, p_ay / p_a)
    lo, hi = sensitivity_row(p_ay, p_a, 1e9)
    assert np.allclose(lo, p_ay, atol=1e-6) and np.allclose(hi, 1 - (p_a - p_ay), atol=1e-6)


@needs_scipy
def test_sensitivity_contains_truth_when_lambda_is_right():
    rng = np.random.default_rng(0)
    for _ in range(100):
        lam = 1 + 3 * rng.uniform()
        pw, base = rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)
        odds = base / (1 - base)
        pi = np.array([odds * lam, odds / lam])
        pi = pi / (1 + pi)
        q = rng.dirichlet(np.ones(3), size=2)
        w_prob = np.array([pw, 1 - pw])
        p_ay = (w_prob * pi) @ q
        truth = w_prob @ q
        p_a = p_ay.sum()
        o_obs = p_a / (1 - p_a)
        lam_eff = max(max(pi / (1 - pi)) / o_obs, o_obs / min(pi / (1 - pi)))
        lo, hi = sensitivity_row(p_ay, p_a, lam_eff)
        assert np.all(lo - 1e-7 <= truth) and np.all(truth <= hi + 1e-7)
        b_lo, b_hi = sensitivity_row_box(RegimeBox.point(p_ay[None, None, :]), 0, 0, lam_eff)
        assert np.all(b_lo - 1e-7 <= truth) and np.all(truth <= b_hi + 1e-7)


@needs_scipy
def test_sensitivity_box_never_refutes_rare_or_tiny_actions():
    P = np.array([[[1e-7, 2e-7], [0.5, 0.4999997]]])
    lo, hi = sensitivity_row_box(RegimeBox.point(P), 0, 0, 1.5)
    assert (hi - lo).max() < 0.5
    for p in (1e-6, 1e-7, 1e-8):
        P = np.array([[[p / 2, p / 2], [0.5, 0.5 - p]]])
        b = RegimeBox.point(P)
        near = RegimeBox(b.cell_lo, b.cell_hi + 1e-15, b.comp_lo, b.comp_hi + 1e-15, b.n)
        lo, hi = sensitivity_row_box(near, 0, 0, 1.5)
        t_lo, t_hi = sensitivity_row(P[0, 0], p, 1.5)
        assert np.all(lo <= t_lo + 1e-9) and np.all(t_hi <= hi + 1e-9)


@needs_scipy
# --- the node-level builder -----------------------------------------------------------------
@pytest.mark.parametrize("method", ["auto", "intersection"])
@pytest.mark.parametrize("interval_method", ["kl", "clopper_pearson"])
def test_rows_contain_the_interventional_kernel(method, interval_method):
    rng = np.random.default_rng(0)
    mdp = random_confounded_mdp(rng, n_states=3, n_actions=2, n_regimes=3)
    data = as_data(*pooled(mdp, rng))
    res = confounded_interval_set(model_for(3, 2), data, "s'", "a", "k", method=method, delta=0.05,
                                  sampling="trajectory", interval_method=interval_method)
    assert contains(res, mdp.interventional_kernel(), 3, 2)
    expected = "lp" if method == "auto" else "intersection"
    assert set(res.method_used) <= {expected, "unvisited"} and expected in set(res.method_used)
    assert res.set.guarantee == "pac_under_model" and res.set.level == pytest.approx(0.95)
    assert len(res.assumptions) == 5


@pytest.mark.parametrize("device", available_devices())
def test_layout_follows_the_dag_parent_order_and_device(device):
    rng = np.random.default_rng(3)
    mdp = random_confounded_mdp(rng, n_states=3, n_actions=2, n_regimes=2)
    s, a, s2, k = pooled(mdp, rng, n_episodes=80)
    base = confounded_interval_set(model_for(3, 2), as_data(s, a, s2, k), "s'", "a", "k",
                                   method="intersection", sampling="trajectory")
    flipped = confounded_interval_set(model_for(3, 2, device=device, action_first=True),
                                      as_data(s, a, s2, k, device), "s'", "a", "k",
                                      method="intersection", sampling="trajectory")
    assert flipped.parents == ("a", "s") and base.parents == ("s", "a")
    assert flipped.set.lower.device.type == torch.device(device).type
    for x, y in zip(rows_as_sas(base, 3, 2), rows_as_sas(flipped, 3, 2)):
        np.testing.assert_allclose(x, y, atol=1e-5)          # float32 on MPS


def test_unvisited_configuration_and_unused_action_get_the_simplex():
    s = np.array([0, 0, 1, 1])
    a = np.array([0, 1, 0, 0])
    s2 = np.array([1, 0, 0, 1])
    k = np.array([0, 0, 1, 1])
    res = confounded_interval_set(model_for(3, 2), as_data(s, a, s2, k), "s'", "a", "k", method="intersection")
    assert res.method_used[2] == "unvisited"
    lo, hi = rows_as_sas(res, 3, 2)
    assert np.all(lo[2] == 0) and np.all(hi[2] == 1)
    unseen = res.set.unseen.reshape(3, 2)
    assert bool(unseen[2].all()) and bool(unseen[1, 1]) and not bool(unseen[1, 0])  # action 1 never at s=1


def test_regime_acting_on_the_environment_is_refuted():
    n = 400
    s = np.zeros(2 * n, int)
    a = np.tile(np.r_[np.zeros(9, int), 1], 2 * n // 10)
    s2 = np.r_[np.zeros(n, int), np.ones(n, int)]
    k = np.r_[np.zeros(n, int), np.ones(n, int)]
    data = as_data(s, a, s2, k)
    with pytest.raises(RefutedAssumption):
        confounded_interval_set(model_for(2, 2), data, "s'", "a", "k", on_refuted="raise")
    flagged = confounded_interval_set(model_for(2, 2), data, "s'", "a", "k")
    assert flagged.method_used[0] == "refuted" and bool(flagged.refuted[0]) and flagged.notes
    assert bool(flagged.set.unseen.reshape(2, 2)[0].all())


@needs_scipy
def test_sensitivity_with_the_exact_lambda_is_sound_and_tighter_than_manski():
    rng = np.random.default_rng(4)
    mdp = random_confounded_mdp(rng, n_states=2, n_actions=2, n_regimes=2, eps=0.3)
    lam = 1.0
    for kk in range(2):
        pi = mdp.policies[kk]
        p_obs = pi.mean(1, keepdims=True)
        odds, odds_obs = pi / (1 - pi), p_obs / (1 - p_obs)
        lam = max(lam, float(np.max(odds / odds_obs)), float(np.max(odds_obs / odds)))
    data = as_data(*pooled(mdp, rng))
    sens = confounded_interval_set(model_for(2, 2), data, "s'", "a", "k", method="sensitivity", lam=lam,
                                   sampling="trajectory", interval_method="clopper_pearson")
    man = confounded_interval_set(model_for(2, 2), data, "s'", "a", "k", method="intersection",
                                  sampling="trajectory", interval_method="clopper_pearson")
    assert contains(sens, mdp.interventional_kernel(), 2, 2)
    assert float(sens.set.width().mean()) <= float(man.set.width().mean()) + 1e-9


@needs_scipy
def test_declared_randomised_logger_narrows_rows_after_dropping_unvisited_regimes():
    rng = np.random.default_rng(9)
    mdp = random_confounded_mdp(rng, n_states=2, n_actions=2, n_regimes=3)
    mdp.policies[2] = 0.5
    s, a, s2, k = pooled(mdp, rng, n_episodes=200)
    keep = ~((k == 1) & (s == 1))                       # regime 1 never visits state 1
    data = as_data(s[keep], a[keep], s2[keep], k[keep])
    plain = confounded_interval_set(model_for(2, 2), data, "s'", "a", "k", method="lp", sampling="trajectory")
    dec = confounded_interval_set(model_for(2, 2), data, "s'", "a", "k", method="lp", sampling="trajectory",
                                  declared_unconfounded=[2])
    assert contains(dec, mdp.interventional_kernel(), 2, 2)
    w_p = rows_as_sas(plain, 2, 2)
    w_d = rows_as_sas(dec, 2, 2)
    assert (w_d[1] - w_d[0])[1].mean() < (w_p[1] - w_p[0])[1].mean() - 0.05


def test_validation():
    rng = np.random.default_rng(0)
    mdp = random_confounded_mdp(rng, n_states=2, n_actions=2, n_regimes=2)
    data = as_data(*pooled(mdp, rng, n_episodes=5, horizon=5))
    m = model_for(2, 2)
    with pytest.raises(ValueError, match="needs lam"):
        confounded_interval_set(m, data, "s'", "a", "k", method="sensitivity")
    with pytest.raises(ValueError, match="only used"):
        confounded_interval_set(m, data, "s'", "a", "k", lam=2.0)
    with pytest.raises(ValueError, match="not a parent"):
        confounded_interval_set(m, data, "s'", "s'", "k")
    with pytest.raises(KeyError):
        confounded_interval_set(m, data, "s'", "a", "nope")
    with pytest.raises(ValueError, match="out of range"):
        confounded_interval_set(m, data, "s'", "a", "k", declared_unconfounded=[7])
    with pytest.raises(ValueError, match="n_regimes"):
        confounded_interval_set(m, data, "s'", "a", "k", n_regimes=1)


def test_scipy_free_intersection_path(monkeypatch):
    """method='intersection' with the default KL box needs no SciPy."""
    import builtins

    real_import = builtins.__import__

    def no_scipy(name, *args, **kwargs):
        if name.startswith("scipy"):
            raise ImportError("no scipy")
        return real_import(name, *args, **kwargs)

    rng = np.random.default_rng(0)
    mdp = random_confounded_mdp(rng, n_states=2, n_actions=2, n_regimes=2)
    data = as_data(*pooled(mdp, rng, n_episodes=40))
    monkeypatch.setattr(builtins, "__import__", no_scipy)
    res = confounded_interval_set(model_for(2, 2), data, "s'", "a", "k", method="intersection")
    auto = confounded_interval_set(model_for(2, 2), data, "s'", "a", "k", method="auto")
    assert set(auto.method_used) <= {"intersection (no LP: SciPy not installed)", "unvisited"}  # labelled fallback
    assert torch.equal(res.set.lower, auto.set.lower)


def test_gate():
    class Result:
        def __init__(self, p):
            self.p_value = p

        def rejected(self, alpha=0.05):
            return self.p_value < alpha

    with pytest.raises(TypeError, match="bool"):
        confounding_gate([1])                         # a regime list belongs to the builder
    assert confounding_gate(False).use_confounded_rows
    assert not confounding_gate(True).use_confounded_rows
    assert not confounding_gate(True, Result(0.4)).use_confounded_rows
    forced = confounding_gate(True, Result(0.001))
    assert forced.use_confounded_rows and "rejected" in forced.reason
    assert confounding_gate(False, Result(0.9)).use_confounded_rows      # a pass never certifies


# --- the value level (RQ2 C8) ---------------------------------------------------------------
def _worst_case_row(lo, hi, v):
    p = lo.astype(float).copy()
    rest = 1.0 - p.sum()
    for j in np.argsort(v, kind="stable"):
        add = min(hi[j] - p[j], rest)
        p[j] += add
        rest -= add
    return p


def _robust_value(lo, hi, reward, policy, gamma, iters=2000):
    S, A, _ = lo.shape
    v = np.zeros(S)
    for _ in range(iters):
        q = np.array([[_worst_case_row(lo[s, a], hi[s, a], v) @ v for a in range(A)] for s in range(S)])
        v = reward + gamma * (policy * q).sum(1)
    return v


def _rows(mdp, regimes, method):
    S, A = mdp.n_states, mdp.n_actions
    lo, hi = np.zeros((S, A, S)), np.ones((S, A, S))
    for s in range(S):
        P = np.stack([mdp.observed(kk)[s] for kk in regimes])
        lo[s], hi[s], _ = rows_from_box(RegimeBox.point(P), method)
    return lo, hi


@needs_scipy
def test_row_level_intersection_strictly_beats_value_level_max():
    S1, S2, GOOD, BAD = range(4)
    env = np.zeros((4, 2, 2, 4))
    for s, nxt in ((S1, S2), (S2, GOOD)):
        env[s, 0, :, BAD] = 1.0
        env[s, 1, 0, nxt], env[s, 1, 0, BAD] = 0.9, 0.1
        env[s, 1, 1, nxt], env[s, 1, 1, BAD] = 0.5, 0.5
    for s in (GOOD, BAD):
        env[s, :, :, s] = 1.0
    pol = np.zeros((2, 4, 2, 2))
    pol[:, :, :, 0] = 1.0
    pol[0, S1, 0], pol[0, S1, 1] = [0.2, 0.8], [0.7, 0.3]
    pol[1, S2, 0], pol[1, S2, 1] = [0.3, 0.7], [0.8, 0.2]
    mdp = ConfoundedMDP(env, pol)
    reward = np.array([0.0, 0.0, 1.0, 0.0])
    policy = np.array([[0, 1], [0, 1], [1, 0], [1, 0]], float)
    P = mdp.interventional_kernel()
    truth = np.linalg.solve(np.eye(4) - 0.9 * np.einsum("sa,sat->st", policy, P), reward)
    per_regime = [_robust_value(*_rows(mdp, [kk], "intersection"), reward, policy, 0.9) for kk in (0, 1)]
    value_level = np.maximum(*per_regime)
    row_level = _robust_value(*_rows(mdp, [0, 1], "intersection"), reward, policy, 0.9)
    for v in [*per_regime, row_level]:
        assert np.all(v <= truth + 1e-8)
    assert np.all(row_level >= value_level - 1e-8)
    assert row_level[S1] > value_level[S1] + 0.1


@needs_scipy
@pytest.mark.slow
@pytest.mark.parametrize("method", ["auto", "intersection"])
def test_joint_coverage(method):
    rng = np.random.default_rng(11)
    mdp = random_confounded_mdp(rng, n_states=3, n_actions=2, n_regimes=2)
    truth = mdp.interventional_kernel()
    reps, misses = 40, 0
    for _ in range(reps):
        data = as_data(*pooled(mdp, rng, n_episodes=40, horizon=15))
        res = confounded_interval_set(model_for(3, 2), data, "s'", "a", "k", method=method, delta=0.1,
                                      sampling="trajectory")
        misses += not contains(res, truth, 3, 2)
    assert misses / reps <= 0.1



# --- review regressions (#324) --------------------------------------------------------------
def test_declarations_unused_by_non_lp_methods_are_noted():
    rng = np.random.default_rng(5)
    mdp = random_confounded_mdp(rng, n_states=2, n_actions=2, n_regimes=2)
    data = as_data(*pooled(mdp, rng, n_episodes=60))
    res = confounded_interval_set(model_for(2, 2), data, "s'", "a", "k", method="intersection",
                                  declared_unconfounded=[1])
    assert any("not used" in n for n in res.notes)


@needs_scipy
def test_vectorised_intersection_equals_the_per_configuration_path():
    from nbn.uncertainty.confounded import RegimeBox, rows_from_box

    rng = np.random.default_rng(2)
    mdp = random_confounded_mdp(rng, n_states=3, n_actions=2, n_regimes=3)
    data = as_data(*pooled(mdp, rng))
    res = confounded_interval_set(model_for(3, 2), data, "s'", "a", "k", method="intersection",
                                  interval_method="clopper_pearson", sampling="trajectory")
    lo, hi = rows_as_sas(res, 3, 2)
    # per configuration through the box path (as the LP path does)
    from nbn.uncertainty.binomial import binomial_interval
    from nbn.uncertainty.builders import per_interval_delta
    from nbn.uncertainty.counts import count_table_for
    from nbn.core.variables import DiscreteVariable
    spec = dict(model_for(3, 2).variables)
    spec["__regime__"] = DiscreteVariable("__regime__", 3)
    N = count_table_for(spec, {**data, "__regime__": data["k"]}, "s'", ("s", "__regime__", "a"))[0].reshape(3, 3, 2, 3)
    trials = N.sum(dim=(2, 3), keepdim=True).expand_as(N)
    d = per_interval_delta(0.05, 3 * 3 * 2 * 2 * 3, trials, "trajectory")
    cl, ch = binomial_interval(N, trials, d, method="clopper_pearson")
    ml, mh = binomial_interval(N.sum(-1, keepdim=True) - N, trials, d, method="clopper_pearson")
    for u in range(3):
        seen = (N[u].sum(dim=(1, 2)) > 0).numpy()
        box = RegimeBox(cl[u].numpy()[seen], ch[u].numpy()[seen], ml[u].numpy()[seen], mh[u].numpy()[seen],
                        N[u].sum(dim=(1, 2)).numpy()[seen])
        r_lo, r_hi, _ = rows_from_box(box, "intersection")
        np.testing.assert_allclose(lo[u], r_lo, atol=1e-12)
        np.testing.assert_allclose(hi[u], r_hi, atol=1e-12)


def test_float32_cast_rounds_outward():
    from nbn.uncertainty.confounded import _outward

    lo = np.array([0.30000001192092896 - 1e-9, 0.1])
    hi = np.array([0.30000001192092896 + 1e-9, 0.7])
    lo32, hi32 = _outward(lo, hi, torch.float32, "cpu")
    assert bool((lo32.double() <= torch.as_tensor(lo)).all()) and bool((hi32.double() >= torch.as_tensor(hi)).all())


@pytest.mark.parametrize("device", available_devices())
def test_results_live_on_the_model_device(device):
    rng = np.random.default_rng(0)
    mdp = random_confounded_mdp(rng, n_states=2, n_actions=2, n_regimes=2)
    data = as_data(*pooled(mdp, rng, n_episodes=30))                     # data on the CPU
    res = confounded_interval_set(model_for(2, 2, device=device), data, "s'", "a", "k", method="intersection")
    dev = torch.device(device).type
    assert res.set.lower.device.type == dev and res.refuted.device.type == dev and res.counts.device.type == dev
    _ = res.counts[res.refuted]                                            # same device: no error



# --- review round 2 regressions (#324) ------------------------------------------------------
def test_fractional_regime_ids_are_rejected():
    rng = np.random.default_rng(0)
    mdp = random_confounded_mdp(rng, n_states=2, n_actions=2, n_regimes=2)
    data = as_data(*pooled(mdp, rng, n_episodes=5, horizon=5))
    data["k"] = data["k"].float() * 0.5
    with pytest.raises(ValueError, match="whole numbers"):
        confounded_interval_set(model_for(2, 2), data, "s'", "a", "k", method="intersection")


def test_binary_child_uses_one_event_per_cell():
    rng = np.random.default_rng(1)
    mdp = random_confounded_mdp(rng, n_states=2, n_actions=2, n_regimes=2)
    data = as_data(*pooled(mdp, rng, n_episodes=80))
    res = confounded_interval_set(model_for(2, 2), data, "s'", "a", "k", method="intersection",
                                  sampling="trajectory")
    assert contains(res, mdp.interventional_kernel(), 2, 2)


@needs_scipy
def test_lp_program_reuse_equals_a_fresh_build():
    rng = np.random.default_rng(4)
    P, _ = random_iv_instance(rng, 2, 3, 2)
    P2, _ = random_iv_instance(rng, 2, 3, 2)
    prog = IVProgram(RegimeBox.point(P))
    fresh = IVProgram(RegimeBox.point(P2)).rows()
    reused = prog.rebind(RegimeBox.point(P2)).rows()
    np.testing.assert_allclose(fresh[0], reused[0], atol=1e-9)
    np.testing.assert_allclose(fresh[1], reused[1], atol=1e-9)


def test_refuted_rows_are_exported_as_refuted(tmp_path):
    from nbn.structure import export_factored_sets, load_factored_sets

    n = 400
    s = np.zeros(2 * n, int)
    a = np.tile(np.r_[np.zeros(9, int), 1], 2 * n // 10)
    s2 = np.r_[np.zeros(n, int), np.ones(n, int)]
    k = np.r_[np.zeros(n, int), np.ones(n, int)]
    res = confounded_interval_set(model_for(2, 2), as_data(s, a, s2, k), "s'", "a", "k", method="intersection")
    ex = load_factored_sets(export_factored_sets(tmp_path / "c", {"s'": res}, model_for(2, 2), ["s"], ["a"]))
    rec = ex.manifest["factors"][0]
    assert rec["confounded"]["refuted_configurations"] >= 1
    assert bool(ex.array("f0", "refuted")[:2].all())                      # s = 0 rows: refuted, not just unseen


def test_lp_unavailable_reason_names_the_limit():
    from nbn.uncertainty.confounded import lp_ok_reason

    if HAS_SCIPY:
        assert "max_vars" in lp_ok_reason(2, 10, 20, max_types=20_000, max_vars=1_000)
        assert "max_types" in lp_ok_reason(3, 10, 2, max_types=100, max_vars=200_000)


# --- third review round (#324) ----------------------------------------------------------------
def test_gate_accepts_numpy_and_torch_booleans():
    for flag in (np.bool_(True), torch.tensor(True), np.all([True, True])):
        assert confounding_gate(flag).use_confounded_rows is False
    assert confounding_gate(torch.tensor(False)).use_confounded_rows is True
    with pytest.raises(TypeError, match="not a collection"):
        confounding_gate([0, 1])
    with pytest.raises(TypeError, match="not a collection"):
        confounding_gate(torch.tensor([True, False]))
    with pytest.raises(TypeError, match="must be a bool"):
        confounding_gate(1)


def test_lp_exceptions_are_exported():
    from nbn.uncertainty import LPFailure, ProgramTooLarge, RefutedAssumption  # noqa: F401


@needs_scipy
def test_binary_child_bounds_follow_from_the_y0_solves():
    from nbn.uncertainty.confounded import IVProgram as Program

    rng = np.random.default_rng(3)
    for _ in range(10):
        K, nA = int(rng.integers(1, 4)), int(rng.integers(2, 4))
        P, _ = random_iv_instance(rng, nA, 2, K)
        comp = P.sum(-1, keepdims=True) - P
        w = rng.uniform(0, 0.05, size=P.shape)
        box = RegimeBox(np.clip(P - w, 0, 1), np.clip(P + w, 0, 1), np.clip(comp - w, 0, 1),
                        np.clip(comp + w, 0, 1), np.full(K, 100.0))
        prog = Program(box)
        lo, hi = prog.rows()
        for a in range(nA):
            full_lo, full_hi = prog.bounds(prog.indicator(a, 1))   # the solve the shortcut skips
            assert lo[a, 1] == pytest.approx(np.clip(full_lo, 0, 1), abs=1e-7)
            assert hi[a, 1] == pytest.approx(np.clip(full_hi, 0, 1), abs=1e-7)


def test_closed_form_tightening_matches_the_formula():
    from nbn.uncertainty.confounded import _closed_form

    rng = np.random.default_rng(5)
    lo = rng.uniform(0, 0.4, size=(6, 3, 4))
    hi = np.clip(lo + rng.uniform(0, 0.5, size=lo.shape), 0, 1)
    t_lo, t_hi, _, _ = _closed_form(lo, hi)
    np.testing.assert_allclose(t_lo, np.clip(np.maximum(lo, 1 - (hi.sum(-1, keepdims=True) - hi)), 0, 1))
    np.testing.assert_allclose(t_hi, np.clip(np.minimum(hi, 1 - (lo.sum(-1, keepdims=True) - lo)), 0, 1))


@needs_scipy
def test_auto_stays_within_the_lp_budget():
    rng = np.random.default_rng(0)
    mdp = random_confounded_mdp(rng, n_states=3, n_actions=2, n_regimes=3)
    data = as_data(*pooled(mdp, rng))
    small = confounded_interval_set(model_for(3, 2), data, "s'", "a", "k", method="auto", max_lp_solves=1)
    inter = confounded_interval_set(model_for(3, 2), data, "s'", "a", "k", method="intersection")
    labels = set(small.method_used) - {"unvisited"}
    assert len(labels) == 1 and "max_lp_solves=1" in labels.pop()
    assert torch.equal(small.set.lower, inter.set.lower) and torch.equal(small.set.upper, inter.set.upper)
    lp = confounded_interval_set(model_for(3, 2), data, "s'", "a", "k", method="lp", max_lp_solves=1)
    assert "lp" in set(lp.method_used)                   # an explicit "lp" ignores the budget
