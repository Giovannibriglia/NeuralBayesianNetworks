# Contributing to NBN

Thank you for helping. Bug reports, fixes, new mechanisms or engines,
documentation and examples are all welcome. By taking part you agree to follow
the [code of conduct](https://github.com/Giovannibriglia/NeuralBayesianNetworks/blob/master/CODE_OF_CONDUCT.md).

- **Found a bug?** Open an
  [issue](https://github.com/Giovannibriglia/NeuralBayesianNetworks/issues/new/choose)
  with the bug template: a minimal script that reproduces it, what you
  expected, what happened, and your environment.
- **Want a feature?** Open a feature request first, so the design can be
  agreed before you write the code.
- **Want to fix something?** Comment on the issue, then follow the steps
  below.

## Development setup

You need Python 3.10 or newer and git. With [uv](https://docs.astral.sh/uv/),
nothing else: `uv run` builds the environment from `pyproject.toml` on first
use.

```bash
git clone https://github.com/Giovannibriglia/NeuralBayesianNetworks.git
cd NeuralBayesianNetworks
uv run --extra dev --extra all python -m pytest -n 8      # fast gate
```

With pip and a virtual environment:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e ".[dev,all]"
pytest -n auto
```

For an NVIDIA GPU, install a CUDA build of PyTorch first
([pytorch.org](https://pytorch.org/get-started/locally/)). On Apple Silicon the
PyPI wheel already includes the `mps` backend. The version comes from git tags
(setuptools-scm), so work in a clone with its history, not in a source
archive.

## Running the tests

The default `pytest` selection is the **fast gate**: every test except those
marked `slow`, `subprocess`, `network`, `perf`, `cuda` and `multi_gpu`. Run it
before every push.

| Marker | Meaning | Run with |
|---|---|---|
| `slow` | more than about 3 s of CPU | `-m slow` |
| `subprocess` | launches benchmark cell workers or the `nbn-bench` CLI | `-m subprocess` |
| `network` | may download bnlearn networks when the cache is cold | `-m network` |
| `perf` | wall-clock or speed-ratio assertions; needs a quiet machine | `-p no:xdist -m perf tests/perf` |
| `cuda` | needs a CUDA device (added automatically to `device="cuda"` cases) | `-m cuda` |
| `multi_gpu` | needs two or more CUDA devices | `-m multi_gpu` |
| `mps` | needs Apple Metal (added automatically to `device="mps"` cases) | `-m mps` |

```bash
pytest -n auto                              # fast gate (the default selection)
pytest -n auto -m "slow or subprocess"      # benchmark-harness lane
pytest -n auto -m "slow or not slow"        # everything this host can run
pytest -m mps                               # Metal tests (macOS); -m cuda / -m multi_gpu on NVIDIA hosts
MPLBACKEND=Agg pytest --nbmake examples/    # execute the example notebooks
```

Device-marked tests skip with a reason on a host without the device; they
never pass by silently running on the CPU. Tests not marked `network` cannot
download: they read the bnlearn networks vendored in `tests/data/bnlearn/`.

## Style

- **Lint:** `ruff check nbn/ tests/ examples/ scripts/` (configuration in
  `ruff.toml`).
- **Types:** `mypy nbn/core nbn/mechanisms` (the type-checked scope; see
  `mypy.ini`). Annotate new public functions everywhere.
- **Docstrings:** numpy style (`Parameters`, `Returns`, `Notes`, ...), since the
  API reference is generated from them. Free-form numpy-style headings
  ("Gradients", "Shape contracts") are fine. Explain *why* a non-obvious rule
  holds where the code relies on it, in the docstring or comment itself.
- Keep the core dependencies minimal (`torch`, `networkx`, `numpy`,
  `opt-einsum`). Anything else goes in an extra and is imported lazily.

## Device rules

NBN runs on Linux (CPU, CUDA, several GPUs, containers with a cgroup memory
limit) and on macOS (CPU and Apple MPS). Code and tests must work on all of
them.

- **No hard-coded device strings.** Take the device from the model
  (`model.device`), the input tensors, or `nbn.utils.device.resolve_device`.
  Never write `"cuda"` or `.cuda()` in library code.
- **Tests parametrise over devices** with `available_devices()` from
  `tests/conftest.py` (`@pytest.mark.parametrize("device", available_devices())`),
  so the macOS CI lane runs them on a real `mps` device. Device-*transfer*
  tests use `available_accelerators()`, so a CPU-only host skips them instead
  of passing vacuously.
- **Move and cast with `nbn.utils.device.to_device_dtype(t, device, dtype)`**,
  not a fused `t.to(device=..., dtype=...)`: it picks the order every backend
  supports (a `float64` target is reached by moving first, any other dtype by
  casting first).
- **Never cast an MPS tensor to `float64`**: Metal has no float64. Accumulate
  in `nbn.utils.device.accum_dtype(device)` (float32 on MPS, float64
  elsewhere) and run small linear solves on `linalg_device(device)`.
- Library kernels must not rely on `PYTORCH_ENABLE_MPS_FALLBACK`; CI runs the
  `mps` tests with it set to `0`.

## Bug fixes and features

- **Every bug fix comes with a regression test that fails before the fix**
  and passes after it. Say in the pull request that you checked it fails on
  `master`.
- **Every feature comes with its documentation:** docstrings, the matching
  page of the user guide in `docs/`, and the README or an example notebook
  when the feature changes how people use NBN. Notebooks in `examples/` are
  stored executed; re-run them (`jupyter nbconvert --to notebook --execute
  --inplace`) when their output changes.
- Changes to the public API need a note in the pull request description; the
  release notes are written from them.

## Building the documentation

The site at <https://nbn.readthedocs.io> is built with Sphinx from `docs/`
(MyST Markdown pages, the notebooks rendered from their stored outputs, the API
reference generated from the docstrings). Warnings are errors, as on Read the
Docs and in CI:

```bash
uv run --extra docs sphinx-build -W --keep-going -b html docs docs/_build/html
# or: pip install -e ".[docs]" && sphinx-build -W --keep-going -b html docs docs/_build/html
open docs/_build/html/index.html             # xdg-open on Linux
```

## Pull requests

1. Branch from an up-to-date `master` (`fix/<issue>-short-name`,
   `feat/...`, `docs/...`).
2. Make the change with its tests and documentation; run ruff, mypy, the fast
   gate and, if you touched the docs, the docs build.
3. Open the pull request against `master` and fill in the template. Link the
   issue (`Closes #123`).
4. CI must be green: lint and types, the fast gate on Python 3.10 and 3.14,
   the benchmark lane, the notebooks, the macOS (MPS) lane and the docs
   build.
5. A maintainer reviews and **squash-merges**; the branch is deleted after the
   merge.

## Commit messages

Use the [Conventional Commits](https://www.conventionalcommits.org/) prefix the
history already follows, an optional scope, and the imperative mood:

```text
fix(inference): honour do= in AmortizedVIEngine (#287)
feat(scm): flow mechanisms and calibrated uncertainty
docs: example notebooks for every NBN feature
```

Common types: `feat`, `fix`, `docs`, `test`, `perf`, `refactor`, `ci`,
`chore`. The squash-merge commit takes the pull request title, so make that
title a good commit message; put the why in the body.

## Releases

Maintainers release by pushing a tag `v<major>.<minor>.<patch>`:

```bash
git tag v0.23.0 && git push origin v0.23.0
```

`.github/workflows/publish.yml` then builds the sdist and the wheel (the
version comes from the tag through setuptools-scm; nothing is edited by hand),
installs the wheel in a clean environment as a smoke test, and uploads to PyPI
with Trusted Publishing. A manual run of the same workflow uploads to TestPyPI
as a dry run. Write the release notes on GitHub from the merged pull requests.
