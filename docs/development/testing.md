# Tests and CI

## Layout

| Directory | Contents |
|---|---|
| `tests/unit/` | library tests: core, mechanisms, inference, learning, update, SCM, probability sets |
| `tests/integration/` | end-to-end library examples |
| `tests/bench/` | `nbn.bench` tests; `tests/bench/integration/` runs the runner, the CLI and bnlearn |
| `tests/perf/` | wall-clock assertions (`-m perf`) |
| `tests/data/` | fixtures, including the vendored bnlearn `.bif` networks |
| `tests/conftest.py` | markers, `available_devices()` / `available_accelerators()`, one GPU per xdist worker, offline bnlearn cache, a fixed seed per test |

## Lanes

The markers are declared in `pyproject.toml` with `--strict-markers`, so a
typo in a marker fails the run. The default selection (`pytest` with no `-m`)
is the fast gate; a later `-m` replaces it.

| Lane | Selection | Where |
|---|---|---|
| fast gate | `pytest -n auto` | every pull request, Linux (Python 3.10 and 3.14) and macOS |
| benchmark harness | `-m "(slow or subprocess) and not network and not perf and not cuda and not multi_gpu"` plus two `nbn-bench` smoke runs | every pull request, Linux |
| MPS | `-m mps` with `PYTORCH_ENABLE_MPS_FALLBACK=0` | every pull request, macOS arm64 |
| notebooks | `pytest --nbmake examples/` | every pull request, Linux |
| network and full suite | everything, at the newest dependency versions | nightly (`nightly.yml`) |
| CUDA | `-m "cuda or multi_gpu"` and the slow suite | by hand on a self-hosted GPU runner (`gpu.yml`) |
| perf | `pytest -p no:xdist -m perf tests/perf` | by hand, on a quiet machine |

## Writing tests

- A bug fix needs a regression test that **fails before the fix**.
- Parametrise device-dependent tests with
  `@pytest.mark.parametrize("device", available_devices())`; the `cuda` /
  `mps` markers are added automatically.
- Mark anything over about 3 s of CPU `slow`, anything that downloads
  `network`, anything that spawns benchmark workers `subprocess`.
- Statistical tests: fix the seed and assert against a tolerance derived from
  the Monte-Carlo error, not a hand-tuned constant.

## CI workflows

| Workflow | Trigger | Jobs |
|---|---|---|
| `ci.yml` | pull requests, pushes to `master`, by hand | `lint` (environment check, ruff, mypy), `test` (fast gate, Python 3.10 and 3.14), `bench`, `notebooks`, `macos` (MPS), `docs` (Sphinx build with warnings as errors) |
| `nightly.yml` | daily at 03:00 UTC, by hand | the full suite at the newest resolvable dependency versions, to catch upstream API removals |
| `gpu.yml` | by hand | the CUDA lane on a self-hosted runner |
| `publish.yml` | a pushed `v*` tag; by hand for TestPyPI | build, smoke-install, upload to PyPI ({doc}`release`) |

A newer push to a pull request cancels the older CI run; runs on `master` are
never cancelled.
