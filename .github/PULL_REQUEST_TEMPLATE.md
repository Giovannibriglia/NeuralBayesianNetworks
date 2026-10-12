## What and why

<!-- What does this change, and why? Link the issue: "Closes #123". -->

## How it was tested

<!-- Tests added or changed. For a bug fix: the regression test fails on master
     and passes here. Devices you ran it on (CPU / CUDA / MPS). -->

## Checklist

- [ ] `ruff check nbn/ tests/ examples/ scripts/` and `mypy nbn/core nbn/mechanisms` pass
- [ ] the fast gate passes (`pytest -n auto`)
- [ ] bug fix: a regression test that fails before the fix
- [ ] new or changed behaviour is documented (docstrings, `docs/`, README or notebooks)
- [ ] no hard-coded device strings; device-dependent tests use `available_devices()`
- [ ] public API changes are described above, for the release notes
