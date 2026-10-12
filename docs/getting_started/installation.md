# Installation

NBN needs **Python 3.10 or newer** and **PyTorch 2.2 or newer**. The PyPI
package is `nbn`, and so is the import name. Use a virtual environment.

```bash
pip install nbn                  # core library
pip install "nbn[neural]"        # + normalising flows (zuko)
pip install "nbn[all]"           # + benchmark suite, every baseline, notebook extras
```

## Extras

| Extra | Adds |
|---|---|
| `neural` | zuko, for `NormalizingFlowMechanism` / `ConditionalFlowMechanism` and `nbn.scm.FlowNoiseMechanism` |
| `bench` | the `nbn-bench` runner, pgmpy, pomegranate, pandas, plotting (also needed by `NeuralBayesianNetwork.from_bif`) |
| `mcmc` | the Pyro baseline for the benchmark suite |
| `gp` | GPyTorch (an optional benchmark baseline) |
| `examples` | matplotlib, JupyterLab and zuko, to run the notebooks in `examples/` |
| `all` | all of the above |
| `docs` | Sphinx and the theme, to build this site (from a checkout) |
| `dev` | tests, linters and type checker (from a checkout) |

## Linux

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install "nbn[neural]"
```

**NVIDIA GPUs.** Install a CUDA build of PyTorch first, with the selector at
[pytorch.org](https://pytorch.org/get-started/locally/); `pip install nbn`
then keeps the build you have. Check it with:

```bash
python -c "import torch; print(torch.cuda.is_available())"   # True
```

Several GPUs, containers and cgroup memory limits are supported: the device is
chosen per model (`device="cuda:1"`), and the variable-elimination memory
guard reads the container's memory limit, not the host's
({doc}`../user_guide/devices`).

## macOS (Apple Silicon and Intel)

On Apple-Silicon Macs (M1 and later), NBN uses the GPU through PyTorch's Metal
backend (`mps`). The standard PyPI PyTorch wheel includes it, so there is
nothing extra to install or configure.

1. **Get Python 3.10 or newer.** Check with `python3 --version`; if it is
   missing or older, install it with Homebrew (`brew install python@3.12`) or
   the installer from [python.org](https://www.python.org/downloads/macos/).
2. **Create a virtual environment and install NBN.**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -U pip
   pip install "nbn[neural]"        # or "nbn[all]" for the benchmark suite
   ```

3. **Check that the GPU is visible.**

   ```bash
   python -c "import torch; print(torch.backends.mps.is_available())"   # True
   ```

With the `bench` extra, `nbn-bench check-env` checks the whole environment and
ends with a line like
`platform darwin; accelerator mps (Apple Metal; float64 unavailable, accumulations run in float32)`.

```{note}
**Intel Macs** run NBN on the CPU. PyTorch stopped publishing Intel-macOS
wheels after 2.2, so pip installs torch 2.2.x there. If you then see a NumPy
ABI error, run `pip install "numpy<2"`.
```

## From source

```bash
git clone https://github.com/Giovannibriglia/NeuralBayesianNetworks.git
cd NeuralBayesianNetworks
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e ".[dev,all]"
pytest -n auto                   # the fast test suite
```

With [uv](https://docs.astral.sh/uv/), `uv run --extra dev --extra all pytest -n auto`
creates the environment from `pyproject.toml` on the fly.
{doc}`../development/contributing` has the full developer setup.

## Check the install

```python
import nbn
from nbn.utils.device import accelerator_summary, resolve_device

print(nbn.__version__)
print(resolve_device("auto"), "|", accelerator_summary())
```
