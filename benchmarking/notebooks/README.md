# Colab notebooks for the paper-scale benchmarks

One notebook per experiment, ready to open in Google Colab on a GPU runtime. Every notebook has the same eight
sections: runtime check → install (all extras) → device check → optional Drive mount → launch → inspect → plot →
save. Only the config, the CLI subcommand and the results-dir glob differ.

| Notebook | Config | Subcommand | |
|---|---|---|---|
| [`inference_speed.ipynb`](inference_speed.ipynb) | `benchmarking/configs/synthetic/speed/inference_speed.yaml` | `inference` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Giovannibriglia/NeuralBayesianNetworks/blob/master/benchmarking/notebooks/inference_speed.ipynb) |
| [`learning_curves.ipynb`](learning_curves.ipynb) | `benchmarking/configs/synthetic/learning_curves/learning_curves.yaml` | `param-learning` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Giovannibriglia/NeuralBayesianNetworks/blob/master/benchmarking/notebooks/learning_curves.ipynb) |
| [`parameter_learning_complete.ipynb`](parameter_learning_complete.ipynb) | `benchmarking/configs/synthetic/complete/parameter_learning_complete.yaml` | `param-learning` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Giovannibriglia/NeuralBayesianNetworks/blob/master/benchmarking/notebooks/parameter_learning_complete.ipynb) |
| [`inference_complete.ipynb`](inference_complete.ipynb) | `benchmarking/configs/synthetic/complete/inference_complete.yaml` | `inference` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Giovannibriglia/NeuralBayesianNetworks/blob/master/benchmarking/notebooks/inference_complete.ipynb) |
| [`inference_scalability_complete.ipynb`](inference_scalability_complete.ipynb) | `benchmarking/configs/synthetic/complete/inference_scalability_complete.yaml` | `inference` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Giovannibriglia/NeuralBayesianNetworks/blob/master/benchmarking/notebooks/inference_scalability_complete.ipynb) |
| [`bnlearn_inference_complete.ipynb`](bnlearn_inference_complete.ipynb) | `benchmarking/configs/bnlearn/complete/inference_complete.yaml` | `inference` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Giovannibriglia/NeuralBayesianNetworks/blob/master/benchmarking/notebooks/bnlearn_inference_complete.ipynb) |

The badges assume the notebooks are on `master` of the public GitHub repo; for a branch, open Colab and use
**File ▸ Open notebook ▸ GitHub** with the branch name.

## Notes

- **GPU**: each notebook is tagged `accelerator: GPU`, so Colab picks a GPU runtime when it can. Section 1 aborts if
  none is visible so you do not install for five minutes and then run on CPU.
- **Session limits**: free Colab caps sessions at roughly 12 h and disconnects on idle. The complete configs are
  paper-scale (10–20 h on a CUDA server). The runner appends one JSONL line per finished cell, so a disconnect keeps
  the cells done so far, but the parquet and figures are only written at the end. Mount Drive in section 4 to keep
  a copy, or trim `n_nodes_list` / `seeds` in a copy of the config for a shorter run.
- **Memory**: a free runtime has ~12 GiB host RAM and a 15 GiB T4. The largest synthetic sizes (n ≥ 5000) and the
  biggest bnlearn networks (barley, munin*, pigs) can exceed that for some baselines; the per-cell memory cap turns
  those into `oom` sentinel rows instead of killing the run.
- **bnlearn**: discrete `.bif` files are downloaded from bnlearn.com on first use (network access required).

The notebooks are generated from a single template so they stay in sync; regenerate them with the script in
`benchmarking/notebooks/_make_notebooks.py` after editing.
