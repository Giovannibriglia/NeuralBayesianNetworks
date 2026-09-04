"""Generate the six Colab-ready benchmark notebooks under benchmarking/notebooks/.

Run from the repo root:  python <this file>
Every notebook is built from the same cell template so they stay consistent;
only the per-experiment table below differs.
"""
from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

REPO_ROOT = Path(__file__).resolve()
while not (REPO_ROOT / "pyproject.toml").exists():
    REPO_ROOT = REPO_ROOT.parent
OUT_DIR = REPO_ROOT / "benchmarking" / "notebooks"

GITHUB_URL = "https://github.com/Giovannibriglia/NeuralBayesianNetworks"
COLAB_BASE = "https://colab.research.google.com/github/Giovannibriglia/NeuralBayesianNetworks/blob/master/benchmarking/notebooks"

# name, subcommand, config path (repo-relative), benchmark, config_name, title, description, runtime note
EXPERIMENTS = [
    dict(
        name="inference_speed",
        sub="inference",
        config="benchmarking/configs/synthetic/speed/inference_speed.yaml",
        benchmark="synthetic",
        config_name="batch_speed",
        title="Inference speed (batched queries)",
        description=(
            "Timing-only benchmark of batched inference on synthetic networks "
            "(n=50, four families, five seeds). Sweeps `batch_sizes` "
            "[1, 64, 256, 512, 1024] over every baseline whose adapter supports "
            "batching; pinned baselines (pgmpy, pyro) run once at B=1. "
            "No accuracy scoring — metrics are `timing` only."
        ),
        runtime="Several hours on a T4. Non-batchable baselines process every query sequentially and may hit the per-cell timeout (documented, expected).",
    ),
    dict(
        name="learning_curves",
        sub="param-learning",
        config="benchmarking/configs/synthetic/learning_curves/learning_curves.yaml",
        benchmark="synthetic",
        config_name="learning_curves",
        title="Learning curves (parameter learning vs n_train)",
        description=(
            "Parameter-learning benchmark that sweeps the training-set size "
            "`n_train_sweep` [64 … 65536] on synthetic networks (n=30, three "
            "families, five seeds). Emits held-out log-likelihood, parameter "
            "recovery (TV / KL vs the true CPTs) for discrete families and "
            "predictive calibration (PIT-KS, SD-ratio) for continuous families."
        ),
        runtime="Several hours on a T4 (ten n_train values × 14 baselines × 15 problems).",
    ),
    dict(
        name="parameter_learning_complete",
        sub="param-learning",
        config="benchmarking/configs/synthetic/complete/parameter_learning_complete.yaml",
        benchmark="synthetic",
        config_name="param_learning_complete",
        title="Parameter learning (complete)",
        description=(
            "Full parameter-learning benchmark on synthetic networks from n=10 "
            "up to n=20000 nodes (three families, five seeds, n_train=20480). "
            "Emits held-out log-likelihood, parameter recovery (TV / KL) for "
            "discrete families and predictive calibration (PIT-KS, SD-ratio) "
            "for continuous families."
        ),
        runtime="Paper-scale: 10+ hours. The largest sizes (n ≥ 5000) may exceed the RAM / VRAM of a free Colab runtime; those cells are recorded as `oom` rows, not crashes.",
    ),
    dict(
        name="inference_complete",
        sub="inference",
        config="benchmarking/configs/synthetic/complete/inference_complete.yaml",
        benchmark="synthetic",
        config_name="complete",
        title="Inference accuracy + timing (complete)",
        description=(
            "Full inference benchmark on synthetic networks from n=10 up to "
            "n=20000 nodes (three families, five seeds, 128 uniformly random "
            "queries per cell). Emits TV / JSD / W1 per node against the "
            "oracle posterior plus fit and query timing for every baseline."
        ),
        runtime="Paper-scale: 15–20 h on a CUDA server (docs/SERVER_RUN.md). A free Colab session is capped at ~12 h, so expect to need Colab Pro or to trim `n_nodes_list` / `seeds` in the config.",
    ),
    dict(
        name="inference_scalability_complete",
        sub="inference",
        config="benchmarking/configs/synthetic/complete/inference_scalability_complete.yaml",
        benchmark="synthetic",
        config_name="scalability_complete",
        title="Inference scalability (complete)",
        description=(
            "Scalability variant of the inference benchmark: same network grid "
            "(n=10 … 20000, three families, five seeds) but the "
            "`heaviest_by_role` query selector and 12 queries per cell, so the "
            "measurement targets the hardest queries in each network."
        ),
        runtime="Paper-scale: many hours. Per-query and per-fit budgets are 1200 s each; the largest networks are expected to time out or OOM for some baselines (recorded as sentinel rows).",
    ),
    dict(
        name="bnlearn_inference_complete",
        sub="inference",
        config="benchmarking/configs/bnlearn/complete/inference_complete.yaml",
        benchmark="bnlearn",
        config_name="bnlearn_complete",
        title="bnlearn repository inference (complete)",
        description=(
            "Inference benchmark on 31 real networks from the bnlearn "
            "repository (asia … munin4, plus the continuous / hybrid "
            "ecoli70, magic-niab, magic-irri, arth150, healthcare, sangiovese, "
            "mehra). Discrete `.bif` files are downloaded on demand from "
            "bnlearn.com into `~/.cache/nbn/bnlearn/`; the continuous ones "
            "ship with the repo under `benchmarking/data/bnlearn/`."
        ),
        runtime="Paper-scale: many hours. Needs outbound network access for the .bif downloads. The largest networks (barley, munin*, pigs) are memory-heavy — free-tier RAM may not be enough for every baseline.",
    ),
]


def build(exp: dict) -> nbformat.NotebookNode:
    name = exp["name"]
    badge = f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB_BASE}/{name}.ipynb)"
    cells = []

    # ── 0. title ──────────────────────────────────────────────────────────
    cells.append(new_markdown_cell(f"""# NBN benchmark — {exp['title']}

{badge}

{exp['description']}

| | |
|---|---|
| Config | `{exp['config']}` |
| CLI subcommand | `nbn-bench {exp['sub']}` |
| Results dir | `benchmarking/results/benchmark_{exp['benchmark']}_{exp['config_name']}_<timestamp>/` |
| Expected runtime | {exp['runtime']} |

**GPU:** this notebook is tagged for a GPU runtime, so Colab should offer one automatically when you open it.
If cell 1 reports no GPU, go to **Runtime ▸ Change runtime type ▸ Hardware accelerator ▸ GPU** and re-run from the top.

**Session limits:** Colab recycles idle / long sessions. The runner appends one JSONL line per finished cell, so a
disconnect keeps every cell completed so far, but the parquet + figures are produced only at the end. Mount Google
Drive in section 4 to keep a copy of the results outside the ephemeral VM. For unattended multi-day runs prefer a
server (`docs/SERVER_RUN.md`).

Sections: 1 runtime check → 2 install → 3 device check → 4 (optional) Drive → 5 launch → 6 inspect → 7 plot → 8 save.
"""))

    # ── 1. runtime check ──────────────────────────────────────────────────
    cells.append(new_markdown_cell("## 1. Runtime check\n\nConfirms a CUDA GPU is attached *before* spending minutes on installation."))
    cells.append(new_code_cell("""import shutil
import subprocess

try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
print("Running in Google Colab:", IN_COLAB)

if shutil.which("nvidia-smi"):
    print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
else:
    raise SystemExit(
        "No NVIDIA GPU visible. In Colab: Runtime > Change runtime type > "
        "Hardware accelerator > GPU, then re-run this cell."
    )"""))

    # ── 2. install ────────────────────────────────────────────────────────
    cells.append(new_markdown_cell("""## 2. Install

Clones the repository (full history, so the package version recorded in the results is exact) and installs it with
**all four extras** — `bench`, `neural`, `gp`, `mcmc`. Omitting any of them makes the corresponding baselines emit
`not_supported` rows instead of failing loudly, so do not trim this line. Colab ships a CUDA-enabled torch that
satisfies the `torch>=2.2` requirement; it is left untouched."""))
    cells.append(new_code_cell(f"""import os
from pathlib import Path

REPO_URL = "{GITHUB_URL}.git"
BRANCH = "master"           # change to test a feature branch
REPO_DIR = Path("/content/NeuralBayesianNetworks") if IN_COLAB else Path.cwd()

if IN_COLAB:
    if not (REPO_DIR / "pyproject.toml").exists():
        !git clone --branch {{BRANCH}} {{REPO_URL}} {{REPO_DIR}}
    else:
        print("Repository already present; pulling latest", BRANCH)
        !git -C {{REPO_DIR}} pull --ff-only
else:
    # Running locally: walk up to the repo root so relative config paths resolve.
    for parent in [REPO_DIR, *REPO_DIR.parents]:
        if (parent / "pyproject.toml").exists():
            REPO_DIR = parent
            break
    else:
        raise SystemExit(f"no pyproject.toml above {{REPO_DIR}}; run this notebook from inside the repo")

os.chdir(REPO_DIR)
print("Working directory:", Path.cwd())
!git log -1 --format='%h %ad %s' --date=short"""))
    cells.append(new_code_cell("""%pip install -q -e ".[bench,neural,gp,mcmc]"
print("install done")"""))

    # ── 3. device check ───────────────────────────────────────────────────
    cells.append(new_markdown_cell("""## 3. Device check

Verifies that torch sees the GPU and that every optional dependency imports. Stop here and fix the install if
anything fails — a multi-hour run with a silently missing extra is the expensive way to find out."""))
    cells.append(new_code_cell("""import importlib

import psutil
import torch

print("torch", torch.__version__, "| CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("torch cannot see a CUDA device — switch the runtime to GPU (see section 1).")
props = torch.cuda.get_device_properties(0)
print(f"GPU: {props.name} | VRAM: {props.total_memory / 1024**3:.1f} GiB")

# Quick kernel smoke test on the device
x = torch.randn(1024, 1024, device="cuda")
print("matmul on cuda OK:", (x @ x).sum().item() != 0)

print(f"Host RAM: {psutil.virtual_memory().total / 1024**3:.1f} GiB total, "
      f"{psutil.virtual_memory().available / 1024**3:.1f} GiB available")

missing = []
for mod in ["nbn", "benchmarking", "zuko", "gpytorch", "pyro", "pgmpy", "pomegranate",
            "pandas", "pyarrow", "matplotlib", "seaborn"]:
    try:
        m = importlib.import_module(mod)
        print(f"  ok  {mod:12s} {getattr(m, '__version__', '')}")
    except Exception as exc:  # noqa: BLE001
        missing.append(mod)
        print(f"  FAIL {mod:12s} {exc}")
if missing:
    raise SystemExit(f"Missing imports: {missing} — re-run the install cell.")
!python -m benchmarking.cli --help | head -n 5"""))

    # ── 4. optional drive ─────────────────────────────────────────────────
    cells.append(new_markdown_cell("""## 4. (Optional) Google Drive

Set `USE_DRIVE = True` to mount Drive. Section 8 then copies the run directory (parquet, JSONL, `run.log`) and the
figures into `MyDrive/nbn_benchmarks/<experiment>/`. Leave `False` to skip; you can still download a zip at the end."""))
    cells.append(new_code_cell(f"""USE_DRIVE = False
DRIVE_DIR = None

if USE_DRIVE and IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    DRIVE_DIR = Path("/content/drive/MyDrive/nbn_benchmarks/{name}")
    DRIVE_DIR.mkdir(parents=True, exist_ok=True)
    print("Results will be copied to", DRIVE_DIR)
else:
    print("Drive not mounted; results stay on the runtime VM until you download them (section 8).")"""))

    # ── 5. launch ─────────────────────────────────────────────────────────
    cells.append(new_markdown_cell(f"""## 5. Launch

Runs `nbn-bench {exp['sub']}` on the config through `python -m benchmarking.cli` (identical to the console script,
but independent of `PATH`). `--device auto` resolves to CUDA for every baseline without an explicit `device:` in
the YAML; baselines pinned to `cpu` (pyro) or `cuda` (kde / knn / flexcode) keep their pin.

The console shows the tqdm progress bar plus warnings (the bar needs a terminal, which Colab's `!` provides;
do not pipe the command through `tee` or the bar disappears). The full INFO stream, including per-cell
subprocess stderr, lands in `run.log` inside the results dir. Each cell runs in its own subprocess with
a memory cap, so an OOM in one cell becomes an `oom` row rather than killing the run.

Expected runtime: {exp['runtime']}"""))
    cells.append(new_code_cell(f"""CONFIG = "{exp['config']}"
SUBCOMMAND = "{exp['sub']}"
DEVICE = "auto"        # 'auto' | 'cuda' | 'cpu'

assert Path(CONFIG).exists(), f"config not found: {{CONFIG}} (is the working directory the repo root?)"
!python -u -m benchmarking.cli {{SUBCOMMAND}} --config {{CONFIG}} --device {{DEVICE}}"""))

    # ── 6. inspect ────────────────────────────────────────────────────────
    cells.append(new_markdown_cell("""## 6. Inspect the results

Locates the newest results directory for this config, tails `run.log`, and summarises the parquet (row count and
per-status counts per baseline). `ok` rows carry metrics; `timeout` / `oom` / `error` / `not_supported` /
`not_applicable` are sentinel rows that the plotting step turns into the success-rate figure."""))
    cells.append(new_code_cell(f"""import pandas as pd

RESULTS_ROOT = Path("benchmarking/results")
candidates = sorted(RESULTS_ROOT.glob("benchmark_{exp['benchmark']}_{exp['config_name']}_*"))
if not candidates:
    raise SystemExit("No results directory found — did the launch cell finish?")
RUN_DIR = candidates[-1]
print("Run directory:", RUN_DIR)
print("Contents:", sorted(p.name for p in RUN_DIR.iterdir()))

print("\\n--- tail of run.log ---")
!tail -n 20 {{RUN_DIR}}/run.log

parquets = sorted(RUN_DIR.glob("*_metrics.parquet"))
if not parquets:
    raise SystemExit(
        "No *_metrics.parquet in the run dir. The run may have been interrupted; the per-cell JSONL is still there. "
        "Convert it with: from benchmarking.core.output import jsonl_to_parquet; "
        f"jsonl_to_parquet(Path('{{RUN_DIR}}/metrics.jsonl'), Path('{{RUN_DIR}}/{exp['config_name']}_metrics.parquet'))"
    )
PARQUET = parquets[0]
df = pd.read_parquet(PARQUET)
print(f"\\n{{PARQUET.name}}: {{len(df)}} rows, {{df.shape[1]}} columns")
if "status" in df.columns:
    print("\\nStatus counts:")
    print(df["status"].value_counts().to_string())
    if "baseline" in df.columns:
        print("\\nStatus per baseline:")
        print(pd.crosstab(df["baseline"], df["status"]).to_string())
df.head()"""))

    # ── 7. plot ───────────────────────────────────────────────────────────
    cells.append(new_markdown_cell("""## 7. Plot

`nbn-bench plot` reads the parquet and writes the paper figures (PDF) and LaTeX tables per family and coverage
subset into `--output-dir` (layout in `docs/v0.13-paper-figures.md`). The second cell renders the PDFs inline so
you can eyeball them without downloading anything."""))
    cells.append(new_code_cell("""FIG_DIR = RUN_DIR / "figures"
AGGREGATION = "iqm_iqr"     # 'iqm_iqr' (paper default) | 'mean_std'

!python -m benchmarking.cli plot {PARQUET} --output-dir {FIG_DIR} --aggregation {AGGREGATION}

pdfs = sorted(FIG_DIR.rglob("*.pdf"))
texs = sorted(FIG_DIR.rglob("*.tex"))
print(f"\\n{len(pdfs)} figures, {len(texs)} LaTeX tables under {FIG_DIR}")
overview = FIG_DIR.rglob("_subsets_overview.txt")
for ov in overview:
    print(f"\\n--- {ov.relative_to(FIG_DIR)} ---")
    print(ov.read_text())"""))
    cells.append(new_code_cell("""# Render the PDF figures inline (PyMuPDF rasterises them; no poppler needed).
%pip install -q pymupdf
import pymupdf
from IPython.display import Image, Markdown, display

MAX_FIGURES = 40          # raise if you want to see every subset
SHOW_ONLY = "all"         # 'all' shows the mixed-coverage subset; None shows every subset

shown = 0
for pdf in pdfs:
    if SHOW_ONLY and SHOW_ONLY not in pdf.parts:
        continue
    if shown >= MAX_FIGURES:
        print(f"... {len(pdfs) - shown} more figures not shown (raise MAX_FIGURES)")
        break
    page = pymupdf.open(pdf)[0]
    png = page.get_pixmap(dpi=110).tobytes("png")
    display(Markdown(f"**{pdf.relative_to(FIG_DIR)}**"))
    display(Image(data=png))
    shown += 1
print(f"Displayed {shown} figures.")"""))

    # ── 8. save ───────────────────────────────────────────────────────────
    cells.append(new_markdown_cell("""## 8. Save the artifacts

Zips the run directory (parquet, JSONL, `run.log`, figures, tables) and either copies it to Google Drive (if
mounted in section 4) or triggers a browser download. The zip is the complete, reproducible record of this run."""))
    cells.append(new_code_cell("""import shutil

ARCHIVE = shutil.make_archive(str(RUN_DIR), "zip", root_dir=RUN_DIR.parent, base_dir=RUN_DIR.name)
print("Archive:", ARCHIVE, f"({Path(ARCHIVE).stat().st_size / 1024**2:.1f} MiB)")

if DRIVE_DIR is not None:
    dest = DRIVE_DIR / Path(ARCHIVE).name
    shutil.copy2(ARCHIVE, dest)
    print("Copied to Drive:", dest)
elif IN_COLAB:
    from google.colab import files
    files.download(ARCHIVE)
else:
    print("Not in Colab; archive left at", ARCHIVE)"""))

    nb = new_notebook(cells=cells)
    nb.metadata = {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4", "toc_visible": True, "name": f"{name}.ipynb"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    return nb


def write_readme() -> None:
    rows = "\n".join(
        f"| [`{e['name']}.ipynb`]({e['name']}.ipynb) | `{e['config']}` | `{e['sub']}` | "
        f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB_BASE}/{e['name']}.ipynb) |"
        for e in EXPERIMENTS
    )
    (OUT_DIR / "README.md").write_text(f"""# Colab notebooks for the paper-scale benchmarks

One notebook per experiment, ready to open in Google Colab on a GPU runtime. Every notebook has the same eight
sections: runtime check → install (all extras) → device check → optional Drive mount → launch → inspect → plot →
save. Only the config, the CLI subcommand and the results-dir glob differ.

| Notebook | Config | Subcommand | |
|---|---|---|---|
{rows}

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
""")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for exp in EXPERIMENTS:
        nb = build(exp)
        nbformat.validate(nb)
        path = OUT_DIR / f"{exp['name']}.ipynb"
        with path.open("w", encoding="utf-8") as fh:
            nbformat.write(nb, fh)
        print("wrote", path.relative_to(REPO_ROOT))
    write_readme()
    print("wrote", (OUT_DIR / "README.md").relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
