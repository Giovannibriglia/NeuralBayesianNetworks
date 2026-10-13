"""Sphinx configuration for the NBN documentation site.

Build locally (warnings are errors, as on Read the Docs and in CI)::

    pip install -e ".[docs]"        # or: uv run --extra docs ...
    sphinx-build -W --keep-going -b html docs docs/_build/html

Pages are MyST Markdown. The executed notebooks in ``examples/`` are rendered
from their stored outputs (myst-nb, never re-executed); they are copied into
``docs/examples/`` at the start of each build (see ``_copy_notebooks``), and
those copies are git-ignored, so ``examples/`` stays the only source.
"""
from __future__ import annotations

import os
import re
import shutil
import sys
import warnings
from pathlib import Path

DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent
sys.path.insert(0, str(ROOT))

GITHUB = "https://github.com/Giovannibriglia/NeuralBayesianNetworks"

# -- Project -----------------------------------------------------------------

project = "NBN"
author = "Giovanni Briglia"
copyright = "Giovanni Briglia"

try:
    from nbn import __version__ as release
except Exception:  # pragma: no cover - a docs build without the package
    release = "unknown"
version = release

# -- General -----------------------------------------------------------------

extensions = [
    "myst_nb",  # MyST Markdown pages + .ipynb notebooks (includes myst_parser)
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "sphinx_design",
]

# .rst first: autosummary writes its generated API pages with the first suffix.
source_suffix = {".rst": "restructuredtext", ".md": "myst-nb", ".ipynb": "myst-nb"}
root_doc = "index"
templates_path = ["_templates"]

exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
    # Design notes and audits that are not part of the site (being removed
    # from the repository; their history stays in git).
    "audits/**",
    "v0.6b/**",
    "v0.6c-a/**",
    "v0.6c-d/**",
    "phase*-design-draft.md",
    "bnlearn-complete-launch-plan.md",
    "v0.13-[!o]*.md",
    "v0.14-*.md",
    "v0.18-*.md",
]

# -- MyST / notebooks --------------------------------------------------------

# The stored notebooks predate nbformat cell ids; reading them is still valid.
warnings.filterwarnings("ignore", message="Cell is missing an id field")

myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "fieldlist"]
# GitHub-style anchors for "#section" links inside the Markdown guides.
myst_heading_anchors = 3
nb_execution_mode = "off"  # the notebooks in examples/ are stored executed
nb_merge_streams = True

# -- API reference -----------------------------------------------------------

autosummary_generate = True
autosummary_imported_members = False
# Members are listed by the autosummary templates in _templates/autosummary/
# (one page per class / function); module pages carry only the docstring.
autodoc_default_options = {"member-order": "bysource"}
# Class docstring plus the __init__ docstring (TensorVariableElimination
# documents its arguments there); see _drop_foreign_init_doc for the rest.
autoclass_content = "both"
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented_params"
autodoc_preserve_defaults = True
# Optional third-party backends (imported lazily by NBN); mock them so the
# API pages build with the docs extra only.
autodoc_mock_imports = ["zuko", "pgmpy", "pyro", "gpytorch"]



def _docstring_headings() -> list[str]:
    """Underlined (numpy-style) headings used in NBN docstrings.

    Besides the standard sections, the docstrings use free-form ones
    ("Gradients", "Shape contracts", ...). Napoleon only ends a section at a
    heading it knows, so each one is registered as a generic custom section
    (rendered as a rubric); otherwise the text after it would be read as more
    parameters.
    """
    from sphinx.ext.napoleon import Config
    from sphinx.ext.napoleon.docstring import NumpyDocstring

    standard = set(NumpyDocstring("", Config())._sections)
    heading = re.compile(r"^[ \t]*(\S[^\n]*?)[ \t]*\n[ \t]*(-{3,})[ \t]*$", re.MULTILINE)
    found = set()
    for path in (ROOT / "nbn").rglob("*.py"):
        if "bench" in path.parts:
            continue
        for m in heading.finditer(path.read_text(encoding="utf-8")):
            if len(m.group(2)) == len(m.group(1)) and m.group(1).lower() not in standard:
                found.add(m.group(1))
    return sorted(found)


napoleon_custom_sections = _docstring_headings()
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False
napoleon_use_ivar = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# -- HTML --------------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = "NBN"
html_static_path: list[str] = []
html_theme_options = {
    "github_url": GITHUB,
    "use_edit_page_button": False,
    "navigation_with_keys": False,
    "show_toc_level": 2,
    "header_links_before_dropdown": 7,
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/nbn/",
            "icon": "fa-brands fa-python",
        },
    ],
}
html_context = {
    "github_user": "Giovannibriglia",
    "github_repo": "NeuralBayesianNetworks",
    "github_version": "master",
    "doc_path": "docs",
}

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True



# GitHub and the PyTorch docs build their "#anchors" in JavaScript.
linkcheck_anchors_ignore_for_url = [r"https://github\.com/.*", r"https://docs\.pytorch\.org/.*"]

# -- Build steps -------------------------------------------------------------


def _copy_notebooks(app) -> None:
    """Copy ``examples/*.ipynb`` into ``docs/examples/`` (git-ignored).

    ``copy2`` keeps the modification time, so an unchanged notebook is not
    re-read on an incremental build.
    """
    dest = DOCS / "examples"
    dest.mkdir(exist_ok=True)
    wanted = set()
    for nb in sorted((ROOT / "examples").glob("*.ipynb")):
        wanted.add(nb.name)
        target = dest / nb.name
        if not target.exists() or os.path.getmtime(target) != os.path.getmtime(nb):
            shutil.copy2(nb, target)
    for stale in dest.glob("*.ipynb"):
        if stale.name not in wanted:
            stale.unlink()


_MD_LINK = re.compile(r"\[([^\]]+)\]\((?!https?://|#|mailto:)([^)\s#]+)(#[^)\s]*)?\)")


def _rewrite_repo_links(app, docname, source) -> None:
    """Make relative Markdown links in the benchmark guides site-safe.

    The guides under ``docs/`` are also read on GitHub, so they link files
    by repository path. A link to another page of the site is kept; a link to
    a repository file outside ``docs/`` points at GitHub; a link to anything
    else (a removed or excluded note) keeps its text and loses the link.
    """
    if not docname or "/" in docname or docname == "index":
        return  # only the top-level guides (BENCHMARKS.md, ...) need this
    text = source[0]

    def fix(m: re.Match) -> str:
        label, target, anchor = m.group(1), m.group(2), m.group(3) or ""
        path = (DOCS / target).resolve()
        if DOCS in path.parents:
            page = path.relative_to(DOCS).with_suffix("").as_posix()
            if page in app.env.found_docs:
                return m.group(0)
        if path.exists() and ROOT in path.parents and DOCS not in path.parents:
            kind = "tree" if path.is_dir() else "blob"
            rel = path.relative_to(ROOT).as_posix()
            return f"[{label}]({GITHUB}/{kind}/master/{rel}{anchor})"
        return label

    source[0] = _MD_LINK.sub(fix, text)


def _drop_foreign_init_doc(app, what, name, obj, options, lines) -> None:
    """Drop an ``__init__`` docstring inherited from outside NBN.

    With ``autoclass_content = "both"`` a class whose ``__init__`` has no
    docstring would show ``torch.nn.Module.__init__``'s ("Initialize internal
    Module state ...") on every mechanism and engine page.
    """
    if what != "class":
        return
    # The class whose __init__ docstring autodoc shows (it walks the MRO).
    owner, doc = None, None
    for klass in getattr(obj, "__mro__", ()):
        init = vars(klass).get("__init__")
        if init is not None and getattr(init, "__doc__", None):
            owner, doc = klass, init.__doc__
            break
    if doc is None or owner.__module__.startswith("nbn"):
        return
    from sphinx.util.docstrings import prepare_docstring

    tail = [line for line in prepare_docstring(doc) if line.strip()]
    body = [line for line in lines if line.strip()]
    if tail and body[-len(tail):] == tail:
        cut = len(lines)
        while tail:
            cut -= 1
            if lines[cut].strip():
                tail.pop()
        del lines[cut:]


def setup(app) -> None:
    app.connect("builder-inited", _copy_notebooks)
    app.connect("source-read", _rewrite_repo_links)
    app.connect("autodoc-process-docstring", _drop_foreign_init_doc)
