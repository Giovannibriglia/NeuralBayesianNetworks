"""Reference answers for the continuous queries (no budget): LW with many particles.

For each non-Gaussian problem used by ``cont_infer`` this runs NBN likelihood
weighting with ``--particles`` particles (in chunks) on the true model and
writes ``results/cont_infer_ref.jsonl`` with the estimate and its effective
sample size, so the budgeted answers can be scored against it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cell  # noqa: E402

from nbn.inference.likelihood_weighting import LikelihoodWeightingEngine  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--particles", type=int, default=1_000_000)
    ap.add_argument("--chunk", type=int, default=100_000)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    out = HERE / "results" / "cont_infer_ref.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(l)["problem"] for l in out.read_text().splitlines() if l.strip()}
    for p in sorted((HERE / "problems").glob("ngs_*.pt")):
        pr = torch.load(p, weights_only=False)
        if pr["name"] in done:
            continue
        model = pr["true_model"]
        eng = LikelihoodWeightingEngine(n_samples=a.chunk)
        answers, ess = [], []
        for tgt, ev, thr in cell.continuous_queries(pr):
            num = den = den2 = 0.0
            for _ in range(a.particles // a.chunk):
                lw, buf = eng._run(model, [tgt], {k: torch.tensor([v]) for k, v in ev.items()},
                                   {}, a.chunk, prune=True)
                from nbn.inference.state import get_inference_state
                st = get_inference_state(model, [tgt], tuple(sorted(ev)), (), eng._cache)
                x = buf[0, :, st.node_slices[st.node_to_idx[tgt]]].reshape(-1)
                w = torch.exp(lw[0].double() - 0.0)
                num += float((w * (x > thr).double()).sum())
                den += float(w.sum())
                den2 += float((w * w).sum())
            answers.append(num / den if den > 0 else float("nan"))
            ess.append(den * den / den2 / a.particles if den2 > 0 else 0.0)
        rec = dict(problem=pr["name"], n=pr["n"], deg=pr["deg"], answers=answers,
                   ess_frac=ess, particles=a.particles)
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(pr["name"], [round(x, 3) for x in answers], flush=True)


if __name__ == "__main__":
    main()
