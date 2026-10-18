"""Merge round-4 results from several result folders into results/r4_e*.jsonl.

* e1: drop rows with an int degree (a killed stale process wrote them).
* e2: rows in r4_e2_collapsed.jsonl (rerun after commit 870d2ab, the fixed
  collapse="auto" rule) replace the collapsed rows of the same host/device/threads.
"""
import json, sys
from pathlib import Path
R = Path(sys.argv[1]); srcs = [R / s for s in sys.argv[2:]]
def rd(p): return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
for e in ("e1", "e2", "e3"):
    rows = []
    for s in srcs:
        rs = rd(s / f"r4_{e}.jsonl")
        if e == "e1":
            rs = [r for r in rs if not isinstance(r.get("deg"), int)]
        if e == "e2":
            new = rd(s / "r4_e2_collapsed.jsonl")
            keys = {(r["host"], r["device"], r["threads"]) for r in new}
            rs = [r for r in rs if not (r["mode"] == "collapsed"
                                        and (r["host"], r["device"], r["threads"]) in keys)]
            rs += [dict(r, auto_rule="batch") for r in new]
        rows += rs
    (R / f"r4_{e}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(e, len(rows))
