"""Extract the shared saddle setup from saddle_discussion_v1.ipynb.

The cell sources are copied verbatim. Only two trailing display blocks are
dropped, and the script asserts on the exact text it drops so a change to the
notebook fails here rather than passing silently.
"""
import io, json, os

ROOT = r"C:\Users\jerem\OneDrive\Desktop\GitHub\TrustRegionRadius.jl"
NB = os.path.join(ROOT, "notebooks", "Saddle", "saddle_discussion_v1.ipynb")
OUT = os.path.join(ROOT, "notebooks", "Saddle", "saddle_problem.jl")

nb = json.load(io.open(NB, encoding="utf-8"))
cells = ["".join(c["source"]) for c in nb["cells"]]

def cut(idx, marker):
    """Cell `idx` up to `marker`, which must occur exactly once."""
    src = cells[idx]
    assert src.count(marker) == 1, (
        "cell %d: marker %r occurs %d times" % (idx, marker[:40], src.count(marker)))
    return src[:src.index(marker)].rstrip()

consts   = cells[9].rstrip()                                  # ETA1 .. KMAX
critpts  = cut(8, 'println("Critical points of f')            # drop the printed table
testfun  = cells[7].rstrip()                                  # f, grad, hess, make_nlp
runners  = cut(15, "let st = run_cfg(")                       # drop the demo run

# Guard: the pieces we rely on must actually be in what we copied.
for needle, blob in (("const ETA1, ETA2", consts),
                     ("const G1, G2, G3", consts),
                     ("const ZETA", consts),
                     ("const MU0, DELTA0", consts),
                     ("const MU_BAR", consts),
                     ("const TOL, KMAX", consts),
                     ("const ORIGIN", critpts),
                     ("const SADDLE", critpts),
                     ("const MINP", critpts),
                     ("const MINM", critpts),
                     ("const X0_DEFAULT", critpts),
                     ("const CRITPTS", critpts),
                     ("f(p) =", testfun),
                     ("grad(p) =", testfun),
                     ("hess(p) =", testfun),
                     ("make_nlp(", testfun),
                     ("function run_cfg(", runners),
                     ("function solve_path(", runners),
                     ("function which_crit(", runners)):
    assert needle in blob, "missing after extraction: %r" % needle

header = '''# =============================================================================
# notebooks/Saddle/saddle_problem.jl
#
# The shared setup for the saddle studies, extracted verbatim from
# `saddle_discussion_v1.ipynb` by `scratchpad/extract_saddle.py` rather than
# retyped, so the two cannot drift. The notebook is the source and is left
# untouched.
#
# Two display blocks were dropped in the copy, the printed table of critical
# points and a single demonstration run. Nothing else was changed, added or
# reordered inside a block. The blocks themselves are ordered constants first.
#
# The caller supplies the packages:
#
#     using TrustRegionRadius, ADNLPModels, LinearAlgebra, Printf
#     include("saddle_problem.jl")
#
# ON THE CONSTANTS. Section~"Settings" of Survey-part3-v1.tex says that every
# number used in the paper is fixed there, and carries a TODO for the table
# rather than the table. There is therefore nothing in the paper to check these
# against. They differ from `benchmark/config.jl` in two places, eta2 (0.75 here,
# 0.9 there) and zeta (0.5 here, 100.0 there). Both files are internally
# consistent, and which set the paper adopts is the author's to settle.
# =============================================================================

'''

body = "\n\n".join([
    "# ---------------------------------------------------------------- constants",
    consts,
    "# ------------------------------------------------------- critical points",
    critpts,
    testfun,
    "# ------------------------------------------------------------------ runners",
    runners,
])

io.open(OUT, "w", encoding="utf-8", newline="\n").write(header + body + "\n")
print("wrote", OUT, "(%d chars)" % len(header + body))
