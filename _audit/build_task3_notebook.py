"""Assemble mu_cap_gradient_descent_v1.ipynb from the verified driver.

The code cells are cut from `_audit/task3_driver.jl` at its `#%%` markers, so
the notebook cannot drift from the file that was run and checked. Only the two
paths that depend on the working directory are rewritten, and the script asserts
on both.
"""
import io, json, os, re

ROOT = r"C:\Users\jerem\OneDrive\Desktop\GitHub\TrustRegionRadius.jl"
SRC = os.path.join(ROOT, "_audit", "task3_driver.jl")
OUT = os.path.join(ROOT, "notebooks", "Saddle", "mu_cap_gradient_descent_v1.ipynb")

src = io.open(SRC, encoding="utf-8").read()

# The driver lives in _audit/ and the notebook in notebooks/Saddle/, so the two
# relative paths differ by one level. Everything else is copied unchanged.
def repath(s):
    a = 'include(joinpath(@__DIR__, "..", "notebooks", "Saddle", "saddle_problem.jl"))'
    b = 'include(joinpath(@__DIR__, "saddle_problem.jl"))'
    assert s.count(a) == 1, "include line not found once"
    s = s.replace(a, b)
    c = 'const REPDIR = joinpath(@__DIR__, "..", "report", "task3-mu-cap")'
    d = 'const REPDIR = joinpath(@__DIR__, "..", "..", "report", "task3-mu-cap")'
    assert s.count(c) == 1, "REPDIR line not found once"
    return s.replace(c, d)

src = repath(src)

# Split on the markers, dropping the file header before the first one.
parts = re.split(r"^#%% ", src, flags=re.M)
assert parts[0].lstrip().startswith("# ====="), "unexpected header"
blocks = {}
order = []
for p in parts[1:]:
    name, _, body = p.partition("\n")
    name = name.strip()
    order.append(name)
    blocks[name] = body.rstrip()

expected = ["setup", "grid", "runner", "classify", "sweep", "table1",
            "reconcile", "angle-fit", "limits", "figures", "latex"]
assert order == expected, "markers changed: %r" % (order,)

MD = {
"setup": """# A small cap shrinks the region until the method is gradient descent

`R-grad` capped bounds the multiplier by $\\bar\\mu$, which the asymptotic theory
needs. The radius $\\Delta_k = \\mu_k\\|g_k\\|$ is then small everywhere. A small
region makes the linear term of the model dominate its quadratic term, the step
that comes back points along $-g_k$, and the model Hessian has stopped
influencing the direction. `R-DFO` does the same for small $\\zeta$, which holds
$\\Delta_k$ below $\\zeta\\omega_k$.

Two subsolvers give two versions of the same degeneration. Truncated conjugate
gradient degenerates **exactly**: CG starts from $s = 0$ along $-g$, and if the
region truncates it there the returned step is $-\\Delta_k g_k/\\|g_k\\|$ with
$\\cos(s_k,-g_k) = 1$ to machine precision. The exact Moré-Sorensen solver
degenerates **asymptotically**: $s^\\ast = -(H+\\sigma I)^{-1}g$ approaches the
gradient direction as $\\Delta \\to 0$ without ever equalling it.

Every run goes through `tr_solve`. The setup comes from `saddle_problem.jl`,
extracted from `saddle_discussion_v1.ipynb` so that the two cannot drift.""",

"grid": """## The grid

Three axes, since the claim is about an interaction. The cap sweep and the
$\\zeta$ sweep are two ways of holding the radius down, so they share every other
setting.

Keyword arguments everywhere. `RGrad` and `RGradCapped` were renumbered, with
$\\gamma_2$ now the mild contraction and $\\gamma_3$ the expansion, so a positional
call can pass silently under the wrong reading.""",

"runner": """## Running one configuration

`run_cfg` builds its model inline and returns only the statistics, so the
evaluation counters are unreachable afterwards. We need a work measure that
survives a change of budget, and an iteration count is not one. The wrapper below
keeps the model. The iteration is still entirely `tr_solve`'s, and the path comes
from the solver's own callback.""",

"classify": """## What we measure

A single Cauchy percentage merges two different events, so we separate three
classes of step. Truncated at the boundary along $-g_k$, where the Hessian has
touched neither the direction nor the length. Stopped inside after one CG
iteration, where the direction is $-g_k$ and the Hessian sets the length. Two or
more CG iterations.

`eig_deviation` is $\\sin$ of the angle between $H_kg_k$ and $g_k$, which vanishes
exactly when $g_k$ is an eigenvector of $H_k$. `z1_ratio` is $\\|z_1\\|/\\Delta_k$,
where $z_1$ is the first Steihaug iterate. Steihaug truncates when that ratio
reaches one. Both are needed in the reconciliation below.""",

"sweep": """## The runs""",

"table1": """## Three classes of step

The degeneration is total below $\\bar\\mu = 0.1$. With the exact Hessian every
iteration falls in the truncated class and the minimum cosine over the run is
one. The method is exactly gradient descent for two thousand iterations.

Class two is almost never occupied. The degeneration proceeds by truncation
rather than by a short interior step.""",

"reconcile": """## Two Cauchy statistics, reconciled

`hypotheses_report(st).cauchy_fraction` tests $\\cos(s_k,-g_k) \\approx 1$ within
$10^{-12}$. Counting `cg_iters == 1` tests the Krylov dimension. The two disagree
at $\\bar\\mu = 4$ and $8$, and the cosine test counts more.

The natural explanation is that $g_k$ is close to an eigenvector of $H_k$, so the
second CG direction adds nothing to the direction while still being counted. On
this function the iterates approach the axis $p_2 = 0$, where the Hessian is
diagonal and the gradient is an eigenvector, so the explanation is plausible.

**It is wrong.** The `sin@dis` and `sin@agree` columns compare the disagreeing
iterations against the agreeing ones. At $\\bar\\mu = 4$ they are equal to three
figures, and both sit near $1/\\sqrt2$, far from an eigenvector. At $\\bar\\mu = 8$
the disagreeing iterations are *further* from the eigenvector case.

The mechanism is the boundary crossing, in the last two columns. On the
disagreeing iterations $\\|z_1\\|/\\Delta_k$ sits within a part in $10^8$ of one, so
CG does not truncate, takes a second iteration, and immediately backtracks to the
boundary by a negligible distance. The step is on the boundary and parallel to
$-g_k$ to within $10^{-12}$, and `cg_iters` records two.

Neither statistic is wrong. One measures the direction that came back, the other
the work that produced it. They separate on a bookkeeping event rather than on a
property of the spectrum.""",

"angle-fit": """## The angle against the radius

For truncated conjugate gradient on the truncated branch $1 - \\cos(s_k,-g_k)$ is
identically zero. Not small, zero. That is the contrast.

For the exact solver we fit the exponent rather than assuming it. Fit (a) is what
the brief asks for, $1-\\cos$ against $\\Delta_k$, and it is poor.

The reason is that $\\Delta_k$ is not the parameter the expansion is in. Writing
$s^\\ast = -(H+\\sigma I)^{-1}g$ in the eigenbasis of $H$ and expanding for large
$\\sigma$, with $\\sigma \\sim \\|g\\|/\\Delta$, gives

$$1 - \\cos(s^\\ast, -g) \\;\\sim\\; \\tfrac12 \\operatorname{Var}_w(\\lambda)\\,\\mu^2,
  \\qquad \\mu = \\Delta/\\|g\\|,$$

with $w_i$ the weights of $g$ on the eigenvectors of $H$. A cap pins $\\mu_k$ at
$\\bar\\mu$, so the radius goes to zero only because $\\|g_k\\|$ does, and no
exponent in $\\mu$ can be fitted from one run. Fit (b) pools the runs across caps
and recovers the second order.

The residual decrease of $1-\\cos$ along a run therefore comes from
$\\operatorname{Var}_w(\\lambda)$ rather than from $\\Delta_k$. The iterates approach
the degenerate origin, the gradient aligns with an eigenvector, and the weighted
variance collapses. The near-eigenvector effect is real, and it is what drives
the exact solver's angle to zero. It is not what separates the two Cauchy
statistics.""",

"limits": """## Limit points

`which_crit` returns `other` when nothing is within $10^{-4}$, and a category
column invites the reading that a trapped run converged elsewhere. We report the
distance to the nearest critical point and $\\|g_k\\|$ at the last iterate, with a
work measure, so that the row survives a change of budget. Nothing is dropped.
The trapped configurations are the evidence.

**Scope.** With the exact Hessian these runs end at the degenerate origin, where
$\\nabla^2 f$ is singular. Part II's inactivity theory assumes a positive definite
Hessian at the limit, so the activity column is not evidence about inactivity.
The constraint stays active there for a reason that has nothing to do with the
cap.""",

"figures": """## Figures""",

"latex": """## LaTeX fragments for the report

Every number the report states is written here, so none is transcribed by
hand.""",
}

cells = [{
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
    "source": ['using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))\n', 'Pkg.status()'],
}]
for name in order:
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": MD[name].splitlines(keepends=True)})
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": blocks[name].splitlines(keepends=True)})

nb = {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Julia 1.11.0", "language": "julia",
                       "name": "julia-1.11"},
        "language_info": {"file_extension": ".jl", "mimetype": "application/julia",
                          "name": "julia", "version": "1.11.0"}},
      "nbformat": 4, "nbformat_minor": 5}

io.open(OUT, "w", encoding="utf-8", newline="\n").write(
    json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print("wrote", OUT, "with", len(cells), "cells")
