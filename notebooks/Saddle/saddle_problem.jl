# =============================================================================
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

# ---------------------------------------------------------------- constants

# ---- standard parameters (change these and re-run) -----------------------
const ETA1, ETA2 = 0.1, 0.75        # acceptance / very-successful thresholds
const G1, G2, G3 = 0.25, 0.5, 2.0   # gamma_1, gamma_2, gamma_3
const ZETA       = 0.5              # R-DFO criticality threshold
const MU0, DELTA0 = 0.5, 1.0        # initial mu and radius
const MU_BAR     = 2.0              # cap for the bounded-mu variant of R-grad

const TOL, KMAX = 1e-9, 2000

# ------------------------------------------------------- critical points

# ---------------------------------------------------------------- critical points
const ORIGIN = [0.0, 0.0]                                   # degenerate, singular, PSD
const SADDLE = [0.75, 0.0]                                  # nonsingular saddle
const MINP   = [(1+sqrt(5))/4,  sqrt((sqrt(5)-1)/4)]        # local minimum
const MINM   = [(1+sqrt(5))/4, -sqrt((sqrt(5)-1)/4)]        # local minimum

const X0_DEFAULT = [-0.5, 0.6]

const CRITPTS = [("origin",  ORIGIN), ("saddle", SADDLE),
                 ("min+",    MINP),   ("min-",   MINM)]

# ---------------------------------------------------------------- test function
f(p) = p[1]^4 - p[1]^3 + (0.25 - p[1]/2)*p[2]^2 + p[2]^4/4

grad(p) = [4p[1]^3 - 3p[1]^2 - p[2]^2/2,
           p[2]^3 + 2p[2]*(0.25 - p[1]/2)]

hess(p) = [6p[1]*(2p[1]-1)   -p[2];
           -p[2]             (-p[1] + 3p[2]^2 + 0.5)]

# the same f, as the model the solver actually sees
make_nlp(x0 = X0_DEFAULT) = ADNLPModel(f, collect(float.(x0)), name = "quartic")

# ------------------------------------------------------------------ runners

"Run one configuration. Returns the GenericExecutionStats from tr_solve, unchanged."
function run_cfg(; rule, model = ExactHessian(), subsolver = SteihaugCG(),
                   x0 = X0_DEFAULT, tol = TOL, tol_H = -1.0, kmax = KMAX,
                   Δ0 = DELTA0, η = ETA1)
    return tr_solve(make_nlp(x0); rule = rule, model = model, subsolver = subsolver,
                    params = TRParams(η = η, η1 = ETA1, η2 = ETA2, Δ0 = Δ0,
                                      tol = tol, tol_H = tol_H, max_iterations = kmax),
                    trace = true)
end

"Same, plus the iterate path collected through the solver's callback."
function solve_path(; x0 = X0_DEFAULT, kwargs...)
    xs = [collect(float.(x0))]
    st = tr_solve(make_nlp(x0);
                  params = TRParams(η = get(kwargs, :η, ETA1), η1 = ETA1, η2 = ETA2,
                                    Δ0 = get(kwargs, :Δ0, DELTA0),
                                    tol = get(kwargs, :tol, TOL),
                                    tol_H = get(kwargs, :tol_H, -1.0),
                                    max_iterations = get(kwargs, :kmax, KMAX)),
                  rule      = kwargs[:rule],
                  model     = get(kwargs, :model, ExactHessian()),
                  subsolver = get(kwargs, :subsolver, SteihaugCG()),
                  trace = true,
                  callback = (_n, _s, stats) -> push!(xs, copy(stats.solution)))
    return st, xs
end

"Classify a limit point by distance to the four known critical points."
function which_crit(x; tol = 1e-4)
    best, bd = "other", Inf
    for (nm, p) in CRITPTS
        d = norm(x .- p); d < bd && (bd = d; best = nm)
    end
    return bd < tol ? best : "other"
end

# `tail_active` used to be defined here, hand-rolled, with
#     k0 = max(1, floor(Int, 0.9 * length(a)))
# which is off by one against the package: at n = 41 it takes six points where
# `active_fraction` takes five, because floor(0.9n) is the last index before the
# tail and the tail starts at floor(0.9n) + 1. Use the package, which does the
# slice once and correctly, and takes `stats` rather than `stats.solver_specific`.
