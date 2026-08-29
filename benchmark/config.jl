# =============================================================================
# benchmark/config.jl
#
# Central configuration for every experiment script.
#
# Rules are given as (name, factory) pairs. The factory matters: RGrad and the
# quasi-Newton models carry mutable state, so each (problem, rule) run must get
# a fresh instance or results depend on the order problems were visited.
# =============================================================================

# -----------------------------------------------------------------------------
# Solver parameters -- identical for every mechanism, deliberately.
# Per-rule tuning would measure tuning effort rather than algorithmic merit.
# -----------------------------------------------------------------------------
const SOLVER_PARAMS = TRParams(
    η              = 0.1,   # acceptance:  step taken iff ρ ≥ η
    η1             = 0.1,   # scaling:     contract below this
    η2             = 0.9,   # scaling:     "very successful" above this
    Δ0             = 1.0,
    max_iterations = 10_000,
    tol            = 1e-5,
    max_time       = 120.0,
)

# -----------------------------------------------------------------------------
# The mechanisms under comparison
# -----------------------------------------------------------------------------
# ONE set of constants for every rule. A comparison at mixed constants measures
# the constants as much as the mechanism, and the roster previously ran RStep at
# γ2 = 0.80, the Hei pair at γ1 = 0.0625, γ3 = 4.0, RDFO at ζ = 100, and a radius
# floor of 1e-14 on three rules and 0 on the rest.
#
# γ3 = 2.0 satisfies the Hei requirement γ3 > 1 + γ2 = 1.5, so the adaptive pair
# runs at the same factors as everything else. Δmin = 0 everywhere: the
# asymptotic claims of Part II are about the unguarded rules, and a floor on some
# rules and not others is the confound rather than a fix for it.
const G1, G2, G3 = 0.25, 0.50, 2.0

const RULES = [
    ("RDelta",        () -> RDelta(       γ1 = G1, γ2 = G2, γ3 = G3, Δmin = 0.0)),
    ("RStep",         () -> RStep(        γ1 = G1, γ2 = G2, γ3 = G3, Δmin = 0.0)),
    ("RDFO",          () -> RDFO(         γ1 = G1, γ2 = G2, γ3 = G3, ζ = 100.0,
                                          Δmin = 0.0)),
    ("RGrad",         () -> RGrad(        γ1 = G1, γ2 = G2, γ3 = G3, μ = 1.0,
                                          Δmin = 0.0)),
    ("RGradCapped",   () -> RGradCapped(  γ1 = G1, γ2 = G2, γ3 = G3, μ = 1.0,
                                          μ_max = 128.0, Δmin = 0.0)),
    ("RRTR",          () -> RRTR(         γ1 = G1, γ2 = G2, γ3 = G3, Δmin = 0.0)),
    ("RAdaptiveStep", () -> RAdaptiveStep(λ1 = 5.0, λ2 = 5.0, Δmin = 0.0)),
    ("RAdaptiveGrad", () -> RAdaptiveGrad(μ = 1.0, λ1 = 5.0, λ2 = 5.0, Δmin = 0.0)),
]

# RRTRGrad is implemented and is deliberately not in the roster above: it is the
# ninth mechanism and the paper's Table of the roster names eight. Add it here to
# close the anchor-by-ratio cell, at the cost of one more column everywhere.
const RRTRGRAD_CONFIG =
    ("RRTRGrad", () -> RRTRGrad(γ1 = G1, γ2 = G2, γ3 = G3, μ = 1.0, Δmin = 0.0))

# -----------------------------------------------------------------------------
# Second-order anchoring
#
# Same mechanisms, criticality measured by τ = max{‖g‖, −λ_min(B)} instead of
# ‖g‖. The update logic is identical — SecondOrder is a wrapper that overrides
# `criticality` and forwards `update_radius!` — so pairing each τ-rule with its
# ‖g‖-twin isolates the measure and nothing else.
#
# Three things must line up or the run silently degrades to a first-order one:
# the measure (here), a model that can report λ_min < 0 (ExactHessian or
# SR1Model), and a subsolver that can move along it (EigenPoint or ExactMS).
# `second_order_configs` pairs them; using it with LBFGSModel gives τ ≡ ‖g‖ and
# a `:second_order` status that certifies nothing.
# -----------------------------------------------------------------------------
const TAU_RULES = [
    ("RGradTau",           () -> RGradTau(γ1 = 0.25, γ2 = 0.50, γ3 = 2.0, μ = 1.0)),
    ("RGradCappedTau",     () -> RGradCappedTau(γ1 = 0.25, γ2 = 0.50, γ3 = 2.0,
                                                μ = 1.0, μ_max = 128.0)),
    ("RDFOTau",            () -> RDFOTau(γ1 = 0.25, γ2 = 0.50, γ3 = 2.0, ζ = 100.0)),
    ("RAdaptiveGradTau",   () -> RAdaptiveGradTau()),
]

"The ‖g‖-anchored rule each τ-rule should be compared against, by name."
const TAU_PAIRS = [
    ("RGrad",            "RGradTau"),
    ("RGradCapped",      "RGradCappedTau"),
    ("RDFO",             "RDFOTau"),
    ("RAdaptiveGrad",    "RAdaptiveGradTau")
]

"""
Parameters for a second-order run: `tol_H > 0` turns on the second-order
stopping test, so the run refuses to stop at a saddle and reports
`:second_order` when it certifies `λ_min(B) ≥ −tol_H`.

`tol_H = -1` in `SOLVER_PARAMS` disables it, which is the default and costs no
curvature estimate.
"""
const SECOND_ORDER_PARAMS = TRParams(
    η              = 0.1,
    η1             = 0.2,
    η2             = 0.9,
    Δ0             = 1.0,
    max_iterations = 10_000,
    tol            = 1e-5,
    tol_H          = 1e-6,
    max_time       = 120.0,
)

# Full configurations (rule + model + subsolver) for the interaction study.
const DEFAULT_MODEL     = () -> ExactHessian()
const DEFAULT_SUBSOLVER = () -> SteihaugCG(max_iters = 1_000)

"Wrap the RULES list into the (name, factory-of-NamedTuple) form run_experiment wants."
rule_configs(rules = RULES; model = DEFAULT_MODEL, subsolver = DEFAULT_SUBSOLVER) =
    [(nm, () -> (rule = f(), model = model(), subsolver = subsolver()))
     for (nm, f) in rules]

"""
    second_order_configs(; model, subsolver) -> Vector

The τ-anchored rules, paired by default with the model and subsolver that make
them mean something: `ExactHessian` reports genuine negative curvature, and
`EigenPoint` wrapping Steihaug takes a step along it with a guaranteed
`½|λ_min|Δ²` decrease.

Both defaults are deliberate. With a positive semidefinite model τ ≡ ‖g‖ and the
whole apparatus is an expensive no-op that will still report `:second_order` at a
saddle; with a plain Steihaug subsolver the radius stays alive but the step goes
wherever the CG recurrence happened to reach. Experiment 9 varies exactly these
two to show that neither alone suffices.
"""
second_order_configs(rules = TAU_RULES;
                     model = () -> ExactHessian(),
                     subsolver = () -> EigenPoint(SteihaugCG())) =
    [(nm, () -> (rule = f(), model = model(), subsolver = subsolver()))
     for (nm, f) in rules]

"""
    paired_configs(; model, subsolver) -> Vector

Each ‖g‖-anchored rule immediately followed by its τ-twin, so a table reads as
adjacent pairs and the measure is the only difference within a pair.
"""
function paired_configs(; model = () -> ExactHessian(),
                          subsolver = () -> EigenPoint(SteihaugCG()))
    lookup(list, nm) = something(findfirst(p -> p[1] == nm, list), 0)
    out = Tuple{String, Function}[]
    for (gname, tname) in TAU_PAIRS
        i = lookup(RULES, gname); j = lookup(TAU_RULES, tname)
        (i == 0 || j == 0) && continue
        fg, ft = RULES[i][2], TAU_RULES[j][2]
        push!(out, (gname, () -> (rule = fg(), model = model(), subsolver = subsolver())))
        push!(out, (tname, () -> (rule = ft(), model = model(), subsolver = subsolver())))
    end
    return out
end

# -----------------------------------------------------------------------------
# CUTEst problem selection
# -----------------------------------------------------------------------------
const MIN_VAR = 2
const MAX_VAR = 1_000      # SteihaugCG is the default subsolver, so n is not
                         # capped by ExactMS's dense limit; keep it under that
                         # (200) so an exact-solver arm stays runnable.
const MAX_CON = 0        # 0 = unconstrained only
const PROBLEM_LIMIT = nothing   
                         # A campaign, not a smoke test. Three two-variable
                         # problems cannot resolve an asymptotic claim: runs are
                         # ten iterations long, so a 10% tail is one iteration
                         # and every tail statistic is quantised to {0, ½, 1}.
                         # Drop to 3 deliberately when debugging a script, and
                         # never report a table produced at that setting.

const PROBLEM_SELECTION = Dict(
    "min_var" => MIN_VAR, "max_var" => MAX_VAR,
    "max_con" => MAX_CON,
)

"""
    flat_well(ε; x0 = [0.0, 1.2]) -> ADNLPModel

The family P2 of the manuscript, `f(x,y) = ½x² + ε(¼y⁴ − ½y²)` with `ε ∈ (0, ½)`.

A nonsingular saddle at the origin with `∇²f = diag(1, −ε)`, and minimisers at
`(0, ±1)` with `∇²f = diag(1, 2ε)`. So

    λ* = λ_min(∇²f(0,±1)) = 2ε,    κ̄ = 8/λ* = 4/ε,    1/λ* = 1/(2ε),

and `ε` tunes the curvature at the solution without moving the solution. The
restriction `ε < ½` is what makes `2ε` the smaller of the two eigenvalues; above
it the roles swap and `λ*` is `1`, so the thresholds below stop being about `ε`.

The line `{x = 0}` is invariant, since `∂f/∂x = x`, and the Hessian is diagonal,
so a run started on it stays on it and is exactly the scalar problem
`φ(y) = ε(¼y⁴ − ½y²)`. Every `y₀ > 0` converges monotonically to `y = 1`.

This is the problem on which the radius thresholds are measurable rather than
estimated: the two constants of `eqn: three thresholds` are `1/(2ε)` and `0.1/ε`
exactly, so a bisection on `ε` reads them off directly.
"""
flat_well(ε; x0 = [0.0, 1.2]) =
    ADNLPModel(p -> 0.5p[1]^2 + ε * (p[2]^4 / 4 - p[2]^2 / 2), copy(float.(x0)),
               name = "P2(eps=$ε)")

"φ''(y) for the scalar problem the invariant line reduces to: ε(3y² − 1)."
flat_well_curvature(ε, y) = ε * (3y^2 - 1)

"The three thresholds of `eqn: three thresholds` at `λ* = 2ε`, as a named tuple."
flat_well_thresholds(ε; γ2 = 0.5, γ3 = 2.0) =
    (rdfo  = (1 - γ2) / (2ε * (γ3 + 1 - γ2)),   # ζ must sit below this
     exact = 1 / (2ε),                          # μ̄ must sit below this
     kbar  = 4 / ε)                             # 8/λ*, the sufficient constant

"The problem set: CUTEst when available, the analytic set otherwise."
function default_problems()
    ps = cutest_problems(min_var = MIN_VAR, max_var = MAX_VAR,
                         max_con = MAX_CON, limit = PROBLEM_LIMIT)
    isempty(ps) ? analytic_problems() : ps
end
