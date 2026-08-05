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
const RULES = [
    ("RDelta",  () -> RDelta(γ1 = 0.25, γ2 = 0.50, γ3 = 2.0)),
    ("RStep",   () -> RStep( γ1 = 0.25, γ2 = 0.80, γ3 = 2.0)),
    ("RDFO",    () -> RDFO(  γ1 = 0.25, γ2 = 0.50, γ3 = 2.0, ζ = 100.0)),
    ("RGrad",   () -> RGrad( γ1 = 0.25, γ2 = 0.50, γ3 = 2.0, μ = 1.0)),
    ("RGradCapped", () -> RGradCapped(γ1 = 0.25, γ2 = 0.50, γ3 = 2.0,
                                     μ = 1.0, μ_max = 128.0)),
    ("RAdaptiveStep",    () -> RAdaptiveStep()),
    ("RAdaptiveGrad",    () -> RAdaptiveGrad()),
    # ("RRTR",             () -> RRTR()),
    # ("RRTRGrad",         () -> RRTRGrad(μ = 1.0)),
]

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
    η1             = 0.1,
    η2             = 0.9,
    Δ0             = 1.0,
    max_iterations = 10_000,
    tol            = 1e-5,
    tol_H          = 1e-6,
    max_time       = 120.0,
)

# Full configurations (rule + model + subsolver) for the interaction study.
const DEFAULT_MODEL     = () -> ExactHessian()
const DEFAULT_SUBSOLVER = () -> SteihaugCG()

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
const MAX_VAR = 2
const MAX_CON = 0        # 0 = unconstrained only
const PROBLEM_LIMIT = 3  # e.g. 50 for a quick pass

const PROBLEM_SELECTION = Dict(
    "min_var" => MIN_VAR, "max_var" => MAX_VAR,
    "max_con" => MAX_CON,
)

"The problem set: CUTEst when available, the analytic set otherwise."
function default_problems()
    ps = cutest_problems(min_var = MIN_VAR, max_var = MAX_VAR,
                         max_con = MAX_CON, limit = PROBLEM_LIMIT)
    isempty(ps) ? analytic_problems() : ps
end
