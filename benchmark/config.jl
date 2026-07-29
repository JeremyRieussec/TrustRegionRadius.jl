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
    η₁             = 0.1,
    η₂             = 0.9,
    Δ₀             = 1.0,
    max_iterations = 10_000,
    tol            = 1e-5,
    max_time       = 120.0,
)

# -----------------------------------------------------------------------------
# The mechanisms under comparison
# -----------------------------------------------------------------------------
const RULES = [
    ("RDelta",  () -> RDelta(γ₁ = 0.25, γ₂ = 0.50, γ₃ = 2.0)),
    ("RStep",   () -> RStep( γ₁ = 0.25, γ₂ = 0.80, γ₃ = 2.0)),
    ("RDFO",    () -> RDFO(  γ₁ = 0.25, γ₂ = 0.50, γ₃ = 2.0, ζ = 1.0)),
    ("RGrad",   () -> RGrad( γ₁ = 0.25, γ₂ = 2.00, μ = 1.0)),
    ("RGradCapped", () -> RGradCapped(μ = 1.0, μ_max = 1.0)),
    ("RAdaptiveStep",    () -> RAdaptiveStep()),
    # ("RAdaptiveGrad",    () -> RAdaptiveGrad()),
    ("RAdaptiveFanYuan", () -> RAdaptiveFanYuan(μ = 1.0)),
    # ("RRTR",             () -> RRTR()),
    # ("RRTRGrad",         () -> RRTRGrad(μ = 1.0)),
]

# Full configurations (rule + model + subsolver) for the interaction study.
const DEFAULT_MODEL     = () -> ExactHessian()
const DEFAULT_SUBSOLVER = () -> SteihaugCG()

"Wrap the RULES list into the (name, factory-of-NamedTuple) form run_experiment wants."
rule_configs(rules = RULES; model = DEFAULT_MODEL, subsolver = DEFAULT_SUBSOLVER) =
    [(nm, () -> (rule = f(), model = model(), subsolver = subsolver()))
     for (nm, f) in rules]

# -----------------------------------------------------------------------------
# CUTEst problem selection
# -----------------------------------------------------------------------------
const MIN_VAR = 2
const MAX_VAR = 100
const MAX_CON = 0        # 0 = unconstrained only
const PROBLEM_LIMIT = 20   # e.g. 50 for a quick pass

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
