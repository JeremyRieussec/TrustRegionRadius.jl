# =============================================================================
# benchmark/config.jl
#
# Central configuration for all benchmark scripts.
# Edit this file to change solver parameters, radius update rules, or the
# CUTEst problem selection criteria.  run_benchmark.jl (and the experiment
# scripts) include this file to pick up the values below.
# =============================================================================

# -----------------------------------------------------------------------------
# Solver parameters
# -----------------------------------------------------------------------------
const SOLVER_PARAMS = TRSolverParams(
    η₁             = 0.1,
    η₂             = 0.9,
    Δ₀             = 1.0,
    max_iterations = 10_000,
    tol            = 1e-5,
)

# -----------------------------------------------------------------------------
# Radius update rules
# Factory functions so each (problem, rule) run gets a fresh instance.
# Mutable rules (R4, HeiFanYuanUpdate) MUST use a factory so each run starts
# from the same initial state.
# -----------------------------------------------------------------------------
const RULES = [
    ("R1",  () -> R1ClassicalUpdate(0.25, 0.50, 2.0)),
    ("R2",  () -> R2StepSizeUpdate(0.25, 0.80, 2.0)),
    # ("R3",  () -> R3DFOLikeUpdate(0.25, 0.50, 2.0, 1.0)),
    # ("R4",  () -> R4RelativeGradUpdate(0.25, 2.0, 1.0)),
    # ("R4-Alt", () -> R4RelativeGradUpdate(0.25, 2.0, 0.5)),
    # #           η     β     γ₁    γ₂    M      λ₁    λ₂
    # ("Hei",    () -> HeiUpdate(       0.1, 0.1, 0.1, 0.5, 10.0, 0.5, 0.1)),
    # ("HeiG",   () -> HeiGradUpdate(   0.1, 0.1, 0.1, 0.5, 10.0, 0.5, 0.1)),
    # ("HFY",    () -> HeiFanYuanUpdate(0.1, 0.1, 0.1, 0.1, 0.5, 10.0, 0.5, 0.1)),
    #                                  μ    η    β    γ₁   γ₂    M    λ₁   λ₂
]

# -----------------------------------------------------------------------------
# CUTEst problem selection
# -----------------------------------------------------------------------------
const MIN_VAR = 500      # minimum number of variables
const MAX_VAR = 500    # maximum number of variables
const MAX_CON = 0      # maximum number of constraints (0 = unconstrained only)

# to start run :
#  julia --project=benchmark benchmark/run_benchmark.jl [--force]