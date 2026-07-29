module TrustRegionRadius

using LinearAlgebra
using Printf

using NLPModels
using SolverCore
using Krylov
using LinearOperators

import SolverCore: solve!, reset!

# =============================================================================
# Includes.  Order matters: the three axes are defined before the solver that
# threads them together, and the benchmark layer last.
# =============================================================================

include("Radius_updates/main.jl")     # RadiusRule and every mechanism
include("Model_Hessians/main.jl")     # ModelHessian and every model
include("Subproblem/main.jl")         # SubproblemSolver and every subsolver
include("Trust-region/main.jl")       # TRParams, TRSolver, tr_solve
include("Benchmark/main.jl")          # profiles and the run matrix

# =============================================================================
# Exports
#
# Three orthogonal axes, one solver, one benchmarking layer.
# Each name appears exactly once.
# =============================================================================

# ---- Axis 1: radius mechanisms ----------------------------------------------
export RadiusRule
export RDelta, RStep, RDFO, RGrad, RGradCapped
export RAdaptiveStep, RAdaptiveGrad, RAdaptiveFanYuan
export RRTR, RRTRGrad
export initial_radius, update_radius!, reset_rule!
export needs_retrospective, is_criticality_anchored, retrospective_ratio
export asymptotic_regime, validate_thresholds, check_factors

# ---- Axis 2: model Hessians -------------------------------------------------
export ModelHessian
export ExactHessian, LBFGSModel, SR1Model, ScaledIdentity, SPDTarget
export hessian_op, dense_hessian, model_hprod!, update_model!, reset_model!
export phi_target

# ---- Axis 3: subproblem solvers ---------------------------------------------
export SubproblemSolver
export SteihaugCG, ExactMS, KrylovCG, KrylovCGLanczos
export solve_subproblem!, cg_step_info

# ---- Solver -----------------------------------------------------------------
export TRParams, TRResult, TRSolver, tr_solve

# ---- Benchmarking -----------------------------------------------------------
export performance_profile, data_profile, profile_to_pgfplots
export run_matrix, summarise, TRConfig, sweep_configs

"""
    greet()

One-line description of the package.
"""
greet() = print("TrustRegionRadius: a testbed for trust-region radius update mechanisms.")

end # module TrustRegionRadius
