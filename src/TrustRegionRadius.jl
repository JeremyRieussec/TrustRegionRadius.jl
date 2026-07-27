module TrustRegionRadius

using NLPModels
using SolverCore
using Krylov
using LinearAlgebra

import Base: show, println, print, showerror
import SolverCore: solve!, reset!

greet() = print("Hello World! This is the package for testing Trust Region Radius update mechanisms.")

# ============================================================
# Abstract types
# ============================================================

abstract type AbstractState end
abstract type AbstractTrustRegionState <: AbstractState end
abstract type AbstractGradientDescentState <: AbstractState end

abstract type AbstractAlgorithmInfo end
abstract type AbstractTrustRegionParameters end
abstract type AbstractStoppingCriteria end
"""
    AbstractRadiusUpdate

Abstract supertype for the four canonical radius update rules R1–R4.
Concrete subtypes implement `update_radius!(rule, Δ, ρ, η₁, η₂, s_norm, g_norm_old, g_norm_new)`.
"""
abstract type AbstractRadiusUpdate end
abstract type AbstractTRSubproblemSolver end

# ============================================================
# Includes
# ============================================================

# include("State/main.jl")
# include("Saving_info/main.jl")
include("Radius_updates/main.jl")
# include("Stopping_tests/main.jl")
include("Subproblem/main.jl")
# include("Plotting_graphs/main.jl")
include("Trust-region/main.jl")
# include("Line-Search/main.jl")

# ============================================================
# Exports
# ============================================================

# Legacy parameter types (existing mechanisms, backward compat)
export SimpleTointGouldTointParameters, TointGouldTointParameters,
       SimpleScheinbergParameters, ScheinbergParameters,
       YuanFanParameters,
       HeiParameters, HeiGradParameters, HeiFanYuanParameters
        
# Canonical R1-R4 radius update rules
export AbstractRadiusUpdate
export R1ClassicalUpdate, R2StepSizeUpdate, R3DFOLikeUpdate, R4RelativeGradUpdate
export update_radius!, initial_radius, reset_rule!

# Hei-family canonical rules
export HeiUpdate, HeiGradUpdate, HeiFanYuanUpdate

# Solver parameters and legacy output
export TRSolverParams, TROutput
export StoppingCriteriaGradient
export AlgorithmInfoTR, AlgorithmInfoGD
export trust_region_with_cg, trust_region_solver, gradient_descent, LS_steepest_backtrack

# JSO-compatible solver interface (SolverCore)
export TRRSolver, trust_region_radius

# Subproblem solver interface
export AbstractTRSubproblemSolver, solve_subproblem!
export SteihaugTointCG, KrylovCG, KrylovCGLanczos

end # module TrustRegionRadius
