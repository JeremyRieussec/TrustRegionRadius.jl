module TrustRegionRadius

using NLPModels
using CUTEst
using LinearAlgebra
using Plots, LaTeXStrings
using Test

import Base:show,println, print, Base.showerror

greet() = print("Hello World! This is the package for testing Trust Region Raidus update mechanisms.")

abstract type AbstractState end
abstract type AbstractTrustRegionState <: AbstractState end
abstract type AbstractGradientDescentState <: AbstractState end

abstract type AbstractAlgorithmInfo end

abstract type AbstractTrustRegionParameters end

abstract type AbstractStoppingCriteria end

abstract type AbstractRadiusUpdate end


include("State/main.jl")
include("Saving_info/main.jl")
include("Radius_updates/main.jl")
include("Stopping_tests/main.jl")
include("Subproblem/main.jl")
include("Plotting_graphs/main.jl")
include("Trust-region/main.jl")
include("Line-Search/main.jl")


# Legacy parameter types (existing mechanisms)
export SimpleTointGouldTointParameters, TointGouldTointParameters,
        SimpleScheinbergParameters, ScheinbergParameters,
        YuanFanParameters,
        HeiParameters, HeiGradParameters,
        HeiFanYuanParameters

# Canonical R1–R4 radius update rules
export AbstractRadiusUpdate
export R1ClassicalUpdate, R2StepSizeUpdate, R3DFOLikeUpdate, R4RelativeGradUpdate
export update_radius!, initial_radius

# Hei-family canonical rules
export HeiUpdate, HeiGradUpdate, HeiFanYuanUpdate

# Solver parameters and output
export TRSolverParams, TROutput

export StoppingCriteriaGradient

export AlgorithmInfoTR, AlgorithmInfoGD

export trust_region_with_cg, trust_region_solver, gradient_descent, LS_steepest_backtrack

end # module TrustRegionRadius