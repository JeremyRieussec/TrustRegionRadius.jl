module TrustRegionRadius

using NLPModels
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


include("State/main.jl")
include("Saving_info/main.jl")
include("Radius_updates/main.jl")
include("Stopping_tests/main.jl")
include("Subproblem/main.jl")
include("Plotting_graphs/main.jl")
include("Trust-region/main.jl")
include("Line-Search/main.jl")


export SimpleTointGouldTointParameters, TointGouldTointParameters, 
        SimpleScheinbergParameters, ScheinbergParameters, 
        YuanFanParameters, 
        HeiParameters, HeiGradParameters, 
        HeiFanYuanParameters

export StoppingCriteriaGradient

export AlgorithmInfoTR, AlgorithmInfoGD

export trust_region_with_cg, gradient_descent, LS_steepest_backtrack

end # module TrustRegionRadius