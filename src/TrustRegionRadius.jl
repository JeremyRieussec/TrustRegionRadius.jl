module TrustRegionRadius

using NLPModels, QuadraticModels
using LinearAlgebra, QuadGK, Polynomials
using Plots, LaTeXStrings
using Test

import Base:show,println, print, Base.showerror

plotlyjs()

greet() = print("Hello World! This is the package for testing Trust Region Raidus update mechanisms.")


end # module TrustRegionRadius
