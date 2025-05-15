# TrustRegionRadius

TrustRegionRadius is a Julia package for implementing and testing trust-region methods, particularly focusing on trust-region radius update mechanisms. 

It includes tools for solving optimization problems using trust-region methods and visualizing optimization paths.

## Features

- Implementation of the Truncated Conjugate Gradient (CG) method for solving trust-region subproblems.
- Visualization of optimization paths and convergence using `Plots.jl`.
- Support for trust-region state management and updates.
- Integration with `NLPModels` and `QuadraticModels` for optimization problem definitions.

## Installation

To install the package, clone the repository and activate it in Julia:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()

