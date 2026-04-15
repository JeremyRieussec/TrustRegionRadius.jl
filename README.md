# TrustRegionRadius

[![CI](https://github.com/JeremyRieussec/TrustRegionRadius.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/JeremyRieussec/TrustRegionRadius.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/JeremyRieussec/TrustRegionRadius.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JeremyRieussec/TrustRegionRadius.jl)

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

## Benchmarking

using BenchmarkTools
using TrustRegionRadius

### Example benchmark
```julia
julia --project=benchmark benchmark/run_benchmark.jl
```
This will run the benchmark script located in the `benchmark` directory, which includes various tests for the trust-region methods implemented in the package.

```julia
julia --project=benchmark benchmark/generate_all_figures.jl
```
This will generate all the figures used in the documentation and examples, allowing you to visualize the results of the optimization processes.





