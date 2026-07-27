# =============================================================================
# src/Benchmark/main.jl
#
# Benchmarking layer: performance and data profiles, and the run matrix.
#
# `profiles.jl` is self-contained (pure functions of a cost matrix).
# `run_matrix.jl` needs `tr_solve` from Trust-region/, so this directory is
# included last.
# =============================================================================

include("profiles.jl")
include("run_matrix.jl")
