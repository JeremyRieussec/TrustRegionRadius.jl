# =============================================================================
# benchmark/initialisation.jl
#
# Load the package, the harness, and every experiment, so a notebook or REPL
# session can call any of them by name.
#
#   include("initialisation.jl")
#   comparison(); trajectories(); ...; second_order()
# =============================================================================

using TrustRegionRadius
using ADNLPModels, NLPModels, LinearAlgebra
using CUTEst
using TOML, Dates, Printf, JLD2
using Plots, StatsPlots

# Order matters: harness.jl defines RunRecord and the run loop, archive.jl the
# storage layer it writes through, config.jl the shared rule lists and parameters.
# The experiment files include NONE of these themselves — they are loaded once,
# here, so that config.jl's `const`s are evaluated exactly once.
include("harness.jl")
include("archive.jl")
include("config.jl")

include("experiments/exp1_comparison.jl")
include("experiments/exp2_trajectories.jl")
include("experiments/exp3_zeta_sweep.jl")
include("experiments/exp4_mu_sweep.jl")
include("experiments/exp5_inactivity.jl")
include("experiments/exp6_interaction.jl")
include("experiments/exp7_convergence_rate.jl")
# include("experiments/exp8_single_problem.jl")   # MISSING -- see note below
include("experiments/exp9_second_order.jl")

# The sampled experiments. They exercise the two sampled problem classes, so
# they need the Sampling layer rather than just the three deterministic axes.
include("experiments/exp11_bhhh.jl")
include("experiments/exp12_sampling_examples.jl")

# -----------------------------------------------------------------------------
# Two gaps, flagged rather than papered over
# -----------------------------------------------------------------------------
#
# exp8_single_problem.jl does not exist. The file that carried that NAME held
# experiment 7 (`CVRate`): every filename from exp2 to exp8 was shifted by one
# against its contents, so `exp8_single_problem.jl` was really
# exp7_convergence_rate, `exp7_convergence_rate.jl` was really exp6_interaction,
# and so on down to `exp2_trajectories.jl`, which held exp1. The files here are
# renamed so each matches what it defines -- which leaves experiment 8 with no
# source at all. `single_problem_experiment(::String)`, which the notebook calls,
# is therefore undefined until that file is written.
#
# exp10 is absent as well: the suite jumps from exp9_second_order to exp11_bhhh.
#
# -----------------------------------------------------------------------------
# Entry points
# -----------------------------------------------------------------------------
#
#   1  comparison()          5  inactivity()          9  second_order()
#   2  trajectories()        6  interaction()        11  bhhh_study()
#   3  zeta_sweep()          7  CVRate()             12  sampling_examples()
#   4  mu_sweep()            8  (missing)
#
# Experiments 9, 11 and 12 previously all defined `main`, and 9 and 12 both
# defined `run_grid` and `grid_table`. Since everything is included into Main,
# the last file loaded silently won: calling `main()` ran experiment 12 whatever
# you meant. exp12 also defined `const RULES`, shadowing config.jl's list of the
# eight radius mechanisms that experiments 1-7 compare.
