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
using Random
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
include("experiments/exp8_single_problem.jl")
include("experiments/exp9_second_order.jl")

# The sampled experiments. They exercise the two sampled problem classes, so
# they need the Sampling layer rather than just the three deterministic axes.
include("experiments/exp11_bhhh.jl")
include("experiments/exp12_sampling_examples.jl")
include("experiments/exp13_flat_well.jl")

# -----------------------------------------------------------------------------
# Two gaps, flagged rather than papered over
# -----------------------------------------------------------------------------
#
# ONE gap, not two. The filename shift is real history: every name from exp2 to
# exp8 was once one ahead of its contents, and the files here are renamed so each
# matches what it defines. The note that used to stand here concluded from that
# that experiment 8 had no source left. It does: exp8_single_problem.jl is
# tracked, its header self-identifies as EXPERIMENT 8, and its subject -- one
# problem, every mechanism, the remaining-active countdown -- is experiment 8's.
# It was excluded because it defined `main`, which experiments 9, 11 and 12 also
# defined at the time; those three have since been renamed to their own entry
# points, so the collision is gone. Its entry point is now
# `single_problem_experiment`, which is the name the notebook already called, and
# it shares no other name with any experiment here.
#
# exp10 is the gap that remains: the suite jumps from exp9_second_order to
# exp11_bhhh.
#
# -----------------------------------------------------------------------------
# Entry points
# -----------------------------------------------------------------------------
#
#   1  comparison()          5  inactivity()          9  second_order()
#   2  trajectories()        6  interaction()        11  bhhh_study()
#   3  zeta_sweep()          7  CVRate()             12  sampling_examples()
#   4  mu_sweep()            8  single_problem_experiment(name)
#
# Experiments 9, 11 and 12 previously all defined `main`, and 9 and 12 both
# defined `run_grid` and `grid_table`. Since everything is included into Main,
# the last file loaded silently won: calling `main()` ran experiment 12 whatever
# you meant. exp12 also defined `const RULES`, shadowing config.jl's list of the
# eight radius mechanisms that experiments 1-7 compare.
