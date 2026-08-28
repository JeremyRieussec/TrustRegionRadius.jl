# =============================================================================
# benchmark/run_campaign.jl
#
# Run a named set of experiments in one Julia session, one archive each, and
# keep going if one of them fails.
#
#   julia --project=benchmark benchmark/run_campaign.jl                 # all six
#   julia --project=benchmark benchmark/run_campaign.jl mu zeta         # a subset
#   TRR_ALLOW_ANALYTIC=1 julia --project=benchmark benchmark/run_campaign.jl
#
# One session for the whole campaign because `initialisation.jl` and the first
# solve cost a minute or two of compilation; paying that once rather than six
# times matters on a remote box.
#
# THE PRECONDITION THIS SCRIPT ENFORCES
#
# `save_config` records the git commit and a `git_dirty` flag taken from
# `git status --porcelain`, which counts UNTRACKED files as dirty. An archive
# written from a dirty tree cannot be reproduced: the commit it names is not the
# code that ran, and no diff recovers the difference. Every archive in
# `_Thesis_FINAL/Article3/experiments` carries `git_dirty = true` for that
# reason. This script refuses to start on a dirty tree unless you override it.
#
# It also refuses to start if CUTEst is unavailable, because `default_problems()`
# falls back to the eight analytic problems with only a `@warn`, and a campaign
# that silently ran on 8 problems instead of 185 looks exactly like one that did
# not. Set TRR_ALLOW_ANALYTIC=1 if the fallback is what you want.
# =============================================================================

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

const CAMPAIGN = [
    ("comparison",  "exp1  like-for-like comparison",              comparison),
    ("zeta",        "exp3  the ζ sweep of RDFO",                   zeta_sweep),
    ("mu",          "exp4  the μ_max sweep of RGradCapped",        mu_sweep),
    ("inactivity",  "exp5  tail activity, eventual inactivity",    inactivity),
    ("interaction", "exp6  rule × model grid",                     interaction),
    ("cvrate",      "exp7  local convergence order",               CVRate),
]

function preflight()
    root = normpath(joinpath(@__DIR__, ".."))
    println("="^70)
    println("PREFLIGHT")
    println("="^70)

    commit = try readchomp(setenv(`git rev-parse HEAD`; dir = root)) catch; "" end
    porcelain = try readchomp(setenv(`git status --porcelain`; dir = root)) catch; "" end
    dirty = !isempty(porcelain)
    @printf("  julia          %s\n", VERSION)
    @printf("  commit         %s\n", isempty(commit) ? "UNKNOWN" : commit[1:min(12, end)])
    @printf("  git_dirty      %s\n", dirty)
    @printf("  hostname       %s\n", gethostname())
    @printf("  threads        %d\n", Threads.nthreads())

    if dirty
        println("\n  Uncommitted or untracked files:")
        for line in split(porcelain, '\n')[1:min(10, end)]
            println("    ", line)
        end
        if get(ENV, "TRR_ALLOW_DIRTY", "0") == "1"
            @warn "Dirty tree, continuing because TRR_ALLOW_DIRTY=1. The archives " *
                  "will record git_dirty = true and will not be reproducible."
        else
            error("Refusing to start from a dirty tree. Commit (or .gitignore) the " *
                  "files above, or set TRR_ALLOW_DIRTY=1 to override.")
        end
    end

    # Report the selection parameters, not just the count. A small problem set has
    # two quite different causes and the remedy differs: either CUTEst is missing
    # and `default_problems()` fell back to the analytic set, or CUTEst is fine
    # and MIN_VAR / MAX_VAR / PROBLEM_LIMIT in config.jl are narrowing it.
    probs = default_problems()
    n = length(probs)
    lim = PROBLEM_LIMIT === nothing ? "none" : string(PROBLEM_LIMIT)
    @printf("  problem set    %d problems
", n)
    @printf("  selection      min_var = %s, max_var = %s, max_con = %s, limit = %s
",
            MIN_VAR, MAX_VAR, MAX_CON, lim)
    @printf("  source         %s
", HAS_CUTEST ? "CUTEst" : "analytic fallback")
    if n <= 10
        println("    ", [p[1] for p in probs])
        cause = if !HAS_CUTEST
            "CUTEst is unavailable, so default_problems() fell back to the " *
            "analytic set. Install CUTEst."
        else
            "CUTEst is available and working; the set is narrow because of " *
            "config.jl: min_var = $(MIN_VAR), max_var = $(MAX_VAR), " *
            "limit = $(lim). Section 5 of the paper reports 185 problems, which " *
            "needs max_var = 200 and PROBLEM_LIMIT = nothing."
        end
        if get(ENV, "TRR_ALLOW_ANALYTIC", "0") == "1"
            @warn "Continuing on $n problems because TRR_ALLOW_ANALYTIC=1. " *
                  "These archives are a rehearsal, not Section 5 numbers. " * cause
        else
            error("Only $n problems. " * cause *
                  " Set TRR_ALLOW_ANALYTIC=1 to run on this set anyway.")
        end
    end
    end
    println()
    return nothing
end

function main(args)
    wanted = isempty(args) ? [c[1] for c in CAMPAIGN] : args
    known = [c[1] for c in CAMPAIGN]
    unknown = setdiff(wanted, known)
    if !isempty(unknown)
        bad = join(unknown, ", ")
        ok  = join(known, ", ")
        error("unknown experiment(s): " * bad * ". Known: " * ok)
    end
    preflight()

    results = Tuple{String, Symbol, Float64, String}[]
    for (key, label, f) in CAMPAIGN
        key in wanted || continue
        println("="^70)
        @printf("START  %-12s %s   %s\n", key, label, Dates.format(now(), "HH:MM:SS"))
        println("="^70)
        flush(stdout)
        t0 = time()
        status, note = try
            f()
            (:ok, "")
        catch e
            bt = sprint(showerror, e)
            @error "experiment $key failed" exception = (e, catch_backtrace())
            (:failed, first(replace(bt, '\n' => ' '), 200))
        end
        dt = time() - t0
        push!(results, (key, status, dt, note))
        @printf("END    %-12s %s in %.1f min\n\n", key, status, dt / 60)
        flush(stdout)
    end

    println("="^70)
    println("CAMPAIGN SUMMARY   ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
    println("="^70)
    for (key, status, dt, note) in results
        @printf("  %-12s %-8s %8.1f min  %s\n", key, status, dt / 60, note)
    end
    nfail = count(r -> r[2] === :failed, results)
    @printf("\n  %d of %d succeeded\n", length(results) - nfail, length(results))
    println("\n  Archives written under benchmark/results/:")
    for d in sort(readdir(joinpath(@__DIR__, "results")))
        startswith(d, "exp_") && println("    ", d)
    end
    exit(nfail == 0 ? 0 : 1)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
