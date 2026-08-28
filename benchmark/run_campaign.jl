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

include(joinpath(@__DIR__, "initialisation.jl"))

using Printf, Dates

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

    probs = default_problems()
    n = length(probs)
    @printf("  problem set    %d problems\n", n)
    if n <= 10
        println("    ", [p[1] for p in probs])
        if get(ENV, "TRR_ALLOW_ANALYTIC", "0") == "1"
            @warn "Running on the analytic fallback, not CUTEst, because " *
                  "TRR_ALLOW_ANALYTIC=1."
        else
            error("Only $n problems: CUTEst is unavailable and default_problems() " *
                  "fell back to the analytic set. Install CUTEst, or set " *
                  "TRR_ALLOW_ANALYTIC=1 if that is what you want.")
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
