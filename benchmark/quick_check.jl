# =============================================================================
# benchmark/quick_check.jl
#
# A quick check against a deposited archive: re-solve the problems of the
# like-for-like comparison on which every archived run was short, and compare
# each run with the archived one.
#
#   julia --project=benchmark benchmark/quick_check.jl \
#         path/to/exp_2026-08-29_02-28-45_comparison  [n_problems]
#
# Without `n_problems` it takes all of them: 176 of the 196 problems, 1056 runs,
# 19 s of solver time on the campaign machine. Most of the wall clock goes into
# decoding and compiling the SIF files. With `n_problems` it takes the cheapest.
#
# WHY THE MACHINE DOES NOT MATTER
#
# A run is unsolved when it exhausts 10^4 iterations or 120 s of wall clock, and
# only the second test depends on the machine. The check therefore takes only
# problems on which every archived run ended well inside that budget: none on
# :max_time, and each under QC_MAX_TIME seconds on the campaign machine. On
# those the solver is deterministic, so the status, the iteration count and the
# evaluation counts must agree exactly, however fast or slow the machine. The
# times are printed only to estimate how this machine compares with that one.
#
# It compares the rules that the current `RULES` and the archive share. The
# comparison ran eight rules; `RRTR` and `RAdaptiveStep` were commented out of
# `RULES` afterwards, so at v0.1.0 six are compared.
# =============================================================================

include(joinpath(@__DIR__, "initialisation.jl"))

const QC_MAX_TIME = 1.0      # seconds per archived run: over 100× inside the budget
const QC_MIN_TIMED = 0.1     # only runs at least this long enter the speed ratio

function quick_check(dir::AbstractString, n::Int = typemax(Int))
    cfg = TOML.parsefile(joinpath(dir, "experiment_config.toml"))
    get(cfg, "experiment", "") == "exp1_comparison" ||
        error("$dir is not the comparison archive (experiment = " *
              "$(get(cfg, "experiment", "?")))")
    HAS_CUTEST || error("CUTEst is unavailable; the comparison ran on CUTEst")

    # --- the archived runs, keyed by (problem, rule) ------------------------
    archived = Dict{Tuple{String, String}, RunRecord}()
    for f in readdir(joinpath(dir, "data"))
        endswith(f, ".jld2") || continue
        r = _record_from_data(JLD2.load(joinpath(dir, "data", f)))
        archived[(r.problem, r.config)] = r
    end
    in_archive = Set(last.(collect(keys(archived))))
    configs = [c for c in rule_configs() if c[1] in in_archive]
    names   = first.(configs)
    skipped = setdiff(in_archive, Set(names))
    isempty(skipped) ||
        println("Not in the current RULES, so not compared: ", join(sort(collect(skipped)), ", "))

    # --- the cheapest problems whose runs all ended far inside the budget ---
    available = Set(first.(cutest_problems(min_var = MIN_VAR, max_var = MAX_VAR,
                                           max_con = MAX_CON)))
    candidates = Tuple{Float64, String}[]
    for p in unique(first.(collect(keys(archived))))
        runs = [get(archived, (p, c), nothing) for c in names]
        any(isnothing, runs) && continue
        all(r -> r.status !== :max_time && r.solve_time <= QC_MAX_TIME, runs) || continue
        if p in available
            push!(candidates, (sum(r.solve_time for r in runs), p))
        else
            println("In the archive but not in this CUTEst collection: ", p)
        end
    end
    sort!(candidates)
    chosen = last.(candidates[1:min(n, length(candidates))])
    isempty(chosen) && error("no problem qualifies")
    @printf("%d of the %d archived problems qualify; checking %d.\n\n",
            length(candidates), length(unique(first.(collect(keys(archived))))),
            length(chosen))

    problems = [(p, () -> CUTEstModel(p)) for p in chosen]
    records  = run_experiment(problems, configs; params = SOLVER_PARAMS)

    # --- compare ------------------------------------------------------------
    fields = (:status, :iterations, :f_evals, :g_evals, :h_evals)
    bad = bad_status = 0
    t_here = t_there = 0.0
    for r in records
        a = archived[(r.problem, r.config)]
        diff = [f for f in fields if getfield(r, f) != getfield(a, f)]
        if !isempty(diff)
            bad += 1
            :status in diff && (bad_status += 1)
            println("DIFFERS  ", r.problem, " / ", r.config, ": ",
                    join(("$f $(getfield(a, f)) -> $(getfield(r, f))" for f in diff), ", "))
        end
        # Shorter runs are timer noise and first-call compilation, not speed.
        if a.solve_time >= QC_MIN_TIMED
            t_here += r.solve_time; t_there += a.solve_time
        end
    end
    @printf("\n%d runs (%d problems × %d rules): %d agree exactly on %s; %d differ, %d of them in status.\n",
            length(records), length(chosen), length(configs),
            length(records) - bad, join(fields, ", "), bad, bad_status)
    t_there > 0 &&
        @printf("Solve time here / on the campaign machine, over the runs of at least %.1f s: %.2f\n",
                QC_MIN_TIMED, t_here / t_there)
    return bad == 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    isempty(ARGS) && error("usage: quick_check.jl ARCHIVE_DIR [n_problems]")
    ok = length(ARGS) >= 2 ? quick_check(ARGS[1], parse(Int, ARGS[2])) : quick_check(ARGS[1])
    exit(ok ? 0 : 1)
end
