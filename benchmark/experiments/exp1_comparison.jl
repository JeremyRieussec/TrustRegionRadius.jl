# =============================================================================
# benchmark/experiments/exp1_comparison.jl
#
# EXPERIMENT 1 -- like-for-like comparison of the radius mechanisms.
#
# Every rule runs on the same problems with identical η's, γ's, model Hessian
# and subproblem solver, so the only variable is the radius rule.
#
#   julia --project=benchmark benchmark/experiments/exp1_comparison.jl
# =============================================================================

function comparison()
    arch = ExperimentArchive(tag = "comparison")
    save_config(arch; rules = RULES, params = SOLVER_PARAMS,
                problem_selection = PROBLEM_SELECTION,
                extra = Dict("experiment" => "exp1_comparison"))

    problems = default_problems()
    configs  = rule_configs()

    @info "Experiment 1: $(length(problems)) problems × $(length(configs)) mechanisms"
    records = run_experiment(problems, configs;
                             params = SOLVER_PARAMS, archive = arch)

    save_table(arch, "exp1_success_rate.txt",
               success_table(records, problems, configs))

    labels = [c[1] for c in configs]

    # --- performance profiles ---
    for (metric, fname, xlab) in (
            (:iter, "exp1_perf_profile_iter.pdf",  "iterations"),
            (:obj,  "exp1_perf_profile_fevals.pdf", "objective evaluations"))
        M = metric_matrix(records, problems, configs, metric)
        τ, prof = performance_profile(M)
        plt = plot(τ, prof; xscale = :log10, label = reshape(labels, 1, :),
                   xlabel = "τ (performance ratio, $xlab)", ylabel = "π(τ)",
                   legend = :bottomright, lw = 2, ylims = (0, 1.02))
        savefig_archived(arch, fname, plt)
        open(joinpath(arch.tables, replace(fname, ".pdf" => ".tex")), "w") do io
            profile_to_pgfplots(io, τ, prof, labels)
        end
    end

    # --- data profile: fairer when dimensions vary ---
    N = metric_matrix(records, problems, configs, :obj)
    dims = [begin nlp = p[2](); d = nlp.meta.nvar; finalize(nlp); d end
            for p in problems]
    κ, dp = data_profile(N, dims)
    plt = plot(κ, dp; label = reshape(labels, 1, :),
               xlabel = "budget (simplex gradients, (n+1) evaluations)",
               ylabel = "fraction solved", legend = :bottomright,
               lw = 2, ylims = (0, 1.02))
    savefig_archived(arch, "exp1_data_profile.pdf", plt)

    finalize_archive(arch; notes = """
        Like-for-like comparison of the radius mechanisms. Parameters are held
        identical across rules; only the radius update differs.

        Read the two performance profiles together: the intercept π(1) is
        efficiency (how often a rule is fastest) and the right-hand asymptote is
        reliability (how many problems it solves at all). The survey's finding
        is that these two move independently -- a parameter choice can leave
        efficiency untouched while collapsing reliability.

        The `tail active` column of exp1_success_rate.txt reports the fraction
        of late iterations on which the trust-region constraint bound. It is the
        observable that separates mechanisms whose first-order behaviour is
        indistinguishable.
        """)
end

if abspath(PROGRAM_FILE) == @__FILE__
    comparison()
end