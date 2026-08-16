# =============================================================================
# benchmark/experiments/exp4_mu_sweep.jl
#
# EXPERIMENT 4 -- the cap μ_max of RGradCapped, and the uncapped RGrad.
#
# Two claims under test.
#
# 1. The cap has a threshold at κ̄ = 4/λ*_min, below which the trust region
#    binds for ever. Unlike ζ, this one is not merely a reliability issue: with
#    a truncated-CG subsolver a small cap makes CG truncate on its first
#    iteration, so the step is the Cauchy point and the model Hessian stops
#    influencing the direction. The method becomes gradient descent, and with a
#    quasi-Newton model that can change which critical point is reached.
#
# 2. Uncapped RGrad crosses any threshold on its own, since μ grows
#    geometrically, so its inactivity is unconditional. It should recover from
#    a starting μ far below the threshold at a cost of O(log(κ̄/μ0)) iterations.
#
#   julia --project=benchmark benchmark/experiments/exp4_mu_sweep.jl
# =============================================================================

const MUS = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 8.0, 128.0]

function mu_sweep()
    arch = ExperimentArchive(tag = "mu_sweep")

    # Typed as `Function`, not left to inference: a comprehension of closures has
    # one concrete closure type, so pushing the uncapped arm onto an untyped
    # vector fails to convert. That is why the arm was commented out.
    configs = Tuple{String, Function}[
        (@sprintf("mu_max=%g", μ),
         () -> (rule = RGradCapped(μ = μ, μ_max = μ),
                model = DEFAULT_MODEL(), subsolver = DEFAULT_SUBSOLVER()))
        for μ in MUS]
    # The uncapped arm is the point of the experiment, not an extra: it is the
    # only configuration testing the one unconditional inactivity result in the
    # survey, that μ_k climbs past any threshold on its own. It starts far below
    # κ̄ on purpose, so that the climb is what is being measured.
    push!(configs, ("uncapped",
                    () -> (rule = RGrad(μ = 0.1), model = DEFAULT_MODEL(),
                           subsolver = DEFAULT_SUBSOLVER())))

    save_config(arch; params = SOLVER_PARAMS, problem_selection = PROBLEM_SELECTION,
                rules = vcat([(@sprintf("mu_max=%g", μ), () -> RGradCapped(μ = μ, μ_max = μ))
                              for μ in MUS], [("uncapped", () -> RGrad(μ = 0.1))]),
                extra = Dict("experiment" => "exp4_mu_sweep", "mu_values" => MUS))

    problems = default_problems()
    records  = run_experiment(problems, configs; params = SOLVER_PARAMS, archive = arch)
    labels   = [c[1] for c in configs]

    save_table(arch, "exp4_mu_summary.txt", success_table(records, problems, configs))

    # The climb, measured. Corollary [mu three regimes] bounds the number of
    # expansions after the iterates settle by log_{γ₃}(γ₃C/μ₀); `:expand` counts
    # them, and `:expand_capped` counts the iterations on which the user's cap
    # refused the climb, which is the mechanism that traps a run below κ̄.
    io = IOBuffer()
    @printf(io, "%-16s %-16s %8s %10s %12s %14s\n",
            "problem", "config", "expand", "capped", "k*", "tail active")
    println(io, "-"^82)
    for r in records
        b = branch_counts(RecordView(r))
        ki = inactivity_index(RecordView(r))
        @printf(io, "%-16s %-16s %8d %10d %12s %14.3f\n", r.problem, r.config,
                get(b, :expand, 0), get(b, :expand_capped, 0),
                ki === nothing ? "never" : string(ki),
                active_fraction(RecordView(r)))
    end
    save_table(arch, "exp4_climb.txt", String(take!(io)))

    M = metric_matrix(records, problems, configs, :iter)
    τ, prof = performance_profile(M)
    plt = plot(τ, prof; xscale = :log10, label = reshape(labels, 1, :),
               xlabel = "τ (performance ratio, iterations)", ylabel = "π(τ)",
               legend = :bottomright, lw = 2, ylims = (0, 1.02))
    savefig_archived(arch, "exp4_perf_profile_mu.pdf", plt)
    open(joinpath(arch.tables, "exp4_perf_profile_mu.tex"), "w") do io
        profile_to_pgfplots(io, τ, prof, labels)
    end

    rate = [count(r -> r.config == c[1] && solved(r), records) / length(problems)
            for c in configs]
    plt = bar(labels, rate; xlabel = "", ylabel = "fraction solved",
              legend = false, xrotation = 45, ylims = (0, 1.02))
    savefig_archived(arch, "exp4_solverate_vs_mu.pdf", plt)

    # --- the Cauchy-point diagnostic ---
    # cos(s,-g) = 1 with a single CG iteration certifies that the model Hessian
    # played no part in the direction.
    io = IOBuffer()
    @printf(io, "%12s %10s %12s %14s\n", "mu_max", "CG iters", "active", "cos(s,-g)")
    println(io, "-"^52)
    probe = first(problems)
    nlp = probe[2]()
    x = copy(nlp.meta.x0); g = grad(nlp, x); gn = norm(g)
    for μ in MUS
        info = cg_step_info(SteihaugCG(), ExactHessian(), nlp, x, g, μ * gn)
        @printf(io, "%12g %10d %12s %14.12f\n", μ, info.cg_iters,
                string(info.active), info.cos_cauchy)
    end
    finalize(nlp)
    save_table(arch, "exp4_cauchy_diagnostic.txt", String(take!(io)))

    finalize_archive(arch; notes = """
        μ_max sweep for RGradCapped over $(MUS), plus uncapped RGrad.

        exp4_cauchy_diagnostic.txt is the mechanism behind any threshold seen in
        the profile: where CG iters = 1 and cos(s,-g) = 1 to machine precision,
        the returned step is exactly the Cauchy point and the model Hessian has
        not influenced the search direction at all.

        The uncapped row is the comparison that matters. If it solves at least
        as much as the best capped setting, that supports preferring it: its
        eventual inactivity needs no hypothesis on κ̄, whereas every capped
        setting does.
        """)
end

if abspath(PROGRAM_FILE) == @__FILE__
    mu_sweep()
end