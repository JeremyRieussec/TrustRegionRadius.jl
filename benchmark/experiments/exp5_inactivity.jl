# =============================================================================
# benchmark/experiments/exp5_inactivity.jl
#
# EXPERIMENT 5 -- direct measurement of trust-region inactivity.
#
# The claim under test is that the mechanisms split cleanly into those whose
# trust-region constraint eventually stops binding and those where it binds for
# ever, and that this split is invisible to a first-order convergence test:
# both groups have ‖g_k‖ decreasing monotonically to zero.
#
# The observable is the fraction of late iterations with ‖s_k‖ = Δ_k, which the
# solver records in :active_trajectory when trace = true.
#
#   julia --project=benchmark benchmark/experiments/exp5_inactivity.jl
# =============================================================================

"""
Minimum iterations before a tail fraction means anything.

On a ten-iteration run the last 10% is one iteration and `tail_active` can only
be 0 or 1, so a table of such values measures rounding rather than asymptotics.
Runs shorter than this are reported as `--` and counted, not averaged in.
"""
const MIN_ITERS_FOR_TAIL = 30

"Fraction of the last `frac` of iterations on which the constraint bound."
tail_active(r::RunRecord; frac::Float64 = 0.1) =
    length(r.active_traj) < MIN_ITERS_FOR_TAIL ? NaN :
        active_fraction(RecordView(r); tail = frac)

function inactivity()
    arch = ExperimentArchive(tag = "inactivity")

    # Rules spanning both families, plus a deliberately-too-small parameter.
    rules = [
        ("RDelta",       () -> RDelta()),
        ("RStep",        () -> RStep()),
        ("RDFO(z=1)",    () -> RDFO(ζ = 1.0)),
        ("RDFO(z=0.01)", () -> RDFO(ζ = 0.01)),
        ("RGrad",        () -> RGrad()),
        ("RGradCapped",  () -> RGradCapped(μ = 0.05, μ_max = 0.05)),
    ]
    configs = rule_configs(rules)

    save_config(arch; rules = rules, params = SOLVER_PARAMS,
                problem_selection = PROBLEM_SELECTION,
                extra = Dict("experiment" => "exp5_inactivity"))

    problems = default_problems()
    # `hessian_norm` so that Σ Δ_k²/M_k can be formed: that, not Σ Δ_k², is what
    # Part II proves finite for the criticality-anchored rules.
    records  = run_experiment(problems, configs; params = SOLVER_PARAMS,
                              trace = true, archive = arch,
                              solver_kwargs = (hessian_norm = true,))
    labels   = [c[1] for c in configs]

    # tail activity per configuration
    io = IOBuffer()
    @printf(io, "%-16s %8s %8s %10s %14s %14s\n",
            "rule", "runs", "solved", "measured", "mean tail act", "never inactive")
    println(io, "-"^74)
    means = Float64[]
    for (cname, _) in configs
        # No `solved` filter. A configuration trapped below its threshold is the
        # one that exhausts the budget, so filtering on success removes exactly
        # the evidence this experiment exists to collect.
        rs = filter(r -> r.config == cname, records)
        ta = filter(!isnan, [tail_active(r) for r in rs])
        m  = isempty(ta) ? NaN : sum(ta)/length(ta)
        never = count(r -> inactivity_index(RecordView(r)) === nothing, rs)
        push!(means, m)
        @printf(io, "%-16s %8d %8d %10d %14s %14d\n", cname, length(rs),
                count(solved, rs), length(ta),
                isnan(m) ? "--" : @sprintf("%.3f", m), never)
    end
    println(io)
    println(io, "`measured` counts runs of at least $MIN_ITERS_FOR_TAIL iterations, the")
    println(io, "others being too short for a tail fraction to carry information.")
    println(io, "`never inactive` counts runs whose constraint bound to the last")
    println(io, "iteration; for a rule below its threshold that is the finding.")
    save_table(arch, "exp5_inactivity_summary.txt", String(take!(io)))

    # The three series, per run. Averaging them across problems mixes a divergent
    # series with convergent ones and produces a number with no reading.
    io = IOBuffer()
    @printf(io, "%-16s %-16s %6s %8s %14s %14s %16s\n", "problem", "rule", "iters",
            "status", "Σ Δ_k", "Σ Δ_k²", "Σ Δ_k²/M_k")
    println(io, "-"^96)
    for r in records
        isempty(r.delta_traj) && continue
        rs = radius_sums(RecordView(r))
        @printf(io, "%-16s %-16s %6d %8s %14.4g %14.4g %16s\n",
                r.problem, r.config, r.iterations, string(r.status),
                rs.sum_delta, rs.sum_delta2,
                isnan(rs.sum_delta2_over_M) ? "--" :
                    @sprintf("%.4g", rs.sum_delta2_over_M))
    end
    println(io)
    println(io, "M_k = max_{i≤k}‖H_i‖, i.e. L = 0: the Lipschitz constant is not known,")
    println(io, "and omitting it changes the constant rather than the convergence.")
    println(io, "These are partial sums over a finite run and settle nothing on their")
    println(io, "own; read them as a shape across rules on the same problem.")
    save_table(arch, "exp5_radius_series.txt", String(take!(io)))

    plt = bar(labels, means; ylabel = "mean tail-active fraction",
              legend = false, xrotation = 45, ylims = (0, 1.02))
    hline!(plt, [0.5]; ls = :dash, c = :black)
    savefig_archived(arch, "exp5_tail_active.pdf", plt)

    # activity along the run, on one problem
    pname = problems[1][1]
    plt = plot(; xlabel = "iteration k", ylabel = "cumulative active fraction",
                 title = pname, legend = :best, lw = 2, ylims = (0, 1.02))
    for r in filter(r -> r.problem == pname && solved(r), records)
        isempty(r.active_traj) && continue
        plot!(plt, cumsum(r.active_traj) ./ (1:length(r.active_traj)); label = r.config)
    end
    savefig_archived(arch, "exp5_active_trajectory_$(pname).pdf", plt)

    finalize_archive(arch; notes = """
        Direct measurement of trust-region inactivity.

        The tail-active fraction is the share of the last 10% of iterations on
        which ‖s_k‖ = Δ_k. Values near 0 mean the constraint has stopped
        binding, so the step is the unconstrained model minimiser and the method
        inherits the local rate of the underlying (quasi-)Newton iteration.
        Values near 1 mean it never stops binding.

        RDFO(z=0.01) and RGradCapped(mu=0.05) are included precisely because
        their parameters sit below any plausible κ̄ = 4/λ*_min: they should stay
        pinned near 1 while every other rule falls to 0. Note that all of them
        still drive ‖g_k‖ → 0, which is why a first-order convergence test
        cannot tell the two groups apart.
        """)
end

if abspath(PROGRAM_FILE) == @__FILE__
    inactivity()
end