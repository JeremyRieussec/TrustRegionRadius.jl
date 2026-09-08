# =============================================================================
# benchmark/experiments/exp7_convergence_rate.jl
#
# EXPERIMENT 7 -- observed local convergence order.
#
# Claim: once the trust region goes inactive the step is the unconstrained model
# minimiser, so the observed order is that of the underlying (quasi-)Newton
# iteration and is INDEPENDENT of the radius rule. The mechanisms differ in
# WHEN inactivity is reached, not in the rate afterwards.
#
# The estimate must be conditioned on inactivity. Including active iterations
# mixes a linear phase with the asymptotic one and returns an order between the
# two -- an artefact, not a finding. This script trims each trajectory to its
# inactive tail before fitting.
#
#   julia --project=benchmark benchmark/experiments/exp7_convergence_rate.jl
# =============================================================================

"""
Fewest inactive iterations from which an order is worth fitting.

Below this the regression has two or three points and returns a number that
looks like a rate and is not one.
"""
const MIN_TAIL = 6

"""
    inactive_tail(r) -> Vector{Float64}

The gradient trajectory restricted to the final run of inactive iterations.
Returns an empty vector if the constraint never stopped binding, which is
itself the finding for that configuration.
"""
function inactive_tail(r::RunRecord)
    (isempty(r.active_traj) || isempty(r.grad_traj)) && return Float64[]
    k = length(r.active_traj)
    while k >= 1 && !r.active_traj[k]
        k -= 1
    end
    start = k + 1
    start > length(r.active_traj) && return Float64[]
    lo = min(start, length(r.grad_traj))
    return r.grad_traj[lo:end]
end

function CVRate()
    arch = ExperimentArchive(tag = "convergence_rate")
    configs  = rule_configs()
    save_config(arch; rules = RULES, configs = configs, params = SOLVER_PARAMS,
                problem_selection = PROBLEM_SELECTION,
                extra = Dict("experiment" => "exp7_convergence_rate"))

    problems = default_problems()
    # A tight tolerance is essential: the asymptotic regime is what is measured.
    params = TRParams(η1 = SOLVER_PARAMS.η1, η2 = SOLVER_PARAMS.η2,
                      Δ0 = SOLVER_PARAMS.Δ0,
                      max_iterations = SOLVER_PARAMS.max_iterations,
                      tol = 1e-10, max_time = SOLVER_PARAMS.max_time)

    records = run_experiment(problems, configs; params = params,
                             trace = true, archive = arch)
    labels  = [c[1] for c in configs]

    orders  = Dict(c[1] => Float64[] for c in configs)
    k_stars = Dict(c[1] => Float64[] for c in configs)
    n_short  = Dict(c[1] => 0 for c in configs)
    n_never  = Dict(c[1] => 0 for c in configs)
    n_unconv = Dict(c[1] => 0 for c in configs)
    n_flat   = Dict(c[1] => 0 for c in configs)
    io = IOBuffer()
    @printf(io, "%-16s %-16s %8s %8s %12s %12s %8s %10s\n",
            "problem", "rule", "status", "iters", "k* (perm.)", "1st inactive",
            "tail", "order")
    println(io, "-"^96)
    # No `solved` filter on the ROWS: a run whose constraint never stops binding
    # has no asymptotic phase to fit, and saying so is the result for that
    # configuration rather than a gap in the table. Every run is therefore
    # printed, and the last column says why it did or did not yield an order.
    #
    # The MEDIANS are a different matter, and four exclusions apply to them.
    # `never` and `short` were always there; `unconv` and `flat` were not, and
    # both were silently corrupting the column this experiment exists to report.
    #
    #   unconv  The run did not reach :first_order. A :stalled run has an
    #           inactive tail -- its steps are tiny and interior -- so it passes
    #           `inactivity_index` and `MIN_TAIL` and yields a log-log slope near
    #           1. That is not a convergence order, it is the signature of a run
    #           going nowhere. Measured on the 2026-08-22 archive: 100 of 459
    #           fitted runs had not converged, and pooling them pulled the median
    #           from 1.4218 to 1.3080. The contamination ran from 20% to 31%
    #           across rules, so it was correlated with the very axis under test.
    #
    #   flat    `estimate_convergence_order` returns NaN when fewer than four
    #           tail values survive its finite-and-positive filter, or when the
    #           regression has zero variance in x. The `short` guard cannot catch
    #           this: it tests the RAW tail length, and the five cases in that
    #           archive had tails of ~10 000 that were perfectly flat. The NaNs
    #           reached `_median`, which sorts them last, so the median index
    #           pointed low into the real values and the printed number was
    #           biased downwards without any error being raised.
    for r in records
        isempty(r.active_traj) && continue
        view = RecordView(r)
        ki   = inactivity_index(view)          # first PERMANENTLY inactive iteration
        fi   = findfirst(!, r.active_traj)     # first inactive of any kind
        tail = inactive_tail(r)
        short = length(tail) < MIN_TAIL
        ord   = short ? NaN : estimate_convergence_order(tail)
        # Why this run contributes no order, or "" when it does. Checked in this
        # order so the reported reason is the most specific one available.
        reason = ki === nothing              ? "never"  :
                 r.status !== :first_order   ? "unconv" :
                 short                       ? "short"  :
                 isnan(ord)                  ? "flat"   : ""
        if     reason == "never";  n_never[r.config]  += 1
        elseif reason == "unconv"; n_unconv[r.config] += 1
        elseif reason == "short";  n_short[r.config]  += 1
        elseif reason == "flat";   n_flat[r.config]   += 1
        else
            push!(orders[r.config], ord); push!(k_stars[r.config], Float64(ki))
        end
        @printf(io, "%-16s %-16s %8s %8d %12s %12s %8d %10s\n",
                r.problem, r.config, string(r.status), r.iterations,
                ki === nothing ? "never" : string(ki),
                fi === nothing ? "never" : string(fi - 1),
                length(tail),
                isempty(reason) ? @sprintf("%.3f", ord) : reason)
    end
    println(io)
    println(io, "k* is the first iteration after which the constraint never binds again,")
    println(io, "and is the quantity the mechanism controls. `1st inactive` is the first")
    println(io, "inactive iteration of any kind; the two differ whenever the region")
    println(io, "releases and binds again, and the order below is fitted from k*.")
    println(io)
    println(io, "The last column is the fitted order, or the reason there is none:")
    println(io, "  never   the constraint bound to the last iteration")
    println(io, "  unconv  the run did not reach :first_order, so there is no")
    println(io, "          asymptotic phase; a stalled run still has an inactive")
    println(io, "          tail and would otherwise be fitted a slope near 1")
    println(io, "  short   fewer than $MIN_TAIL inactive iterations, too few to fit")
    println(io, "  flat    the tail is long enough but degenerate: fewer than four")
    println(io, "          finite positive values, or zero variance in the regression")
    println(io, "All four are excluded from the medians and counted separately.")
    save_table(arch, "exp7_conv_order_summary.txt", String(take!(io)))

    io = IOBuffer()
    @printf(io, "%-16s %8s %8s %8s %8s %8s %14s %14s\n",
            "rule", "fitted", "unconv", "short", "flat", "never",
            "median order", "median k*")
    println(io, "-"^96)
    for (cname, _) in configs
        o = orders[cname]; ks = k_stars[cname]
        @printf(io, "%-16s %8d %8d %8d %8d %8d %14s %14s\n", cname, length(o),
                n_unconv[cname], n_short[cname], n_flat[cname], n_never[cname],
                isempty(o)  ? "--" : @sprintf("%.3f", _median(o)),
                isempty(ks) ? "--" : @sprintf("%.0f",  _median(ks)))
    end
    println(io)
    println(io, "The claim is that `median order` is the same across rules while")
    println(io, "`median k*` is not. The four exclusion columns are reported because")
    println(io, "each is a way the median could be biased, and because they are not")
    println(io, "uniform across rules: a median over the runs that reached inactivity")
    println(io, "early favours the mechanisms that do so, and one taken over runs that")
    println(io, "never converged favours nothing at all -- it measures stalling.")
    println(io)
    println(io, "`fitted` counts only runs that reached :first_order with a")
    println(io, "well-conditioned tail, so `median order` is now an order and not a")
    println(io, "blend of orders with stall artefacts.")
    save_table(arch, "exp7_order_by_rule.txt", String(take!(io)))

    # `all`, not `any`: the boxplot recipe takes a quantile of every series, so a
    # single empty one aborts the whole figure. `any` let that through and the
    # experiment died after both tables were already written, which is the worst
    # place to fail -- the archive looked complete and had no figures.
    keep = [i for i in eachindex(configs) if !isempty(orders[configs[i][1]])]
    if !isempty(keep)
        data = [orders[configs[i][1]] for i in keep]
        plt = boxplot(reshape(labels[keep], 1, :), data; legend = false,
                      ylabel = "estimated convergence order", xrotation = 45)
        hline!(plt, [1.0, 2.0]; ls = :dash, c = :black)
        savefig_archived(arch, "exp7_conv_order_boxplot.pdf", plt)
    end
    length(keep) == length(configs) ||
        @info "boxplot omits configurations with no fitted order" dropped =
              [configs[i][1] for i in eachindex(configs) if !(i in keep)]

    plt = plot(; xlabel = "k*, first permanently inactive iteration",
                 ylabel = "estimated order", legend = :best)
    for (cname, _) in configs
        o = orders[cname]; fi = k_stars[cname]
        n = min(length(o), length(fi))
        n == 0 && continue
        scatter!(plt, fi[1:n], o[1:n]; label = cname, ms = 5)
    end
    hline!(plt, [1.0, 2.0]; ls = :dash, c = :black, label = "")
    savefig_archived(arch, "exp7_conv_order_scatter.pdf", plt)

    finalize_archive(arch; notes = """
        Observed local convergence order, estimated on the INACTIVE tail only.

        exp7_order_by_rule.txt reports two columns. If the survey's account
        holds, the median order should be roughly equal across rules -- it is a
        property of the model, not of the radius -- while the median first
        inactive iteration should differ substantially, since that is what the
        radius rule controls.

        Rows with `never` in the first-inactive column are configurations whose
        constraint bound to the end: no order is estimated, and that is the
        result rather than a gap in it.

        Four exclusions apply to the medians, and two of them are new. A run is
        fitted only if it reached :first_order (`unconv` otherwise) with a tail
        that is long enough (`short`) and well conditioned (`flat`), and whose
        constraint eventually released (`never`). All four counts are printed.

        The `unconv` exclusion matters most and is not cosmetic. A :stalled run
        has an inactive tail, because its steps are tiny and interior, so it
        passes every structural guard and yields a log-log slope near 1 -- the
        signature of a run going nowhere rather than a convergence order. On the
        2026-08-22 archive, 100 of 459 otherwise-fitted runs had not converged,
        and including them moved the pooled median from 1.4218 to 1.3080. Their
        share ranged from 20% to 31% across rules, so the contamination was
        correlated with the axis under test and made the mechanisms look more
        different than they are.
        """)
end

if abspath(PROGRAM_FILE) == @__FILE__
    CVRate()
end