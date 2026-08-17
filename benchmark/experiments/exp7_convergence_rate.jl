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
    save_config(arch; rules = RULES, params = SOLVER_PARAMS,
                problem_selection = PROBLEM_SELECTION,
                extra = Dict("experiment" => "exp7_convergence_rate"))

    problems = default_problems()
    configs  = rule_configs()
    # A tight tolerance is essential: the asymptotic regime is what is measured.
    params = TRParams(η1 = SOLVER_PARAMS.η1, η2 = SOLVER_PARAMS.η2,
                      Δ0 = SOLVER_PARAMS.Δ0,
                      max_iterations = SOLVER_PARAMS.max_iterations,
                      tol = 1e-10, max_time = SOLVER_PARAMS.max_time)

    records = run_experiment(problems, configs; params = params,
                             trace = true, archive = arch)
    labels  = [c[1] for c in configs]

    orders = Dict(c[1] => Float64[] for c in configs)
    k_stars = Dict(c[1] => Float64[] for c in configs)
    n_short = Dict(c[1] => 0 for c in configs)
    n_never = Dict(c[1] => 0 for c in configs)
    io = IOBuffer()
    @printf(io, "%-16s %-16s %8s %8s %12s %12s %8s %10s\n",
            "problem", "rule", "status", "iters", "k* (perm.)", "1st inactive",
            "tail", "order")
    println(io, "-"^96)
    # No `solved` filter: a run whose constraint never stops binding has no
    # asymptotic phase to fit, and saying so is the result for that
    # configuration rather than a gap in the table.
    for r in records
        isempty(r.active_traj) && continue
        view = RecordView(r)
        ki   = inactivity_index(view)          # first PERMANENTLY inactive iteration
        fi   = findfirst(!, r.active_traj)     # first inactive of any kind
        tail = inactive_tail(r)
        short = length(tail) < MIN_TAIL
        ord  = short ? NaN : estimate_convergence_order(tail)
        if ki === nothing
            n_never[r.config] += 1
        elseif short
            n_short[r.config] += 1
        else
            push!(orders[r.config], ord); push!(k_stars[r.config], Float64(ki))
        end
        @printf(io, "%-16s %-16s %8s %8d %12s %12s %8d %10s\n",
                r.problem, r.config, string(r.status), r.iterations,
                ki === nothing ? "never" : string(ki),
                fi === nothing ? "never" : string(fi - 1),
                length(tail),
                isnan(ord) ? (ki === nothing ? "never" : "short") :
                             @sprintf("%.3f", ord))
    end
    println(io)
    println(io, "k* is the first iteration after which the constraint never binds again,")
    println(io, "and is the quantity the mechanism controls. `1st inactive` is the first")
    println(io, "inactive iteration of any kind; the two differ whenever the region")
    println(io, "releases and binds again, and the order below is fitted from k*.")
    println(io, "`short` marks a run with fewer than $MIN_TAIL inactive iterations: too")
    println(io, "few to fit, and dropped from the medians rather than reported as a")
    println(io, "number. `never` marks a constraint that bound to the end.")
    save_table(arch, "exp7_conv_order_summary.txt", String(take!(io)))

    io = IOBuffer()
    @printf(io, "%-16s %8s %8s %8s %14s %14s\n",
            "rule", "fitted", "short", "never", "median order", "median k*")
    println(io, "-"^72)
    for (cname, _) in configs
        o = orders[cname]; ks = k_stars[cname]
        @printf(io, "%-16s %8d %8d %8d %14s %14s\n", cname, length(o),
                n_short[cname], n_never[cname],
                isempty(o)  ? "--" : @sprintf("%.3f", _median(o)),
                isempty(ks) ? "--" : @sprintf("%.0f",  _median(ks)))
    end
    println(io)
    println(io, "The claim is that `median order` is the same across rules while")
    println(io, "`median k*` is not. `short` and `never` are reported because a")
    println(io, "median taken over the runs that reached inactivity early is biased")
    println(io, "towards the mechanisms that do so.")
    save_table(arch, "exp7_order_by_rule.txt", String(take!(io)))

    data = [orders[c[1]] for c in configs]
    if any(!isempty, data)
        plt = boxplot(reshape(labels, 1, :), data; legend = false,
                      ylabel = "estimated convergence order", xrotation = 45)
        hline!(plt, [1.0, 2.0]; ls = :dash, c = :black)
        savefig_archived(arch, "exp7_conv_order_boxplot.pdf", plt)
    end

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
        """)
end

if abspath(PROGRAM_FILE) == @__FILE__
    CVRate()
end