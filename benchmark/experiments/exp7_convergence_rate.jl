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

using Plots
include(joinpath(@__DIR__, "..", "archive.jl"))
include(joinpath(@__DIR__, "..", "harness.jl"))
include(joinpath(@__DIR__, "..", "config.jl"))

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

function main()
    arch = ExperimentArchive(tag = "convergence_rate")
    save_config(arch; rules = RULES, params = SOLVER_PARAMS,
                problem_selection = PROBLEM_SELECTION,
                extra = Dict("experiment" => "exp7_convergence_rate"))

    problems = default_problems()
    configs  = rule_configs()
    # A tight tolerance is essential: the asymptotic regime is what is measured.
    params = TRParams(η₁ = SOLVER_PARAMS.η₁, η₂ = SOLVER_PARAMS.η₂,
                      Δ₀ = SOLVER_PARAMS.Δ₀,
                      max_iterations = SOLVER_PARAMS.max_iterations,
                      tol = 1e-10, max_time = SOLVER_PARAMS.max_time)

    records = run_experiment(problems, configs; params = params,
                             trace = true, archive = arch)
    labels  = [c[1] for c in configs]

    orders = Dict(c[1] => Float64[] for c in configs)
    first_inactive = Dict(c[1] => Float64[] for c in configs)
    io = IOBuffer()
    @printf(io, "%-16s %-16s %10s %14s %14s\n",
            "problem", "rule", "iters", "1st inactive", "order")
    println(io, "-"^76)
    for r in filter(solved, records)
        tail = inactive_tail(r)
        ord  = length(tail) >= 4 ? estimate_convergence_order(tail) : NaN
        fi   = findfirst(!, r.active_traj)
        isnan(ord) || push!(orders[r.config], ord)
        fi === nothing || push!(first_inactive[r.config], Float64(fi))
        @printf(io, "%-16s %-16s %10d %14s %14s\n", r.problem, r.config, r.iterations,
                fi === nothing ? "never" : string(fi),
                isnan(ord) ? "--" : @sprintf("%.3f", ord))
    end
    save_table(arch, "exp7_conv_order_summary.txt", String(take!(io)))

    io = IOBuffer()
    @printf(io, "%-16s %10s %14s %18s\n",
            "rule", "n", "median order", "median 1st inactive")
    println(io, "-"^62)
    for (cname, _) in configs
        o = orders[cname]; fi = first_inactive[cname]
        @printf(io, "%-16s %10d %14s %18s\n", cname, length(o),
                isempty(o)  ? "--" : @sprintf("%.3f", _median(o)),
                isempty(fi) ? "--" : @sprintf("%.0f",  _median(fi)))
    end
    save_table(arch, "exp7_order_by_rule.txt", String(take!(io)))

    data = [orders[c[1]] for c in configs]
    if any(!isempty, data)
        plt = boxplot(reshape(labels, 1, :), data; legend = false,
                      ylabel = "estimated convergence order", xrotation = 45)
        hline!(plt, [1.0, 2.0]; ls = :dash, c = :black)
        savefig_archived(arch, "exp7_conv_order_boxplot.pdf", plt)
    end

    plt = plot(; xlabel = "first inactive iteration", ylabel = "estimated order",
                 legend = :best)
    for (cname, _) in configs
        o = orders[cname]; fi = first_inactive[cname]
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

abspath(PROGRAM_FILE) == @__FILE__ && main()
