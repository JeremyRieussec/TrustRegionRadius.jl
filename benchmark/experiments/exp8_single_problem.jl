# =============================================================================
# benchmark/experiments/exp8_single_problem.jl
#
# EXPERIMENT 8 -- one problem, every mechanism, per-iteration diagnostics.
#
# Where exp2 traces a handful of problems to show the family separation, this
# script takes a SINGLE problem and looks at it closely: Δ_k, ‖g_k‖ and ρ_k
# against k for every rule, plus the countdown of remaining active iterations.
#
# The countdown is the point of the experiment. Define
#
#     R_k = #{ j > k : iteration j had ‖s_j‖ = Δ_j },      k = 0, …, K,
#
# so R_0 is the total number of active iterations and R decreases by one at
# every active iteration and is flat on inactive ones. R_k = 0 means every
# iteration after k was inactive, i.e. the trust-region constraint has stopped
# binding for good, so the first k at which the curve reaches zero is exactly
# the onset of the inactive regime -- the quantity Part II's inactivity theorems
# are about. A curve that only reaches zero at k = K has no inactive tail at
# all: the constraint was still binding on the last iteration and the method
# never entered the asymptotic regime.
#
# Reading the two together is what makes the figure worth having. The rules that
# never go inactive still drive ‖g_k‖ → 0 with healthy ρ_k, so the top two
# panels look fine for every mechanism and only the countdown distinguishes
# them.
#
# Run it the way every other experiment here is run -- through
# initialisation.jl, which loads the package, the harness and config.jl exactly
# once. This file deliberately carries no `using` and no `include` of its own:
# it used to include archive.jl, harness.jl and config.jl itself, and re-running
# config.jl over an already-loaded session raised
# `invalid redefinition of constant Main.DEFAULT_MODEL`, which is why the suite
# had it commented out.
#
#   julia --project=benchmark -e 'include("benchmark/initialisation.jl");
#                                 single_problem_experiment("ROSENBR")'
#
# The problem may also come from TRR_PROBLEM.
# =============================================================================

"Problem traced when none is given on the command line or in TRR_PROBLEM."
const DEFAULT_TRACE_PROBLEM = "ROSENBR"

"Clipping window for ρ_k in the plot; ρ is unbounded below and −Inf is possible."
const RHO_CLIP = (-1.0, 2.0)

"Floor applied before a log scale, so that an exactly-zero radius still plots."
const LOG_FLOOR = 1e-300

# -----------------------------------------------------------------------------
# Problem selection
# -----------------------------------------------------------------------------

"""
    chosen_problem(args) -> String

The problem to trace: first command-line argument, else `TRR_PROBLEM`, else
[`DEFAULT_TRACE_PROBLEM`].

`args` is passed in rather than read from `ARGS` directly because `run_all.jl`
includes this file with its own arguments (experiment ids), which must not be
mistaken for a problem name.
"""
function chosen_problem(args)
    isempty(args) || return String(args[1])
    env = get(ENV, "TRR_PROBLEM", "")
    isempty(env) ? DEFAULT_TRACE_PROBLEM : env
end

"""
    select_problem(name) -> (name, thunk)

Locate one problem by name: the analytic set first, then CUTEst. Returns the
`(name, thunk)` pair the harness expects.

The analytic set takes precedence because its critical points are known
exactly, which is what a single-problem diagnostic wants; a CUTEst name that is
not in it is passed straight to `CUTEstModel`, which reports its own error if
the name is unknown.
"""
function select_problem(name::AbstractString)
    for (nm, mk) in analytic_problems()
        nm == name && return (nm, mk)
    end
    if HAS_CUTEST
        @info "«$name» is not in the analytic set; trying CUTEst"
        return (String(name), () -> CUTEstModel(String(name)))
    end
    error("""Unknown problem «$name», and CUTEst is unavailable.
             Analytic problems: $(join([p[1] for p in analytic_problems()], ", "))""")
end

# -----------------------------------------------------------------------------
# The countdown
# -----------------------------------------------------------------------------

"""
    remaining_active(active) -> Vector{Int}

`R_k = #{ j > k : active[j] }` for `k = 0, …, K`, returned as a vector of length
`K+1` whose entry `k+1` is `R_k`.

`R_0` is the total number of active iterations; `R` decreases by one at every
active iteration, is flat on inactive ones, and `R_K = 0` identically. It is a
suffix sum rather than a prefix sum on purpose: the question "is the constraint
done binding?" is a question about the future of the run, not its past.
"""
function remaining_active(active::AbstractVector{Bool})
    K = length(active)
    R = Vector{Int}(undef, K + 1)
    R[K + 1] = 0
    for k in K:-1:1
        R[k] = R[k + 1] + (active[k] ? 1 : 0)
    end
    return R
end

"""
    inactivity_onset(active) -> (k_star, tail)

`k_star` is the index of the last active iteration and `tail = K - k_star` the
length of the inactive run that follows it.

- `tail == 0`: the constraint was active on the final iteration. There is no
  inactive tail; the countdown reaches zero only at `k = K`, which is an
  artefact of the run ending rather than a change of regime.
- `k_star == 0`: the constraint never bound at all.

Since the run stops at the first-order tolerance rather than at infinity, a
positive `tail` is evidence of inactivity over the iterations observed and not
a proof of eventual inactivity. It is the right observable nonetheless: the
theorems say the constraint stops binding at some finite index, and this is that
index when it exists.
"""
function inactivity_onset(active::AbstractVector{Bool})
    K = length(active)
    k = K
    while k >= 1 && !active[k]
        k -= 1
    end
    return k, K - k
end

"Label suffix reporting the onset, or its absence."
function onset_label(active::AbstractVector{Bool})
    isempty(active) && return ""
    k_star, tail = inactivity_onset(active)
    tail == 0 && return "  [never]"
    k_star == 0 && return "  [always inactive]"
    return "  [k*=$(k_star)]"
end

# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------

"Δ_k against k, log scale. Iteration axis is 0:K, since Δ0 precedes iteration 1."
function plot_delta(rs, pname)
    plt = plot(; xlabel = "iteration k", ylabel = "Δ_k", yscale = :log10,
                 title = "$pname: trust-region radius", legend = :best, lw = 2)
    for r in rs
        isempty(r.delta_traj) && continue
        plot!(plt, 0:(length(r.delta_traj) - 1), max.(r.delta_traj, LOG_FLOOR);
              label = r.config)
    end
    return plt
end

"‖g_k‖ against k, log scale, with the stopping tolerance marked."
function plot_grad(rs, pname, tol)
    plt = plot(; xlabel = "iteration k", ylabel = "‖g_k‖", yscale = :log10,
                 title = "$pname: gradient norm", legend = :best, lw = 2)
    for r in rs
        isempty(r.grad_traj) && continue
        plot!(plt, 0:(length(r.grad_traj) - 1), max.(r.grad_traj, LOG_FLOOR);
              label = r.config)
    end
    hline!(plt, [tol]; ls = :dot, c = :black, label = "tol")
    return plt
end

"""
    plot_ratio(rs, pname, params)

ρ_k against k, clipped to [`RHO_CLIP`], with η, η1 and η2 marked.

The three lines are worth drawing separately now that acceptance is decoupled
from scaling: points in [η, η1) are accepted steps that still contract the
radius, a band that does not exist in the coupled formulation.
"""
function plot_ratio(rs, pname, params)
    lo, hi = RHO_CLIP
    plt = plot(; xlabel = "iteration k", ylabel = "ρ_k", ylims = (lo - 0.1, hi + 0.1),
                 title = "$pname: ratio (clipped to [$lo, $hi])",
                 legend = :best, lw = 1.2)
    for r in rs
        isempty(r.ratio_traj) && continue
        plot!(plt, 1:length(r.ratio_traj), clamp.(r.ratio_traj, lo, hi);
              label = r.config)
    end
    hline!(plt, [Float64(params.η)];  ls = :solid, c = :black, label = "η (accept)")
    if params.η1 != params.η
        hline!(plt, [Float64(params.η1)]; ls = :dash, c = :black, label = "η1")
    end
    hline!(plt, [Float64(params.η2)]; ls = :dashdot, c = :black, label = "η2")
    return plt
end

"""
    plot_countdown(rs, pname; normalise = false)

The countdown `R_k` of remaining active iterations. A staircase, so
`:steppost`: `R` changes only at the iterations themselves.

Plotted over `k = 0, …, K-1` rather than `0, …, K`. `R_K = 0` holds for every
run, because no iteration follows the last one, so including it would make every
curve touch zero and destroy the distinction the figure exists to show. Stopping
one short, a curve ends at zero exactly when the final iteration was inactive
and ends at one or more when the constraint was still binding — so "reaches
zero" and "never reaches zero" are both visible, and mean what they say.

With `normalise = true` the axes become `k/K` and `R_k/R_0`, which makes runs of
very different length comparable at the cost of hiding how long each one was.
"""
function plot_countdown(rs, pname; normalise::Bool = false)
    plt = plot(; xlabel = normalise ? "iteration fraction k/K" : "iteration k",
                 ylabel = normalise ? "remaining active / total" :
                                      "remaining active iterations",
                 title = "$pname: active iterations remaining after k",
                 legend = :best, lw = 2)
    for r in rs
        isempty(r.active_traj) && continue
        K = length(r.active_traj)
        R = remaining_active(r.active_traj)[1:K]      # drop the trivial R_K = 0
        xs = normalise ? (0:(K - 1)) ./ max(K, 1) : collect(0:(K - 1))
        ys = normalise ? R ./ max(R[1], 1)          : R
        plot!(plt, xs, ys; label = r.config * onset_label(r.active_traj),
              seriestype = :steppost)

        # Mark the onset: the first k at which the countdown reaches zero.
        k_star, tail = inactivity_onset(r.active_traj)
        tail == 0 && continue
        scatter!(plt, [normalise ? k_star / max(K, 1) : k_star], [0.0];
                 label = "", ms = 5, mc = :black)
    end
    hline!(plt, [0.0]; ls = :dot, c = :black, label = "")
    return plt
end

# -----------------------------------------------------------------------------
# Table
# -----------------------------------------------------------------------------

function countdown_table(rs, K_budget)
    io = IOBuffer()
    @printf(io, "%-18s %10s %8s %10s %8s %8s %8s %10s %10s\n",
            "rule", "status", "iters", "final ‖g‖", "active", "k*",
            "tail", "Σ Δ_k", "Σ Δ_k²")
    println(io, "-"^108)
    for r in rs
        A = isempty(r.active_traj) ? 0 : count(r.active_traj)
        k_star, tail = isempty(r.active_traj) ? (0, 0) : inactivity_onset(r.active_traj)
        sd  = isempty(r.delta_traj) ? NaN : sum(r.delta_traj)
        sd2 = isempty(r.delta_traj) ? NaN : sum(abs2, r.delta_traj)
        @printf(io, "%-18s %10s %8d %10.3e %8d %8s %8d %10.4g %10.4g\n",
                r.config, string(r.status), r.iterations, r.final_grad, A,
                (tail == 0 && A > 0) ? "never" : string(k_star),
                tail, sd, sd2)
    end
    println(io)
    println(io, "active : iterations with ‖s_k‖ = Δ_k")
    println(io, "k*     : index of the last active iteration; \"never\" when that")
    println(io, "         index is the final iteration, so no inactive tail exists")
    println(io, "tail   : K − k*, the number of consecutive inactive iterations")
    println(io, "         at the end of the run (0 means the constraint never")
    println(io, "         stopped binding within the budget of $(K_budget))")
    return String(take!(io))
end

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

function single_problem_experiment(args = String[])
    pname_wanted = chosen_problem(args)
    problem = select_problem(pname_wanted)
    problems = [problem]

    arch = ExperimentArchive(tag = "single_$(problem[1])")
    save_config(arch; rules = RULES, params = SOLVER_PARAMS,
                extra = Dict("experiment" => "exp8_single_problem",
                             "problem" => problem[1]))

    configs = rule_configs()
    records = run_experiment(problems, configs;
                             params = SOLVER_PARAMS, trace = true, archive = arch)

    pname = problem[1]
    rs = filter(r -> r.problem == pname, records)
    if isempty(rs)
        @error "no records for $pname"
        return
    end

    # Runs that produced no trajectory at all (an immediate exception) cannot be
    # plotted, but they belong in the table: a rule that fails on this problem is
    # a result about the rule.
    plottable = filter(r -> !isempty(r.delta_traj), rs)
    unsolved  = filter(!solved, rs)
    isempty(unsolved) ||
        @info "did not reach :first_order" outcomes = [(r.config, r.status) for r in unsolved]

    save_table(arch, "exp8_$(pname)_countdown.txt",
               countdown_table(rs, SOLVER_PARAMS.max_iterations))

    if isempty(plottable)
        @error "no run on $pname produced a trajectory; nothing to plot" 
        finalize_archive(arch; notes = "No traced run on $pname; see the table.")
        return
    end

    savefig_archived(arch, "exp8_$(pname)_delta.pdf",
                     plot_delta(plottable, pname))
    savefig_archived(arch, "exp8_$(pname)_grad.pdf",
                     plot_grad(plottable, pname, Float64(SOLVER_PARAMS.tol)))
    savefig_archived(arch, "exp8_$(pname)_ratio.pdf",
                     plot_ratio(plottable, pname, SOLVER_PARAMS))
    savefig_archived(arch, "exp8_$(pname)_countdown.pdf",
                     plot_countdown(plottable, pname))
    savefig_archived(arch, "exp8_$(pname)_countdown_normalised.pdf",
                     plot_countdown(plottable, pname; normalise = true))

    # One stacked figure, which is the version that goes in the paper: the three
    # trajectories and the countdown share an iteration axis, so the onset of
    # inactivity can be read against what Δ, ‖g‖ and ρ were doing at the time.
    panel = plot(plot_delta(plottable, pname),
                 plot_grad(plottable, pname, Float64(SOLVER_PARAMS.tol)),
                 plot_ratio(plottable, pname, SOLVER_PARAMS),
                 plot_countdown(plottable, pname);
                 layout = (4, 1), size = (900, 1400), left_margin = 8Plots.mm)
    savefig_archived(arch, "exp8_$(pname)_panel.pdf", panel)

    finalize_archive(arch; notes = """
        Single-problem diagnostics on $pname, every mechanism, one figure per
        quantity plus a stacked panel.

        The countdown figure plots R_k, the number of active iterations still to
        come after iteration k. It starts at the total number of active
        iterations, steps down by one at each active iteration, and is flat on
        inactive ones. Where it reaches zero, the trust-region constraint has
        stopped binding for good and the step is thereafter the unconstrained
        model minimiser; the marked point k* is that onset. A curve that reaches
        zero only at the right-hand edge has no inactive tail: the constraint was
        still active on the final iteration, and the run never entered the
        regime in which the local rate is available. Those runs are labelled
        [never].

        Read the countdown against the first two panels. Every mechanism here
        drives ‖g_k‖ below the tolerance with ρ_k mostly above η2, so the
        gradient and ratio panels look much the same for all of them; the
        countdown is where they separate. That is the content of the claim that
        first-order diagnostics cannot see the distinction between mechanisms
        whose constraint goes inactive and mechanisms whose constraint does not.

        Because the run stops at ‖g‖ ≤ tol, a positive tail is evidence of
        inactivity over the iterations observed rather than a proof of eventual
        inactivity, and a tail of zero may mean the budget ran out before the
        onset. Both readings are visible in the table: compare tail against
        iters.
        """)
end

# The notebook and the suite call this by name; `main` was the old spelling and
# collided with experiments 9, 11 and 12, all of which have since been renamed to
# their own entry points.
single_problem_experiment(name::AbstractString) = single_problem_experiment([name])

# No `PROGRAM_FILE` guard: this file is not runnable on its own any more, by
# design. Direct execution would need the harness, and loading it from here is
# exactly the double-definition that kept experiment 8 out of the suite.
