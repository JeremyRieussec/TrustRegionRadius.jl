# =============================================================================
# benchmark/harness.jl
#
# Shared machinery for the experiment scripts: problem sets, the run loop, and
# the conversion from results to the metric matrices that profiles consume.
#
# Every experiment file includes this and `archive.jl`, then does its own thing.
# =============================================================================

using TrustRegionRadius
using NLPModels
using Printf
using JLD2

# -----------------------------------------------------------------------------
# Problem sets
# -----------------------------------------------------------------------------

"""
    cutest_problems(; min_var, max_var, max_con = 0, limit = nothing) -> Vector

CUTEst problems as **thunks**, each returning a fresh `CUTEstModel`.

Thunks rather than models because each `CUTEstModel` holds a handle to a
compiled SIF problem and only one may be live at a time; the runner opens and
finalises them one by one.

Returns an empty vector with a warning if `CUTEst` is not installed, so an
experiment file can degrade to its analytic problem set rather than failing.
"""
function cutest_problems(; min_var::Int = 2, max_var::Int = 500,
                           max_con::Int = 0, limit = nothing)
    if !_has_cutest()
        @warn "CUTEst.jl not available -- skipping the CUTEst problem set"
        return Tuple{String, Function}[]
    end
    CUTEst = Base.require(Base.PkgId(
        Base.UUID("1b53aba6-35b6-5f92-a507-53c67d53f819"), "CUTEst"))
    names = try
        CUTEst.select_sif_problems(min_var = min_var, max_var = max_var,
                                   max_con = max_con, only_free_var = true)
    catch
        CUTEst.select(min_var = min_var, max_var = max_var,
                      max_con = max_con, only_free_var = true)
    end
    sort!(names)
    limit === nothing || (names = names[1:min(limit, length(names))])
    @info "CUTEst: $(length(names)) problems selected"
    return [(nm, () -> CUTEst.CUTEstModel(nm)) for nm in names]
end

function _has_cutest()
    try
        Base.require(Base.PkgId(
            Base.UUID("1b53aba6-35b6-5f92-a507-53c67d53f819"), "CUTEst"))
        true
    catch
        false
    end
end

"""
    analytic_problems() -> Vector{Tuple{String, Function}}

A small set of classical unconstrained problems as `ADNLPModel` thunks. Used
when CUTEst is unavailable, and for the diagnostic experiments where a problem
whose critical points are known exactly matters more than breadth.
"""
function analytic_problems()
    P = Tuple{String, Function, Vector{Float64}}[
        ("ROSENBR",   x -> 100(x[2]-x[1]^2)^2 + (1-x[1])^2,            [-1.2, 1.0]),
        ("BEALE",     x -> (1.5-x[1]+x[1]*x[2])^2 + (2.25-x[1]+x[1]*x[2]^2)^2 +
                           (2.625-x[1]+x[1]*x[2]^3)^2,                 [ 1.0, 1.0]),
        ("HIMMELBLAU",x -> (x[1]^2+x[2]-11)^2 + (x[1]+x[2]^2-7)^2,     [ 0.0, 0.0]),
        ("POWELLSG",  x -> (x[1]+10x[2])^2 + 5(x[3]-x[4])^2 +
                           (x[2]-2x[3])^4 + 10(x[1]-x[4])^4,     [3.0,-1.0,0.0,1.0]),
        ("WOOD",      x -> 100(x[2]-x[1]^2)^2 + (1-x[1])^2 +
                           90(x[4]-x[3]^2)^2 + (1-x[3])^2 +
                           10.1*((x[2]-1)^2+(x[4]-1)^2) +
                           19.8*(x[2]-1)*(x[4]-1),         [-3.0,-1.0,-3.0,-1.0]),
        ("ILLCOND",   x -> x[1]^2 + 1000x[2]^2 + 0.01x[1]*x[2],        [ 1.0, 1.0]),
        ("EXTROSEN3", x -> 100(x[2]-x[1]^2)^2 + (1-x[1])^2 +
                           100(x[3]-x[2]^2)^2 + (1-x[2])^2,      [-1.2, 1.0,-1.2]),
        ("TRIGQUAD",  x -> cos(x[1])*cos(x[2]) + 0.1(x[1]^2+x[2]^2),   [ 1.0, 0.5]),
    ]
    return [(nm, () -> ADNLPModel(f, x0, name = nm)) for (nm, f, x0) in P]
end

# -----------------------------------------------------------------------------
# The run loop
# -----------------------------------------------------------------------------

"""
    RunRecord

One (problem, configuration) result. Flat and concrete so a vector of these can
be reduced into a metric matrix without further dispatch.
"""
struct RunRecord
    problem::String
    config::String
    n::Int
    status::Symbol
    iterations::Int
    f_evals::Int
    g_evals::Int
    h_evals::Int
    final_grad::Float64
    final_obj::Float64
    solve_time::Float64
    delta_traj::Vector{Float64}
    grad_traj::Vector{Float64}
    obj_traj::Vector{Float64}
    ratio_traj::Vector{Float64}
    active_traj::Vector{Bool}
end

solved(r::RunRecord) = r.status === :first_order

"""
    run_experiment(problems, configs; params, trace, archive, verbose) -> Vector{RunRecord}

Run every configuration on every problem.

`problems` is a vector of `(name, thunk)`; `configs` a vector of
`(name, factory)` where `factory()` returns a `NamedTuple` with fields `rule`,
and optionally `model` and `subsolver`. Factories, not instances: rules and
quasi-Newton models carry mutable state and each run must start clean.

If `archive` is given, each record is also written to `data/` as JLD2 so the
run can be re-analysed without recomputing.
"""
function run_experiment(problems, configs;
                        params::TRParams = TRParams(),
                        trace::Bool = true,
                        archive = nothing,
                        verbose::Bool = true)
    records = RunRecord[]

    if verbose
        @printf("%-20s", "problem")
        for (nm, _) in configs
            @printf("%14s", first(nm, 14))
        end
        println()
        println("-"^(20 + 14length(configs)))
    end

    for (pname, mk) in problems
        nlp = mk()
        row = String[]
        for (cname, factory) in configs
            cfg = factory()
            rule  = cfg.rule
            model = hasproperty(cfg, :model)     ? cfg.model     : ExactHessian()
            sub   = hasproperty(cfg, :subsolver) ? cfg.subsolver : SteihaugCG()
            rec = try
                NLPModels.reset!(nlp)
                st = tr_solve(nlp; rule = rule, model = model,
                              subsolver = sub, params = params, trace = trace)
                ss = st.solver_specific
                RunRecord(pname, cname, nlp.meta.nvar, st.status, st.iter,
                          neval_obj(nlp), neval_grad(nlp), neval_hprod(nlp),
                          Float64(st.dual_feas), Float64(st.objective),
                          st.elapsed_time,
                          get(ss, :delta_trajectory,  Float64[]),
                          get(ss, :grad_trajectory,   Float64[]),
                          get(ss, :obj_trajectory,    Float64[]),
                          get(ss, :ratio_trajectory,  Float64[]),
                          get(ss, :active_trajectory, Bool[]))
            catch err
                err isa InterruptException && rethrow()
                verbose && @warn "run failed" problem=pname config=cname err
                RunRecord(pname, cname, nlp.meta.nvar, :exception, 0, 0, 0, 0,
                          NaN, NaN, 0.0, Float64[], Float64[], Float64[],
                          Float64[], Bool[])
            end
            push!(records, rec)
            push!(row, solved(rec) ? @sprintf("%14d", rec.iterations) :
                                     @sprintf("%14s", string(rec.status)))
            archive === nothing || save_data(archive, "$(pname)_$(cname).jld2";
                problem_name = rec.problem, rule_name = rec.config, n = rec.n,
                status = rec.status, iterations = rec.iterations,
                f_evals = rec.f_evals, g_evals = rec.g_evals,
                final_grad_norm = rec.final_grad, final_obj = rec.final_obj,
                solve_time = rec.solve_time,
                delta_trajectory = rec.delta_traj,
                grad_norm_trajectory = rec.grad_traj,
                obj_trajectory = rec.obj_traj,
                ratio_trajectory = rec.ratio_traj,
                active_trajectory = rec.active_traj)
        end
        verbose && (@printf("%-20s", first(pname, 20)); println(join(row)))
        finalize(nlp)
    end
    return records
end

# -----------------------------------------------------------------------------
# Records → matrices
# -----------------------------------------------------------------------------

"""
    metric_matrix(records, problems, configs, metric; failure = Inf) -> Matrix{Float64}

Assemble the `(n_problems × n_configs)` matrix a profile consumes.

`metric` is `:iter`, `:obj`, `:grad`, `:time`, or a function of a `RunRecord`.
Unsolved entries get `failure`, which is what makes the right-hand asymptote of
a performance profile read as reliability.
"""
function metric_matrix(records::Vector{RunRecord}, problems, configs,
                       metric = :iter; failure::Float64 = Inf)
    pnames = [p[1] for p in problems]
    cnames = [c[1] for c in configs]
    idx = Dict((r.problem, r.config) => r for r in records)

    f = metric isa Function ? metric :
        metric === :iter ? (r -> Float64(r.iterations)) :
        metric === :obj  ? (r -> Float64(r.f_evals))    :
        metric === :grad ? (r -> Float64(r.g_evals))    :
        metric === :time ? (r -> r.solve_time)          :
        throw(ArgumentError("metric_matrix: unknown metric $metric"))

    M = fill(failure, length(pnames), length(cnames))
    for (i, p) in enumerate(pnames), (j, c) in enumerate(cnames)
        r = get(idx, (p, c), nothing)
        r !== nothing && solved(r) && (M[i, j] = f(r))
    end
    return M
end

"""
    success_table(records, problems, configs) -> String

Per-configuration solved counts, median and mean iterations, and mean tail
activity — the fraction of the last decile of iterations on which the
trust-region constraint bound.

That last column is the one worth reading. Two configurations can have
identical iteration counts and identical first-order behaviour while one keeps
the constraint permanently active and the other does not, and only the activity
column distinguishes them.
"""
function success_table(records::Vector{RunRecord}, problems, configs)
    io = IOBuffer()
    @printf(io, "%-22s %10s %10s %10s %12s\n",
            "configuration", "solved", "median", "mean", "tail active")
    println(io, "-"^70)
    npb = length(problems)
    for (cname, _) in configs
        rs = filter(r -> r.config == cname, records)
        ok = filter(solved, rs)
        iters = Float64[r.iterations for r in ok]
        acts = Float64[]
        for r in ok
            isempty(r.active_traj) && continue
            k0 = max(1, floor(Int, 0.9 * length(r.active_traj)))
            push!(acts, count(r.active_traj[k0:end]) / length(r.active_traj[k0:end]))
        end
        @printf(io, "%-22s %5d/%-4d %10s %10s %12s\n", cname, length(ok), npb,
                isempty(iters) ? "--" : @sprintf("%.1f", _median(iters)),
                isempty(iters) ? "--" : @sprintf("%.1f", sum(iters)/length(iters)),
                isempty(acts)  ? "--" : @sprintf("%.3f", sum(acts)/length(acts)))
    end
    return String(take!(io))
end

function _median(v::AbstractVector)
    s = sort(collect(v)); n = length(s)
    n == 0 && return NaN
    isodd(n) ? s[(n+1)÷2] : (s[n÷2] + s[n÷2+1]) / 2
end

"""
    estimate_convergence_order(grad_traj; tail = 20) -> Float64

Least-squares slope of `log‖g_{k+1}‖` on `log‖g_k‖` over the last `tail`
iterations: 1 for linear convergence, 2 for quadratic.

!!! warning "Condition on inactivity"
    The estimate is meaningful only over iterations where the trust-region
    constraint is *inactive*. Including active iterations mixes a linear phase
    with the asymptotic one and returns an order between the two, which is an
    artefact rather than a finding. Trim the trajectory with the activity flag
    before calling this.
"""
function estimate_convergence_order(grad_traj::Vector{Float64}; tail::Int = 20)
    g = filter(x -> isfinite(x) && x > 0, grad_traj)
    length(g) < 4 && return NaN
    n = min(tail, length(g) - 1)
    x = log.(g[end-n:end-1])
    y = log.(g[end-n+1:end])
    x̄, ȳ = sum(x)/length(x), sum(y)/length(y)
    den = sum((x .- x̄).^2)
    den <= 0 && return NaN
    return sum((x .- x̄) .* (y .- ȳ)) / den
end
